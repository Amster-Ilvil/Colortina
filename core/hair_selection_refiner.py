"""Offline, supervised refinement of a rough hair selection.

This is an *editable selection* algorithm, not a hair-detection checkpoint.
A user marks a rough hair area and ideally paints foreground (+hair) and
background (-skin/clothes) strokes. A seeded GrabCut segmentation uses the
mc-v2 colourized preview plus the source ink as a boundary prior, but it
never overwrites arbitrary pixels. Only the returned mask is used by the
existing selection-edit pipeline. Background strokes always veto selection.

No network, pretrained model, or post-generation pixel manipulation here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class HairRefineInfo:
    rough_pixels: int
    selected_pixels: int
    added_pixels: int
    removed_pixels: int
    positive_pixels: int
    negative_pixels: int
    max_growth_px: int
    used_fallback: bool
    reason: str
    model_boundary_veto_pixels: int = 0


def _binary(mask, shape):
    if mask is None:
        return np.zeros(shape, np.uint8)
    data = np.asarray(mask)
    if data.ndim == 3:
        data = np.max(data, axis=2)
    if data.shape != shape:
        raise ValueError('selection, source and scribble dimensions must match')
    return (data > 0).astype(np.uint8)


def _ellipse(radius):
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r + 1, 2*r + 1))


def _components_touching_seeds(mask, seeds, rough):
    """Keep disconnected bangs if each part has seed or overlaps the rough hair."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    keep = np.zeros(mask.shape, np.uint8)
    for i in range(1, n):
        piece = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if np.any(seeds[piece]) or (area >= 18 and
            np.count_nonzero(rough[piece]) >= max(8, int(area * 0.55))):
            keep[piece] = 1
    return keep


def _skin_like_mask(color_bgr: np.ndarray) -> np.ndarray:
    """Conservative skin-like estimate from the colourized preview.

    This is only used to veto obvious forehead / eyebrow pockets inside a
    rough hair selection. It is intentionally conservative and should not be
    treated as a generic skin parser.
    """
    ycc = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    skin = ((cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (y >= 95))
    return skin.astype(np.uint8)


def _build_auto_negative(source_bgr: np.ndarray,
                         colorized_bgr: np.ndarray,
                         local_rough: np.ndarray,
                         local_pos: np.ndarray,
                         local_neg: np.ndarray) -> np.ndarray:
    """Infer conservative background seeds for forehead / eyebrow pockets.

    The failure case we want to suppress is: a rough hair lasso around bangs
    causes GrabCut to absorb bright forehead skin near the eyebrows.
    We only veto pixels that are simultaneously skin-like in the colourized
    preview, low-ink in the source page, and close to the *lower* boundary of
    the rough selection.
    """
    h, w = local_rough.shape
    if h < 8 or w < 8:
        return np.zeros((h, w), np.uint8)
    skin = _skin_like_mask(colorized_bgr).astype(bool)
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 170).astype(np.float32)
    ink_density = cv2.blur(ink, (11, 11))
    bottom = np.full(w, -1, np.int32)
    ys, xs = np.nonzero(local_rough)
    if xs.size == 0:
        return np.zeros((h, w), np.uint8)
    for x in range(w):
        cy = np.flatnonzero(local_rough[:, x] > 0)
        if cy.size:
            bottom[x] = int(cy.max())
    smooth_bottom = bottom.copy()
    for x in range(w):
        lo = max(0, x - 9)
        hi = min(w, x + 10)
        vals = bottom[lo:hi]
        vals = vals[vals >= 0]
        if vals.size:
            smooth_bottom[x] = int(np.median(vals))
    band = int(np.clip(min(24, max(8, h * 0.16)), 8, 24))
    yy = np.arange(h, dtype=np.int32)[:, None]
    lower_band = np.zeros((h, w), np.uint8)
    for x in range(w):
        b = int(smooth_bottom[x])
        if b >= 0:
            lower_band[:, x] = (yy[:, 0] >= max(0, b - band)).astype(np.uint8)
    auto_neg = (
        (local_rough > 0) & skin & (gray > 150) & (ink_density < 0.16) &
        (lower_band > 0) & (yy > int(h * 0.25))
    )
    if np.any(local_pos):
        # Do not fight explicit user-confirmed hair marks.
        auto_neg &= cv2.dilate(local_pos, _ellipse(4)) == 0
    if np.any(local_neg):
        auto_neg |= (local_neg > 0)
    return auto_neg.astype(np.uint8)


def _suppress_skin_pockets(chosen: np.ndarray,
                           source_bgr: np.ndarray,
                           colorized_bgr: np.ndarray,
                           local_pos: np.ndarray,
                           auto_neg: np.ndarray) -> np.ndarray:
    """Small post-pass that trims remaining forehead/eyebrow-like pockets.

    Only removes components that overlap the automatically inferred background
    seeds and are not explicitly protected by positive scribbles.
    """
    if not np.any(chosen) or not np.any(auto_neg):
        return chosen
    protected = cv2.dilate(local_pos, _ellipse(3)) if np.any(local_pos) else np.zeros_like(chosen)
    trim = (auto_neg > 0) & (protected == 0) & (chosen > 0)
    if not np.any(trim):
        return chosen
    out = chosen.copy()
    out[trim] = 0
    out = _components_touching_seeds(out, local_pos if np.any(local_pos) else chosen, chosen)
    if np.any(local_pos):
        out[local_pos > 0] = 1
    return out



def _generator_evidence_boundary_veto(
        chosen: np.ndarray, source_bgr: np.ndarray,
        generator_output_bgr: np.ndarray, positive: np.ndarray,
        negative: np.ndarray, *, band_px: int = 10,
        margin: float = 2.0) -> tuple[np.ndarray, int]:
    """Conservative *model-output* veto for skin-coloured hair-mask edges.

    This is NOT a pretrained semantic parser. It asks the already-run mc-v2
    generator whether high-confidence RGB evidence agrees with the user's
    hair and not-hair anchor colours. It can only DELETE low-confidence
    boundary pixels, never grow the mask or change confirmed hair scribbles.
    Ambiguous/model-monochrome cases keep the incoming selection exactly.
    """
    binary = (np.asarray(chosen) > 0).astype(np.uint8)
    pos = (np.asarray(positive) > 0)
    neg = (np.asarray(negative) > 0)
    if (np.count_nonzero(pos) < 12 or np.count_nonzero(neg) < 12
            or not np.any(binary)):
        return binary, 0
    original_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    confident_light = original_gray > 175
    pos_samples = pos & confident_light
    neg_samples = neg & confident_light
    if np.count_nonzero(pos_samples) < 12 or np.count_nonzero(neg_samples) < 12:
        return binary, 0
    lab = cv2.cvtColor(generator_output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour = lab[:, :, 1:3]
    hair = np.median(colour[pos_samples], axis=0)
    skin = np.median(colour[neg_samples], axis=0)
    # If mc-v2 gives indistinguishable neutral colours, its output cannot
    # provide semantic evidence and the current selection must be preserved.
    if float(np.linalg.norm(hair - skin)) < 6.0:
        return binary, 0
    dist_hair = np.linalg.norm(colour - hair, axis=2)
    dist_skin = np.linalg.norm(colour - skin, axis=2)
    inner_distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    near_boundary = inner_distance <= max(1, min(16, int(band_px)))
    positive_guard = cv2.dilate(pos.astype(np.uint8), _ellipse(2)) > 0
    # Only trust a skin-colour veto on the face-facing side of the instance:
    # it must be nearer an explicit not-hair seed than the known hair seed.
    # This prevents bright highlights near the top of the hairstyle being
    # erased simply because the model painted them nearly white.
    to_skin = cv2.distanceTransform((~neg).astype(np.uint8), cv2.DIST_L2, 5)
    to_hair = cv2.distanceTransform((~pos).astype(np.uint8), cv2.DIST_L2, 5)
    skin_side = to_skin < to_hair * 1.25
    veto = ((binary > 0) & near_boundary & (original_gray >= 170)
            & (dist_skin + float(margin) < dist_hair) & skin_side & ~positive_guard)
    count = int(np.count_nonzero(veto))
    if count == 0 or count > int(np.count_nonzero(binary) * 0.08):
        return binary, 0
    out = binary.copy()
    out[veto] = 0
    out[pos] = 1
    out[neg] = 0
    return out, count

def refine_hair_selection(source_bgr: np.ndarray,
                          colorized_bgr: np.ndarray,
                          rough_mask: np.ndarray,
                          *, positive: np.ndarray | None = None,
                          negative: np.ndarray | None = None,
                          grow_px: int = 22,
                          iterations: int = 5,
                          max_growth_ratio: float = 1.80,
                          ) -> tuple[np.ndarray, HairRefineInfo]:
    """Refine hair using coloured preview and user-provided strokes.

    `rough_mask` may be a rough lasso or rectangle. The mask can expand at
    most `grow_px` outside it, or to the explicitly positive painted strokes.
    Negative strokes and their 1px collar remain unselected. No user strokes
    means conservative eroded-core seeding (GUI explains positives work best).
    If the segmentation leaks across large regions, return the *original*
    rough selection with negative strokes removed (never auto-flood the page).
    """
    if source_bgr is None or colorized_bgr is None:
        raise ValueError('source and colorized preview are required')
    if source_bgr.shape != colorized_bgr.shape or source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
        raise ValueError('source and colourized preview must be same-size BGR images')
    h, w = source_bgr.shape[:2]
    shape = (h, w)
    rough = _binary(rough_mask, shape)
    pos = _binary(positive, shape)
    neg = _binary(negative, shape)
    neg = cv2.dilate(neg, _ellipse(1)) if np.any(neg) else neg
    pos[neg > 0] = 0  # explicit background always wins
    rough_count = int(np.count_nonzero(rough))
    if not rough_count:
        raise ValueError('rough selection is empty')
    # All positive marks must belong to this *one* instance.  A single valid
    # mark must not authorize other, remote marks on another character.
    mark_limit = cv2.dilate(rough, _ellipse(48))
    if np.any(pos & (mark_limit == 0)):
        raise ValueError('positive hair strokes are outside the rough hair selection')

    # Limit the algorithm to this page-local ROI. Extent expansion is bounded,
    # while explicit positive scribbles are allowed slightly beyond the lasso.
    grow_px = int(np.clip(grow_px, 0, 96))
    shell = cv2.dilate(rough, _ellipse(max(1, grow_px))) if grow_px else rough.copy()
    if np.any(pos):
        shell = np.maximum(shell, cv2.dilate(pos, _ellipse(3)))
    shell[neg > 0] = 0

    # Only a small box around the selection is passed into GMM/GrabCut.
    ys, xs = np.nonzero(shell)
    if not len(xs):
        raise ValueError('no editable region remains after negative strokes')
    pad = 5
    x0, x1 = max(0, int(xs.min())-pad), min(w, int(xs.max())+pad+1)
    y0, y1 = max(0, int(ys.min())-pad), min(h, int(ys.max())+pad+1)
    local = np.ascontiguousarray(colorized_bgr[y0:y1, x0:x1], dtype=np.uint8)
    local_rough = rough[y0:y1, x0:x1]
    local_pos = pos[y0:y1, x0:x1]
    local_neg = neg[y0:y1, x0:x1]
    local_shell = shell[y0:y1, x0:x1]
    labels = np.full(local_rough.shape, cv2.GC_BGD, np.uint8)
    labels[local_shell > 0] = cv2.GC_PR_BGD
    labels[local_rough > 0] = cv2.GC_PR_FGD

    auto_neg = _build_auto_negative(
        source_bgr[y0:y1, x0:x1], local, local_rough, local_pos, local_neg)

    if not np.any(local_pos):
        # Best-effort automatic seeding when there are no positive strokes.
        # Unlike rectangle-wide foreground, eroded cores should not force
        # face/background into the foreground for narrow selections.
        dist = cv2.distanceTransform(local_rough, cv2.DIST_L2, 3)
        core = (dist >= max(2.5, min(10., float(dist.max()) * 0.50)))
        if int(np.count_nonzero(core)) < 10:
            yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            core[yy,xx] = True
        local_pos = core.astype(np.uint8)

    # Force only deliberate scribbles / inferred central hair core.
    labels[local_pos > 0] = cv2.GC_FGD
    labels[local_neg > 0] = cv2.GC_BGD
    labels[auto_neg > 0] = cv2.GC_BGD
    if int(np.count_nonzero(labels == cv2.GC_BGD)) < 5:
        labels[0, :] = cv2.GC_BGD
        labels[-1, :] = cv2.GC_BGD
        labels[:, 0] = cv2.GC_BGD
        labels[:, -1] = cv2.GC_BGD
    if int(np.count_nonzero(labels == cv2.GC_FGD)) < 1:
        raise ValueError('no foreground seed remains')

    fallback_reason = ''
    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(local, labels, None, bg_model, fg_model,
                    max(1, int(np.clip(iterations, 1, 12))), cv2.GC_INIT_WITH_MASK)
        chosen = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    except cv2.error as exc:
        chosen = None
        fallback_reason = 'grabcut_failed: ' + str(exc).splitlines()[0][:140]

    if chosen is not None:
        chosen &= local_shell
        chosen[local_neg > 0] = 0
        # Narrow ink gaps between predicted hair sections can be filled, but
        # never fill gaps where a negative skin/face stroke was painted.
        chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, _ellipse(2))
        chosen &= local_shell
        chosen[local_neg > 0] = 0
        chosen = _components_touching_seeds(chosen, local_pos, local_rough)
        chosen = _suppress_skin_pockets(chosen, source_bgr[y0:y1, x0:x1], local,
                                        local_pos, auto_neg)
        # The model-native evidence is intentionally only a veto. It cannot
        # propagate colours into unreviewed eyebrow/skin areas.
        chosen, _model_vetoed = _generator_evidence_boundary_veto(
            chosen, source_bgr[y0:y1, x0:x1], local, local_pos, local_neg)
        chosen[local_pos > 0] = 1  # explicit foreground authority
        chosen[local_neg > 0] = 0
        count = int(np.count_nonzero(chosen))
        # A compact rough polygon may be larger than true hair area. Shrinking
        # is allowed; uncontrolled *growth* is a much bigger risk.
        if count <= 0 or count > int(rough_count * max(1.05, float(max_growth_ratio))):
            fallback_reason = 'area_safety_limit'
        elif np.any(chosen & (1 - local_shell)):
            fallback_reason = 'scope_violation'
        else:
            full = np.zeros(shape, np.uint8)
            full[y0:y1, x0:x1] = chosen
            full[neg > 0] = 0
            n = int(np.count_nonzero(full))
            return full * 255, HairRefineInfo(
                rough_count, n, int(np.count_nonzero((full>0)&(rough==0))),
                int(np.count_nonzero((full==0)&(rough>0))),
                int(np.count_nonzero(pos)), int(np.count_nonzero(neg)),
                grow_px, False, 'grabcut_guided', int(_model_vetoed))

    # Never fall back to "entire face" without making the user aware.
    safe = rough.copy()
    safe[neg > 0] = 0
    count = int(np.count_nonzero(safe))
    return safe * 255, HairRefineInfo(
        rough_count, count, 0, rough_count-count,
        int(np.count_nonzero(pos)), int(np.count_nonzero(neg)),
        grow_px, True, fallback_reason or 'no_valid_refinement')
