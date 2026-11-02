"""Optional soft, unsegmented hints for the ORIGINAL mc-v2 generator.

This deliberately does not estimate a hair mask, fill connected components,
or modify the generator checkpoint. All points are drawn in the same resize/
pad coordinates as MangaColorizator.set_image; soft alpha avoids hard paint
stamps while preserving the unhinted generator path when there are no marks.
"""
from __future__ import annotations

import numpy as np

from core.hint_spec import HintSpec
from core.hint_rasterizer import model_geometry


def rasterize_native_soft_hints(h: int, w: int, size: int, specs,
                                *, peak_alpha: float = 0.65,
                                radius_factor: float = 1.0):
    if h <= 0 or w <= 0 or size <= 0 or not np.isfinite(peak_alpha):
        raise ValueError('invalid image geometry or alpha')
    if not 0.0 < float(peak_alpha) <= 1.0:
        raise ValueError('peak_alpha must be in (0,1]')
    if not np.isfinite(radius_factor) or not 0.25 <= radius_factor <= 4.0:
        raise ValueError('radius_factor outside [0.25,4]')
    ph, pw, rh, rw = model_geometry(h, w, size)
    hint = np.full((ph, pw, 3), 128, np.uint8)
    alpha = np.zeros((ph, pw), np.float32)
    owner_priority = np.full((ph, pw), -1e10, np.float32)
    ordered = sorted((HintSpec.from_any(s) for s in specs or []),
                     key=lambda s: (s.priority or 0, s.effective_strength))
    for spec in ordered:
        if spec.source == 'style_only' or spec.effective_strength <= 0:
            continue
        px = min(rw-1, max(0, int(round(spec.x_norm * rw))))
        py = min(rh-1, max(0, int(round(spec.y_norm * rh))))
        radius = max(1, min(48, int(round(spec.radius_norm * rw * radius_factor))))
        x0, x1 = max(0, px-radius), min(rw,px+radius+1)
        y0, y1 = max(0, py-radius), min(rh,py+radius+1)
        yy,xx = np.ogrid[y0:y1,x0:x1]
        d2 = ((xx-px)**2+(yy-py)**2)/max(1,radius*radius)
        falloff = np.maximum(0.,1.-d2)**2
        local = (falloff * float(peak_alpha) *
                 float(np.clip(spec.effective_strength,0.,1.))).astype(np.float32)
        crop_alpha = alpha[y0:y1,x0:x1]
        crop_priority = owner_priority[y0:y1,x0:x1]
        priority = float(spec.priority or 0)
        changed = (local > 0) & ((priority > crop_priority) |
                    ((priority == crop_priority) & (local >= crop_alpha)))
        hint[y0:y1,x0:x1][changed] = np.asarray(spec.rgb,dtype=np.uint8)
        crop_priority[changed] = priority
        # Respect colour ownership: a lower-priority brush must not leave
        # invisible alpha behind under another brush's RGB command.
        crop_alpha[changed] = local[changed]
    return hint,alpha


def assess_native_hint_candidate(baseline_bgr, candidate_bgr, *,
                                 positive=None, negative=None,
                                 target_rgb=None, tolerance=8.0):
    """Report evidence, not segmentation accuracy or automatic acceptance.

    Positive/negative marks must be supplied by the user if a controlled
    improvement assessment is desired. Without both, no numeric acceptance
    decision is possible. No model output is composited or silently modified.
    """
    import cv2
    b=np.asarray(baseline_bgr)
    c=np.asarray(candidate_bgr)
    if b.shape!=c.shape or b.ndim!=3 or b.shape[2]!=3 or b.dtype!=np.uint8 or c.dtype!=np.uint8:
        raise ValueError('baseline/candidate must be same-size BGR uint8')
    diff=np.max(np.abs(c.astype(np.int16)-b.astype(np.int16)),axis=2)
    report={'changed_pixels_gt_tolerance':int(np.count_nonzero(diff>float(tolerance))),
            'mean_abs_max_change':float(diff.mean()), 'manual_review_required':True}
    if positive is None or negative is None or target_rgb is None:
        report['reason']='missing_reviewed_positive_negative_and_target'
        return report
    pos=np.asarray(positive)>0;neg=np.asarray(negative)>0
    if pos.shape!=diff.shape or neg.shape!=diff.shape or not pos.any() or not neg.any():
        report['reason']='missing_or_invalid_reviewed_markers'
        return report
    target=np.array([[[int(target_rgb[2]),int(target_rgb[1]),int(target_rgb[0])]]],np.uint8)
    t=cv2.cvtColor(target,cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    bl=cv2.cvtColor(b,cv2.COLOR_BGR2LAB).astype(np.float32)
    cl=cv2.cvtColor(c,cv2.COLOR_BGR2LAB).astype(np.float32)
    dist0=np.linalg.norm(bl-t,axis=2)
    dist1=np.linalg.norm(cl-t,axis=2)
    report.update(positive_mean_delta=float(diff[pos].mean()),
                  negative_mean_delta=float(diff[neg].mean()),
                  negative_changed_pixels_gt_tolerance=int(np.count_nonzero((diff>tolerance)&neg)),
                  target_lab_improvement_at_positive=float((dist0[pos]-dist1[pos]).mean()),
                  reason='review_evidence_only_not_a_segmentation_score')
    return report


def inspect_hint_spots(baseline_rgb, candidate_rgb, specs, *,
                       core_radius_factor: float = 0.70,
                       ring_inner_factor: float = 1.35,
                       ring_outer_factor: float = 2.20,
                       min_core_chroma: float = 4.5,
                       min_center_ring_gap: float = 3.5):
    """Detect localized colour islands left by native point hints.

    Compare the *same mc-v2 model*, same page/denoising, with and without
    hints. This reports colour-change topology rather than semantic hair
    membership. A hint whose chroma difference is concentrated in a circular
    core relative to the surrounding ring is a visible spot risk. Two such
    hints veto the whole candidate; we never replace only the offending
    circle, since that would cause new seams and destroy line-art continuity.
    """
    import cv2
    base = np.asarray(baseline_rgb)
    cand = np.asarray(candidate_rgb)
    if base.shape != cand.shape or base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError('baseline/candidate must have matching RGB image shapes')
    if base.dtype != np.uint8:
        base = np.clip(base.astype(np.float32) * (255.0 if np.max(base) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if cand.dtype != np.uint8:
        cand = np.clip(cand.astype(np.float32) * (255.0 if np.max(cand) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    lab_base = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_cand = cv2.cvtColor(cand, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab_cand[:, :, 1:] - lab_base[:, :, 1:], axis=2)
    luminance = np.abs(lab_cand[:, :, 0] - lab_base[:, :, 0])
    h,w=chroma.shape
    rows=[]
    risky=0
    specs=[HintSpec.from_any(s) for s in (specs or [])]
    for spec in specs:
        if spec.source=='style_only' or spec.effective_strength<=0:
            continue
        cx=int(round(spec.x_norm * (w-1)))
        cy=int(round(spec.y_norm * (h-1)))
        rad=max(2, round(spec.radius_norm * w))
        bound=int(np.ceil(rad*ring_outer_factor))
        x0=max(0,cx-bound);x1=min(w,cx+bound+1)
        y0=max(0,cy-bound);y1=min(h,cy+bound+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        r2=(xx-cx)**2+(yy-cy)**2
        core=r2<=(rad*core_radius_factor)**2
        ring=(r2>=(rad*ring_inner_factor)**2)&(r2<=(rad*ring_outer_factor)**2)
        if not np.any(core) or not np.any(ring):
            continue
        inner=float(chroma[y0:y1,x0:x1][core].mean())
        outer=float(chroma[y0:y1,x0:x1][ring].mean())
        gap=inner-outer
        light_core=float(luminance[y0:y1,x0:x1][core].mean())
        light_ring=float(luminance[y0:y1,x0:x1][ring].mean())
        light_gap=light_core-light_ring
        spotted=((inner>=min_core_chroma and gap>=min_center_ring_gap)
                 or (light_core>=6.0 and light_gap>=5.0))
        risky+=int(spotted)
        rows.append({'x':cx,'y':cy,'radius_px':rad,
                     'core_chroma_change':round(inner,3),
                     'ring_chroma_change':round(outer,3),
                     'center_ring_gap':round(gap,3),
                     'luminance_center_ring_gap':round(light_gap,3),
                     'visible_spot_risk':bool(spotted)})
    return {'spots':risky, 'hints_checked':len(rows),
            'reject':bool(risky>=1),
            'hints':rows,
            'policy':'reject full candidate; do not paint over model output'}
