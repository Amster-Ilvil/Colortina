from __future__ import annotations

import cv2
import numpy as np

from core.natural_tint import apply_natural_tint


def _compute_scope_weight(gray: np.ndarray, scope: str,
                          result_bgr: np.ndarray | None = None) -> np.ndarray:
    if scope == "page":
        return np.ones_like(gray, dtype=np.float32)
    if result_bgr is not None:
        # 优先用真实角色分割（SkyTNT anime-seg）；对上色结果推理，
        # 比黑白线稿更接近模型的训练域。失败则回退启发式。
        try:
            from core.character_scope import character_likelihood
            seg = character_likelihood(result_bgr)
        except Exception:
            seg = None
        # 覆盖率异常（几乎全零或几乎全一）说明分割对这页不可靠——
        # 例如极简线稿或非角色页面——此时回退到启发式而不是把
        # 角色范围整体压到 0.10。
        if seg is not None:
            coverage = float(seg.mean())
            if not (0.02 <= coverage <= 0.98):
                seg = None
        if seg is not None:
            if scope == "characters":
                return (0.10 + 0.90 * seg).astype(np.float32)
            if scope == "background":
                return (0.10 + 0.90 * (1.0 - seg)).astype(np.float32)
    edges = cv2.Canny(gray, 70, 150)
    density = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 9)
    if float(density.max()) > 1e-6:
        density = density / float(density.max())
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dist = np.sqrt(((xx - cx) / max(1.0, w * 0.55)) ** 2 + ((yy - cy) / max(1.0, h * 0.58)) ** 2)
    center = np.clip(1.0 - dist, 0.0, 1.0)
    char_like = np.clip(0.60 * density + 0.40 * center, 0.0, 1.0)
    if scope == "characters":
        return 0.22 + 0.78 * char_like
    if scope == "background":
        return 0.22 + 0.78 * (1.0 - char_like)
    return np.ones_like(gray, dtype=np.float32)


def _compute_tone_weight(l_channel: np.ndarray, tone_range: str) -> np.ndarray:
    l_norm = np.clip(l_channel.astype(np.float32) / 255.0, 0.0, 1.0)
    if tone_range == "highlights":
        return np.clip((l_norm - 0.58) / 0.30, 0.0, 1.0)
    if tone_range == "shadows":
        return np.clip((0.42 - l_norm) / 0.30, 0.0, 1.0)
    if tone_range == "midtones":
        return np.clip(1.0 - np.abs(l_norm - 0.52) / 0.28, 0.0, 1.0)
    return np.ones_like(l_norm, dtype=np.float32)


def _skin_protection_weight(result_bgr: np.ndarray) -> np.ndarray:
    """Return 0..1 skin-likelihood used only as a protection mask.

    The deliberately broad YCrCb rule protects common light/dark manga skin
    colours without claiming semantic face detection. A blurred mask avoids
    hard halos at the boundary.
    """
    ycrcb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    _y, cr, cb = cv2.split(ycrcb)
    skin = (
        np.clip((cr - 128.0) / 18.0, 0.0, 1.0) *
        np.clip((178.0 - cr) / 22.0, 0.0, 1.0) *
        np.clip((138.0 - cb) / 20.0, 0.0, 1.0) *
        np.clip((cb - 72.0) / 22.0, 0.0, 1.0)
    )
    return cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 2.0)


def _lineart_protection_weight(gray: np.ndarray) -> np.ndarray:
    dark = np.clip((112.0 - gray.astype(np.float32)) / 88.0, 0.0, 1.0)
    edges = cv2.Canny(gray, 55, 135).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.2)
    return np.clip(np.maximum(dark, edges), 0.0, 1.0)


def _saturation_protection_weight(result_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    return np.clip((saturation - 0.48) / 0.34, 0.0, 1.0)


def apply_global_color_bias(result_bgr: np.ndarray,
                            source_bw_bgr: np.ndarray | None,
                            rgb: tuple[int, int, int],
                            strength: float = 0.35,
                            scope: str = "page",
                            tone_range: str = "all",
                            protect_skin: bool = False,
                            protect_lineart: bool = False,
                            protect_saturated: bool = False) -> np.ndarray:
    """Nudge the whole page toward one selected colour while preserving detail.

    This is not a flat overlay. It biases only colourable pixels, protects paper
    white and black line-art, and preserves local texture / luminance so the page
    still looks like an mc-v2 result rather than a tinted glass layer.
    """
    if result_bgr is None:
        return result_bgr
    strength = float(np.clip(strength, 0.0, 2.0))
    if strength <= 1e-6:
        return result_bgr

    h, w = result_bgr.shape[:2]
    if source_bw_bgr is None:
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        src = source_bw_bgr
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Protect black line-art. Paper protection must be measured from the
    # *colourized result*, not only from the original black/white page: manga
    # character interiors are usually white in the source even after mc-v2 has
    # assigned them a real colour. The old source-white gate therefore rejected
    # most faces, clothes and backgrounds and made the control appear broken.
    ink_gate = np.clip((gray.astype(np.float32) - 22.0) / 42.0, 0.0, 1.0)
    chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2)
    paper_likelihood = (
        np.clip((lab[..., 0] - 238.0) / 15.0, 0.0, 1.0) *
        np.clip((12.0 - chroma) / 10.0, 0.0, 1.0)
    )
    paper_gate = 1.0 - 0.985 * paper_likelihood

    # Treat either source screentone or visible result chroma as evidence that a
    # pixel is colourable. This keeps true blank paper safe while allowing pale
    # skin, clothing and backgrounds to receive the requested colour tendency.
    source_tone = np.clip((250.0 - gray.astype(np.float32)) / 42.0, 0.0, 1.0)
    result_colour = np.clip(chroma / 18.0, 0.0, 1.0)
    # Eligibility, not amount: any pixel with moderate evidence (pale skin
    # coloured by mc-v2, light backgrounds) deserves the FULL requested
    # bias. A linear gate made pale regions permanently half-strength
    # relative to dark screentone, which read as "the control barely
    # works". Saturate the evidence so the gate answers yes/no smoothly.
    evidence = np.maximum(source_tone, result_colour)
    colourable_gate = 0.12 + 0.88 * np.clip(evidence / 0.35, 0.0, 1.0)
    scope_gate = _compute_scope_weight(gray, str(scope or "page"),
                                       result_bgr=result_bgr)
    tone_gate = _compute_tone_weight(lab[..., 0], str(tone_range or "all"))
    protection_gate = np.ones_like(gray, dtype=np.float32)
    if protect_skin:
        protection_gate *= (1.0 - 0.42 * _skin_protection_weight(result_bgr))
    if protect_lineart:
        protection_gate *= (1.0 - 0.94 * _lineart_protection_weight(gray))
    if protect_saturated:
        protection_gate *= (1.0 - 0.40 * _saturation_protection_weight(result_bgr))
    # The multiplied 0..1 gates systematically collapse alpha: a typical
    # colourable pixel landed near 0.4, so the UI's "35%" only moved colours
    # ~15% of the way to the target. Renormalize the *eligibility* gates by a
    # robust high percentile so the requested strength applies at full value
    # on clearly-eligible pixels. Protections multiply AFTER normalization —
    # normalizing them away on uniform pages would silently disable them.
    # The 0.6 ceiling leaves the 100-200% authority range real headroom.
    base = np.clip(
        ink_gate * paper_gate * colourable_gate * scope_gate * tone_gate,
        0.0, 1.0)
    eligible = base > 1e-4
    if np.any(eligible):
        scale = float(np.percentile(base[eligible], 95.0))
        base = np.clip(base / max(scale, 0.30), 0.0, 1.0)
    alpha = np.clip(base * protection_gate * 0.6, 0.0, 1.0)
    # Hard-zero effectively-ineligible pixels (paper, ink cores). The
    # 100-200% authority curve consumes *remaining headroom*, so without
    # this cutoff a 0.003-alpha paper pixel would jump to ~0.7 at 200%
    # and the page would look like a flat tinted overlay.
    alpha[alpha < 0.04] = 0.0

    active = alpha > 1e-4
    if not np.any(active):
        return result_bgr.copy()

    return apply_natural_tint(
        result_bgr, tuple(int(np.clip(v, 0, 255)) for v in rgb), alpha,
        active=active, authority=strength, texture_retention=0.45,
        chroma_retention=0.38, tone_strength=0.035)
