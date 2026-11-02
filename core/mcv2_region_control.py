"""Explicit region commands for manga-colorization-v2.

mc-v2 treats colour hints as soft guidance.  This module adds an inference-time
control contract and a deterministic post-check so an explicitly selected
region cannot silently drift back to the model's preferred colour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import cv2
import numpy as np


def is_black_like_rgb(rgb: tuple[int, int, int]) -> bool:
    r, g, b = [int(v) for v in rgb]
    return max(rgb) <= 96 and (max(rgb) - min(rgb)) <= 34


def is_neutral_ink_rgb(rgb: tuple[int, int, int]) -> bool:
    """True for the neutral Ink Black family, not blue/cool black presets."""
    r, g, b = [int(v) for v in rgb]
    return max(rgb) <= 48 and (max(rgb) - min(rgb)) <= 6


@dataclass(frozen=True)
class RegionControlSettings:
    enabled: bool = False
    target_rgb: tuple[int, int, int] = (28, 28, 32)
    generator_hint_rgb: tuple[int, int, int] | None = None
    hint_opacity: float = 0.50
    # v5.18: direct control is part of Generator.forward itself.  Legacy
    # three-stage mixing is retained behind ``dual_route`` for A/B tests.
    generator_control_mode: str = "direct_tunnel2"
    strict_output_mask: bool = True
    deep_strength: float = 0.38
    middle_strength: float = 0.74
    shallow_strength: float = 0.94
    feature_gain: float = 1.75
    deep_shrink_px: int = 8
    middle_shrink_px: int = 3
    shallow_shrink_px: int = 0
    mask_feather_px: int = 3
    final_color_lock: float = 0.92
    luminance_shift: float = 0.28
    protect_lineart: bool = True
    preserve_highlights: bool = True
    use_adapter: bool = False
    source_bias_strength: float = 0.0
    source_bias_tone: int = 0
    source_bias_feather_px: int = 3
    neutralize_chroma: float = 0.0
    warm_suppression: float = 0.0
    protect_lineart_threshold: int = 88
    # For black-like targets, rebuild the complete page-local hair object and
    # use that mask for generator control, locking and final compositing.
    # Explicit opt-in only: user selection is authoritative by default.
    complete_hair_object: bool = False
    hair_object_max_ratio: float = 2.15
    # Set by the UI when semantic selection already returned the complete hair
    # object. Manual add/erase corrections remain authoritative and the local
    # recolour path must not rebuild/cut the mask a second time.
    hair_object_mask_prebuilt: bool = False
    # v5.20.1: disabled by default for local manual edits because the
    # automatic warm-edge recovery can spill onto adjacent skin when the user
    # already supplied a reviewed hair mask. Keep opt-in switches so A/B
    # research remains possible.
    expand_control_mask: bool = False
    expand_merge_mask: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        rgb = tuple(int(np.clip(v, 0, 255)) for v in self.target_rgb)
        if len(rgb) != 3:
            rgb = (28, 28, 32)
        object.__setattr__(self, "target_rgb", rgb)
        hint_rgb = self.generator_hint_rgb
        if hint_rgb is not None:
            hint_rgb = tuple(int(np.clip(v, 0, 255)) for v in hint_rgb)
            if len(hint_rgb) != 3:
                hint_rgb = None
        object.__setattr__(self, "generator_hint_rgb", hint_rgb)
        for name in ("hint_opacity", "deep_strength", "middle_strength",
                     "shallow_strength", "final_color_lock", "luminance_shift",
                     "source_bias_strength", "neutralize_chroma",
                     "warm_suppression"):
            object.__setattr__(self, name,
                               float(np.clip(getattr(self, name), 0.0, 1.0)))
        object.__setattr__(self, "feature_gain",
                           float(np.clip(self.feature_gain, 0.5, 4.0)))
        for name in ("deep_shrink_px", "middle_shrink_px",
                     "shallow_shrink_px", "mask_feather_px",
                     "source_bias_feather_px"):
            object.__setattr__(self, name, max(0, int(getattr(self, name))))
        object.__setattr__(self, "protect_lineart_threshold",
                           int(np.clip(int(self.protect_lineart_threshold), 0, 255)))
        object.__setattr__(self, "source_bias_tone",
                           int(np.clip(int(self.source_bias_tone), 0, 255)))
        mode = str(self.generator_control_mode or "direct_tunnel2").strip().lower()
        if mode not in {"direct_tunnel2", "direct", "tunnel2", "dual_route", "legacy_dual_route"}:
            mode = "direct_tunnel2"
        object.__setattr__(self, "generator_control_mode", mode)
        object.__setattr__(self, "strict_output_mask", bool(self.strict_output_mask))
        object.__setattr__(self, "protect_lineart", bool(self.protect_lineart))
        object.__setattr__(self, "preserve_highlights", bool(self.preserve_highlights))
        object.__setattr__(self, "use_adapter", bool(self.use_adapter))
        object.__setattr__(self, "complete_hair_object", bool(self.complete_hair_object))
        object.__setattr__(self, "hair_object_mask_prebuilt",
                           bool(self.hair_object_mask_prebuilt))
        object.__setattr__(self, "expand_control_mask",
                           bool(self.expand_control_mask))
        object.__setattr__(self, "expand_merge_mask",
                           bool(self.expand_merge_mask))
        object.__setattr__(self, "hair_object_max_ratio",
                           float(np.clip(self.hair_object_max_ratio, 1.10, 4.0)))

        if is_black_like_rgb(rgb):
            # Real generator calibration (page-197 regression, stock mc-v2): a
            # literal near-black hint does not produce neutral black; depending
            # on strength it drifts warm or cyan.  The dark teal compensation
            # below is the inverse command that makes the *generator output*
            # land close to neutral black before the deterministic guard.
            if self.generator_hint_rgb is None:
                object.__setattr__(self, "generator_hint_rgb", (0, 32, 48))
            if float(self.neutralize_chroma) <= 0.0:
                object.__setattr__(
                    self, "neutralize_chroma",
                    1.00 if is_neutral_ink_rgb(rgb) else 0.94)
            if float(self.warm_suppression) <= 0.0:
                object.__setattr__(self, "warm_suppression", 0.90)
            if int(self.protect_lineart_threshold) <= 88:
                object.__setattr__(self, "protect_lineart_threshold", 108)

    @classmethod
    def from_any(cls, value: Any) -> "RegionControlSettings":
        if isinstance(value, cls):
            return value
        if not value:
            return cls()
        payload = dict(value)
        payload.pop("mask", None)
        if "rgb" in payload and "target_rgb" not in payload:
            payload["target_rgb"] = payload.pop("rgb")
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_dict(self) -> dict:
        out = asdict(self)
        out["target_rgb"] = list(self.target_rgb)
        if self.generator_hint_rgb is not None:
            out["generator_hint_rgb"] = list(self.generator_hint_rgb)
        return out


def normalize_region_mask(mask: np.ndarray | None,
                          shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, np.uint8)
    work = np.asarray(mask)
    if work.ndim == 3:
        work = np.max(work, axis=2)
    if work.shape != shape:
        work = cv2.resize(work.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return np.where(work > 0, 255, 0).astype(np.uint8)


def control_payload(settings: RegionControlSettings | dict | None,
                    mask: np.ndarray | None) -> dict | None:
    cfg = RegionControlSettings.from_any(settings)
    if not cfg.enabled or mask is None or not np.any(mask):
        return None
    data = cfg.to_dict()
    data["mask"] = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    return data


def region_control_signature(control: dict | RegionControlSettings | None):
    if not control:
        return None
    cfg = RegionControlSettings.from_any(control)
    mask = control.get("mask") if isinstance(control, dict) else None
    if not cfg.enabled or mask is None or not np.any(mask):
        return None
    mask = np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8))
    digest = hashlib.blake2b(mask.tobytes(), digest_size=16).digest()
    return (
        tuple(cfg.target_rgb), tuple(cfg.generator_hint_rgb or cfg.target_rgb),
        round(cfg.hint_opacity, 4), cfg.generator_control_mode, cfg.strict_output_mask,
        round(cfg.deep_strength, 4), round(cfg.middle_strength, 4),
        round(cfg.shallow_strength, 4), round(cfg.feature_gain, 4),
        cfg.deep_shrink_px, cfg.middle_shrink_px, cfg.shallow_shrink_px,
        cfg.mask_feather_px, round(cfg.final_color_lock, 4),
        round(cfg.luminance_shift, 4), cfg.protect_lineart,
        cfg.preserve_highlights, cfg.use_adapter,
        round(cfg.source_bias_strength, 4), cfg.source_bias_tone,
        cfg.source_bias_feather_px, round(cfg.neutralize_chroma, 4),
        round(cfg.warm_suppression, 4), cfg.protect_lineart_threshold,
        cfg.complete_hair_object, round(cfg.hair_object_max_ratio, 3),
        cfg.hair_object_mask_prebuilt, cfg.expand_control_mask,
        cfg.expand_merge_mask,
        mask.shape, digest,
    )


def expand_black_hair_control_mask(current_bgr: np.ndarray,
                                   original_bgr: np.ndarray,
                                   region_mask: np.ndarray,
                                   settings: RegionControlSettings | dict,
                                   *, radius: int = 10):
    """Recover narrow warm hair edges/gaps before the controlled model run.

    The semantic selector intentionally protects face/skin aggressively.  On
    warm mc-v2 hair this can leave a few orange strips just outside the blue
    mask.  For an explicit black-like command, grow only through nearby pixels
    that share the selected object's dominant hue, are meaningfully saturated,
    and are not extremely bright.  Growth is geodesic and bounded, so it cannot
    jump to a distant object.
    """
    cfg = RegionControlSettings.from_any(settings)
    image = np.asarray(current_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask = normalize_region_mask(region_mask, image.shape[:2])
    if not cfg.enabled or not is_black_like_rgb(cfg.target_rgb) or not np.any(mask):
        return mask, {"added_pixels": 0, "enabled": False}

    radius = max(1, int(radius))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    light = hls[..., 1]
    selected_hues = h[(mask > 0) & (s > 100)]
    if selected_hues.size < 16:
        return mask, {"added_pixels": 0, "enabled": True}
    target_hue = int(np.bincount(selected_hues, minlength=180).argmax())
    hue_delta = np.minimum(
        np.abs(h.astype(np.int16) - target_hue),
        180 - np.abs(h.astype(np.int16) - target_hue))

    vicinity = cv2.dilate(
        (mask > 0).astype(np.uint8),
        np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8), iterations=1) > 0
    yy, cr, cb = cv2.split(cv2.cvtColor(image.astype(np.uint8),
                                         cv2.COLOR_BGR2YCrCb))
    skin_like = ((yy > 135) & (yy < 252) &
                 (cr > 120) & (cr < 192) &
                 (cb > 48) & (cb < 155))
    # Warm hair normally has much higher Cr/saturation than skin. Keep that
    # explicit escape hatch while preventing edge recovery from entering face,
    # hand or blush regions next to bangs.
    hair_over_skin = (cr >= 195) | ((s >= 210) & (hue_delta <= 10))
    candidate = (vicinity & (mask == 0) & (hue_delta <= 20) &
                 (s >= 75) & (v > 40) & (light <= 195) &
                 ((~skin_like) | hair_over_skin))
    eligible = ((mask > 0) | candidate).astype(np.uint8)
    grown = (mask > 0).astype(np.uint8)
    for _ in range(radius):
        nearby = cv2.dilate(grown, np.ones((3, 3), np.uint8), iterations=1)
        grown = ((nearby > 0) & (eligible > 0)).astype(np.uint8)
    out = np.where(grown > 0, 255, 0).astype(np.uint8)
    return out, {
        "enabled": True,
        "target_hue": target_hue,
        "added_pixels": int(np.count_nonzero((out > 0) & (mask == 0))),
    }


def expand_black_hair_merge_mask(base_bgr: np.ndarray,
                                  generated_bgr: np.ndarray,
                                  region_mask: np.ndarray,
                                  settings: RegionControlSettings | dict,
                                  *, radius: int = 20):
    """Keep black-hair pixels that the controlled generator fixed just outside
    the original semantic boundary.

    Decoder receptive fields often correct a narrow fringe even when the blue
    mask missed it.  Merge only nearby warm base pixels whose generated result
    became genuinely dark; this retains corrected bangs/edges without pulling
    face or clothing into the selection.
    """
    cfg = RegionControlSettings.from_any(settings)
    base = np.asarray(base_bgr)
    generated = np.asarray(generated_bgr)
    if base.shape[:2] != generated.shape[:2]:
        generated = cv2.resize(generated, (base.shape[1], base.shape[0]),
                               interpolation=cv2.INTER_AREA)
    mask = normalize_region_mask(region_mask, base.shape[:2])
    if not cfg.enabled or not is_black_like_rgb(cfg.target_rgb) or not np.any(mask):
        return mask, {"added_pixels": 0, "enabled": False}

    radius = max(1, int(radius))
    bh, bs, bv = cv2.split(cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2HSV))
    _gh, gs, _gv = cv2.split(cv2.cvtColor(generated.astype(np.uint8), cv2.COLOR_BGR2HSV))
    gl = cv2.cvtColor(generated.astype(np.uint8), cv2.COLOR_BGR2HLS)[..., 1]
    vicinity = cv2.dilate(
        (mask > 0).astype(np.uint8),
        np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8), iterations=1) > 0
    by, bcr, bcb = cv2.split(cv2.cvtColor(base.astype(np.uint8),
                                           cv2.COLOR_BGR2YCrCb))
    skin_like = ((by > 135) & (by < 252) &
                 (bcr > 120) & (bcr < 192) &
                 (bcb > 48) & (bcb < 155))
    warm_base = (((bh < 25) | (bh > 170)) & (bs > 80) & (bv > 80))
    hair_over_skin = (bcr >= 195) | (bs >= 210)
    candidate = (vicinity & (mask == 0) & warm_base &
                 ((~skin_like) | hair_over_skin) &
                 (gl < 110) & (gs < 210))
    eligible = ((mask > 0) | candidate).astype(np.uint8)
    grown = (mask > 0).astype(np.uint8)
    for _ in range(radius):
        nearby = cv2.dilate(grown, np.ones((3, 3), np.uint8), iterations=1)
        grown = ((nearby > 0) & (eligible > 0)).astype(np.uint8)
    out = np.where(grown > 0, 255, 0).astype(np.uint8)
    return out, {
        "enabled": True,
        "added_pixels": int(np.count_nonzero((out > 0) & (mask == 0))),
    }


def _inward_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary.astype(np.float32)
    if feather_px <= 0:
        return binary.astype(np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    out = np.clip(dist / max(1.0, float(feather_px)), 0.0, 1.0)
    out[binary == 0] = 0.0
    return out.astype(np.float32)


def enforce_region_color(generated_bgr: np.ndarray,
                         original_bgr: np.ndarray,
                         region_mask: np.ndarray,
                         settings: RegionControlSettings | dict):
    """Move selected pixels to the commanded chroma while preserving shading.

    Pixels outside the selection and protected solid ink are restored byte for
    byte after the colour-space round trip.
    """
    cfg = RegionControlSettings.from_any(settings)
    image = np.asarray(generated_bgr).astype(np.uint8, copy=True)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = normalize_region_mask(region_mask, image.shape[:2])
    if not cfg.enabled or cfg.final_color_lock <= 0 or not np.any(mask):
        return image, {"pixels": 0, "mean_delta": 0.0}

    original = np.asarray(original_bgr).astype(np.uint8, copy=False)
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if original.shape[:2] != image.shape[:2]:
        original = cv2.resize(original, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_AREA)

    chroma_alpha = (mask > 0).astype(np.float32)
    luminance_alpha = _inward_alpha(mask, cfg.mask_feather_px)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    protected_ink = np.zeros_like(gray, dtype=bool)
    if cfg.protect_lineart:
        protected_ink = gray <= int(cfg.protect_lineart_threshold)
    chroma_alpha *= cfg.final_color_lock
    luminance_alpha *= cfg.final_color_lock
    active = chroma_alpha > 1e-5
    if not np.any(active):
        return image, {"pixels": 0, "mean_delta": 0.0}

    target_bgr = np.uint8([[[cfg.target_rgb[2], cfg.target_rgb[1],
                             cfg.target_rgb[0]]]])
    target_hls = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)[0, 0].astype(np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)

    hue_weight = float(np.clip(target_hls[2] / 24.0, 0.0, 1.0))
    hue_delta = ((float(target_hls[0]) - hls[..., 0] + 90.0) % 180.0) - 90.0
    hls[..., 0] = (hls[..., 0] + hue_delta * chroma_alpha * hue_weight) % 180.0
    hls[..., 2] = (hls[..., 2] * (1.0 - chroma_alpha) +
                   float(target_hls[2]) * chroma_alpha)

    if cfg.luminance_shift > 0:
        center = float(np.median(hls[..., 1][active]))
        target_center = max(22.0, float(target_hls[1]))
        shift = (target_center - center) * cfg.luminance_shift
        lum_alpha = luminance_alpha.copy()
        if cfg.preserve_highlights:
            lum_alpha *= np.clip((248.0 - hls[..., 1]) / 96.0, 0.06, 1.0)
        hls[..., 1] = np.clip(hls[..., 1] + shift * lum_alpha, 0, 255)

    output = cv2.cvtColor(np.clip(hls, 0, 255).astype(np.uint8),
                          cv2.COLOR_HLS2BGR)
    output[~active] = image[~active]

    if cfg.neutralize_chroma > 0:
        out_f = output.astype(np.float32)
        gray_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray3 = np.repeat(gray_out[..., None], 3, axis=2)
        alpha = chroma_alpha[..., None] * float(cfg.neutralize_chroma)
        # For commanded black hair, highlights should retain luminance but not
        # the model's orange/purple chroma. Previous highlight attenuation left
        # exactly the bright orange streaks users reported. Non-black colours
        # keep the older chroma-preserving highlight behaviour.
        if cfg.preserve_highlights and not is_black_like_rgb(cfg.target_rgb):
            alpha *= np.clip((248.0 - gray_out) / 118.0, 0.10, 1.0)[..., None]
        out_f = out_f * (1.0 - alpha) + gray3 * alpha
        output = np.clip(out_f + 0.5, 0, 255).astype(np.uint8)

    if cfg.warm_suppression > 0:
        out_f = output.astype(np.float32)
        gray_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warmness = np.clip(out_f[..., 2] - np.maximum(out_f[..., 0], out_f[..., 1]), 0.0, 255.0)
        warm_mask = warmness > 1.0
        sup = chroma_alpha * float(cfg.warm_suppression)
        if cfg.preserve_highlights and not is_black_like_rgb(cfg.target_rgb):
            sup *= np.clip((248.0 - gray_out) / 108.0, 0.08, 1.0)
        sup *= warm_mask.astype(np.float32)
        out_f[..., 2] -= warmness * sup
        green_excess = np.clip(out_f[..., 1] - gray_out, 0.0, 255.0)
        out_f[..., 1] -= green_excess * sup * 0.55
        output = np.clip(out_f + 0.5, 0, 255).astype(np.uint8)

    if cfg.protect_lineart and np.any(protected_ink):
        protected = protected_ink & (mask > 0)
        output[protected] = original[protected]
    output[mask == 0] = image[mask == 0]
    diff = np.abs(output.astype(np.int16) - image.astype(np.int16))
    return output, {
        "pixels": int(np.count_nonzero(np.any(diff > 0, axis=2) & (mask > 0))),
        "mean_delta": float(diff[mask > 0].mean()),
        "target_rgb": tuple(cfg.target_rgb),
        "source_bias_strength": float(cfg.source_bias_strength),
        "neutralize_chroma": float(cfg.neutralize_chroma),
        "warm_suppression": float(cfg.warm_suppression),
    }
