"""Selection-focused mc-v2 inference with strictly local compositing.

The recolor job always composites only inside the blue selection. By default it
still runs mc-v2 on a full-page B/W image, but the pixels outside the selected
area can be weakened to white (with an optional context band) so the model
focuses on the chosen region without receiving a hard crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from core.hint_manager import Hint, HintManager


@dataclass
class LocalModelRecolorResult:
    result_bgr: np.ndarray
    ai_result_bgr: np.ndarray
    filter_base_bgr: np.ndarray
    generated_full_bgr: np.ndarray
    generated_filter_base_bgr: np.ndarray
    used_hint_count: int
    selection_pixels: int
    changed_pixels: int
    diagnostics: dict


def normalize_selection_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a binary uint8 mask at ``shape`` without growing its boundary."""
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    work = np.asarray(mask)
    if work.ndim == 3:
        work = np.max(work, axis=2)
    if work.shape != shape:
        work = cv2.resize(work.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return np.where(work > 0, 255, 0).astype(np.uint8)


def inward_feather_alpha(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Build alpha that is zero outside and feathers only *inside* selection."""
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    if not np.any(binary):
        return binary.astype(np.float32)
    radius = max(0, int(radius))
    if radius == 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(max(1, radius)), 0.0, 1.0)
    alpha[binary == 0] = 0.0
    return alpha.astype(np.float32)


def merge_inside_selection(base_bgr: np.ndarray, generated_bgr: np.ndarray,
                           mask: np.ndarray, feather: int = 4) -> np.ndarray:
    """Merge generated pixels into base while keeping outside bit-identical."""
    if base_bgr.shape[:2] != generated_bgr.shape[:2]:
        generated_bgr = cv2.resize(
            generated_bgr, (base_bgr.shape[1], base_bgr.shape[0]),
            interpolation=cv2.INTER_AREA)
    alpha = inward_feather_alpha(mask, feather)[..., None]
    out = np.rint(
        base_bgr.astype(np.float32) * (1.0 - alpha) +
        generated_bgr.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    # Explicit assignment guarantees exact outside preservation even if future
    # numeric implementations change rounding behaviour.
    out[mask == 0] = base_bgr[mask == 0]
    return out


def preview_black_and_white(current_bgr: np.ndarray, original_bgr: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """Non-destructive display preview: B&W only inside the selection."""
    if original_bgr.shape[:2] != current_bgr.shape[:2]:
        original_bgr = cv2.resize(
            original_bgr, (current_bgr.shape[1], current_bgr.shape[0]),
            interpolation=cv2.INTER_AREA)
    out = current_bgr.copy()
    active = mask > 0
    out[active] = original_bgr[active]
    return out


def _ensure_3ch_uint8(image: np.ndarray) -> np.ndarray:
    work = np.asarray(image)
    if work.ndim == 2:
        work = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif work.ndim == 3 and work.shape[2] == 1:
        work = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return work.astype(np.uint8, copy=False)


def build_focus_inference_image(original_bgr: np.ndarray, selection_mask: np.ndarray,
                                *, context_expand_px: int = 32,
                                fade_expand_px: int = 96,
                                outside_mode: str = "fade_white") -> np.ndarray:
    """Create a local-focus B/W page for mc-v2.

    The blue selection stays untouched. A context band around it keeps the
    original line-art, while farther-away pixels are weakened so the model
    focuses on the selected area instead of the whole page.

    ``outside_mode``:
      - ``none``: return the unmodified original page.
      - ``white``: keep the selection + context, turn everything else white.
      - ``fade_white``: keep the selection + context, then gradually fade the
        outer ring to white.
    """
    original = _ensure_3ch_uint8(original_bgr)
    mask = normalize_selection_mask(selection_mask, original.shape[:2])
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    if not np.any(binary):
        return original.copy()

    mode = str(outside_mode or "none").lower()
    if mode in {"", "none", "off", "disabled"}:
        return original.copy()

    context_expand_px = max(0, int(context_expand_px))
    fade_expand_px = max(context_expand_px, int(fade_expand_px))

    def dilate_region(src: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return src.copy()
        k = radius * 2 + 1
        return cv2.dilate(src, np.ones((k, k), np.uint8), iterations=1)

    context = dilate_region(binary, context_expand_px)
    outer = dilate_region(binary, fade_expand_px)

    if mode == "white":
        out = np.full_like(original, 255)
        keep = context > 0
        out[keep] = original[keep]
        return out

    # Default: fade_white
    out = np.full_like(original, 255)
    keep = context > 0
    out[keep] = original[keep]
    fade_ring = (outer > 0) & (~keep)
    if np.any(fade_ring):
        inv_context = np.where(context > 0, 0, 1).astype(np.uint8)
        distance = cv2.distanceTransform(inv_context, cv2.DIST_L2, 5)
        span = max(1.0, float(max(0, fade_expand_px - context_expand_px)))
        alpha = np.clip(1.0 - ((distance - float(context_expand_px)) / span), 0.0, 1.0)
        alpha[~fade_ring] = 0.0
        alpha3 = alpha[..., None].astype(np.float32)
        out = np.rint(original.astype(np.float32) * alpha3 + out.astype(np.float32) * (1.0 - alpha3)).clip(0, 255).astype(np.uint8)
        out[keep] = original[keep]
    return out


def _hint_hits_mask(hint: Hint, expanded_mask: np.ndarray) -> bool:
    h, w = expanded_mask.shape
    x = int(np.clip(round(float(hint.x_norm) * max(0, w - 1)), 0, max(0, w - 1)))
    y = int(np.clip(round(float(hint.y_norm) * max(0, h - 1)), 0, max(0, h - 1)))
    return bool(expanded_mask[y, x] > 0)


def _clone_hint_for_model(hint: Hint, *, classic_points: bool = False) -> Hint:
    cloned = Hint.from_dict(hint.to_dict())
    # AI selection recolor must preserve the user's exact RGB instruction.
    #
    # Standard route keeps ``manual`` as ``manual`` and uses the pipeline's
    # ``mixed`` renderer.  That renderer fills the line-art region touched by
    # the dab with the exact selected RGB value, instead of first converting it
    # to ``manual_region`` and splitting it into several sparse style tiers.
    # The old tier route was easy for mc-v2 to ignore, allowing its common warm
    # / yellow prior to win regardless of the requested colour.
    #
    # Classic route intentionally remains a sparse original-style point hint.
    if cloned.source == "manual":
        if classic_points:
            cloned.source = "eyedropper_hint"
        cloned.priority = max(100, int(cloned.priority or 0))
        cloned.strength = max(0.95, float(cloned.strength))
    return cloned


def filtered_hint_manager(source: HintManager, selection_mask: np.ndarray,
                          *, margin_px: int = 16,
                          only_selection_hints: bool = True,
                          classic_point_hints: bool = False) -> HintManager:
    """Clone hints for a local model run, optionally dropping distant hints.

    When the user places at least one manual / eyedropper hint inside the
    active selection, those manual instructions must dominate the local mc-v2
    run. To prevent the classic "new colour mixed with stale old colour"
    failure mode, automatic hints whose centres are inside the blue selection
    are suppressed for this run. Distant automatic hints outside the selection
    can still survive when allowed by ``only_selection_hints`` / ``margin_px``
    so broader page context is not discarded unnecessarily.
    """
    binary = np.where(selection_mask > 0, 255, 0).astype(np.uint8)
    margin_px = max(0, int(margin_px))
    if margin_px:
        k = margin_px * 2 + 1
        expanded = cv2.dilate(binary, np.ones((k, k), np.uint8), iterations=1)
    else:
        expanded = binary

    def keep(hint: Hint) -> bool:
        # Future-compatible explicit global sources are always retained.
        if str(getattr(hint, "source", "")) in {"global", "global_manual"}:
            return True
        return (not only_selection_hints) or _hint_hits_mask(hint, expanded)

    manual_hints = [_clone_hint_for_model(h, classic_points=classic_point_hints)
                    for h in source.manual_hints if keep(h)]
    has_local_manual_override = any(
        str(getattr(h, "source", "")) in {"manual", "manual_region", "eyedropper_hint"}
        for h in manual_hints)

    auto_hints: list[Hint] = []
    for hint in source.auto_hints:
        if not keep(hint):
            continue
        if has_local_manual_override and _hint_hits_mask(hint, binary):
            continue
        auto_hints.append(Hint.from_dict(hint.to_dict()))

    local = HintManager(auto_hints=auto_hints, manual_hints=manual_hints)
    local.last_diagnostics = {
        "manual_override_active": bool(has_local_manual_override),
        "suppressed_auto_hints_in_selection": int(
            sum(1 for h in source.auto_hints if keep(h) and _hint_hits_mask(h, binary))
        ) if has_local_manual_override else 0,
    }
    return local


def count_hints_in_selection(source: HintManager, selection_mask: np.ndarray,
                             *, margin_px: int = 16) -> int:
    local = filtered_hint_manager(
        source, selection_mask, margin_px=margin_px,
        only_selection_hints=True)
    return len(local.auto_hints) + len(local.manual_hints)


def clear_manual_hints_in_selection(source: HintManager, selection_mask: np.ndarray,
                                    *, margin_px: int = 0) -> int:
    """Delete manual/model hints whose centres lie inside the selected area."""
    binary = np.where(selection_mask > 0, 255, 0).astype(np.uint8)
    if margin_px > 0:
        k = int(margin_px) * 2 + 1
        binary = cv2.dilate(binary, np.ones((k, k), np.uint8), iterations=1)
    before = len(source.manual_hints)
    source.manual_hints = [h for h in source.manual_hints if not _hint_hits_mask(h, binary)]
    return before - len(source.manual_hints)


def _target_hls_from_rgb(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (int(np.clip(v, 0, 255)) for v in rgb)
    px = np.array([[[b, g, r]]], dtype=np.uint8)
    hls = cv2.cvtColor(px, cv2.COLOR_BGR2HLS)[0, 0]
    return float(hls[0]), float(hls[1]), float(hls[2])


def apply_manual_hint_color_lock(
        generated_bgr: np.ndarray,
        original_bgr: np.ndarray,
        selection_mask: np.ndarray,
        local_hints: HintManager,
        *, strength: float = 1.0) -> tuple[np.ndarray, int]:
    """Lock user-selected hue/chroma after mc-v2 while preserving shading.

    mc-v2 can occasionally ignore even valid hints and fall back to a warm or
    yellow prior.  This pass does *not* blend the old coloured page back in.
    It works only on the newly generated image: the model's HLS luminance is
    preserved, while hue and saturation are replaced by the user's selected
    RGB inside the line-art region touched by each manual / eyedropper hint.

    The correction is restricted to both the active blue selection and the
    connected line-art region, so neighbouring skin, speech bubbles and other
    selected-but-unhinted regions are not recoloured.
    """
    generated = _ensure_3ch_uint8(generated_bgr).copy()
    original = _ensure_3ch_uint8(original_bgr)
    if original.shape[:2] != generated.shape[:2]:
        original = cv2.resize(original, (generated.shape[1], generated.shape[0]),
                              interpolation=cv2.INTER_AREA)
    mask = normalize_selection_mask(selection_mask, generated.shape[:2])
    manual = [h for h in local_hints.manual_hints
              if str(getattr(h, "source", "")) in
              {"manual", "manual_region", "eyedropper_hint"}]
    if not manual or not np.any(mask):
        return generated, 0

    region_map = local_hints.region_map
    labels = getattr(region_map, "labels", None)
    if labels is not None and labels.shape != generated.shape[:2]:
        labels = cv2.resize(labels.astype(np.int32),
                            (generated.shape[1], generated.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    h, w = generated.shape[:2]
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(generated, cv2.COLOR_BGR2HLS).astype(np.float32)
    corrected = np.zeros((h, w), dtype=bool)
    global_strength = float(np.clip(strength, 0.0, 1.0))

    for hint in manual:
        ix = min(w - 1, max(0, int(round(float(hint.x_norm) * max(0, w - 1)))))
        iy = min(h - 1, max(0, int(round(float(hint.y_norm) * max(0, h - 1)))))
        region = np.zeros((h, w), dtype=bool)
        rid = int(getattr(hint, "region_id", 0) or 0)
        if labels is not None:
            if rid <= 0:
                rid = int(labels[iy, ix])
            if rid > 0:
                region = labels == rid

        # Fallback for an unclosed / broken-line region: constrain a local disk
        # to the blue selection rather than colouring the whole rectangle.
        if not np.any(region):
            radius = max(5, int(round(max(0.006, float(hint.radius_norm)) * w * 2.2)))
            cv2.circle(region.view(np.uint8), (ix, iy), radius, 1, -1)

        region &= mask > 0
        # Never colour solid line art. Antialiased gray edge pixels remain so
        # the generated shading does not develop a white halo.
        region &= gray >= 28
        if not np.any(region):
            continue

        target_h, _target_l, target_s = _target_hls_from_rgb(hint.color)
        hint_strength = float(np.clip(max(global_strength, float(hint.strength)), 0.0, 1.0))

        current_h = hls[..., 0][region]
        # Hue is circular on OpenCV's [0, 180) interval.
        delta = ((target_h - current_h + 90.0) % 180.0) - 90.0
        hls[..., 0][region] = (current_h + delta * hint_strength) % 180.0
        hls[..., 2][region] = (
            hls[..., 2][region] * (1.0 - hint_strength) +
            target_s * hint_strength)
        corrected |= region

    if not np.any(corrected):
        return generated, 0
    locked = cv2.cvtColor(np.clip(hls, 0, 255).astype(np.uint8), cv2.COLOR_HLS2BGR)
    locked[~corrected] = generated[~corrected]
    return locked, int(np.count_nonzero(corrected))


def recolor_selection_with_model(
        original_bgr: np.ndarray,
        current_result_bgr: np.ndarray,
        current_ai_result_bgr: np.ndarray | None,
        current_filter_base_bgr: np.ndarray | None,
        selection_mask: np.ndarray,
        hint_manager: HintManager,
        *,
        feather: int = 4,
        hint_margin_px: int = 16,
        gap_close: int = 4,
        only_selection_hints: bool = True,
        classic_point_hints: bool = False,
        focus_outside_mode: str = "fade_white",
        focus_context_expand_px: int = 32,
        focus_fade_expand_px: int = 96,
        colorize_fn: Callable | None = None,
        colorize_kwargs: dict | None = None) -> LocalModelRecolorResult:
    """Run mc-v2 on a selection-focused full page and composite only the selection."""
    if original_bgr is None or current_result_bgr is None:
        raise ValueError("original and current colour layers are required")
    shape = current_result_bgr.shape[:2]
    mask = normalize_selection_mask(selection_mask, shape)
    selection_pixels = int(np.count_nonzero(mask))
    if selection_pixels <= 0:
        raise ValueError("selection is empty")

    # Keep the imported black-and-white page at its native resolution for
    # inference. Generated output is resized only at the final compositing step
    # when a legacy/upscaled current layer has a different shape.
    original = _ensure_3ch_uint8(original_bgr)
    focus_input = build_focus_inference_image(
        original, mask,
        context_expand_px=focus_context_expand_px,
        fade_expand_px=focus_fade_expand_px,
        outside_mode=focus_outside_mode)

    local_hints = filtered_hint_manager(
        hint_manager, mask, margin_px=hint_margin_px,
        only_selection_hints=only_selection_hints,
        classic_point_hints=classic_point_hints)
    local_hints.bind_source_image(original, gap_close=max(0, int(gap_close)))
    used_hint_count = len(local_hints.auto_hints) + len(local_hints.manual_hints)

    if colorize_fn is None:
        from pipeline import colorize_page
        colorize_fn = colorize_page
    kwargs = dict(colorize_kwargs or {})
    # Keep local-selection fixes isolated from the ordinary whole-page pipeline.
    # Custom colour bias remains a first-class feature, but for local recolor we
    # apply it in this module *after* the manual-hint lock so the yellow-fix
    # does not silently cancel the user-facing colour-bias control.
    custom_bias_cfg = dict(kwargs.get("custom_color_bias") or {})
    if custom_bias_cfg:
        isolated_bias_cfg = dict(custom_bias_cfg)
        isolated_bias_cfg["enabled"] = False
        kwargs["custom_color_bias"] = isolated_bias_cfg
    kwargs.update({
        "hint_manager": local_hints,
        "regenerate_auto": False,
        "preserve_empty_auto_hints": True,
        "learn_identity": False,
        "return_filter_base": True,
        "hint_render_mode": ("legacy" if classic_point_hints else "mixed"),
    })
    payload = colorize_fn(focus_input, **kwargs)
    if isinstance(payload, tuple) and len(payload) == 2:
        generated_full, generated_filter_base = payload
    else:
        generated_full = payload
        generated_filter_base = payload
    generated_full_before_local_post = np.asarray(generated_full).copy()
    generated_filter_base_before_local_post = np.asarray(generated_filter_base).copy()
    if generated_full is None:
        raise RuntimeError("mc-v2 returned no image")

    manual_lock_pixels = 0
    if not classic_point_hints and local_hints.manual_hints:
        generated_full, manual_lock_pixels = apply_manual_hint_color_lock(
            generated_full, original, mask, local_hints, strength=1.0)
        generated_filter_base, _ = apply_manual_hint_color_lock(
            generated_filter_base, original, mask, local_hints, strength=1.0)

    custom_bias_pixels = 0
    if custom_bias_cfg and bool(custom_bias_cfg.get("enabled")):
        from core.custom_color_bias import apply_global_color_bias
        bias_rgb = tuple(custom_bias_cfg.get("rgb") or (255, 160, 160))
        bias_strength = float(custom_bias_cfg.get("strength", 35)) / 100.0
        bias_scope = str(custom_bias_cfg.get("scope", "page"))
        bias_tone_range = str(custom_bias_cfg.get("tone_range", "all"))
        generated_full = apply_global_color_bias(
            generated_full, original, bias_rgb, bias_strength,
            bias_scope, bias_tone_range,
            protect_skin=bool(custom_bias_cfg.get("protect_skin", True)),
            protect_lineart=bool(custom_bias_cfg.get("protect_lineart", True)),
            protect_saturated=bool(custom_bias_cfg.get("protect_saturated", True)),
        )
        generated_filter_base = apply_global_color_bias(
            generated_filter_base, original, bias_rgb, bias_strength,
            bias_scope, bias_tone_range,
            protect_skin=bool(custom_bias_cfg.get("protect_skin", True)),
            protect_lineart=bool(custom_bias_cfg.get("protect_lineart", True)),
            protect_saturated=bool(custom_bias_cfg.get("protect_saturated", True)),
        )
        custom_bias_pixels = int(np.count_nonzero(
            np.any(generated_full != generated_full_before_local_post, axis=2)
        ))

    ai_base = (current_ai_result_bgr if current_ai_result_bgr is not None
               else current_result_bgr)
    filter_base = (current_filter_base_bgr if current_filter_base_bgr is not None
                   else current_result_bgr)
    merged_result = merge_inside_selection(
        current_result_bgr, generated_full, mask, feather)
    merged_ai = merge_inside_selection(ai_base, generated_full, mask, feather)
    merged_filter = merge_inside_selection(
        filter_base, generated_filter_base, mask, feather)
    changed_pixels = int(np.count_nonzero(
        np.any(merged_result != current_result_bgr, axis=2) & (mask > 0)))

    diagnostics = dict(getattr(local_hints, "last_diagnostics", {}) or {})
    diagnostics.update({
        "local_model_recolor": True,
        "local_selection_pixels": selection_pixels,
        "local_hint_count": used_hint_count,
        "local_hint_margin_px": int(hint_margin_px),
        "local_gap_close_px": int(gap_close),
        "local_only_selection_hints": bool(only_selection_hints),
        "local_classic_point_hints": bool(classic_point_hints),
        "local_feather_px": int(feather),
        "local_changed_pixels": int(changed_pixels),
        "local_manual_color_lock_pixels": int(manual_lock_pixels),
        "local_manual_color_lock_enabled": bool(
            not classic_point_hints and bool(local_hints.manual_hints)),
        "local_custom_color_bias_enabled": bool(custom_bias_cfg and custom_bias_cfg.get("enabled")),
        "local_custom_color_bias_pixels": int(custom_bias_pixels),
        "local_focus_outside_mode": str(focus_outside_mode or "none"),
        "local_focus_context_expand_px": int(focus_context_expand_px),
        "local_focus_fade_expand_px": int(focus_fade_expand_px),
    })
    return LocalModelRecolorResult(
        result_bgr=merged_result,
        ai_result_bgr=merged_ai,
        filter_base_bgr=merged_filter,
        generated_full_bgr=np.asarray(generated_full).copy(),
        generated_filter_base_bgr=np.asarray(generated_filter_base).copy(),
        used_hint_count=used_hint_count,
        selection_pixels=selection_pixels,
        changed_pixels=changed_pixels,
        diagnostics=diagnostics,
    )
