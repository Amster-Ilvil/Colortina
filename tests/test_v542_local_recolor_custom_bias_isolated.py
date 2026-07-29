import numpy as np

from core.hint_manager import HintManager
from core.local_model_recolor import recolor_selection_with_model


def _page(h=84, w=112):
    original = np.full((h, w, 3), 245, np.uint8)
    original[18:66, 34:36] = 0
    current = np.full((h, w, 3), (80, 80, 80), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[18:66, 34:88] = 255
    return original, current, mask


def _mean_bgr(image, mask):
    region = image[mask > 0]
    return region.mean(axis=0)


def test_local_recolor_reapplies_custom_color_bias_after_manual_lock():
    original, current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (40, 80, 235), 0.03, source="manual")

    seen = {}
    def fake_colorize(image, **kwargs):
        seen["custom_bias_enabled_seen_by_pipeline"] = bool((kwargs.get("custom_color_bias") or {}).get("enabled"))
        generated = np.full_like(image, (120, 120, 120))
        return generated, generated.copy()

    payload_plain = recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=0, hint_margin_px=8, colorize_fn=fake_colorize,
        colorize_kwargs={"custom_color_bias": {"enabled": False}},
    )
    plain_mean = _mean_bgr(payload_plain.result_bgr, mask)

    payload_bias = recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=0, hint_margin_px=8, colorize_fn=fake_colorize,
        colorize_kwargs={"custom_color_bias": {
            "enabled": True,
            "rgb": (40, 220, 60),
            "strength": 200,
            "scope": "page",
            "tone_range": "all",
            "protect_skin": False,
            "protect_lineart": False,
            "protect_saturated": False,
        }},
    )
    bias_mean = _mean_bgr(payload_bias.result_bgr, mask)

    assert seen["custom_bias_enabled_seen_by_pipeline"] is False
    # Bias-on result must differ materially from the no-bias local result.
    assert np.linalg.norm(bias_mean - plain_mean) > 8.0
    assert payload_bias.diagnostics["local_custom_color_bias_enabled"] is True
    assert payload_bias.diagnostics["local_custom_color_bias_pixels"] > 0
