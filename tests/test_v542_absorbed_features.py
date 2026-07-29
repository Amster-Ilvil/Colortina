"""V5.4.2 — 吸收的三个能力：文字气泡保护、角色分割范围、Real-ESRGAN。

模型文件缺失时相关断言自动跳过（特性设计即为优雅降级）。
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import Config

ROOT = Path(__file__).resolve().parents[1]

_HAS_TEXT = os.path.isfile(getattr(Config, "TEXT_DETECTOR_PATH", ""))
_HAS_SEG = os.path.isfile(getattr(Config, "CHAR_SEG_PATH", ""))
_HAS_ESRGAN = os.path.isfile(getattr(Config, "ESRGAN_MODEL_PATH", ""))


def _text_page() -> np.ndarray:
    page = np.full((800, 600), 255, np.uint8)
    cv2.ellipse(page, (300, 420), (140, 180), 0, 0, 360, 0, 4)
    cv2.ellipse(page, (150, 120), (110, 70), 0, 0, 360, 0, 2)
    for i, line in enumerate(["What is", "going on", "here?!"]):
        cv2.putText(page, line, (75, 95 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


@pytest.mark.skipif(not _HAS_TEXT, reason="comictextdetector.pt.onnx missing")
def test_text_guard_masks_glyphs_not_artwork():
    from core.text_guard import text_mask
    page = _text_page()
    mask = text_mask(page)
    assert mask is not None and mask.shape == page.shape[:2]
    assert mask[65:180, 60:250].mean() > 0.08      # 气泡文字区被覆盖
    assert mask[300:600, 180:430].mean() < 0.03    # 人物区几乎无误报


@pytest.mark.skipif(not _HAS_TEXT, reason="comictextdetector.pt.onnx missing")
def test_protect_text_regions_restores_source_pixels():
    from core.text_guard import protect_text_regions, text_mask
    page = _text_page()
    tinted = np.full_like(page, 200)
    tinted[..., 2] = 255  # 整页染成粉色
    out = protect_text_regions(tinted, page)
    mask = text_mask(page) > 0.5
    assert mask.any()
    restored = np.abs(out.astype(int) - page.astype(int))[mask].mean()
    untouched = np.abs(out.astype(int) - tinted.astype(int))[~mask].mean()
    assert restored < 12.0
    assert untouched < 1.0




def test_protect_text_regions_does_not_restore_midtone_art_on_false_positive_mask(monkeypatch):
    from core import text_guard
    page = np.full((120, 160, 3), 255, np.uint8)
    # Mid-tone artwork region that must not be restored even if the detector
    # falsely covers it.
    cv2.rectangle(page, (12, 18), (68, 96), (132, 132, 132), -1)
    # Legitimate speech/text area: white paper with black glyphs.
    cv2.putText(page, "Hi", (96, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    tinted = np.full_like(page, (210, 150, 255))
    monkeypatch.setattr(text_guard, "text_mask",
                        lambda *_args, **_kwargs: np.ones(page.shape[:2], np.float32))
    out = text_guard.protect_text_regions(tinted, page)

    # The false-positive mid-tone art block should stay almost entirely tinted.
    art_diff = np.abs(out[24:88, 18:62].astype(int) - tinted[24:88, 18:62].astype(int)).mean()
    assert art_diff < 6.0

    # The true bubble/text area should still restore strongly toward the source.
    source_like = np.abs(out[28:80, 88:145].astype(int) - page[28:80, 88:145].astype(int)).mean()
    tinted_like = np.abs(out[28:80, 88:145].astype(int) - tinted[28:80, 88:145].astype(int)).mean()
    assert source_like < tinted_like

def _anime_figure() -> np.ndarray:
    img = np.full((768, 576, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 500), (576, 768), (190, 210, 230), -1)
    cv2.ellipse(img, (288, 260), (95, 115), 0, 0, 360, (200, 214, 246), -1)
    cv2.ellipse(img, (288, 175), (115, 90), 0, 180, 360, (60, 60, 160), -1)
    cv2.ellipse(img, (288, 300), (14, 8), 0, 0, 180, (80, 80, 140), 2)
    for cx in (250, 326):
        cv2.circle(img, (cx, 255), 14, (255, 255, 255), -1)
        cv2.circle(img, (cx, 255), 9, (140, 80, 40), -1)
        cv2.circle(img, (cx, 255), 4, (20, 20, 20), -1)
    cv2.rectangle(img, (218, 370), (358, 620), (140, 100, 60), -1)
    cv2.rectangle(img, (238, 620), (278, 740), (110, 120, 140), -1)
    cv2.rectangle(img, (298, 620), (338, 740), (110, 120, 140), -1)
    cv2.ellipse(img, (288, 260), (95, 115), 0, 0, 360, (60, 50, 50), 3)
    cv2.rectangle(img, (218, 370), (358, 620), (60, 50, 50), 3)
    return img


@pytest.mark.skipif(not _HAS_SEG, reason="isnetis.onnx missing")
def test_character_scope_segments_anime_figure():
    from core.character_scope import character_likelihood
    img = _anime_figure()
    seg = character_likelihood(img)
    assert seg is not None and seg.shape == img.shape[:2]
    assert seg[200:600, 230:350].mean() > 0.5
    assert seg[40:200, 20:150].mean() < 0.3


def test_scope_weight_falls_back_on_degenerate_segmentation():
    from core.custom_color_bias import _compute_scope_weight
    blank = np.full((400, 300, 3), 255, np.uint8)
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    weight = _compute_scope_weight(gray, "characters", result_bgr=blank)
    # 分割覆盖率退化时必须回退启发式，而不是把整页压到 0.10。
    assert float(weight.mean()) > 0.2


@pytest.mark.skipif(not _HAS_ESRGAN, reason="realesrgan_anime6b.pth missing")
def test_esrgan_upscale_runs_without_basicsr():
    from core.upscaler import upscale
    img = np.full((64, 48, 3), 255, np.uint8)
    cv2.circle(img, (24, 32), 16, (180, 120, 240), -1)
    out = upscale(img, scale=4, weights_path=Config.ESRGAN_MODEL_PATH)
    assert out.shape == (256, 192, 3)
    lanczos = cv2.resize(img, (192, 256), interpolation=cv2.INTER_LANCZOS4)
    assert float(np.abs(out.astype(int) - lanczos.astype(int)).mean()) > 0.5


def test_pipeline_and_ui_wiring():
    pipe = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "protect_text: bool = True" in pipe
    assert "protect_text_regions" in pipe
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "_chk_protect_text" in main
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")
    assert "protect_text=self._protect_text" in worker
    ups = (ROOT / "core" / "upscaler.py").read_text(encoding="utf-8")
    assert "basicsr" not in ups
    assert "vendor.realesrgan_min" in ups
