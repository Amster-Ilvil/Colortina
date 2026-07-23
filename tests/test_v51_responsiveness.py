from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.model_downloader import ModelDownloadError, _gdrive_download, models_ready
from core.region_classifier import RegionClassifier


class V51ResponsivenessTests(unittest.TestCase):
    def test_clip_does_not_silently_download_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COLORTINA_ALLOW_CLIP_DOWNLOAD", None)
            classifier = RegionClassifier(model_path="not-cached/model")
        self.assertFalse(classifier._allow_download)

    def test_model_readiness_requires_nontrivial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "denoiser"), exist_ok=True)
            open(os.path.join(tmp, "generator.zip"), "wb").close()
            open(os.path.join(tmp, "denoiser", "net_rgb.pth"), "wb").close()
            self.assertFalse(models_ready(tmp))
            with open(os.path.join(tmp, "generator.zip"), "wb") as f:
                f.truncate(1024 * 1024)
            with open(os.path.join(tmp, "denoiser", "net_rgb.pth"), "wb") as f:
                f.truncate(64 * 1024)
            self.assertTrue(models_ready(tmp))

    def test_failed_gdown_is_reported_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "core.model_downloader._download_via_gdown", return_value=None):
            dest = os.path.join(tmp, "generator.zip")
            with self.assertRaises(ModelDownloadError):
                _gdrive_download("bad", dest, "test", min_bytes=1024)
            self.assertFalse(os.path.exists(dest))

    def test_get_colorizer_can_skip_duplicate_download(self):
        import pipeline
        old = pipeline._colorizer
        pipeline._colorizer = None
        fake = SimpleNamespace(device_name="cpu")
        cfg = SimpleNamespace(
            ML_DEVICE="cpu", GENERATOR_WEIGHTS_PATH="g",
            EXTRACTOR_WEIGHTS_PATH="e", DENOISER_WEIGHTS_DIR="d")
        try:
            with patch("pipeline.MangaColorizer", return_value=fake) as ctor, patch(
                    "core.model_downloader.ensure_models_downloaded") as downloader:
                result = pipeline.get_colorizer(cfg, ensure_weights=False)
            self.assertIs(result, fake)
            downloader.assert_not_called()
            ctor.assert_called_once()
        finally:
            pipeline._colorizer = old

    def test_right_panel_distributes_groups_into_blank_height(self):
        source = (Path(__file__).resolve().parents[1] /
                  "ui" / "main_window.py").read_text(encoding="utf-8")
        start = source.index("    def _build_right_panel")
        end = source.index("\n    def ", start + 10)
        body = source[start:end]
        self.assertIn("render_layout.addWidget(style_group, stretch=3)", body)
        self.assertIn("render_layout.addWidget(auto_group, stretch=2)", body)
        self.assertIn("reference_layout.addWidget(character_group, stretch=3)", body)
        self.assertNotIn("render_layout.addStretch", body)
        self.assertNotIn("reference_layout.addStretch", body)
        self.assertIn("_auto_status_label", body)

    def test_worker_always_emits_completion(self):
        source = (Path(__file__).resolve().parents[1] /
                  "ui" / "worker.py").read_text(encoding="utf-8")
        self.assertIn("finally:\n            self.finished_all.emit()", source)
        self.assertIn("page_done = Signal(str, object)", source)


if __name__ == "__main__":
    unittest.main()
