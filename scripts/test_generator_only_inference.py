"""Real CPU forward pass using a temporary randomly initialized generator.

This tests the offline generator-only inference path without downloading a
checkpoint. It does NOT test image quality or reproduce trained model output.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from core.model_downloader import models_ready
from vendor.manga_colorization_v2.colorizator import MangaColorizator
from vendor.manga_colorization_v2.networks.models import Colorizer


def main() -> None:
    torch.set_num_threads(2)
    os.environ["COLORTINA_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory(prefix="colortina-generator-forward-") as tmp:
        root = Path(tmp)
        generator = root / "generator.zip"
        source = Colorizer()
        torch.save(source.generator.state_dict(), generator)
        del source
        gc.collect()
        assert models_ready(tmp), "Generator-only checkpoint not detected"
        engine = MangaColorizator(
            device="cpu", generator_path=str(generator),
            extractor_path=str(root / "nonexistent-extractor.pth"),
            denoiser_weights_dir=str(root / "denoiser"),
        )
        assert not engine.denoiser.available
        sample = np.full((80, 80, 1), 210, dtype=np.uint8)
        engine.set_image(sample, size=384, apply_denoise=True, denoise_sigma=15)
        result = engine.colorize()
        assert result.ndim == 3 and result.shape[-1] == 3
        assert np.all(np.isfinite(result))
        print("PASS: offline mc-v2 CPU forward with only generator.zip",
              result.shape, flush=True)


if __name__ == "__main__":
    main()
