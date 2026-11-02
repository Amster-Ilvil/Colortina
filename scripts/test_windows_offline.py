"""Offline regression checks for generator-only distributions (no real weights)."""
from __future__ import annotations

from pathlib import Path
import sys
import os
import tempfile
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_downloader import (
    ModelDownloadError, ensure_models_downloaded, generator_file_ready,
    model_paths, models_ready,
)


def main() -> None:
    previous = os.environ.get("COLORTINA_OFFLINE")
    os.environ["COLORTINA_OFFLINE"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="colortina-offline-test-") as tmp:
            gen, extractor, denoiser_dir, denoiser_path = model_paths(tmp)
            assert not models_ready(tmp)
            try:
                ensure_models_downloaded(tmp, allow_download=True)
            except ModelDownloadError:
                pass
            else:
                raise AssertionError("Offline mode must not download missing generator")
            with zipfile.ZipFile(gen, "w", compression=zipfile.ZIP_STORED) as f:
                f.writestr("fake/data.pkl", b"0" * (1024 * 1024))
            assert generator_file_ready(gen)
            assert models_ready(tmp)
            assert not Path(extractor).exists()
            assert not Path(denoiser_path).exists()
            with patch("core.model_downloader._download_via_gdown",
                       side_effect=AssertionError("No network access expected")):
                paths = ensure_models_downloaded(tmp, allow_download=True)
            assert paths == (gen, extractor, denoiser_dir)
            print("PASS: generator-only offline readiness + zero downloads")

            # Synthetic checkpoint: mock deserialization, not real inference.
            from vendor.manga_colorization_v2 import colorizator as mod

            class Generator:
                def load_state_dict(self, state, strict=True):
                    assert state == {}
                    assert strict is True

            class Colorizer:
                def __init__(self):
                    self.generator = Generator()

                def to(self, _device):
                    return self

                def eval(self):
                    return self

            class Denoiser:
                def __init__(self, _device, _weights_dir):
                    self.available = False

            with patch.object(mod, "Colorizer", Colorizer), \
                 patch.object(mod, "FFDNetDenoiser", Denoiser), \
                 patch.object(mod.torch, "load", return_value={}) as load:
                engine = mod.MangaColorizator(
                    device="cpu", generator_path=gen,
                    extractor_path=extractor,
                    denoiser_weights_dir=denoiser_dir,
                )
                assert engine.denoiser.available is False
                assert load.call_count == 1, "Only generator should be loaded"
                assert load.call_args.args[0] == gen
            print("PASS: colorizer init without optional checkpoints")
            # A synthetic checkpoint cannot validate real image inference.
    finally:
        if previous is None:
            os.environ.pop("COLORTINA_OFFLINE", None)
        else:
            os.environ["COLORTINA_OFFLINE"] = previous


if __name__ == "__main__":
    main()
