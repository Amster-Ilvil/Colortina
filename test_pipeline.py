"""Phase 1 smoke test: run the Auto pipeline on a single image.

Usage:
    python test_pipeline.py path/to/page.png
    python test_pipeline.py path/to/page.png --no-guided   # skip CLIP hints
    python test_pipeline.py path/to/page.png --device cpu  # force CPU

First run downloads mc-v2 weights (~400MB, from Google Drive via gdown)
into models/weights/. Needs a real internet connection on your Mac.
"""

import argparse
import os
import sys
import time

import cv2

from config import Config
from core.model_downloader import ensure_models_downloaded
from pipeline import colorize_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a black-and-white manga/comic page")
    ap.add_argument("--out", default=None, help="Output path (default: <name>_colorized.png)")
    ap.add_argument("--no-guided", action="store_true",
                     help="Disable CLIP-driven auto color hints (plain mc-v2)")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: no such file: {args.image}")
        sys.exit(1)

    if args.no_guided:
        Config.USE_GUIDED_HINTS = False
    if args.device:
        Config.ML_DEVICE = args.device

    print("[1/3] Checking model weights...")
    ensure_models_downloaded(Config.WEIGHTS_DIR, callback=print)

    print("[2/3] Loading image...")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"ERROR: could not read image: {args.image}")
        sys.exit(1)
    print(f"    shape: {image_bgr.shape}")

    print("[3/3] Running Auto pipeline (this loads mc-v2 + CLIP on first call)...")
    t0 = time.time()
    result = colorize_page(image_bgr)
    dt = time.time() - t0
    print(f"    done in {dt:.2f}s")

    out_path = args.out or (os.path.splitext(args.image)[0] + "_colorized.png")
    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
