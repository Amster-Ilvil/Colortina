#!/usr/bin/env python3
"""Render detected manga barriers and accepted local gap repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.region_map import build_region_map  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--line-low", type=int, default=75)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    region_map = build_region_map(
        image, line_low=args.line_low, gap_close=args.gap)

    overlay = image.copy()
    barrier = region_map.barrier > 0
    repaired = region_map.repaired > 0
    # Existing detected lines: cyan; newly accepted bridges: green.
    overlay[barrier] = (
        overlay[barrier].astype(np.float32) * 0.35 +
        np.asarray((255, 220, 0), np.float32) * 0.65).astype(np.uint8)
    overlay[repaired] = (0, 255, 0)

    output = Path(args.output) if args.output else Path(args.image).with_name(
        Path(args.image).stem + f"_boundary_gap{args.gap}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)

    counts = np.bincount(region_map.labels.ravel())
    report = {
        "input": str(Path(args.image).resolve()),
        "output": str(output.resolve()),
        "gap_px": int(args.gap),
        "line_low": int(args.line_low),
        "barrier_ratio": float(np.mean(barrier)),
        "repaired_pixels": int(np.count_nonzero(repaired)),
        "region_count": int(region_map.labels.max()),
        "largest_region_ratio": (float(counts[1:].max() / region_map.labels.size)
                                 if len(counts) > 1 else 0.0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
