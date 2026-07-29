#!/usr/bin/env python3
"""Evaluate user-supplied real manga regression cases from a JSON manifest.

This tool evaluates existing results; it does not invoke mc-v2.  It is useful
for comparing two Colortina versions with the same black-and-white pages and
sample points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation import (  # noqa: E402
    assignment_metrics, identity_delta_metrics, line_bleed_ratio,
    median_rgb_at_points,
)


def _read(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    return image


def evaluate(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    base = manifest_path.parent
    samples = []
    pages = []
    diagnostics = []

    for case in manifest.get("cases", []):
        source_path = base / case["source"]
        result_path = base / case["result"]
        source = _read(source_path)
        result = _read(result_path)
        page_report = {
            "name": case.get("name", result_path.stem),
            "source": str(source_path),
            "result": str(result_path),
            "line_bleed_ratio": line_bleed_ratio(source, result),
        }
        diag_path = case.get("diagnostics")
        if diag_path:
            with (base / diag_path).open("r", encoding="utf-8") as f:
                diag = json.load(f)
            diagnostics.append(diag)
            page_report["diagnostics"] = diag

        for sample in case.get("samples", []):
            rgb = median_rgb_at_points(
                result, sample.get("points", []),
                radius=int(sample.get("radius", 4)))
            samples.append({
                "page": page_report["name"],
                "character": sample.get("character", ""),
                "attribute": sample.get("attribute", ""),
                "rgb": rgb,
            })
        pages.append(page_report)

    valid_bleed = [p["line_bleed_ratio"] for p in pages]
    return {
        "manifest": str(manifest_path),
        "pages": pages,
        "identity": identity_delta_metrics(samples),
        "assignments": assignment_metrics(diagnostics),
        "line_bleed_ratio_mean": (sum(valid_bleed) / len(valid_bleed)
                                   if valid_bleed else None),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.manifest.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
