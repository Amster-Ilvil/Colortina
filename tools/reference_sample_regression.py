#!/usr/bin/env python3
"""Run a reproducible reference/target regression without hiding limitations.

The JSON manifest can include manual character boxes and sample points.  The
script always runs style extraction, manual identity enrolment, target
segmentation and descriptor diagnostics.  It records whether the local mc-v2
weights are actually present; it never labels an analysis-only run as completed
colourization.

Example:
    python tools/reference_sample_regression.py manifest.json --output report_dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from core.character_library import CharacterLibrary
from core.manual_reference import manual_head_features, sample_hex_at
from core.region_segmenter import segment_regions
from core.style_analyzer import StyleAnalyzer


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def draw_overlay(image: np.ndarray, boxes: list[dict], out_path: Path) -> None:
    canvas = image.copy()
    for index, item in enumerate(boxes, start=1):
        x, y, w, h = map(int, item["head_bbox"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 245), 5)
        # OpenCV's built-in font cannot render CJK reliably. Use a stable
        # ASCII id in the bitmap; the JSON/Markdown report maps it back to the
        # full character or target label.
        label = str(item.get("overlay_id") or f"R{index}")
        cv2.putText(canvas, label, (x + 4, max(30, y + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (40, 40, 245), 3,
                    cv2.LINE_AA)
        for attr, point in (item.get("points") or {}).items():
            px, py = map(int, point)
            color = {"hair": (0, 210, 255), "skin": (30, 220, 70),
                     "eyes": (255, 210, 30), "clothing": (230, 80, 210)}.get(
                         attr, (255, 255, 255))
            cv2.circle(canvas, (px, py), 9, color, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(out_path.suffix or ".jpg", canvas)[1].tofile(str(out_path))


def local_model_status() -> dict:
    generator = Path(Config.GENERATOR_WEIGHTS_PATH)
    extractor = Path(Config.EXTRACTOR_WEIGHTS_PATH)
    denoiser = Path(Config.DENOISER_WEIGHTS_DIR)
    status = {
        "generator": generator.is_file(),
        "extractor": extractor.is_file(),
        "denoiser": denoiser.is_dir() and any(denoiser.iterdir()),
        "paths": {
            "generator": str(generator), "extractor": str(extractor),
            "denoiser": str(denoiser),
        },
    }
    status["full_inference_available"] = bool(
        status["generator"] and status["extractor"] and status["denoiser"])
    return status


def descriptor_scores(library: CharacterLibrary, image: np.ndarray,
                      head_bbox: tuple[int, int, int, int]) -> list[dict]:
    tone, hist, aspect, area, lineart = manual_head_features(image, head_bbox)
    scored = []
    for character in library.characters:
        distance = library._feature_distance(  # diagnostic use only
            tone, hist, aspect, area, [], lineart, character)
        scored.append({
            "character_id": character.char_id,
            "name": character.name,
            "distance": round(float(distance), 5),
            "score": round(float(max(0.0, 1.0 - distance)), 5),
        })
    scored.sort(key=lambda item: item["distance"])
    if len(scored) >= 2:
        margin = scored[1]["score"] - scored[0]["score"]
        # Scores are inverse distance, so the useful margin is top1-top2.
        margin = scored[0]["score"] - scored[1]["score"]
    else:
        margin = scored[0]["score"] if scored else 0.0
    for item in scored:
        item["top_margin"] = round(float(margin), 5)
    return scored


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    references = manifest.get("references", [])
    targets = manifest.get("targets", [])

    analyzer = StyleAnalyzer()
    style_images = []
    library = CharacterLibrary()
    reference_results = []

    for ref_index, ref in enumerate(references):
        path = str(ref["path"])
        image = read_image(path)
        if ref.get("use_for_style", True):
            style_images.append(image)
        characters = ref.get("characters", [])
        if characters:
            overlay_characters = []
            for character_index, item in enumerate(characters, start=1):
                overlay_item = dict(item)
                overlay_item["overlay_id"] = f"C{character_index}"
                overlay_characters.append(overlay_item)
            draw_overlay(image, overlay_characters,
                         output_dir / f"reference_{ref_index:02d}_overlay.jpg")
        enrolled = []
        for item in characters:
            colors = dict(item.get("colors") or {})
            for attr, point in (item.get("points") or {}).items():
                if attr not in colors:
                    colors[attr] = sample_hex_at(
                        image, int(point[0]), int(point[1]),
                        radius_px=int(item.get("sample_radius", 8)))
            profile = library.add_manual_reference(
                image, tuple(item["head_bbox"]), colors=colors,
                name=str(item.get("name", "")),
                rotation=int(item.get("rotation", 0)),
                classifier=None, merge_same_name=True)
            enrolled.append({
                "name": profile.name, "character_id": profile.char_id,
                "colors": dict(profile.colors),
                "reference_samples": profile.reference_samples,
            })
        reference_results.append({
            "path": path, "shape": list(image.shape),
            "manual_enrolments": enrolled,
        })

    descriptor = (analyzer.analyze_many(style_images, name="sample-regression",
                                        classifier=None)
                  if len(style_images) > 1 else
                  analyzer.analyze(style_images[0], name="sample-regression",
                                   classifier=None)) if style_images else None
    if descriptor is not None:
        descriptor.save(str(output_dir / "sample_reference.ccstyle"))
    library.save(str(output_dir / "sample_characters.ccpalette"))

    target_results = []
    for target_index, target in enumerate(targets):
        path = str(target["path"])
        image = read_image(path)
        seg = segment_regions(image)
        boxes = [{"head_bbox": item["head_bbox"],
                  "name": item.get("label", f"target-{i + 1}"),
                  "overlay_id": f"T{i + 1}"}
                 for i, item in enumerate(target.get("head_boxes", []))]
        if boxes:
            draw_overlay(image, boxes,
                         output_dir / f"target_{target_index:02d}_overlay.jpg")
        head_results = []
        for item in target.get("head_boxes", []):
            scores = descriptor_scores(library, image, tuple(item["head_bbox"]))
            head_results.append({
                "label": item.get("label", ""),
                "head_bbox": item["head_bbox"],
                "candidates": scores[:5],
                # Analysis-only descriptor matching follows the same strict
                # fallback gate used by the application when CLIP is absent.
                "safe_auto_lock": bool(
                    scores and scores[0]["score"] >= max(0.62, library.min_match_score) and
                    scores[0]["top_margin"] >= max(0.075, library.min_margin)),
            })
        target_results.append({
            "path": path, "shape": list(image.shape),
            "segmented_regions": len(seg.regions),
            "manual_head_diagnostics": head_results,
        })

    model = local_model_status()
    report = {
        "schema": "colortina-reference-regression-v1",
        "manifest": str(manifest_path),
        "analysis_only": not model["full_inference_available"],
        "model_status": model,
        "style": ({
            "temperature": descriptor.temperature,
            "saturation": descriptor.saturation,
            "contrast": descriptor.contrast,
            "global_warm_cool": descriptor.global_warm_cool,
            "global_saturation": descriptor.global_saturation,
            "reference_palette": descriptor.reference_palette,
            "semantic_region_samples": descriptor.region_samples,
            "note": "K-Means fallback supplies atmosphere only; it does not invent character semantics.",
        } if descriptor is not None else None),
        "character_profiles": [
            {"id": ch.char_id, "name": ch.name, "colors": ch.colors,
             "manual": ch.manual, "reference_samples": ch.reference_samples}
            for ch in library.characters
        ],
        "references": reference_results,
        "targets": target_results,
        "limitations": ([] if model["full_inference_available"] else [
            "Local mc-v2 generator/extractor/denoiser weights are missing; no generated color page was claimed.",
            "The report validates reference extraction, identity separation, segmentation and matching diagnostics only.",
        ]),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sample_regression"))
    args = parser.parse_args()
    report = run(args.manifest, args.output)
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "analysis_only": report["analysis_only"],
        "character_profiles": len(report["character_profiles"]),
        "targets": len(report["targets"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
