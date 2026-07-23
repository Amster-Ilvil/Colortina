from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from core.hint_manager import HintManager
from core.presets import get_style
from core.style_post import apply_style_grade
from pipeline import _apply_pastel_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate style fine-tuning and manual-hint strength on user-provided "
            "images without requiring mc-v2 weights."
        )
    )
    parser.add_argument("images", nargs="+", type=Path, help="Input manga page image(s)")
    parser.add_argument(
        "--output", type=Path, default=Path("VALIDATION_EFFECT_CONTROLS.md"),
        help="Markdown report path (default: VALIDATION_EFFECT_CONTROLS.md)",
    )
    parser.add_argument(
        "--style", choices=("light2", "light3"), default="light2",
        help="Built-in style baseline to validate",
    )
    return parser.parse_args()


def load_page(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not load image: {path}")
    return image


def proxy_colorize(source_bgr: np.ndarray) -> np.ndarray:
    """Create deterministic pseudo-color for post-processing validation."""
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    warm = cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)
    cool = cv2.applyColorMap(255 - gray, cv2.COLORMAP_OCEAN)
    blended = cv2.addWeighted(warm, 0.62, cool, 0.38, 0)
    return cv2.addWeighted(
        blended, 0.72, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.28, 0
    )


def mean_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def mean_chroma(image: np.ndarray) -> float:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(
        np.mean(np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2))
    )


def validate_style_tuning(image: np.ndarray, style_key: str) -> dict[str, float]:
    proxy = proxy_colorize(image)
    base = get_style(style_key)
    weak = _apply_pastel_tuning(base, {
        "color_strength": 55,
        "brightness": 90,
        "warmth": 85,
        "highlight_preserve": 85,
        "softness": 80,
        "flatten": 80,
        "light3_intensity": 60,
    })
    strong = _apply_pastel_tuning(base, {
        "color_strength": 175,
        "brightness": 130,
        "warmth": 160,
        "highlight_preserve": 155,
        "softness": 165,
        "flatten": 150,
        "light3_intensity": 170,
    })
    output_weak = apply_style_grade(proxy, image, weak, context=None)
    output_strong = apply_style_grade(proxy, image, strong, context=None)
    return {
        "delta": mean_abs_delta(output_weak, output_strong),
        "chroma_weak": mean_chroma(output_weak),
        "chroma_strong": mean_chroma(output_strong),
    }


def validate_manual_strength(image: np.ndarray) -> dict[str, float]:
    manager = HintManager()
    manager.bind_source_image(image)
    manager.add_manual_hint(0.5, 0.5, (255, 120, 120), 0.02)
    low = [
        hint for hint in manager.merge_specs(image_bgr=image, manual_strength=0.25)
        if hint.source == "manual"
    ][0]
    high = [
        hint for hint in manager.merge_specs(image_bgr=image, manual_strength=1.0)
        if hint.source == "manual"
    ][0]
    return {
        "low_strength": float(low.strength),
        "high_strength": float(high.strength),
        "low_radius": float(low.radius_norm),
        "high_radius": float(high.radius_norm),
    }


def main() -> int:
    args = parse_args()
    rows: list[tuple[str, dict[str, float]]] = []
    all_ok = True

    for path in args.images:
        image = load_page(path)
        result = validate_style_tuning(image, args.style)
        rows.append((path.name, result))
        all_ok &= result["delta"] >= 2.0
        all_ok &= abs(result["chroma_strong"] - result["chroma_weak"]) >= 1.0

    first_image = load_page(args.images[0])
    manual = validate_manual_strength(first_image)
    manual_ok = (
        manual["high_strength"] > manual["low_strength"]
        and manual["high_radius"] > manual["low_radius"]
    )
    all_ok &= manual_ok

    lines = [
        "# Effect-control validation",
        "",
        f"Style baseline: `{args.style}`",
        "",
        "The report uses a deterministic pseudo-color proxy and does not require mc-v2 weights.",
        "",
        "## Style fine-tuning",
        "",
    ]
    for name, result in rows:
        lines.append(
            f"- `{name}`: mean delta={result['delta']:.2f}; "
            f"chroma={result['chroma_weak']:.2f} → {result['chroma_strong']:.2f}"
        )

    lines.extend([
        "",
        "## Manual hint strength",
        "",
        f"- strength: {manual['low_strength']:.3f} → {manual['high_strength']:.3f}",
        f"- radius: {manual['low_radius']:.5f} → {manual['high_radius']:.5f}",
        "",
        "## Verdict",
        "",
        f"- PASS={all_ok}",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
