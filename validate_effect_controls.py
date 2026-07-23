from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math
import cv2
import numpy as np

from core.hint_manager import HintManager
from core.presets import get_style
from core.style_post import apply_style_grade
from pipeline import _apply_pastel_tuning, build_auto_hints

PAGE_PATHS = [
    Path('/mnt/data/ghostwriter_images/context/4a34e19b-dbbb-50fc-ace5-d5f99f05034c.jpg'),
    Path('/mnt/data/ghostwriter_images/context/34b8f3ba-b738-5801-b535-e5f0ca721c78.jpg'),
    Path('/mnt/data/ghostwriter_images/context/0bf3fbfc-8f10-5ba5-849a-845adbab985c.jpg'),
]


def load_page(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'failed to load {path}')
    return img


def proxy_colorize(src_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    # A deterministic pseudo-color proxy so style/post controls can be tested
    # without requiring mc-v2 weights in CI.
    warm = cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)
    cool = cv2.applyColorMap(255 - gray, cv2.COLORMAP_OCEAN)
    blend = cv2.addWeighted(warm, 0.62, cool, 0.38, 0)
    return cv2.addWeighted(blend, 0.72, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.28, 0)


def mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mean_chroma(img: np.ndarray) -> float:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)))


def test_monochrome_tuning(img: np.ndarray) -> dict:
    proxy = proxy_colorize(img)
    base = get_style('monochrome')
    weak = _apply_pastel_tuning(base, {
        'color_strength': 55, 'brightness': 90, 'warmth': 85,
        'person_strength': 70, 'environment_strength': 0,
        'hair_strength': 70, 'skin_strength': 72,
        'eye_strength': 72, 'clothing_strength': 70,
        'softness': 80, 'flatten': 80, 'highlight_preserve': 85,
        'skin_warmth': 90,
    })
    strong = _apply_pastel_tuning(base, {
        'color_strength': 190, 'brightness': 135, 'warmth': 165,
        'person_strength': 180, 'environment_strength': 180,
        'hair_strength': 180, 'skin_strength': 180,
        'eye_strength': 180, 'clothing_strength': 180,
        'softness': 170, 'flatten': 160, 'highlight_preserve': 160,
        'skin_warmth': 160,
    })
    out_a = apply_style_grade(proxy, img, weak, context=None)
    out_b = apply_style_grade(proxy, img, strong, context=None)
    return {
        'delta': mean_abs_delta(out_a, out_b),
        'chroma_a': mean_chroma(out_a),
        'chroma_b': mean_chroma(out_b),
    }


def test_manual_strength(img: np.ndarray) -> dict:
    hm = HintManager()
    hm.bind_source_image(img)
    hm.add_manual_hint(0.5, 0.5, (255, 120, 120), 0.02)
    low = hm.merge_specs(image_bgr=img, manual_strength=0.25)
    high = hm.merge_specs(image_bgr=img, manual_strength=1.0)
    low_m = [h for h in low if h.source == 'manual'][0]
    high_m = [h for h in high if h.source == 'manual'][0]
    return {
        'low_strength': round(float(low_m.strength), 3),
        'high_strength': round(float(high_m.strength), 3),
        'low_radius': round(float(low_m.radius_norm), 5),
        'high_radius': round(float(high_m.radius_norm), 5),
    }


def test_reference_strength_source() -> dict:
    guided = Path('core/guided_colorist.py').read_text(encoding='utf-8')
    return {
        'uses_reference_strength_for_identity': '0.58 * self._reference_strength' in guided,
        'uses_reference_strength_for_scene': '0.30 * self._reference_strength' in guided,
        'pipeline_builds_context': 'build_page_context(' in Path('pipeline.py').read_text(encoding='utf-8'),
    }


def test_worker_passthrough() -> dict:
    worker_text = Path('ui/worker.py').read_text(encoding='utf-8')
    return {
        'colorize_worker_passes_library': 'self._character_library = character_library' in worker_text,
        'colorize_worker_passes_memories': 'self._character_memories = character_memories' in worker_text,
        'batch_worker_passes_library': worker_text.count('self._character_library = character_library') >= 2,
        'batch_worker_passes_memories': worker_text.count('self._character_memories = character_memories') >= 2,
    }


def main() -> int:
    lines = []
    lines.append('# Effect-control validation')
    lines.append('')
    lines.append('This report uses the three user-supplied manga pages as validation inputs.')
    lines.append('The monochrome tuning test uses a deterministic pseudo-color proxy so it can run without mc-v2 weights.')
    lines.append('')

    all_ok = True
    mono_rows = []
    for path in PAGE_PATHS:
        img = load_page(path)
        mono = test_monochrome_tuning(img)
        mono_rows.append((path.name, mono))
        ok = mono['delta'] >= 3.0 and abs(mono['chroma_b'] - mono['chroma_a']) >= 2.0
        all_ok &= ok
    lines.append('## 1) Monochrome fine-tuning changes the page')
    lines.append('')
    for name, mono in mono_rows:
        lines.append(f'- `{name}`: mean pixel delta={mono["delta"]:.2f}, chroma={mono["chroma_a"]:.2f} -> {mono["chroma_b"]:.2f}')
    lines.append('')

    img0 = load_page(PAGE_PATHS[0])
    manual = test_manual_strength(img0)
    manual_ok = manual['high_strength'] > manual['low_strength'] and manual['high_radius'] > manual['low_radius']
    all_ok &= manual_ok
    lines.append('## 2) Manual prompt strength affects composed hints')
    lines.append('')
    lines.append(f'- low strength={manual["low_strength"]}, high strength={manual["high_strength"]}')
    lines.append(f'- low radius={manual["low_radius"]}, high radius={manual["high_radius"]}')
    lines.append('')

    ref = test_reference_strength_source()
    ref_ok = all(ref.values())
    all_ok &= ref_ok
    lines.append('## 3) Reference/identity strength wiring is present again')
    lines.append('')
    for key, value in ref.items():
        lines.append(f'- {key}: {value}')
    lines.append('')

    worker = test_worker_passthrough()
    worker_ok = all(worker.values())
    all_ok &= worker_ok
    lines.append('## 4) UI workers now pass through character/reference objects')
    lines.append('')
    for key, value in worker.items():
        lines.append(f'- {key}: {value}')
    lines.append('')

    lines.append('## Verdict')
    lines.append('')
    lines.append(f'- PASS={all_ok}')

    out = Path('VALIDATION_EFFECT_CONTROLS.md')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out.read_text(encoding='utf-8'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
