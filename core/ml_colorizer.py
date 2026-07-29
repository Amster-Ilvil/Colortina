"""ML-based manga/comic colorization using manga-colorization-v2.

Adds:
- Tiled colorization at native page resolution (preserves fine detail).
- Per-panel colorization (uses panel_detector if requested).
- Screentone pre-cleaning before the model sees the page.
- Graceful OOM fallback: shrink first, retry, then drop to CPU.
"""

import gc
import math
import sys
import os
import threading

# Some ops the generator/denoiser use aren't implemented on Apple's MPS
# backend in every torch version (varies release to release). Without
# this, those ops raise `NotImplementedError` and colorization on Mac
# fails outright; with it, torch transparently runs just that op on CPU
# and continues on MPS for everything else. Must be set before torch
# touches the MPS backend, so this needs to happen at import time here,
# before `import torch` below and before any MangaColorizer is built.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

# Make vendor package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendor.manga_colorization_v2.colorizator import MangaColorizator
from core.masks import deherron_screentones
from core.panel_detector import detect_panels, extract_panel_image
from core.lineart_fill import clip_stamp_to_region


def build_hint_arrays(h: int, w: int, size: int,
                      hint_points, label_map=None,
                      page_gray=None, render_mode: str = "soft_region") -> tuple[np.ndarray, np.ndarray]:
    """Rasterize legacy points or v4 HintSpecs into mc-v2 hint channels."""
    from core.hint_rasterizer import (
        rasterize_hint_specs,
        rasterize_hint_specs_legacy,
    )
    mode = str(render_mode or "soft_region").lower()
    if mode in {"legacy", "classic", "original_points", "original-point"}:
        return rasterize_hint_specs_legacy(
            h, w, size, hint_points, label_map=label_map, page_gray=page_gray)
    if mode in {"mixed", "manual_points"}:
        # mc-v2's colour propagation scales with hint COVERAGE — a single
        # point barely moves it (measured: r=3 dot colours ~0% of a region,
        # r=45 only ~18%). So user-placed hints (brush dabs / eyedropper)
        # are expanded to FILL their enclosed line-art region in the hint
        # channel at full strength: one dab = "this whole region is blue",
        # and the model renders it with its own shading. Auto hints stay
        # soft; dabs that land on lines/huge open areas fall back to a
        # hard point.
        from core.hint_spec import HintSpec
        from core.hint_rasterizer import model_geometry, _resized_labels
        specs = [HintSpec.from_any(v) for v in (hint_points or [])]
        hard_sources = {"manual", "eyedropper_hint"}
        hard = [s for s in specs if str(s.source) in hard_sources]
        soft = [s for s in specs if str(s.source) not in hard_sources]
        hint, alpha = rasterize_hint_specs(
            h, w, size, soft, label_map=label_map, page_gray=page_gray)
        if hard:
            ph, pw, rh, rw = model_geometry(h, w, size)
            labels = _resized_labels(label_map, page_gray, rw, rh, w)
            max_region = int(rh * rw * 0.35)
            leftover = []
            for spec in hard:
                px = min(rw - 1, max(0, int(round(spec.x_norm * rw))))
                py = min(rh - 1, max(0, int(round(spec.y_norm * rh))))
                strength = float(np.clip(spec.effective_strength, 0.0, 1.0))
                filled = False
                if labels is not None and strength > 0.01:
                    rid = int(labels[py, px])
                    if rid == 0:
                        # Dab landed on a line/screentone dot — look for a
                        # region in the immediate neighbourhood.
                        y1, y2 = max(0, py - 6), min(rh, py + 7)
                        x1, x2 = max(0, px - 6), min(rw, px + 7)
                        window = labels[y1:y2, x1:x2]
                        nonzero = window[window > 0]
                        if nonzero.size:
                            rid = int(np.bincount(nonzero).argmax())
                    if rid > 0:
                        region = labels == rid
                        area = int(region.count_nonzero()
                                   if hasattr(region, "count_nonzero")
                                   else np.count_nonzero(region))
                        if 0 < area <= max_region:
                            eroded = cv2.erode(region.astype(np.uint8),
                                               np.ones((3, 3), np.uint8))
                            m = (eroded if int(eroded.sum()) else
                                 region.astype(np.uint8)).astype(bool)
                            # ``labels`` / ``m`` cover only the resized content
                            # area (rh, rw), while ``hint`` and ``alpha`` include
                            # mc-v2's bottom/right padding (ph, pw).  Index the
                            # valid content views explicitly; otherwise pages
                            # whose resized height is not divisible by 32 (for
                            # example 817 padded to 832) raise a boolean-shape
                            # IndexError.
                            hint_valid = hint[:rh, :rw]
                            alpha_valid = alpha[:rh, :rw]
                            hint_valid[m] = np.asarray(spec.rgb, dtype=hint.dtype)
                            alpha_valid[m] = np.maximum(alpha_valid[m], strength)
                            filled = True
                if not filled:
                    leftover.append(spec)
            if leftover:
                hard_hint, hard_alpha = rasterize_hint_specs_legacy(
                    h, w, size, leftover, label_map=label_map,
                    page_gray=page_gray)
                m = hard_alpha > 0
                hint[m] = hard_hint[m].astype(hint.dtype)
                alpha[m] = np.maximum(alpha[m], hard_alpha[m])
        return hint, alpha
    return rasterize_hint_specs(
        h, w, size, hint_points, label_map=label_map, page_gray=page_gray)


class MangaColorizer:
    """Wrapper around manga-colorization-v2 with tiled / per-panel paths."""

    def __init__(self, device: str = "auto",
                 generator_path: str = "",
                 extractor_path: str = "",
                 denoiser_weights_dir: str = ""):
        self._lock = threading.Lock()
        self.device_warning: str | None = None
        self._requested_device = str(device or "auto")
        self._device = self._resolve_device(device)
        self._generator_path = generator_path
        self._extractor_path = extractor_path
        self._denoiser_weights_dir = denoiser_weights_dir
        try:
            self._model = MangaColorizator(
                device=self._device,
                generator_path=generator_path,
                extractor_path=extractor_path,
                denoiser_weights_dir=denoiser_weights_dir,
            )
        except Exception as exc:
            if self._device.type == "mps":
                # Some Mac/torch/macOS combinations advertise MPS as
                # available but still fail on model load (driver
                # mismatches, unsupported ops with fallback disabled by
                # the user's env, etc.) — don't take the whole app down,
                # drop to CPU and keep going.
                print(f"[ml_colorizer] MPS failed to load ({exc}); falling back to CPU")
                self.device_warning = f"MPS 加载失败，已回退到 CPU：{exc}"
                self._device = torch.device("cpu")
                self._model = MangaColorizator(
                    device=self._device,
                    generator_path=generator_path,
                    extractor_path=extractor_path,
                    denoiser_weights_dir=denoiser_weights_dir,
                )
            else:
                raise
        self.device_name = str(self._device)
        self.cuda_available = torch.cuda.is_available()

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Apple Silicon GPU via Metal Performance Shaders. Both
            # is_built() (torch itself compiled with MPS support) and
            # is_available() (the current machine/macOS actually has a
            # usable MPS device) need to be true — checking only
            # is_available() still throws on some non-Apple-Silicon or
            # very old torch builds where the mps backend attribute
            # exists but querying it isn't safe.
            try:
                mps_backend = getattr(torch.backends, "mps", None)
                if (mps_backend is not None
                        and mps_backend.is_built()
                        and mps_backend.is_available()):
                    return torch.device("mps")
            except Exception as exc:  # noqa: BLE001 — never let device probing crash startup
                print(f"[ml_colorizer] MPS probe failed ({exc}); using CPU")
            return torch.device("cpu")
        return torch.device(device)

    def to_device(self, device: str) -> None:
        """Move the loaded model between devices without reloading weights."""
        target = self._resolve_device(device)
        if str(target) == self.device_name:
            return
        with self._lock:
            self._model.colorizer.to(target)
            self._model.denoiser.to(target)
            self._model.device = target
            # Drop any staged tensors from the old device
            self._model.current_image = None
            self._model.current_hint = None
            self._device = target
            self.device_name = str(target)

    def switch_device(self, device: str) -> None:
        self.to_device(device)

    # ── Core single-pass colorize at fixed model size ─────────────────────

    def _colorize_at_size(self, rgb: np.ndarray, size: int,
                          denoise_sigma: int,
                          hint_points=None, label_map=None,
                          page_gray=None, hint_render_mode: str = "soft_region") -> np.ndarray:
        """Run mc-v2 once at *size*, returning float32 RGB [0,1]."""
        # fp16 autocast on CUDA: ~1.5-2x faster on tensor-core GPUs, half VRAM
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=self._device.type == "cuda"):
            self._model.set_image(rgb, size=size,
                                  apply_denoise=denoise_sigma > 0,
                                  denoise_sigma=denoise_sigma)
            if hint_points:
                # Feed the guided-coloring palette into the model's native
                # hint channel — it propagates these colors with its own
                # learned shading instead of inventing a global wash
                hint, mask = build_hint_arrays(
                    rgb.shape[0], rgb.shape[1], size, hint_points,
                    label_map=label_map, page_gray=page_gray,
                    render_mode=hint_render_mode)
                self._model.update_hint(hint, mask)
            return self._model.colorize()

    def _safe_colorize(self, rgb: np.ndarray, size: int,
                       denoise_sigma: int, hint_points=None,
                       label_map=None, page_gray=None,
                       hint_render_mode: str = "soft_region") -> np.ndarray:
        """Robust single-pass colorize with shrink-then-CPU OOM fallback."""
        attempts = [size, max(384, size // 2)]
        last_err: RuntimeError | None = None
        for try_size in attempts:
            try:
                return self._colorize_at_size(rgb, try_size, denoise_sigma,
                                              hint_points=hint_points,
                                              label_map=label_map,
                                              page_gray=page_gray,
                                              hint_render_mode=hint_render_mode)
            except RuntimeError as exc:
                last_err = exc
                msg = str(exc).lower()
                if "out of memory" not in msg:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif self._device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                gc.collect()
                continue

        # All GPU attempts OOM'd — fall back to CPU
        if self.device_name != "cpu":
            self._device = torch.device("cpu")
            self._model = MangaColorizator(
                device=self._device,
                generator_path=self._generator_path,
                extractor_path=self._extractor_path,
                denoiser_weights_dir=self._denoiser_weights_dir,
            )
            self.device_name = "cpu"
            return self._colorize_at_size(rgb, max(384, size // 2), denoise_sigma,
                                          hint_points=hint_points,
                                          label_map=label_map,
                                          hint_render_mode=hint_render_mode)

        raise last_err if last_err else RuntimeError("colorize failed")

    # ── Public colorize ────────────────────────────────────────────────────

    def colorize(self, image: np.ndarray, size: int = 576,
                 denoise_sigma: int = 18,
                 *,
                 tiled: bool = False,
                 tile_size: int = 768,
                 tile_overlap: int = 96,
                 per_panel: bool = False,
                 panel_style: str = "western",
                 deherron: bool = False,
                 deherron_strength: float = 0.6,
                 hint_points=None,
                 label_map=None,
                 hint_render_mode: str = "soft_region") -> np.ndarray:
        """Colorize a single B&W page image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 input.
        size : int
            Target resize dimension (must be /32) for the simple path.
        denoise_sigma : int
            FFDNet denoiser strength (0-255).
        tiled : bool
            If True, run tiled colorization at native resolution and
            blend tiles with feathered alpha.
        tile_size, tile_overlap : int
            Tile size and overlap (pixels) for tiled mode.
        per_panel : bool
            If True, detect panels and colorize each independently.
        deherron : bool
            If True, soften screentones before the model sees the page.
        label_map : np.ndarray, optional
            Connected-component region-id array at the page resolution.
            Simple, panel and tiled paths all preserve/remap hints; local
            crops rebuild their region map from area-resized line art.
        """
        # Ensure 3 channels (mc-v2 expects RGB)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if deherron:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cleaned = deherron_screentones(gray, strength=deherron_strength)
            image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        # Tiled / per-panel paths: denoise the whole page ONCE up front
        # instead of re-denoising every (overlapping) tile or panel.
        if (tiled or per_panel) and denoise_sigma > 0:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self._device.type == "cuda"):
                    image = self._model.denoiser.get_denoised_image(
                        image, sigma=denoise_sigma, max_edge=None)
                denoise_sigma = 0
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                # Page too large for one denoise pass — keep per-tile denoising
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if per_panel:
            return self._colorize_per_panel(
                image, size=size, denoise_sigma=denoise_sigma,
                tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap,
                panel_style=panel_style, hint_points=hint_points,
                hint_render_mode=hint_render_mode,
            )

        if tiled:
            return self._colorize_tiled(
                image, denoise_sigma=denoise_sigma,
                tile_size=tile_size, overlap=tile_overlap,
                hint_points=hint_points,
                hint_render_mode=hint_render_mode,
            )

        return self._colorize_simple(image, size=size, denoise_sigma=denoise_sigma,
                                     hint_points=hint_points, label_map=label_map,
                                     hint_render_mode=hint_render_mode)

    # ── Simple resize-and-go path (legacy) ─────────────────────────────────

    def _colorize_simple(self, image: np.ndarray, size: int,
                         denoise_sigma: int, hint_points=None,
                         label_map=None, hint_render_mode: str = "soft_region") -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # Gray page at full res: build_hint_arrays re-derives region labels
        # at model resolution from this, so brush hints clip to real lines.
        page_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if hint_points else None

        with self._lock:
            result = self._safe_colorize(rgb, size, denoise_sigma,
                                         hint_points=hint_points,
                                         label_map=label_map,
                                         page_gray=page_gray,
                                         hint_render_mode=hint_render_mode)

        result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

        if result_bgr.shape[:2] != (orig_h, orig_w):
            rh, rw = result_bgr.shape[:2]
            interp = cv2.INTER_AREA if (rh > orig_h or rw > orig_w) else cv2.INTER_LANCZOS4
            result_bgr = cv2.resize(result_bgr, (orig_w, orig_h), interpolation=interp)
        return result_bgr

    # ── Per-panel path ─────────────────────────────────────────────────────

    @staticmethod
    def _hints_for_crop(hint_points, crop_x: int, crop_y: int,
                        crop_w: int, crop_h: int,
                        page_w: int, page_h: int,
                        *, canvas_w: int | None = None,
                        canvas_h: int | None = None):
        """Remap page-normalized structured or legacy hints into a crop."""
        if not hint_points or crop_w <= 0 or crop_h <= 0:
            return None
        from core.hint_spec import HintSpec
        canvas_w = int(canvas_w or crop_w)
        canvas_h = int(canvas_h or crop_h)
        remapped = []
        for point in hint_points:
            is_spec = isinstance(point, HintSpec)
            spec = HintSpec.from_any(point)
            local = spec.remap_to_crop(
                crop_x, crop_y, crop_w, crop_h, page_w, page_h,
                canvas_w=canvas_w, canvas_h=canvas_h)
            if local is None:
                continue
            remapped.append(local if is_spec else local.as_legacy_point())
        return remapped or None

    @classmethod
    def _hints_for_panel(cls, hint_points, panel, page_w: int, page_h: int):
        return cls._hints_for_crop(
            hint_points, panel.x, panel.y, panel.width, panel.height,
            page_w, page_h)

    def _colorize_per_panel(self, image: np.ndarray, *,
                            size: int, denoise_sigma: int,
                            tiled: bool, tile_size: int, tile_overlap: int,
                            panel_style: str, hint_points=None,
                            hint_render_mode: str = "soft_region") -> np.ndarray:
        """Detect panels, colorize each, composite back into the page."""
        from models.schemas import PanelRegion

        try:
            panels = detect_panels(image, style=panel_style)
        except Exception:
            panels = []

        # If detection found nothing useful, fall back to tiled or simple
        if not panels or len(panels) <= 1:
            if tiled:
                return self._colorize_tiled(
                    image, denoise_sigma=denoise_sigma,
                    tile_size=tile_size, overlap=tile_overlap,
                    hint_points=hint_points,
                    hint_render_mode=hint_render_mode)
            return self._colorize_simple(image, size=size, denoise_sigma=denoise_sigma,
                                         hint_points=hint_points,
                                         hint_render_mode=hint_render_mode)

        page_h, page_w = image.shape[:2]
        out = image.copy()
        for panel in panels:
            crop = extract_panel_image(image, panel)
            if crop.size == 0:
                continue
            ph, pw = crop.shape[:2]
            if min(ph, pw) < 64:
                continue

            panel_hints = self._hints_for_panel(hint_points, panel, page_w, page_h)

            if tiled and max(ph, pw) > tile_size:
                colored = self._colorize_tiled(
                    crop, denoise_sigma=denoise_sigma,
                    tile_size=tile_size, overlap=tile_overlap,
                    hint_points=panel_hints,
                    hint_render_mode=hint_render_mode,
                )
            else:
                colored = self._colorize_simple(
                    crop, size=size, denoise_sigma=denoise_sigma,
                    hint_points=panel_hints,
                    hint_render_mode=hint_render_mode,
                )

            # Composite back, with a small feather at the seam
            seam = self._panel_feather(ph, pw, feather=8)
            target = out[panel.y:panel.y + ph, panel.x:panel.x + pw].astype(np.float32)
            colored_f = colored.astype(np.float32)
            blended = colored_f * seam + target * (1.0 - seam)
            out[panel.y:panel.y + ph, panel.x:panel.x + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        return out

    @staticmethod
    def _panel_feather(h: int, w: int, feather: int = 8) -> np.ndarray:
        """Soft 1.0 mask with a small fade at the borders."""
        feather = max(1, min(feather, min(h, w) // 4))
        m = np.ones((h, w), dtype=np.float32)
        for i in range(feather):
            v = (i + 1) / (feather + 1)
            if i < h:
                m[i, :] *= v
                m[h - 1 - i, :] *= v
            if i < w:
                m[:, i] *= v
                m[:, w - 1 - i] *= v
        return m[:, :, None]

    # ── Tiled native-resolution path ──────────────────────────────────────

    def _colorize_tiled(self, image: np.ndarray, *,
                        denoise_sigma: int,
                        tile_size: int = 768,
                        overlap: int = 96,
                        hint_points=None,
                        hint_render_mode: str = "soft_region") -> np.ndarray:
        """Colorize at native resolution by tiling.

        Each tile is fed to mc-v2 at exactly *tile_size* (the model's
        native input size — no internal downsampling). Tiles overlap by
        ``overlap`` pixels and are composited with a feathered alpha so
        seams disappear.
        """
        h, w = image.shape[:2]

        # Round tile size to /32 (mc-v2 requirement)
        tile_size = max(384, (tile_size // 32) * 32)
        overlap = max(16, min(overlap, tile_size // 4))

        # If the page already fits in one tile, just run the simple path
        if h <= tile_size and w <= tile_size:
            return self._colorize_simple(
                image, size=tile_size, denoise_sigma=denoise_sigma,
                hint_points=hint_points,
                hint_render_mode=hint_render_mode)

        # Whole-page low-resolution color prior.  It provides consistent hue
        # context to every high-resolution tile without replacing tile linework.
        global_prior = None
        try:
            prior_size = min(512, tile_size)
            global_prior = self._colorize_simple(
                image, size=prior_size, denoise_sigma=denoise_sigma,
                hint_points=hint_points,
                hint_render_mode=hint_render_mode)
        except Exception as exc:
            print(f"[ml_colorizer] global prior unavailable: {exc}")

        # Stride: how far we move each step
        stride = tile_size - overlap

        # Compute tile origins so the last tile lands on the edge
        def _origins(extent: int) -> list[int]:
            if extent <= tile_size:
                return [0]
            origins = list(range(0, extent - tile_size, stride))
            origins.append(extent - tile_size)
            return sorted(set(origins))

        ys = _origins(h)
        xs = _origins(w)

        accum = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # Per-tile feather alpha
        feather = self._tile_feather(tile_size, tile_size, overlap)

        for y in ys:
            for x in xs:
                tile_bgr = image[y:y + tile_size, x:x + tile_size]
                # Pad if at edge (model wants exactly tile_size)
                ph = tile_size - tile_bgr.shape[0]
                pw = tile_size - tile_bgr.shape[1]
                if ph > 0 or pw > 0:
                    tile_bgr = cv2.copyMakeBorder(tile_bgr, 0, ph, 0, pw,
                                                  cv2.BORDER_REFLECT)

                use_h = tile_size - ph
                use_w = tile_size - pw
                tile_hints = self._hints_for_crop(
                    hint_points, x, y, use_w, use_h, w, h,
                    canvas_w=tile_size, canvas_h=tile_size)
                tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
                tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY) \
                    if tile_hints else None
                with self._lock:
                    out = self._safe_colorize(
                        tile_rgb, tile_size, denoise_sigma,
                        hint_points=tile_hints, page_gray=tile_gray,
                        hint_render_mode=hint_render_mode)
                out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

                # Crop padding
                out_bgr = out_bgr[:use_h, :use_w]
                if global_prior is not None:
                    prior_crop = global_prior[y:y + use_h, x:x + use_w]
                    source_crop = image[y:y + use_h, x:x + use_w]
                    if prior_crop.shape[:2] == out_bgr.shape[:2]:
                        out_bgr = self._blend_chroma_prior(
                            out_bgr, prior_crop, source_crop, strength=0.26)
                tile_alpha = feather[:use_h, :use_w]

                accum[y:y + use_h, x:x + use_w] += out_bgr.astype(np.float32) * tile_alpha
                weight[y:y + use_h, x:x + use_w] += tile_alpha

        weight = np.maximum(weight, 1e-6)
        return np.clip(accum / weight, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_chroma_prior(tile_bgr: np.ndarray, prior_bgr: np.ndarray,
                            source_bgr: np.ndarray, strength: float = 0.26) -> np.ndarray:
        """Blend only LAB chroma from a whole-page prior into a tile.

        Tile luminance and high-resolution line detail remain untouched.  Paper
        and ink are protected using the original monochrome source.
        """
        strength = float(np.clip(strength, 0.0, 0.5))
        if strength <= 0.0:
            return tile_bgr
        tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        prior_lab = cv2.cvtColor(prior_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = (cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
                if source_bgr.ndim == 3 else source_bgr)
        from core.masks import combined_neutral_mask
        keep = combined_neutral_mask(gray, line_dilate=1, blur=3).astype(np.float32)
        tile_chroma = np.linalg.norm(tile_lab[..., 1:3] - 128.0, axis=2)
        # Low-chroma drift benefits most from the prior; already decisive local
        # colours retain more of the tile model's choice.
        gate = strength * keep * np.clip(1.15 - tile_chroma / 80.0, 0.35, 1.0)
        tile_lab[..., 1:3] = (
            tile_lab[..., 1:3] * (1.0 - gate[..., None]) +
            prior_lab[..., 1:3] * gate[..., None])
        return cv2.cvtColor(np.clip(tile_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _tile_feather(h: int, w: int, overlap: int) -> np.ndarray:
        """Cosine-tapered alpha mask (h, w, 1) peaking at center."""
        ramp_h = np.ones(h, dtype=np.float32)
        ramp_w = np.ones(w, dtype=np.float32)
        if overlap > 0:
            t = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            cos_in = 0.5 - 0.5 * np.cos(t * math.pi)
            ramp_h[:overlap] = cos_in
            ramp_h[-overlap:] = cos_in[::-1]
            ramp_w[:overlap] = cos_in
            ramp_w[-overlap:] = cos_in[::-1]
        return (ramp_h[:, None] * ramp_w[None, :])[:, :, None]

    def unload(self):
        """Release model and free GPU memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) is not None \
                    and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
