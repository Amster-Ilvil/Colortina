"""Built-in final-image compression for Colortina.

The editor always keeps a lossless full-resolution working layer.  This module
encodes a separate immutable export asset after colourization/editing, so the
user can export the already-compressed bytes without recompressing a lossy
image or degrading later edits.

Profiles deliberately use codecs already supplied by Pillow.  No external GUI,
network service, or GPL compressor binary is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
import tempfile
from typing import Final

import numpy as np
from PIL import Image, features


PROFILE_SMART_WEBP: Final[str] = "smart_webp"
PROFILE_WEBP_LOSSLESS: Final[str] = "webp_lossless"
PROFILE_PNG_OPTIMIZED: Final[str] = "png_optimized"
PROFILE_JPEG_HIGH: Final[str] = "jpeg_high"

DEFAULT_PROFILE: Final[str] = PROFILE_PNG_OPTIMIZED

_PROFILE_EXTENSIONS: Final[dict[str, str]] = {
    PROFILE_SMART_WEBP: ".webp",
    PROFILE_WEBP_LOSSLESS: ".webp",
    PROFILE_PNG_OPTIMIZED: ".png",
    PROFILE_JPEG_HIGH: ".jpg",
}


@dataclass(frozen=True)
class CompressionResult:
    data: bytes
    profile: str
    extension: str
    codec: str
    variant: str
    width: int
    height: int
    original_bytes: int
    compressed_bytes: int
    lossless: bool
    psnr_db: float | None = None
    mean_abs_error: float | None = None
    large_error_ratio: float | None = None
    dark_mean_abs_error: float | None = None

    @property
    def ratio(self) -> float:
        if self.original_bytes <= 0:
            return 1.0
        return float(self.compressed_bytes) / float(self.original_bytes)

    @property
    def saved_percent(self) -> float:
        return max(0.0, (1.0 - self.ratio) * 100.0)

    def as_metadata(self) -> dict:
        return {
            "profile": self.profile,
            "extension": self.extension,
            "codec": self.codec,
            "variant": self.variant,
            "width": self.width,
            "height": self.height,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "lossless": self.lossless,
            "psnr_db": self.psnr_db,
            "mean_abs_error": self.mean_abs_error,
            "large_error_ratio": self.large_error_ratio,
            "dark_mean_abs_error": self.dark_mean_abs_error,
            "saved_percent": self.saved_percent,
        }


def normalize_profile(profile: str | None) -> str:
    key = str(profile or DEFAULT_PROFILE).strip().lower()
    if key not in _PROFILE_EXTENSIONS:
        return DEFAULT_PROFILE
    return key


def profile_extension(profile: str | None) -> str:
    return _PROFILE_EXTENSIONS[normalize_profile(profile)]


def profile_is_lossless(profile: str | None) -> bool:
    return normalize_profile(profile) in {
        PROFILE_WEBP_LOSSLESS,
        PROFILE_PNG_OPTIMIZED,
    }


def _validate_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a NumPy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
        raise ValueError("image_bgr must have shape HxWx3 or HxWx4")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_bgr)


def _to_pillow(image_bgr: np.ndarray) -> Image.Image:
    if image_bgr.shape[2] == 4:
        rgba = image_bgr[:, :, [2, 1, 0, 3]]
        return Image.fromarray(rgba, mode="RGBA")
    rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def _encode_pillow(image: Image.Image, fmt: str, **save_kwargs) -> bytes:
    out = BytesIO()
    image.save(out, format=fmt, **save_kwargs)
    return out.getvalue()


def _decode_to_bgr(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as decoded:
        decoded.load()
        if decoded.mode == "RGBA":
            rgba = np.asarray(decoded, dtype=np.uint8)
            return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
        rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])


def _quality_metrics(original: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    """Measure perceptual safety without allocating a full float image.

    The calculation is chunked by rows, keeping peak memory bounded even for
    600-dpi manga pages.  The dark-pixel statistic protects black line art and
    text, where codec ringing is most visible.
    """
    if original.shape != decoded.shape:
        return {
            "psnr_db": 0.0,
            "mean_abs_error": 255.0,
            "large_error_ratio": 1.0,
            "dark_mean_abs_error": 255.0,
        }
    total_values = int(original.size)
    abs_sum = 0
    sq_sum = 0
    large_count = 0
    dark_abs_sum = 0
    dark_values = 0
    rows = original.shape[0]
    step = max(16, min(256, 8_000_000 // max(1, original.shape[1] * original.shape[2])))
    channels = min(3, original.shape[2])

    for y0 in range(0, rows, step):
        y1 = min(rows, y0 + step)
        a = original[y0:y1]
        b = decoded[y0:y1]
        delta = a.astype(np.int16) - b.astype(np.int16)
        absolute = np.abs(delta)
        abs_sum += int(absolute.sum(dtype=np.int64))
        sq_sum += int((delta.astype(np.int32) ** 2).sum(dtype=np.int64))
        large_count += int(np.count_nonzero(absolute > 8))

        # Integer BT.601 luma approximation, BGR order.
        base = a[:, :, :3].astype(np.uint16)
        gray = ((29 * base[:, :, 0] + 150 * base[:, :, 1] +
                 77 * base[:, :, 2] + 128) >> 8)
        dark_mask = gray < 96
        if np.any(dark_mask):
            dark_delta = absolute[:, :, :channels][dark_mask]
            dark_abs_sum += int(dark_delta.sum(dtype=np.int64))
            dark_values += int(dark_delta.size)

    mse = float(sq_sum) / float(max(1, total_values))
    psnr = float("inf") if mse <= 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return {
        "psnr_db": psnr,
        "mean_abs_error": float(abs_sum) / float(max(1, total_values)),
        "large_error_ratio": float(large_count) / float(max(1, total_values)),
        "dark_mean_abs_error": (
            float(dark_abs_sum) / float(max(1, dark_values))
            if dark_values else 0.0),
    }


def _smart_webp(image: Image.Image, image_bgr: np.ndarray) -> tuple[bytes, str, bool, dict]:
    if not features.check("webp"):
        # Pillow builds without WebP are rare, but Colortina must still export.
        data = _encode_pillow(image, "PNG", optimize=True, compress_level=7)
        return data, "png-fallback", True, {
            "psnr_db": float("inf"),
            "mean_abs_error": 0.0,
            "large_error_ratio": 0.0,
            "dark_mean_abs_error": 0.0,
        }

    attempts = (
        (96, {"min_psnr": 45.0, "max_mae": 1.25,
              "max_large": 0.0025, "max_dark_mae": 1.55}),
        (98, {"min_psnr": 48.0, "max_mae": 0.90,
              "max_large": 0.0012, "max_dark_mae": 1.15}),
    )
    for quality, limits in attempts:
        data = _encode_pillow(
            image, "WEBP", quality=quality, method=6, lossless=False,
            exact=True)
        decoded = _decode_to_bgr(data)
        metrics = _quality_metrics(image_bgr, decoded)
        if (metrics["psnr_db"] >= limits["min_psnr"] and
                metrics["mean_abs_error"] <= limits["max_mae"] and
                metrics["large_error_ratio"] <= limits["max_large"] and
                metrics["dark_mean_abs_error"] <= limits["max_dark_mae"]):
            return data, f"webp-q{quality}-guarded", False, metrics

    # The quality guard is intentionally conservative.  Difficult pages fall
    # back to lossless WebP rather than silently accepting line-art damage.
    data = _encode_pillow(
        image, "WEBP", quality=100, method=6, lossless=True, exact=True)
    return data, "webp-lossless-fallback", True, {
        "psnr_db": float("inf"),
        "mean_abs_error": 0.0,
        "large_error_ratio": 0.0,
        "dark_mean_abs_error": 0.0,
    }


def encode_image(image_bgr: np.ndarray, profile: str | None = None) -> CompressionResult:
    image_bgr = _validate_bgr(image_bgr)
    profile = normalize_profile(profile)
    image = _to_pillow(image_bgr)
    h, w = image_bgr.shape[:2]
    raw_bytes = int(image_bgr.nbytes)
    metrics: dict[str, float] = {}

    if profile == PROFILE_SMART_WEBP:
        data, variant, lossless, metrics = _smart_webp(image, image_bgr)
        extension = ".webp" if variant.startswith("webp") else ".png"
        codec = "WebP" if extension == ".webp" else "PNG"
    elif profile == PROFILE_WEBP_LOSSLESS:
        if features.check("webp"):
            data = _encode_pillow(
                image, "WEBP", quality=100, method=6, lossless=True,
                exact=True)
            extension, codec, variant = ".webp", "WebP", "webp-lossless"
        else:
            data = _encode_pillow(image, "PNG", optimize=True, compress_level=7)
            extension, codec, variant = ".png", "PNG", "png-fallback"
        lossless = True
        metrics = {
            "psnr_db": float("inf"), "mean_abs_error": 0.0,
            "large_error_ratio": 0.0, "dark_mean_abs_error": 0.0,
        }
    elif profile == PROFILE_PNG_OPTIMIZED:
        data = _encode_pillow(image, "PNG", optimize=True, compress_level=7)
        extension, codec, variant, lossless = ".png", "PNG", "png-optimized", True
        metrics = {
            "psnr_db": float("inf"), "mean_abs_error": 0.0,
            "large_error_ratio": 0.0, "dark_mean_abs_error": 0.0,
        }
    elif profile == PROFILE_JPEG_HIGH:
        # JPEG cannot preserve alpha; generated Colortina pages are normally RGB.
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        data = _encode_pillow(
            image.convert("RGB"), "JPEG", quality=95, subsampling=0,
            optimize=True, progressive=True)
        decoded = _decode_to_bgr(data)
        metrics = _quality_metrics(image_bgr[:, :, :3], decoded[:, :, :3])
        extension, codec, variant, lossless = ".jpg", "JPEG", "jpeg-q95-444", False
    else:  # normalize_profile currently makes this unreachable.
        raise ValueError(f"Unsupported compression profile: {profile}")

    return CompressionResult(
        data=data,
        profile=profile,
        extension=extension,
        codec=codec,
        variant=variant,
        width=w,
        height=h,
        original_bytes=raw_bytes,
        compressed_bytes=len(data),
        lossless=lossless,
        psnr_db=metrics.get("psnr_db"),
        mean_abs_error=metrics.get("mean_abs_error"),
        large_error_ratio=metrics.get("large_error_ratio"),
        dark_mean_abs_error=metrics.get("dark_mean_abs_error"),
    )


def atomic_write_bytes(path: str, data: bytes) -> str:
    path = os.path.abspath(path)
    parent = os.path.dirname(path) or os.curdir
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return path


def compress_to_file(image_bgr: np.ndarray, path: str,
                     profile: str | None = None) -> tuple[str, CompressionResult]:
    result = encode_image(image_bgr, profile)
    base, current_ext = os.path.splitext(path)
    target = base + result.extension if current_ext.lower() != result.extension else path
    return atomic_write_bytes(target, result.data), result
