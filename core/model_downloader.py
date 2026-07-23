"""Download and validate optional model weights.

The downloader is intentionally strict: a failed Google Drive request must not
be reported as success, because that leaves the UI waiting on a model file that
does not exist and makes Auto Colorize appear unresponsive.
"""

from __future__ import annotations

import os
import zipfile



_GENERATOR_ID = "1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t"
_DENOISER_ID = "161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3"


class ModelDownloadError(RuntimeError):
    """Raised when a required weight download is missing or incomplete."""

def _download_via_gdown(url: str, dest: str):
    try:
        import gdown
    except ImportError as exc:
        raise ModelDownloadError(
            "缺少 gdown 依赖，请先执行 pip install -r requirements.txt") from exc
    return gdown.download(url, dest, quiet=False, resume=True)


def model_paths(weights_dir: str) -> tuple[str, str, str, str]:
    generator_path = os.path.join(weights_dir, "generator.zip")
    extractor_path = os.path.join(weights_dir, "extractor.pth")
    denoiser_dir = os.path.join(weights_dir, "denoiser")
    denoiser_path = os.path.join(denoiser_dir, "net_rgb.pth")
    return generator_path, extractor_path, denoiser_dir, denoiser_path


def models_ready(weights_dir: str) -> bool:
    """Return True when the hard-required mc-v2 files are present."""
    generator_path, _extractor_path, _denoiser_dir, denoiser_path = model_paths(weights_dir)
    return (_valid_file(generator_path, min_bytes=1024 * 1024)
            and _valid_file(denoiser_path, min_bytes=64 * 1024))


def _valid_file(path: str, min_bytes: int = 1) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _gdrive_download(file_id: str, dest: str, label: str, callback=None,
                     min_bytes: int = 1):
    if _valid_file(dest, min_bytes=min_bytes):
        if callback:
            callback(f"已找到 {label}")
        return dest

    # Delete zero-byte/HTML/error remnants before retrying.
    try:
        if os.path.exists(dest):
            os.remove(dest)
    except OSError:
        pass

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if callback:
        callback(f"首次运行：正在下载 {label}，请保持网络连接...")

    url = f"https://drive.google.com/uc?id={file_id}"
    result = _download_via_gdown(url, dest)
    if not result or not _valid_file(dest, min_bytes=min_bytes):
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        raise ModelDownloadError(
            f"{label} 下载失败或文件不完整。请检查网络后重试，目标位置：{dest}")

    if callback:
        callback(f"{label} 下载完成")
    return dest


def ensure_models_downloaded(weights_dir: str, callback=None):
    """Download all required mc-v2 weights into ``weights_dir``."""
    os.makedirs(weights_dir, exist_ok=True)
    generator_path, extractor_path, denoiser_dir, denoiser_path = model_paths(weights_dir)

    _gdrive_download(
        _GENERATOR_ID, generator_path, "generator 权重（约 400 MB）",
        callback, min_bytes=1024 * 1024)

    if not _valid_file(extractor_path, min_bytes=64 * 1024):
        # torch.save files are zip containers too; only extract when an actual
        # extractor member exists. The generator itself may already contain the
        # encoder weights, so extractor.pth remains optional.
        if zipfile.is_zipfile(generator_path):
            try:
                with zipfile.ZipFile(generator_path, "r") as zf:
                    names = [n for n in zf.namelist() if "extractor" in n.lower()]
                    if names:
                        if callback:
                            callback("正在提取 extractor 权重...")
                        with zf.open(names[0]) as src, open(extractor_path, "wb") as dst:
                            dst.write(src.read())
            except (OSError, zipfile.BadZipFile):
                pass

    _gdrive_download(
        _DENOISER_ID, denoiser_path, "denoiser 权重（约 7 MB）",
        callback, min_bytes=64 * 1024)

    if callback:
        callback("mc-v2 模型文件准备完成")
    return generator_path, extractor_path, denoiser_dir


def ensure_manganinja_downloaded(config, callback=None):
    from huggingface_hub import hf_hub_download

    weights_dir = config.MANGANINJA_WEIGHTS_DIR
    os.makedirs(weights_dir, exist_ok=True)
    files = [
        ("denoising_unet.pth", config.MANGANINJA_DENOISING_UNET),
        ("reference_unet.pth", config.MANGANINJA_REFERENCE_UNET),
        ("point_net.pth", config.MANGANINJA_POINTNET),
        ("controlnet.pth", config.MANGANINJA_CONTROLNET),
    ]
    for fname, dest_path in files:
        if os.path.exists(dest_path):
            continue
        if callback:
            callback(f"Downloading MangaNinja {fname}...")
        downloaded = hf_hub_download(
            repo_id=config.MANGANINJA_HF_REPO,
            filename=fname,
            local_dir=weights_dir,
        )
        if downloaded != dest_path and os.path.exists(downloaded):
            import shutil
            shutil.move(downloaded, dest_path)
        if callback:
            callback(f"Downloaded MangaNinja {fname}")


def ensure_esrgan_downloaded(config, callback=None):
    dest = config.ESRGAN_MODEL_PATH
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if callback:
        callback("Downloading Real-ESRGAN weights (~17 MB)...")
    import urllib.request
    urllib.request.urlretrieve(config.ESRGAN_MODEL_URL, dest)
    if callback:
        callback("Downloaded Real-ESRGAN weights")
