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


def generator_file_ready(path: str, min_bytes: int = 1024 * 1024) -> bool:
    """Quick structural guard against an HTML response or truncated ZIP.

    Older PyTorch can write legacy pickle-form checkpoints. The safe
    weights_only=True deserializer remains the final authority at load time.
    """
    if not _valid_file(path, min_bytes=min_bytes):
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header.startswith(b"PK\x03\x04"):
            return zipfile.is_zipfile(path)
        return header[:2] == b"\x80\x02" or header[:2] in {b"\x80\x03", b"\x80\x04", b"\x80\x05"}
    except (OSError, zipfile.BadZipFile):
        return False


def models_ready(weights_dir: str) -> bool:
    """Return True when the hard-required mc-v2 generator is present.

    FFDNet is an optional pre-cleaning stage.  The vendored denoiser already
    returns the original image when ``net_rgb.pth`` is absent, therefore the
    GUI must not report the whole colorizer as unavailable merely because the
    optional 3.4 MB denoiser weight has not been downloaded.
    """
    generator_path, _extractor_path, _denoiser_dir, _denoiser_path = model_paths(weights_dir)
    return generator_file_ready(generator_path)


def _valid_file(path: str, min_bytes: int = 1) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _gdrive_download(file_id: str, dest: str, label: str, callback=None,
                     min_bytes: int = 1):
    validate = (lambda path: generator_file_ready(path, min_bytes=min_bytes)
                if file_id == _GENERATOR_ID else _valid_file(path, min_bytes=min_bytes))
    if validate(dest):
        if callback:
            callback(f"已找到 {label}")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if callback:
        callback(f"首次使用：正在下载 {label}，请保持网络连接...")
    temp = dest + ".part"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        result = _download_via_gdown(url, temp)
        if not result or not validate(temp):
            raise ModelDownloadError(
                f"{label} 下载失败或校验未通过；保留 .part 文件供检查：{temp}")
        os.replace(temp, dest)
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise ModelDownloadError(f"{label} 下载失败：{exc}") from exc
    if callback:
        callback(f"{label} 下载完成")
    return dest


def ensure_models_downloaded(weights_dir: str, callback=None, *, allow_download: bool = False):
    """Require explicit opt-in before fetching the base generator.

    Previously an eager UI preload invoked this function during window
    construction, downloading without asking. A valid local generator is
    sufficient; optional denoiser weights are never fetched implicitly.
    """
    generator_path, extractor_path, denoiser_dir, _denoiser_path = model_paths(weights_dir)
    if generator_file_ready(generator_path):
        if callback:
            callback("本地 mc-v2 模型已就绪（不会联网）")
        return generator_path, extractor_path, denoiser_dir
    offline = os.environ.get("COLORTINA_OFFLINE", "").strip().lower() in {
        "1", "true", "yes", "on"}
    if offline:
        raise ModelDownloadError(
            f"离线模式下未找到本地 mc-v2 generator：{generator_path}")
    if not allow_download:
        raise ModelDownloadError(
            "缺少 mc-v2 generator 模型。请在界面确认首次模型下载，"
            "或者将 generator.zip 放入 models/weights 后重试。")
    from core.network_lockdown import confirmed_model_download_window
    with confirmed_model_download_window("mc-v2 generator"):
        os.makedirs(weights_dir, exist_ok=True)
        _gdrive_download(
            _GENERATOR_ID, generator_path, "generator 权重（约 129 MB）",
            callback, min_bytes=1024 * 1024)
    if callback:
        callback("mc-v2 主模型下载完成；可选降噪模型未自动下载")
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


def manga_line_model_ready(weights_path: str) -> bool:
    """Return True when the official erika.pth weight appears complete."""
    return _valid_file(weights_path, min_bytes=64 * 1024 * 1024)


def _download_http_resume(url: str, dest: str, label: str, callback=None,
                          min_bytes: int = 1, cancel_check=None) -> str:
    """Download a large release asset atomically, resuming a .part file."""
    import urllib.request
    from core.cancellation import OperationCancelled, check_cancelled

    check_cancelled(cancel_check)
    if _valid_file(dest, min_bytes=min_bytes):
        if callback:
            callback(f"已找到 {label}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    offset = 0
    try:
        offset = os.path.getsize(part) if os.path.isfile(part) else 0
    except OSError:
        offset = 0
    headers = {"User-Agent": "Colortina/3 MangaLineExtraction downloader"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    if callback:
        callback(
            f"首次使用闭合区域识别：正在下载 {label}"
            + (f"（从 {offset // (1024 * 1024)} MB 继续）" if offset else ""))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            append = offset > 0 and int(status) == 206
            # A server may ignore Range and send 200, in which case restart.
            # If it does send 206, verify the returned starting byte before
            # appending; otherwise the model checkpoint becomes corrupted.
            range_total = None
            range_end = None
            if int(status) == 206:
                import re
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
                expected_start = offset if append else 0
                if not match or int(match.group(1)) != expected_start:
                    raise ModelDownloadError(
                        f"{label} 服务端断点续传响应不匹配：{content_range!r}")
                range_end = int(match.group(2))
                if range_end < expected_start:
                    raise ModelDownloadError(f"{label} 非法 Content-Range：{content_range!r}")
                if match.group(3) != "*":
                    range_total = int(match.group(3))
                    if range_end >= range_total:
                        raise ModelDownloadError(f"{label} 非法 Content-Range 总长：{content_range!r}")
            if not append:
                offset = 0
            total_header = response.headers.get("Content-Length")
            total = offset + int(total_header) if total_header and total_header.isdigit() else 0
            mode = "ab" if append else "wb"
            downloaded = offset
            last_percent = -1
            with open(part, mode) as handle:
                while True:
                    check_cancelled(cancel_check)
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    check_cancelled(cancel_check)
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if callback and total > 0:
                        percent = int(downloaded * 100 / total)
                        if percent >= last_percent + 5:
                            last_percent = percent
                            callback(
                                f"正在下载 {label}：{percent}% "
                                f"({downloaded // (1024 * 1024)}/{total // (1024 * 1024)} MB)")
        check_cancelled(cancel_check)
        # Content-Range provides the complete object size even when the
        # remaining request is a smaller 206 response.
        if range_end is not None and downloaded != range_end + 1:
            raise ModelDownloadError(
                f"{label} 本段响应长度不符：收到 {downloaded}，预计 {range_end + 1}")
        if range_total is not None and downloaded != range_total:
            raise ModelDownloadError(
                f"{label} 文件仍未完整：实际 {downloaded}，总长 {range_total}；"
                "保留 .part 文件供下次续传")
        if total and downloaded != total:
            raise ModelDownloadError(
                f"{label} 下载字节数不符：实际 {downloaded}，预计 {total}；"
                "保留 .part 文件供下次续传")
        if not _valid_file(part, min_bytes=min_bytes):
            raise ModelDownloadError(f"{label} 下载不完整：{part}")
        os.replace(part, dest)
    except OperationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelDownloadError(f"{label} 下载失败：{exc}") from exc
    if callback:
        callback(f"{label} 下载完成")
    return dest


def ensure_manga_line_model_downloaded(config, callback=None, cancel_check=None,
                                       *, allow_download: bool = False) -> str:
    """Load locally or, after confirmation, download the erika checkpoint."""
    path = config.MANGA_LINE_WEIGHTS_PATH
    if manga_line_model_ready(path):
        return path
    if os.environ.get("COLORTINA_OFFLINE", "").strip().lower() in {"1", "true", "on", "yes"}:
        raise ModelDownloadError(f"离线模式下未找到漫画结构线模型：{path}")
    if not allow_download:
        raise ModelDownloadError("需要先确认下载 MangaLineExtraction 模型（约 170 MB）")
    from core.network_lockdown import confirmed_model_download_window
    with confirmed_model_download_window("MangaLineExtraction erika"):
        return _download_http_resume(
            config.MANGA_LINE_MODEL_URL, path,
            "漫画结构线模型 erika.pth（约 170 MB）",
            callback=callback, min_bytes=64 * 1024 * 1024,
            cancel_check=cancel_check)


def optional_hair_model_status(config) -> dict:
    """Compatibility wrapper around the dedicated AI hair model manager."""
    from core.hair_model_manager import model_status
    return model_status(config)


def ensure_hair_models_downloaded(config, keys=None, callback=None, *, force=False, cancel_check=None):
    """Download selected AI hair parsing models after caller confirmation."""
    from core.hair_model_manager import ALL_MODEL_KEYS, ensure_models
    return ensure_models(keys or ALL_MODEL_KEYS, config=config,
                         callback=callback, force=force, cancel_check=cancel_check)
