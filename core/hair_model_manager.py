"""On-demand model manager for AI hair parsing backends.

No optional parsing weight is bundled in the source ZIP.  Downloads are started
only after an explicit UI confirmation.  Every file is written atomically via a
``.part`` file and validated before becoming visible to inference code.
"""
from __future__ import annotations

from core.cancellation import OperationCancelled, check_cancelled

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


@dataclass(frozen=True)
class HairModelSpec:
    key: str
    display_name: str
    filename: str
    role: str
    format: str
    size_mb: float
    source_project: str
    source_url: str
    license_name: str
    download_method: str = "http"
    google_drive_id: str | None = None
    min_bytes: int = 1024 * 1024


MODEL_SPECS: dict[str, HairModelSpec] = {
    "styleanime": HairModelSpec(
        key="styleanime",
        display_name="StyleAnime / Danbooru-Parsing 主头发解析模型",
        filename="styleanime_danbooru_parsing.pth",
        role="主模型：动漫 hair/skin/cloth/eyeglass 语义解析",
        format="pytorch_bisenet19",
        size_mb=53.0,
        source_project="zsl2018/StyleAnime",
        source_url="https://drive.google.com/uc?id=1NhvfYmr9grvHw5lFCfNukurB-Sft-Jvo",
        google_drive_id="1NhvfYmr9grvHw5lFCfNukurB-Sft-Jvo",
        license_name="upstream research checkpoint; architecture MIT",
        download_method="gdrive",
        min_bytes=8 * 1024 * 1024,
    ),
    "anime_face_segmentation": HairModelSpec(
        key="anime_face_segmentation",
        display_name="Anime-Face-Segmentation 轻量备用模型",
        filename="anime_face_unet.pth",
        role="轻量备用：7 类动漫脸/头发分割",
        format="pytorch_mobilenet_unet7",
        size_mb=6.13,
        source_project="siyeong0/Anime-Face-Segmentation",
        source_url=("https://raw.githubusercontent.com/siyeong0/"
                    "Anime-Face-Segmentation/main/model/UNet.pth"),
        license_name="MIT",
        min_bytes=5 * 1024 * 1024,
    ),
    "skytnt_foreground": HairModelSpec(
        key="skytnt_foreground",
        display_name="SkyTNT anime-segmentation 防泄露模型",
        filename="isnetis.onnx",
        role="人物前景约束：阻止发色泄露到背景/对白框",
        format="onnx_isnetis",
        size_mb=176.0,
        source_project="SkyTNT/anime-segmentation",
        source_url="https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
        license_name="Apache-2.0",
        min_bytes=150 * 1024 * 1024,
    ),
    "yakhyo_face_parsing": HairModelSpec(
        key="yakhyo_face_parsing",
        display_name="yakhyo face-parsing 辅助排除模型",
        filename="yakhyo_face_parsing_resnet18.onnx",
        role="辅助排除：皮肤、眼镜、衣服和脸部特征",
        format="onnx_bisenet19",
        size_mb=43.0,
        source_project="yakhyo/face-parsing",
        source_url=("https://github.com/yakhyo/face-parsing/releases/"
                    "download/weights/resnet18.onnx?download=1"),
        license_name="MIT",
        min_bytes=35 * 1024 * 1024,
    ),
}

RECOMMENDED_MODEL_KEYS = ("anime_face_segmentation", "skytnt_foreground")
ALL_MODEL_KEYS = tuple(MODEL_SPECS.keys())


def _weights_dir(config=None) -> Path:
    if config is None:
        from config import Config
        config = Config
    return Path(config.WEIGHTS_DIR)


def model_path(key: str, config=None) -> Path:
    return _weights_dir(config) / MODEL_SPECS[key].filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ("state_dict", "model_state_dict", "model", "net", "network"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return obj


def _validate_torch(path: Path, spec: HairModelSpec) -> tuple[bool, str]:
    try:
        import torch
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # older torch
            obj = torch.load(str(path), map_location="cpu")
        state = _unwrap_state_dict(obj)
        if not isinstance(state, dict) or len(state) < 20:
            return False, "权重不是有效的 state_dict"
        tensor_count = sum(1 for value in state.values() if hasattr(value, "shape"))
        if tensor_count < 20:
            return False, "权重张量数量不足"
        return True, f"state_dict={len(state)}, tensors={tensor_count}"
    except Exception as exc:  # noqa: BLE001
        return False, f"PyTorch 权重校验失败：{type(exc).__name__}: {exc}"


def _validate_onnx(path: Path, spec: HairModelSpec) -> tuple[bool, str]:
    # File-size validation remains useful on installations where onnxruntime is
    # deliberately not installed yet.  When available, session construction is
    # a stronger protobuf/graph check.
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 4
        sess = ort.InferenceSession(str(path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if not inputs or not outputs:
            return False, "ONNX 图缺少输入或输出"
        return True, f"input={inputs[0].name}, outputs={len(outputs)}"
    except ImportError:
        return True, "onnxruntime 未安装；已通过文件完整性门槛"
    except Exception as exc:  # noqa: BLE001
        return False, f"ONNX 图校验失败：{type(exc).__name__}: {exc}"


def validate_model(key: str, path: str | Path | None = None,
                   config=None) -> tuple[bool, str]:
    spec = MODEL_SPECS[key]
    target = Path(path) if path is not None else model_path(key, config)
    try:
        size = target.stat().st_size
    except OSError:
        return False, "文件不存在"
    if size < spec.min_bytes:
        return False, f"文件过小：{size / 1024 / 1024:.2f} MB"
    if spec.format.startswith("pytorch_"):
        return _validate_torch(target, spec)
    if spec.format.startswith("onnx_"):
        return _validate_onnx(target, spec)
    return True, "文件存在"


def model_status(config=None, *, deep_validate: bool = False) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for key, spec in MODEL_SPECS.items():
        path = model_path(key, config)
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        ready = exists and size >= spec.min_bytes
        detail = "已安装" if ready else "未安装"
        if ready and deep_validate:
            ready, detail = validate_model(key, path, config)
        rows[key] = {
            **asdict(spec),
            "path": str(path),
            "exists": exists,
            "ready": bool(ready),
            "actual_size_mb": round(size / 1024 / 1024, 2),
            "detail": detail,
        }
    return rows


def total_size_mb(keys) -> float:
    return round(sum(MODEL_SPECS[key].size_mb for key in keys), 2)


def missing_model_keys(keys=ALL_MODEL_KEYS, config=None) -> list[str]:
    status = model_status(config)
    return [key for key in keys if not status[key]["ready"]]


def main_backend_ready(config=None) -> bool:
    status = model_status(config)
    return bool(status["styleanime"]["ready"] or
                status["anime_face_segmentation"]["ready"])


def _emit(callback, message: str):
    if callback:
        callback(str(message))



def _part_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".part")


def _payload_head(path: Path, limit: int = 4096) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except OSError:
        return b""


def _looks_like_error_payload(path: Path) -> tuple[bool, str]:
    """Detect HTML/JSON/Git-LFS pointer files saved as model weights."""
    head = _payload_head(path)
    if not head:
        return True, "文件为空"
    stripped = head.lstrip().lower()
    if stripped.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return True, "下载到的是 Git LFS 指针，不是模型内容"
    if stripped.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return True, "下载到的是网页/错误页面，不是模型内容"
    if stripped.startswith((b"{", b"[")) and any(
            token in stripped[:1024] for token in (b"error", b"message", b"not found", b"expired")):
        return True, "下载到的是 JSON 错误响应，不是模型内容"
    if any(token in stripped[:2048] for token in (
            b"jwt: expired", b"access denied", b"rate limit exceeded",
            b"repository not found", b"page not found")):
        return True, "下载响应包含访问错误信息"
    return False, ""


def _http_urls(spec: HairModelSpec) -> list[str]:
    urls = [spec.source_url]
    if spec.key == "yakhyo_face_parsing":
        base = ("https://github.com/yakhyo/face-parsing/releases/"
                "download/weights/resnet18.onnx")
        urls = [base + "?download=1", base + f"?download=1&nocache={int(time.time())}"]
    return list(dict.fromkeys(urls))


def _clean_invalid_download(key: str, destination: Path, config=None) -> list[str]:
    """Remove corrupt final/partial files so retry starts from byte zero."""
    removed = []
    part = _part_path(destination)
    for candidate in (destination, part):
        if not candidate.exists():
            continue
        suspicious, _reason = _looks_like_error_payload(candidate)
        size = candidate.stat().st_size
        # A non-suspicious short .part is a legitimate interrupted download and
        # should remain resumable. Full-sized partials must pass graph/checkpoint
        # validation or they will loop forever as corrupt resumes.
        if candidate == part and not suspicious and size < MODEL_SPECS[key].min_bytes:
            continue
        valid = False
        if not suspicious and size >= MODEL_SPECS[key].min_bytes:
            valid, _detail = validate_model(key, candidate, config)
        if suspicious or not valid:
            try:
                candidate.unlink()
                removed.append(str(candidate))
            except OSError:
                pass
    return removed


def _download_http_once(spec: HairModelSpec, url: str, destination: Path,
                        callback=None, *, allow_resume: bool = True,
                        cancel_check=None) -> Path:
    check_cancelled(cancel_check)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(destination)
    if part.exists():
        suspicious, reason = _looks_like_error_payload(part)
        if suspicious:
            _emit(callback, f"检测到损坏的续传文件，已删除：{reason}")
            part.unlink(missing_ok=True)
    offset = part.stat().st_size if allow_resume and part.exists() else 0
    headers = {
        "User-Agent": "Colortina/5.12 AI-Hair-Model-Manager",
        "Accept": "application/octet-stream,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    _emit(callback, f"正在下载 {spec.display_name}（约 {spec.size_mb:g} MB）…")
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            part.unlink(missing_ok=True)
            return _download_http_once(spec, url, destination, callback,
                                       allow_resume=False, cancel_check=cancel_check)
        raise
    with response:
        status = int(getattr(response, "status", response.getcode()) or 200)
        content_type = str(response.headers.get("Content-Type") or "").lower()
        if any(kind in content_type for kind in ("text/html", "application/json", "text/plain")):
            sample = response.read(4096)
            raise RuntimeError(
                f"服务器返回了 {content_type or '文本'}，不是模型文件："
                f"{sample[:160]!r}")
        append = offset > 0 and status == 206
        if not append:
            offset = 0
        length = response.headers.get("Content-Length")
        total = offset + int(length) if length and length.isdigit() else 0
        mode = "ab" if append else "wb"
        downloaded = offset
        last_percent = -1
        with part.open(mode) as handle:
            while True:
                check_cancelled(cancel_check)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_percent + 5:
                        last_percent = percent
                        _emit(callback, (f"{spec.display_name}：{percent}% "
                                         f"({downloaded // 1048576}/{total // 1048576} MB)"))
    suspicious, reason = _looks_like_error_payload(part)
    if suspicious:
        part.unlink(missing_ok=True)
        raise RuntimeError(reason)
    if part.stat().st_size < spec.min_bytes:
        raise RuntimeError(
            f"下载文件过小：{part.stat().st_size / 1048576:.2f} MB，"
            f"预期至少 {spec.min_bytes / 1048576:.2f} MB")
    return part

def _download_http(spec: HairModelSpec, destination: Path, callback=None, *, cancel_check=None) -> Path:
    last_error = None
    for attempt, url in enumerate(_http_urls(spec), start=1):
        check_cancelled(cancel_check)
        try:
            if attempt > 1:
                _part_path(destination).unlink(missing_ok=True)
                _emit(callback, f"首次下载校验失败，正在从零重试 {spec.display_name}…")
            return _download_http_once(
                spec, url, destination, callback, allow_resume=(attempt == 1),
                cancel_check=cancel_check)
        except OperationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= len(_http_urls(spec)):
                break
    raise RuntimeError(f"{spec.display_name} 下载失败：{last_error}")

def _download_gdrive(spec: HairModelSpec, destination: Path, callback=None, *, cancel_check=None) -> Path:
    check_cancelled(cancel_check)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("缺少 gdown，请先安装 requirements.txt") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(destination)
    if part.exists():
        suspicious, _reason = _looks_like_error_payload(part)
        if suspicious:
            part.unlink(missing_ok=True)
    _emit(callback, f"正在下载 {spec.display_name}（约 {spec.size_mb:g} MB）…")
    result = gdown.download(id=spec.google_drive_id, output=str(part),
                            quiet=False, resume=True)
    check_cancelled(cancel_check)
    if not result or not part.exists():
        raise RuntimeError("Google Drive 下载未生成文件")
    suspicious, reason = _looks_like_error_payload(part)
    if suspicious:
        part.unlink(missing_ok=True)
        raise RuntimeError(reason)
    return part


def _extract_if_archive(destination: Path, spec: HairModelSpec):
    """Extract a container ZIP without misclassifying torch.save checkpoints.

    Modern ``torch.save`` files are ZIP containers internally. They must first
    be tested as actual PyTorch checkpoints; blindly opening them as a model
    bundle loses the state_dict and caused every local .pth import to fail.
    """
    if spec.format.startswith("pytorch_"):
        valid, _detail = _validate_torch(destination, spec)
        if valid:
            return
    elif spec.format.startswith("onnx_"):
        valid, _detail = _validate_onnx(destination, spec)
        if valid:
            return
    if not zipfile.is_zipfile(destination):
        return
    with zipfile.ZipFile(destination, "r") as archive:
        candidates = [name for name in archive.namelist()
                      if name.lower().endswith((".pth", ".pt", ".onnx"))]
        if not candidates:
            raise RuntimeError("下载文件是 ZIP，但其中没有可用权重")
        selected = max(candidates, key=lambda name: archive.getinfo(name).file_size)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False,
                                         suffix=Path(selected).suffix) as tmp:
            with archive.open(selected) as src:
                shutil.copyfileobj(src, tmp)
            temp_path = Path(tmp.name)
    os.replace(temp_path, destination)


def _write_manifest(config=None):
    root = _weights_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    rows = model_status(config, deep_validate=False)
    manifest = {
        "schema": "colortina.ai_hair_models.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": rows,
    }
    target = root / "ai_hair_models_manifest.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    os.replace(temp, target)


def ensure_models(keys=ALL_MODEL_KEYS, config=None, callback=None,
                  *, force: bool = False, cancel_check=None) -> dict[str, dict]:
    """Install selected models inside an explicit, temporary network window.

    Callers must obtain user confirmation before invoking this function. Normal
    application runtime remains network-locked, and the lock is restored even
    when a download or validation fails.
    """
    check_cancelled(cancel_check)
    requested = [key for key in keys if key in MODEL_SPECS]
    if not requested:
        return model_status(config)

    pending = []
    for key in requested:
        check_cancelled(cancel_check)
        destination = model_path(key, config)
        valid, _detail = validate_model(key, destination, config)
        if force or not valid:
            pending.append(key)
    if not pending:
        for index, key in enumerate(requested, start=1):
            _emit(callback,
                  f"[{index}/{len(requested)}] 已找到 {MODEL_SPECS[key].display_name}")
        _write_manifest(config)
        return model_status(config, deep_validate=True)

    from core.network_lockdown import confirmed_model_download_window
    with confirmed_model_download_window("AI 头发解析模型"):
        for index, key in enumerate(requested, start=1):
            check_cancelled(cancel_check)
            spec = MODEL_SPECS[key]
            destination = model_path(key, config)
            valid, detail = validate_model(key, destination, config)
            if valid and not force:
                _emit(callback, f"[{index}/{len(requested)}] 已找到 {spec.display_name}")
                continue
            removed = _clean_invalid_download(key, destination, config)
            if removed:
                _emit(callback, f"已清理 {spec.display_name} 的损坏文件，将从零下载")
            _emit(callback, f"[{index}/{len(requested)}] 准备 {spec.display_name}")
            candidate = None
            try:
                if spec.download_method == "gdrive":
                    candidate = _download_gdrive(
                        spec, destination, callback, cancel_check=cancel_check)
                else:
                    candidate = _download_http(
                        spec, destination, callback, cancel_check=cancel_check)
                check_cancelled(cancel_check)
                _extract_if_archive(candidate, spec)
                valid, detail = validate_model(key, candidate, config)
                if not valid:
                    # Invalid protobuf/checkpoint cannot be resumed safely.
                    candidate.unlink(missing_ok=True)
                    raise RuntimeError(detail + "；损坏文件已删除，下次会从零重试")
                os.replace(candidate, destination)
                _emit(callback, f"{spec.display_name} 已安装并通过校验")
            except Exception:
                if destination.exists():
                    destination.unlink(missing_ok=True)
                # Network interruption keeps a plausible partial file; format
                # validation failures remove it above to prevent endless corrupt resumes.
                if candidate is not None and candidate.exists():
                    suspicious, _reason = _looks_like_error_payload(candidate)
                    if suspicious:
                        candidate.unlink(missing_ok=True)
                raise
    check_cancelled(cancel_check)
    _write_manifest(config)
    try:
        from core.ai_hair_parser import clear_cache
        clear_cache()
    except Exception:
        pass
    try:
        from core.character_scope import reset_session
        reset_session()
    except Exception:
        pass
    _emit(callback, "AI 发色解析模型准备完成")
    return model_status(config, deep_validate=True)


def repair_invalid_downloads(config=None, callback=None) -> list[str]:
    """Remove corrupt final/partial model files found by deep validation."""
    removed: list[str] = []
    for key in MODEL_SPECS:
        destination = model_path(key, config)
        rows = _clean_invalid_download(key, destination, config)
        removed.extend(rows)
        for row in rows:
            _emit(callback, f"已删除损坏模型文件：{row}")
    if removed:
        _write_manifest(config)
    return removed


def status_summary(config=None) -> str:
    rows = model_status(config)
    ready = sum(1 for row in rows.values() if row["ready"])
    main = "主模型可用" if main_backend_ready(config) else "主模型未安装"
    return f"AI 发色模型：{ready}/{len(rows)} 已安装；{main}"


def install_local_model(key: str, source_path: str | Path, config=None,
                        callback=None) -> dict[str, dict]:
    """Atomically import and validate a user-selected local model file."""
    if key not in MODEL_SPECS:
        raise KeyError(f"未知 AI 头发模型：{key}")
    spec = MODEL_SPECS[key]
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    destination = model_path(key, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _emit(callback, f"正在导入 {spec.display_name}…")
    suffix = source.suffix or destination.suffix
    with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f"{key}-import-", suffix=suffix,
            delete=False) as tmp:
        temp_path = Path(tmp.name)
        with source.open("rb") as src:
            shutil.copyfileobj(src, tmp, length=1024 * 1024)
    try:
        _extract_if_archive(temp_path, spec)
        valid, detail = validate_model(key, temp_path, config)
        if not valid:
            raise RuntimeError(detail)
        os.replace(temp_path, destination)
        _write_manifest(config)
        try:
            from core.ai_hair_parser import clear_cache
            clear_cache()
        except Exception:
            pass
        try:
            from core.character_scope import reset_session
            reset_session()
        except Exception:
            pass
        _emit(callback, f"{spec.display_name} 已导入并通过校验")
        return model_status(config, deep_validate=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
