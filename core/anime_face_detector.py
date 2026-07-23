"""Lightweight anime/manga face fallback using OpenCV only.

The primary identity path still uses CLIP semantic regions when available.
This module exists for two practical cases:

1. the optional Transformers/CLIP dependency has not been downloaded yet;
2. a dense cover or manga page produces no reliable ``hair`` anchor regions.

It deliberately favours safety over recall.  Detections are filtered by a
simple manga-face structure score, identity matches still pass the normal
Top-2 margin gate, and only regions geometrically attached to a detected face
are eligible for identity locking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import urllib.request

import cv2
import numpy as np


@dataclass
class FaceCandidate:
    bbox: tuple[int, int, int, int]
    head_bbox: tuple[int, int, int, int]
    body_bbox: tuple[int, int, int, int]
    confidence: float
    # Clockwise rotation needed to make the detected face upright.  The
    # dedicated anime cascade is evaluated at 0/90/180/270 degrees so covers
    # with upside-down characters are supported without treating body parts as
    # generic frontal faces.
    rotation: int = 0


def _clip_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x, y, bw, bh = map(int, box)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


def head_bbox_from_face(face: tuple[int, int, int, int],
                        shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = face
    return _clip_bbox((x - 0.34 * w, y - 0.52 * h,
                       1.68 * w, 1.78 * h), shape)


def body_bbox_from_face(face: tuple[int, int, int, int],
                        shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = face
    return _clip_bbox((x - 1.00 * w, y - 0.30 * h,
                       3.00 * w, 5.10 * h), shape)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _face_structure_score(image_bgr: np.ndarray,
                          box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0 or min(crop.shape[:2]) < 20:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:h, 0:w]
    ellipse = (((xx - w * 0.50) / max(1.0, w * 0.39)) ** 2 +
               ((yy - h * 0.55) / max(1.0, h * 0.46)) ** 2) <= 1.0
    central = gray[ellipse]
    if central.size == 0:
        return 0.0
    light_ratio = float(np.mean(central > 115))
    white_ratio = float(np.mean(central > 245))

    eye_band = ((yy > h * 0.24) & (yy < h * 0.62) &
                (xx > w * 0.10) & (xx < w * 0.90) & ellipse)
    dark = ((gray < 105) & eye_band).astype(np.uint8)
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if max(2, int(w * h * 0.0005)) <= area <= max(20, int(w * h * 0.075)):
            comps.append((area, float(cents[i][0]), float(cents[i][1])))
    pair_score = 0.0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            dx, dy = abs(a[1] - b[1]), abs(a[2] - b[2])
            if dx > w * 0.16 and dy < h * 0.16:
                pair_score = max(pair_score, 1.0 - min(1.0, dy / max(1.0, h * 0.16)))

    # Colour references get an additional skin-likeness signal.  On monochrome
    # pages this term stays near zero and does not reject a valid face.
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin = ((Y > 45) & (Y < 250) & (Cr > 118) & (Cr < 188) &
            (Cb > 70) & (Cb < 150) & ellipse)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    colorfulness = float(np.mean(hsv[..., 1] > 20))
    skin_ratio = float(skin.sum() / max(1, ellipse.sum())) if colorfulness > 0.05 else 0.0

    # Text glyphs often trigger tiny Haar boxes.  They tend to have almost no
    # mid-tone face area and no plausible left/right eye pair.
    midtone = float(np.mean((central > 75) & (central < 245)))
    score = (0.26 * np.clip(light_ratio, 0.0, 1.0) +
             0.22 * np.clip(midtone * 1.8, 0.0, 1.0) +
             0.30 * pair_score +
             0.22 * np.clip(skin_ratio * 2.4, 0.0, 1.0))
    if white_ratio > 0.94 and pair_score < 0.2:
        score *= 0.35
    return float(np.clip(score, 0.0, 1.0))


_CASCADE_DOWNLOAD_ATTEMPTED = False

_ANIME_CASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/"
    "lbpcascade_animeface/master/lbpcascade_animeface.xml"
)


def _bundled_cascade_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "lbpcascade_animeface.xml"


def _cached_cascade_path() -> Path:
    root = Path(os.environ.get(
        "COLORTINA_CACHE_DIR",
        Path.home() / ".cache" / "colortina"))
    return root / "lbpcascade_animeface.xml"


def _valid_cascade(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 10000:
        return False
    try:
        return not cv2.CascadeClassifier(str(path)).empty()
    except Exception:
        return False


def ensure_anime_face_cascade(*, allow_download: bool = True) -> str | None:
    """Return the dedicated anime-face cascade path, downloading once if needed.

    Generic OpenCV frontal-face cascades produced unacceptable false positives
    on the supplied manga cover (hands, highlights and lettering).  They are no
    longer used by default.  The small MIT-licensed anime LBP cascade is cached
    in the user's normal cache directory and can also be bundled in ``assets``.
    """
    global _CASCADE_DOWNLOAD_ATTEMPTED
    for path in (_bundled_cascade_path(), _cached_cascade_path()):
        if _valid_cascade(path):
            return str(path)
    if _CASCADE_DOWNLOAD_ATTEMPTED:
        return None
    if not allow_download or os.environ.get("COLORTINA_OFFLINE", "").lower() in {
            "1", "true", "yes"}:
        return None
    _CASCADE_DOWNLOAD_ATTEMPTED = True
    target = _cached_cascade_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            _ANIME_CASCADE_URL,
            headers={"User-Agent": "Colortina/5 anime-face-model-downloader"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read()
        if len(payload) < 10000 or b"<cascade>" not in payload:
            return None
        with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix="animeface-", suffix=".xml",
                delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        if not _valid_cascade(tmp_path):
            tmp_path.unlink(missing_ok=True)
            return None
        os.replace(tmp_path, target)
        return str(target)
    except Exception as exc:
        print(f"[anime_face_detector] anime cascade unavailable: {exc}")
        return None


def _rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    rotation %= 360
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _box_from_rotated(box: tuple[int, int, int, int],
                      original_shape: tuple[int, int],
                      rotation: int) -> tuple[int, int, int, int]:
    """Map an axis-aligned box from a rotated image back to the original."""
    H, W = original_shape
    x, y, w, h = map(float, box)
    corners = np.asarray([
        [x, y], [x + w, y], [x, y + h], [x + w, y + h]
    ], dtype=np.float32)
    rotation %= 360
    if rotation == 90:       # rotated -> original: (xr, yr) => (yr, H-xr)
        mapped = np.column_stack([corners[:, 1], H - corners[:, 0]])
    elif rotation == 180:
        mapped = np.column_stack([W - corners[:, 0], H - corners[:, 1]])
    elif rotation == 270:    # rotated -> original: (xr, yr) => (W-yr, xr)
        mapped = np.column_stack([W - corners[:, 1], corners[:, 0]])
    else:
        mapped = corners
    x1, y1 = np.floor(mapped.min(axis=0)).astype(int)
    x2, y2 = np.ceil(mapped.max(axis=0)).astype(int)
    return _clip_bbox((x1, y1, x2 - x1, y2 - y1), (H, W))


def _box_to_rotated(box: tuple[int, int, int, int],
                    original_shape: tuple[int, int],
                    rotation: int) -> tuple[int, int, int, int]:
    """Map an original-image box into the rotated coordinate system."""
    H, W = original_shape
    x, y, w, h = map(float, box)
    corners = np.asarray([
        [x, y], [x + w, y], [x, y + h], [x + w, y + h]
    ], dtype=np.float32)
    rotation %= 360
    if rotation == 90:       # original -> cw: (x, y) => (H-y, x)
        mapped = np.column_stack([H - corners[:, 1], corners[:, 0]])
        shape = (W, H)
    elif rotation == 180:
        mapped = np.column_stack([W - corners[:, 0], H - corners[:, 1]])
        shape = (H, W)
    elif rotation == 270:    # original -> ccw: (x, y) => (y, W-x)
        mapped = np.column_stack([corners[:, 1], W - corners[:, 0]])
        shape = (W, H)
    else:
        mapped = corners
        shape = (H, W)
    x1, y1 = np.floor(mapped.min(axis=0)).astype(int)
    x2, y2 = np.ceil(mapped.max(axis=0)).astype(int)
    return _clip_bbox((x1, y1, x2 - x1, y2 - y1), shape)


def upright_face_view(image_bgr: np.ndarray,
                      face: FaceCandidate) -> tuple[np.ndarray, FaceCandidate]:
    """Return an upright image/candidate pair for palette/feature sampling."""
    rotation = int(face.rotation) % 360
    if rotation == 0:
        return image_bgr, face
    H, W = image_bgr.shape[:2]
    rotated = _rotate_image(image_bgr, rotation)
    rb = _box_to_rotated(face.bbox, (H, W), rotation)
    rh = _box_to_rotated(face.head_bbox, (H, W), rotation)
    rbody = _box_to_rotated(face.body_bbox, (H, W), rotation)
    return rotated, FaceCandidate(
        bbox=rb, head_bbox=rh, body_bbox=rbody,
        confidence=face.confidence, rotation=0)


def _generic_haar_faces(image_bgr: np.ndarray, max_faces: int) -> list[FaceCandidate]:
    """Opt-in compatibility fallback; disabled because it is noisy on manga."""
    H, W = image_bgr.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    min_face = max(40, int(min(H, W) * 0.035))
    raw = []
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        found = cascade.detectMultiScale(
            gray, scaleFactor=1.075, minNeighbors=4,
            minSize=(min_face, min_face))
        raw.extend(tuple(map(int, box)) for box in found)
    scored = []
    for box in raw:
        score = _face_structure_score(image_bgr, box)
        if score >= 0.45:
            scored.append((score, box))
    scored.sort(reverse=True)
    kept = []
    for score, box in scored:
        if any(_iou(box, other.bbox) > 0.30 for other in kept):
            continue
        kept.append(FaceCandidate(
            bbox=box,
            head_bbox=head_bbox_from_face(box, (H, W)),
            body_bbox=body_bbox_from_face(box, (H, W)),
            confidence=float(score), rotation=0))
        if len(kept) >= max_faces:
            break
    return kept


def detect_anime_faces(image_bgr: np.ndarray, max_faces: int = 32,
                       *, allow_download: bool = True) -> list[FaceCandidate]:
    """Detect anime/manga faces with the dedicated LBP cascade.

    Four orientations are evaluated so rotated cover characters are supported.
    When the optional model cannot be obtained, the safe result is an empty
    list; callers can offer manual reference enrolment instead of polluting the
    character library with false positives.  Generic Haar is available only
    through ``COLORTINA_ALLOW_GENERIC_HAAR=1``.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    H, W = image_bgr.shape[:2]
    cascade_path = ensure_anime_face_cascade(allow_download=allow_download)
    if cascade_path is None:
        if os.environ.get("COLORTINA_ALLOW_GENERIC_HAAR", "").lower() in {
                "1", "true", "yes"}:
            return _generic_haar_faces(image_bgr, max_faces)
        return []

    max_dim = max(H, W)
    scale = min(1.0, 1800.0 / max_dim)
    work = (cv2.resize(image_bgr, (round(W * scale), round(H * scale)),
                       interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image_bgr)
    work_h, work_w = work.shape[:2]
    cascade = cv2.CascadeClassifier(cascade_path)
    candidates: list[tuple[float, FaceCandidate]] = []

    for rotation in (0, 90, 180, 270):
        rotated = _rotate_image(work, rotation)
        gray = cv2.equalizeHist(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
        min_side = min(rotated.shape[:2])
        min_face = max(34, int(min_side * 0.025))
        max_face = max(min_face + 1, int(min_side * 0.48))
        found = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4,
            minSize=(min_face, min_face), maxSize=(max_face, max_face))
        for rx, ry, rw, rh in found:
            if not (0.68 <= rw / max(1.0, rh) <= 1.48):
                continue
            rotated_box = (int(rx), int(ry), int(rw), int(rh))
            structure = _face_structure_score(rotated, rotated_box)
            if structure < 0.20:
                continue
            rhead = head_bbox_from_face(rotated_box, rotated.shape[:2])
            rbody = body_bbox_from_face(rotated_box, rotated.shape[:2])
            box_work = _box_from_rotated(rotated_box, (work_h, work_w), rotation)
            head_work = _box_from_rotated(rhead, (work_h, work_w), rotation)
            body_work = _box_from_rotated(rbody, (work_h, work_w), rotation)

            def full(box):
                x, y, bw, bh = box
                return _clip_bbox((round(x / scale), round(y / scale),
                                   round(bw / scale), round(bh / scale)), (H, W))

            candidate = FaceCandidate(
                bbox=full(box_work), head_bbox=full(head_work),
                body_bbox=full(body_work), confidence=float(structure),
                rotation=rotation)
            candidates.append((structure, candidate))

    candidates.sort(key=lambda item: (
        item[0], item[1].bbox[2] * item[1].bbox[3]), reverse=True)
    kept: list[FaceCandidate] = []
    for score, candidate in candidates:
        if any(_iou(candidate.bbox, other.bbox) > 0.28 for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_faces:
            break
    kept.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
    return kept


def lineart_descriptor(image_bgr: np.ndarray,
                       bbox: tuple[int, int, int, int],
                       rotation: int = 0) -> list[float]:
    """Return a compact pose-tolerant grayscale/edge descriptor."""
    H, W = image_bgr.shape[:2]
    x, y, w, h = _clip_bbox(bbox, (H, W))
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return []
    crop = _rotate_image(crop, int(rotation) % 360)
    gray = (crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    norm = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Low-frequency DCT captures silhouette/face layout; pooled edge magnitude
    # captures bangs, eye line and hair contour without depending on colour.
    dct_gray = cv2.dct(norm)[:8, :8].ravel()[1:]
    dct_edge = cv2.dct(mag)[:8, :8].ravel()[1:]
    vec = np.concatenate([dct_gray, dct_edge]).astype(np.float32)
    vec -= float(vec.mean())
    denom = float(np.linalg.norm(vec))
    if denom > 1e-8:
        vec /= denom
    return vec.astype(float).tolist()


def _hex_from_pixels(pixels: np.ndarray) -> str | None:
    if pixels is None or pixels.size == 0:
        return None
    b, g, r = np.median(pixels.astype(np.float32), axis=0)
    return f"#{int(np.clip(r, 0, 255)):02x}{int(np.clip(g, 0, 255)):02x}{int(np.clip(b, 0, 255)):02x}"


def _dominant_cluster(pixels: np.ndarray, skin_lab: np.ndarray | None = None,
                      prefer_dark: bool = False) -> str | None:
    if pixels is None or len(pixels) < 12:
        return _hex_from_pixels(pixels)
    if len(pixels) > 12000:
        idx = np.linspace(0, len(pixels) - 1, 12000).astype(int)
        pixels = pixels[idx]
    data = pixels.astype(np.float32)
    k = min(4, max(1, len(data) // 400))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.7)
    _c, labels, centers = cv2.kmeans(data, k, None, criteria, 4,
                                     cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float32)
    center_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(center_u8, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(center_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    scores = counts / max(1.0, counts.sum())
    scores *= 0.75 + hsv[:, 1] / 255.0 * 0.65
    if prefer_dark:
        scores *= 0.75 + (255.0 - hsv[:, 2]) / 255.0 * 0.65
    if skin_lab is not None:
        dist = np.linalg.norm(lab - skin_lab[None, :], axis=1)
        scores *= np.clip(dist / 24.0, 0.20, 1.25)
    i = int(np.argmax(scores))
    b, g, r = np.clip(centers[i], 0, 255).astype(np.uint8)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def sample_reference_face_palette(image_bgr: np.ndarray,
                                  face: FaceCandidate) -> dict[str, str]:
    """Estimate hair/skin/iris/clothing identity colours around one face."""
    image_bgr, face = upright_face_view(image_bgr, face)
    H, W = image_bgr.shape[:2]
    fx, fy, fw, fh = face.bbox
    hx, hy, hw, hh = face.head_bbox
    bx, by, bw, bh = face.body_bbox
    face_crop = image_bgr[fy:fy + fh, fx:fx + fw]
    head_crop = image_bgr[hy:hy + hh, hx:hx + hw]
    if face_crop.size == 0 or head_crop.size == 0:
        return {}

    out: dict[str, str] = {}
    f_ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(f_ycrcb)
    yy, xx = np.mgrid[0:fh, 0:fw]
    central = (((xx - fw * 0.50) / max(1.0, fw * 0.40)) ** 2 +
               ((yy - fh * 0.58) / max(1.0, fh * 0.48)) ** 2) <= 1.0
    skin = (central & (Y > 45) & (Y < 248) &
            (Cr > 118) & (Cr < 188) & (Cb > 70) & (Cb < 150))
    skin_pixels = face_crop[skin]
    if len(skin_pixels) >= 24:
        skin_hsv = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3),
                                cv2.COLOR_BGR2HSV).reshape(-1, 3)
        skin_pixels = skin_pixels[(skin_hsv[:, 1] < 175) & (skin_hsv[:, 2] > 45)]
        value = _hex_from_pixels(skin_pixels)
        if value:
            out["skin"] = value

    skin_lab = None
    if "skin" in out:
        s = out["skin"].lstrip("#")
        rgb = np.array([[[int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)]]],
                       np.uint8)
        skin_lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    # Hair zone = expanded head minus central face ellipse.  This works for
    # black, brown and bright hair without assuming a specific hue.
    hyy, hxx = np.mgrid[0:hh, 0:hw]
    face_rel_x = fx - hx
    face_rel_y = fy - hy
    inner_face = (((hxx - (face_rel_x + fw * 0.50)) / max(1.0, fw * 0.48)) ** 2 +
                  ((hyy - (face_rel_y + fh * 0.58)) / max(1.0, fh * 0.58)) ** 2) <= 1.0
    head_hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
    head_ycc = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
    hY, hCr, hCb = cv2.split(head_ycc)
    head_skin = ((hY > 45) & (hY < 248) & (hCr > 118) & (hCr < 188) &
                 (hCb > 70) & (hCb < 150))
    hair_zone = ~inner_face & ~head_skin & (head_hsv[..., 2] > 12) & (head_hsv[..., 2] < 248)
    # Give top and side pixels more weight by duplicating them.
    top_side = hair_zone & ((hyy < hh * 0.62) |
                            (hxx < hw * 0.24) | (hxx > hw * 0.76))
    hair_pixels = head_crop[hair_zone]
    weighted = np.concatenate([hair_pixels, head_crop[top_side]], axis=0) if len(hair_pixels) else hair_pixels
    value = _dominant_cluster(weighted, skin_lab=skin_lab, prefer_dark=True)
    if value:
        out["hair"] = value

    # Iris colours: compact chromatic components in the upper face band.
    f_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    eye_band = ((yy > fh * 0.24) & (yy < fh * 0.63) &
                (xx > fw * 0.08) & (xx < fw * 0.92))
    iris = eye_band & ~skin & (f_hsv[..., 1] > 38) & \
           (f_hsv[..., 2] > 28) & (f_hsv[..., 2] < 238)
    n, comps, stats, cents = cv2.connectedComponentsWithStats(iris.astype(np.uint8), 8)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 2 <= area <= max(120, int(fw * fh * 0.045)):
            candidates.append((i, area, float(cents[i][0]), float(cents[i][1])))
    chosen = []
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            dx, dy = abs(a[2] - b[2]), abs(a[3] - b[3])
            if dx < fw * 0.15 or dy > fh * 0.14:
                continue
            balance = abs(a[1] - b[1]) / max(a[1], b[1])
            score = dy / max(1.0, fh) + balance * 0.25
            if best is None or score < best[0]:
                best = (score, a[0], b[0])
    if best is not None:
        chosen = [best[1], best[2]]
    elif candidates:
        chosen = [max(candidates, key=lambda item: item[1])[0]]
    if chosen:
        value = _hex_from_pixels(face_crop[np.isin(comps, chosen)])
        if value:
            out["eyes"] = value

    # Clothing sample below the face, excluding skin-like pixels.
    body = image_bgr[by:by + bh, bx:bx + bw]
    if body.size:
        byy, bxx = np.mgrid[0:body.shape[0], 0:body.shape[1]]
        rel_face_bottom = max(0, fy + fh - by)
        clothing_zone = ((byy > rel_face_bottom) &
                         (bxx > body.shape[1] * 0.15) &
                         (bxx < body.shape[1] * 0.85))
        ycc = cv2.cvtColor(body, cv2.COLOR_BGR2YCrCb)
        bY, bCr, bCb = cv2.split(ycc)
        bskin = ((bY > 45) & (bY < 248) & (bCr > 118) & (bCr < 188) &
                 (bCb > 70) & (bCb < 150))
        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        clothing_zone &= ~bskin & (hsv[..., 2] > 18) & (hsv[..., 2] < 248)
        value = _dominant_cluster(body[clothing_zone], skin_lab=skin_lab)
        if value:
            out["clothing"] = value
    return out


def hair_tone_features(image_bgr: np.ndarray, face: FaceCandidate
                       ) -> tuple[float, list[float], float, float]:
    image_bgr, face = upright_face_view(image_bgr, face)
    gray = (image_bgr if image_bgr.ndim == 2
            else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    hx, hy, hw, hh = face.head_bbox
    fx, fy, fw, fh = face.bbox
    crop = gray[hy:hy + hh, hx:hx + hw]
    if crop.size == 0:
        return 128.0, [], 1.0, 0.0
    yy, xx = np.mgrid[0:hh, 0:hw]
    face_rel_x, face_rel_y = fx - hx, fy - hy
    inner = (((xx - (face_rel_x + fw * 0.5)) / max(1.0, fw * 0.5)) ** 2 +
             ((yy - (face_rel_y + fh * 0.58)) / max(1.0, fh * 0.60)) ** 2) <= 1.0
    zone = ~inner & ((yy < hh * 0.65) | (xx < hw * 0.25) | (xx > hw * 0.75))
    pixels = crop[zone]
    if not len(pixels):
        pixels = crop.reshape(-1)
    hist, _ = np.histogram(pixels, bins=12, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    tone = float(np.median(pixels))
    area_frac = float(hw * hh / max(1.0, image_bgr.shape[0] * image_bgr.shape[1]))
    return tone, hist.astype(float).tolist(), float(hw / max(1.0, hh)), area_frac


def map_regions_to_face(segmentation, face: FaceCandidate,
                        page_shape: tuple[int, int]) -> dict[str, list[int]]:
    """Attach segmented manga regions to one detected face by geometry."""
    fx, fy, fw, fh = face.bbox
    hx, hy, hw, hh = face.head_bbox
    bx, by, bw, bh = face.body_bbox
    cx_face, cy_face = fx + fw * 0.5, fy + fh * 0.55
    candidates = {"hair": [], "skin": [], "eyes": [], "clothing": []}

    for region in segmentation.regions:
        rx, ry, rw, rh = region.bbox
        cx, cy = rx + rw / 2, ry + rh / 2
        if region.area < 3:
            continue
        in_head = hx <= cx <= hx + hw and hy <= cy <= hy + hh
        in_face = fx <= cx <= fx + fw and fy <= cy <= fy + fh
        in_body = bx <= cx <= bx + bw and by <= cy <= by + bh
        if in_face:
            nx = abs(cx - cx_face) / max(1.0, fw * 0.5)
            ny = (cy - fy) / max(1.0, fh)
            if 0.20 <= ny <= 0.68 and nx <= 0.92 and region.frac <= 0.007 and region.mean_gray < 225:
                candidates["eyes"].append((region.area, int(region.label_id)))
            elif 0.08 <= ny <= 1.05 and nx <= 0.95 and 45 <= region.mean_gray <= 252:
                candidates["skin"].append((region.area, int(region.label_id)))
        if in_head and not (fx + fw * 0.15 <= cx <= fx + fw * 0.85 and
                            fy + fh * 0.18 <= cy <= fy + fh * 0.92):
            if region.mean_gray < 238 and region.frac <= 0.09:
                candidates["hair"].append((region.area, int(region.label_id)))
        if in_body and cy > fy + fh * 0.72 and not in_head:
            if 25 <= region.mean_gray <= 250 and region.frac <= 0.12:
                candidates["clothing"].append((region.area, int(region.label_id)))

    # Keep a small set of strongest regions.  Eye candidates prefer two compact
    # components; hair/skin/clothing can be fragmented by screentones.
    result = {}
    result["hair"] = [rid for _a, rid in sorted(candidates["hair"], reverse=True)[:5]]
    result["skin"] = [rid for _a, rid in sorted(candidates["skin"], reverse=True)[:4]]
    result["eyes"] = [rid for _a, rid in sorted(candidates["eyes"], reverse=True)[:2]]
    result["clothing"] = [rid for _a, rid in sorted(candidates["clothing"], reverse=True)[:5]]
    return result
