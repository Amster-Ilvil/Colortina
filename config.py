import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _runtime_data_dir() -> str:
    """Return a writable data directory for packaged desktop builds.

    Source checkouts keep the historical in-repository layout.  PyInstaller
    bundles, especially a macOS app launched from a DMG or /Applications,
    must not try to download model weights or write project state inside the
    read-only application bundle.
    """
    if not getattr(sys, "frozen", False):
        return BASE_DIR

    if sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    else:
        root = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")

    path = os.path.join(root, "Colortina")
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = _runtime_data_dir()


class Config:
    """Minimal local config. No Flask, no .env parsing — just paths."""

    APP_VERSION = "5.13.27"
    APP_VERSION_LABEL = "V5"

    # Source runs keep using ./models/weights. Frozen desktop releases use a
    # per-user writable directory so first-run downloads work from installed
    # apps as well as directly mounted macOS DMGs.
    WEIGHTS_DIR = os.path.join(DATA_DIR, "models", "weights")
    GENERATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "generator.zip")
    EXTRACTOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "extractor.pth")
    DENOISER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "denoiser")

    # MangaLineExtraction_PyTorch (MIT) — used only by closed-region
    # rectangle/lasso detection on the original black-and-white source.
    MANGA_LINE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "erika.pth")
    MANGA_LINE_MODEL_URL = (
        "https://github.com/ljsabc/MangaLineExtraction_PyTorch/"
        "releases/download/v1/erika.pth")
    MANGA_LINE_MAX_SIDE = int(os.environ.get("MANGA_LINE_MAX_SIDE", "1024"))

    # "auto" picks CUDA > MPS (Apple Silicon) > CPU
    ML_DEVICE = os.environ.get(
        "COLORTINA_DEVICE", os.environ.get("COMICCOLORER_DEVICE", "auto"))
    MCV2_SIZE = int(os.environ.get("MCV2_SIZE", "768"))
    MCV2_DENOISE_SIGMA = int(os.environ.get("MCV2_DENOISE_SIGMA", "15"))

    # Auto color-hint generation (GuidedColorist). False = plain mc-v2
    # auto-colorize with no CLIP-driven palette hints.
    USE_GUIDED_HINTS = False

    # core/presets.py — pick with pipeline.colorize_page(style_key=...,
    # quality_key=...). A caller-supplied StyleProfile (see
    # core.style_engine) overrides DEFAULT_STYLE_KEY when given.
    DEFAULT_STYLE_KEY = "none"
    DEFAULT_QUALITY_KEY = "draft"

    # Character Memory — book-consistent colors for characters whose
    # hair (etc.) differs from every other character's, instead of one
    # shared palette color for the whole "hair" label. See
    # core.character_memory. The UI/worker owns the actual
    # CharacterMemory instances (one per book) and this flag is just the
    # suggested default for new jobs.
    USE_CHARACTER_MEMORY = True

    # Style Engine ("Extract Style" from a color reference page/cover)
    # and Character Memory persistence — one subfolder per project.
    STYLES_DIR = os.path.join(DATA_DIR, "styles")
    CHARACTERS_DIR = os.path.join(DATA_DIR, "characters")

    # Real-ESRGAN anime weights for the "Ultra" quality preset's 4x
    # upscale pass (core.upscaler). Optional — falls back to a Lanczos
    # resize if the weights aren't present or `realesrgan` isn't
    # installed, so this is never a hard requirement.
    ESRGAN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "realesrgan_anime6b.pth")
    # Optional local ONNX models (feature degrades gracefully when absent).
    TEXT_DETECTOR_PATH = os.path.join(WEIGHTS_DIR, "comictextdetector.pt.onnx")
    CHAR_SEG_PATH = os.path.join(WEIGHTS_DIR, "isnetis.onnx")
    ESRGAN_MODEL_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                        "download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth")
