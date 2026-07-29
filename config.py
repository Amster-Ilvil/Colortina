import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Minimal local config. No Flask, no .env parsing — just paths."""

    APP_VERSION = "5.4.0"
    APP_VERSION_LABEL = "V5"

    WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
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
    STYLES_DIR = os.path.join(BASE_DIR, "styles")
    CHARACTERS_DIR = os.path.join(BASE_DIR, "characters")

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
