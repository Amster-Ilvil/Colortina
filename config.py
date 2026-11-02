import os
import sys


def _runtime_data_dir() -> str:
    """Return the writable directory used for models and user data.

    PyInstaller one-file applications unpack Python modules into a temporary
    directory. Model files must stay beside the executable instead, so they
    survive restarts and can be replaced without rebuilding the EXE.
    ``COLORTINA_DATA_DIR`` is an optional override for installations that keep
    large models on another drive.
    """
    override = os.environ.get("COLORTINA_DATA_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            return os.path.join(os.path.expanduser("~"), "Library",
                                "Application Support", "Colortina")
        # Keep the historical Windows portable directory layout. A future
        # per-user MSI can override this with COLORTINA_DATA_DIR.
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _runtime_data_dir()


class Config:
    """Minimal local config. No Flask, no .env parsing — just paths."""

    APP_VERSION = "5.20.0"
    APP_VERSION_LABEL = "V5.20.0 RC"

    WEIGHTS_DIR = os.path.abspath(os.environ.get(
        "COLORTINA_WEIGHTS_DIR",
        os.path.join(BASE_DIR, "models", "weights")))
    GENERATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "generator.zip")
    EXTRACTOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "extractor.pth")
    REGION_ADAPTER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "region_control_adapter.pth")
    # Optional head trained only on frozen generator encoder features. Missing
    # or invalid files fail closed to the existing semantic parser path.
    GENERATOR_SEMANTIC_HEAD_PATH = os.path.join(
        WEIGHTS_DIR, "generator_semantic_head.pth")
    GENERATOR_SEMANTIC_HEAD_ENABLED = os.environ.get(
        "COLORTINA_GENERATOR_SEMANTIC_HEAD", "1").strip().lower() not in {
            "0", "false", "off", "no"}
    GENERATOR_SEMANTIC_HEAD_SIZE = int(
        os.environ.get("COLORTINA_GENERATOR_SEMANTIC_SIZE", "576"))
    GENERATOR_SEMANTIC_HEAD_CONFIDENCE = float(
        os.environ.get("COLORTINA_GENERATOR_SEMANTIC_CONFIDENCE", "0.45"))
    # Optional tiny decoder-side adapter. It is never part of generator.zip.
    TUNNEL2_SEMANTIC_ADAPTER_PATH = os.path.join(
        WEIGHTS_DIR, "tunnel2_semantic_color_adapter.pth")
    TUNNEL2_SEMANTIC_ADAPTER_ENABLED = os.environ.get(
        "COLORTINA_TUNNEL2_SEMANTIC_ADAPTER", "1").strip().lower() not in {
            "0", "false", "off", "no"}
    DENOISER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "denoiser")

    # MangaLineExtraction_PyTorch (MIT) — used only by closed-region
    # rectangle/lasso detection on the original black-and-white source.
    MANGA_LINE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "erika.pth")
    MANGA_LINE_MODEL_URL = (
        "https://github.com/ljsabc/MangaLineExtraction_PyTorch/"
        "releases/download/v1/erika.pth")
    MANGA_LINE_MAX_SIDE = int(os.environ.get("MANGA_LINE_MAX_SIDE", "1024"))

    # "auto" picks CUDA > validated MPS (Apple Silicon) > CPU.
    # On MPS the mc-v2 generator stays on GPU while the optional FFDNet
    # denoiser defaults to CPU for macOS 27 compatibility.
    ML_DEVICE = os.environ.get(
        "COLORTINA_DEVICE", os.environ.get("COMICCOLORER_DEVICE", "auto"))
    MPS_DENOISER_DEVICE = os.environ.get("COLORTINA_MPS_DENOISER", "cpu")
    MCV2_SIZE = int(os.environ.get("MCV2_SIZE", "768"))
    MCV2_DENOISE_SIGMA = int(os.environ.get("MCV2_DENOISE_SIGMA", "15"))

    # Auto color-hint generation (GuidedColorist). False = plain mc-v2
    # auto-colorize with no CLIP-driven palette hints.
    USE_GUIDED_HINTS = False

    # core/presets.py — pick with pipeline.colorize_page(style_key=...,
    # quality_key=...). A caller-supplied StyleProfile (see
    # core.style_engine) overrides DEFAULT_STYLE_KEY when given.
    DEFAULT_STYLE_KEY = "light2"
    DEFAULT_QUALITY_KEY = "draft"

    # Character Memory — book-consistent colors for characters whose
    # hair (etc.) differs from every other character's, instead of one
    # shared palette color for the whole "hair" label. See
    # core.character_memory. The UI/worker owns the actual
    # CharacterMemory instances (one per book) and this flag is just the
    # suggested default for new jobs.
    USE_CHARACTER_MEMORY = True

    # Two-stage book colouring: establish reference colours before style
    # grading, then restore them after style/filter passes.
    ENABLE_TWO_STAGE_IDENTITY = True
    # Purple guard is deliberately limited to non-character pixels.  Hair,
    # eyes, skin and clothing references remain eligible for their own locks.
    SUPPRESS_NON_CHARACTER_PURPLE = True
    PURPLE_GUARD_STRENGTH = 0.62

    # Style Engine ("Extract Style" from a color reference page/cover)
    # and Character Memory persistence — one subfolder per project.
    STYLES_DIR = os.path.join(BASE_DIR, "styles")
    CHARACTERS_DIR = os.path.join(BASE_DIR, "characters")

    # Real-ESRGAN anime weights for the "Ultra" quality preset's 4x
    # upscale pass (core.upscaler). Optional — falls back to a Lanczos
    # resize if the weights aren't present or `realesrgan` isn't
    # installed, so this is never a hard requirement.
    ESRGAN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "realesrgan_anime6b.pth")
    # Optional AI hair parsing backends. Model files are never bundled and are
    # downloaded only after explicit UI confirmation.
    STYLEANIME_PARSING_PATH = os.path.join(WEIGHTS_DIR, 'styleanime_danbooru_parsing.pth')
    STYLEANIME_INPUT_SIZE = int(os.environ.get('COLORTINA_STYLEANIME_INPUT', '512'))
    ANIME_FACE_SEG_PATH = os.path.join(WEIGHTS_DIR, 'anime_face_unet.pth')
    ANIME_FACE_SEG_INPUT_SIZE = int(os.environ.get('COLORTINA_ANIME_FACE_SEG_INPUT', '512'))
    YAKHYO_FACE_PARSING_ONNX_PATH = os.path.join(WEIGHTS_DIR, 'yakhyo_face_parsing_resnet18.onnx')
    YAKHYO_FACE_PARSING_INPUT_SIZE = int(os.environ.get('COLORTINA_YAKHYO_INPUT', '512'))
    # Optional local ONNX models (feature degrades gracefully when absent).
    TEXT_DETECTOR_PATH = os.path.join(WEIGHTS_DIR, "comictextdetector.pt.onnx")
    CHAR_SEG_PATH = os.path.join(WEIGHTS_DIR, "isnetis.onnx")
    ESRGAN_MODEL_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                        "download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth")
