import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Minimal local config. No Flask, no .env parsing — just paths."""

    WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
    GENERATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "generator.zip")
    EXTRACTOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "extractor.pth")
    DENOISER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "denoiser")

    # "auto" picks CUDA > MPS (Apple Silicon) > CPU
    ML_DEVICE = os.environ.get(
        "COLORTINA_DEVICE", os.environ.get("COMICCOLORER_DEVICE", "auto"))
    MCV2_SIZE = int(os.environ.get("MCV2_SIZE", "768"))
    MCV2_DENOISE_SIGMA = int(os.environ.get("MCV2_DENOISE_SIGMA", "15"))

    # Auto color-hint generation (GuidedColorist). False = plain mc-v2
    # auto-colorize with no CLIP-driven palette hints.
    USE_GUIDED_HINTS = True
    LLM_DIRECTOR = False  # static palette only, no external API calls

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
    ESRGAN_MODEL_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                        "download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth")
