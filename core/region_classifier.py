"""CLIP zero-shot region labeling — local semantics, no vision LLM.

Each segmented region's crop is compared against text prototypes
("a manga drawing of a character's hair", "…the sky", …) with the same
CLIP family already used for character memory.  This gives the palette
engine object identity (skin / hair / metal / sky / …) so a region can
receive a semantically sensible color instead of a model's guess.

Degrades gracefully: if CLIP can't load, ``available`` is False and the
guided pipeline simply produces no hints (previous behavior).
"""

import os

import cv2
import numpy as np


# (key, text used to build the CLIP prototype)
REGION_LABELS: list[tuple[str, str]] = [
    ("eyes",     "a character's iris or eye area"),
    ("skin",     "a character's face or bare skin"),
    ("hair",     "a character's hair"),
    ("clothing", "clothing, a shirt, a dress or a fabric garment"),
    ("metal",    "metal armor, a sword, a weapon or machinery"),
    ("wood",     "wooden furniture, a wooden wall or a wooden floor"),
    ("sky",      "the sky or clouds"),
    ("foliage",  "trees, grass, bushes or plants"),
    ("stone",    "a stone wall, rocks or bricks"),
    ("water",    "water, a river or the sea"),
    ("fire",     "fire, flames or an explosion"),
    ("background", "the empty background of an indoor room"),
    ("bubble",   "a speech bubble filled with text"),
]

_PROMPT = "a black and white manga drawing of {}"


class RegionClassifier:
    """Zero-shot labeler for region crops (lazy CLIP load, batched)."""

    def __init__(self, model_path: str | None = None, allow_download: bool | None = None):
        self._model_path = model_path or os.environ.get(
            "GUIDED_CLIP_PATH", "openai/clip-vit-base-patch32")
        if allow_download is None:
            allow_download = os.environ.get("COLORTINA_ALLOW_CLIP_DOWNLOAD", "0") == "1"
        self._allow_download = bool(allow_download)
        self._model = None
        self._processor = None
        self._text_embeds = None  # (n_labels, D), L2-normalized
        self._tried = False
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self):
        if self._tried:
            return
        self._tried = True
        try:
            import torch
            from transformers import (
                CLIPImageProcessor,
                CLIPTokenizer,
                CLIPTextModelWithProjection,
                CLIPVisionModelWithProjection,
            )

            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            # The *WithProjection variants return .text_embeds/.image_embeds
            # explicitly — stable across transformers versions (CLIPModel's
            # get_text_features return type is not)
            load_kwargs = {} if self._allow_download else {"local_files_only": True}
            text_model = CLIPTextModelWithProjection.from_pretrained(
                self._model_path, **load_kwargs).to(device).eval()
            vision_model = CLIPVisionModelWithProjection.from_pretrained(
                self._model_path, **load_kwargs).to(device).eval()
            tokenizer = CLIPTokenizer.from_pretrained(self._model_path, **load_kwargs)
            processor = CLIPImageProcessor.from_pretrained(self._model_path, **load_kwargs)

            prompts = [_PROMPT.format(desc) for _, desc in REGION_LABELS]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.inference_mode():
                text = text_model(**tokens).text_embeds
            text = text / text.norm(dim=-1, keepdim=True)

            # Text encoder is only needed once — free it immediately
            del text_model, tokenizer

            self._model = vision_model
            self._processor = processor
            self._text_embeds = text
            self._device = device
            print(f"[region_classifier] CLIP ready on {device}")
        except Exception as exc:
            self.last_error = str(exc)
            if self._allow_download:
                print(f"[region_classifier] CLIP unavailable: {exc}")
            else:
                print("[region_classifier] CLIP not cached; continuing with plain mc-v2. "
                      "Set COLORTINA_ALLOW_CLIP_DOWNLOAD=1 to allow first-run download.")
            self._model = None

    def _crop_images(self, page_bgr: np.ndarray,
                     bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        crops = []
        H, W = page_bgr.shape[:2]
        for (x, y, w, h) in bboxes:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            crop = page_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = page_bgr[:16, :16]
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return crops

    def embed_bboxes(self, page_bgr: np.ndarray,
                     bboxes: list[tuple[int, int, int, int]],
                     batch_size: int = 32) -> np.ndarray | None:
        """Return normalized visual embeddings for arbitrary page crops."""
        self._ensure_loaded()
        if self._model is None or not bboxes:
            return None
        try:
            import torch
            crops = self._crop_images(page_bgr, bboxes)
            chunks = []
            for start in range(0, len(crops), max(1, batch_size)):
                inputs = self._processor(
                    images=crops[start:start + batch_size], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    self._device, dtype=self._model.dtype)
                with torch.inference_mode():
                    emb = self._model(pixel_values=pixel_values).image_embeds
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                chunks.append(emb.float().cpu().numpy())
            return np.concatenate(chunks, axis=0) if chunks else None
        except Exception as exc:
            print(f"[region_classifier] embed failed: {exc}")
            return None

    def classify(self, page_bgr: np.ndarray, bboxes: list[tuple[int, int, int, int]],
                 ) -> list[tuple[str, float]] | None:
        """Label each bbox crop. Returns [(label_key, confidence)] or None.

        Crops are encoded in bounded batches so pages with many small regions
        (eyes, hair strands, clothing details) do not exhaust GPU/MPS memory.
        """
        img = self.embed_bboxes(page_bgr, bboxes)
        if img is None:
            return None
        try:
            text = self._text_embeds.float().cpu().numpy()
            sims = (img @ text.T) * 100.0
            sims -= sims.max(axis=1, keepdims=True)
            probs = np.exp(sims)
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
            out: list[tuple[str, float]] = []
            for row in probs:
                idx = int(np.argmax(row))
                out.append((REGION_LABELS[idx][0], float(row[idx])))
            return out
        except Exception as exc:
            print(f"[region_classifier] classify failed: {exc}")
            return None

