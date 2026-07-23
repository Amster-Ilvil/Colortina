"""StyleAnalyzer — extracts color language from reference images.

This replaces the old "extract_from_reference -> palette HEX" flow.
Instead of learning "hair = #FFD200", we learn:
  - hair: warm_bias=+6, shadow_bias=-2, shadow_hue_rotate=+3, highlight_bias=+8
  - skin: warm_bias=+4, shadow_desat=0.2 ...

The key insight (from design doc):
  mc-v2 decides BASE COLOR (identity: black hair / blond hair / red hair).
  StyleDescriptor decides STYLE COLOR (language: how shadows fall, how warm
  the mid-tones are, how saturated the highlights stay).

We extract style language by:
  1. Converting the reference to LAB.
  2. For each semantic region (via CLIP classifier or K-Means), split
     pixels into highlight / mid / shadow tiers by their L channel.
  3. Measure how EACH TIER differs from the mid-tier in the A/B (chroma)
     directions — that difference IS the style language.
  4. Store those relative shifts as RegionDescriptor fields.

No absolute hex colors. Nothing that can make all characters the same hue.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.style_descriptor import StyleDescriptor, RegionDescriptor
from core.color_director import DEFAULT_PALETTE


# ── Tier thresholds in LAB L (0-255 scale) ───────────────────────────────────
_L_HI_MIN  = 160   # pixel is "highlight" if L > this
_L_SH_MAX  = 100   # pixel is "shadow"    if L < this
# everything in between is "mid-tone"


def _lab_tier_stats(lab_pixels: np.ndarray) -> dict:
    """Given N×3 float32 LAB pixels (L 0-255, A/B centred at 128),
    split into hi/mid/shadow and return per-tier mean A and B (recentred
    at 0) plus the overall saturation and gradient proxy."""
    if len(lab_pixels) == 0:
        return {}

    L = lab_pixels[:, 0]
    A = lab_pixels[:, 1] - 128.0
    B = lab_pixels[:, 2] - 128.0

    # Region-relative quantiles are robust to a dark night scene or a very
    # bright pastel reference.  Fixed global thresholds often left one tier
    # empty and made unrelated references look identical.
    q_lo, q_hi = np.quantile(L, [0.25, 0.75]) if len(L) >= 8 else (_L_SH_MAX, _L_HI_MIN)
    if q_hi - q_lo < 12:
        q_lo, q_hi = float(np.min(L)), float(np.max(L))
    hi_mask = L >= q_hi
    sh_mask = L <= q_lo
    mid_mask = ~hi_mask & ~sh_mask

    # Fall back: if a tier is empty, use whole region
    if not mid_mask.any():
        mid_mask = np.ones(len(L), dtype=bool)
    mid_A = float(A[mid_mask].mean())
    mid_B = float(B[mid_mask].mean())

    hi_A  = float(A[hi_mask].mean())  if hi_mask.any()  else mid_A
    hi_B  = float(B[hi_mask].mean())  if hi_mask.any()  else mid_B
    sh_A  = float(A[sh_mask].mean())  if sh_mask.any()  else mid_A
    sh_B  = float(B[sh_mask].mean())  if sh_mask.any()  else mid_B

    # Saturation proxy — chroma magnitude of mid-tones
    chroma_mid = float(np.sqrt(A[mid_mask]**2 + B[mid_mask]**2).mean())
    # Gradient proxy — std of L within region
    gradient = float(np.clip(L.std() / 60.0, 0.0, 1.0))
    # Shadow desaturation: how much chroma drops in shadows vs mid-tones
    chroma_sh  = float(np.sqrt(A[sh_mask]**2 + B[sh_mask]**2).mean()) \
                 if sh_mask.any() else chroma_mid
    sh_desat = float(np.clip(1.0 - chroma_sh / max(chroma_mid, 1e-3), 0.0, 1.0))
    hi_desat = (float(np.clip(
        1.0 - float(np.sqrt(A[hi_mask]**2 + B[hi_mask]**2).mean()) /
        max(chroma_mid, 1e-3), 0.0, 1.0)) if hi_mask.any() else 0.0)

    return {
        "mid_A": mid_A, "mid_B": mid_B,
        "hi_A":  hi_A,  "hi_B":  hi_B,
        "sh_A":  sh_A,  "sh_B":  sh_B,
        "chroma_mid": chroma_mid,
        "gradient":   gradient,
        "sh_desat":   sh_desat,
        "hi_desat":   hi_desat,
        "count": int(len(lab_pixels)),
    }


def _region_descriptor_from_stats(stats: dict, global_mid_B: float = 0.0,
                                  saturation_scale: float = 1.0) -> RegionDescriptor:
    """Convert per-tier LAB stats into a RegionDescriptor."""
    if not stats:
        return RegionDescriptor()

    mid_B = stats["mid_B"]
    mid_A = stats["mid_A"]

    # warm_bias = how much warmer (positive B) this region is vs global mid
    warm_bias = float(np.clip(mid_B - global_mid_B, -15.0, 15.0))

    # shadow shifts RELATIVE to the mid-tone of THIS region
    sh_B_shift = float(np.clip(stats["sh_B"] - mid_B, -12.0, 12.0))
    sh_A_shift = float(np.clip(stats["sh_A"] - mid_A, -12.0, 12.0))

    # highlight shifts relative to mid-tone
    hi_B_shift = float(np.clip(stats["hi_B"] - mid_B, -12.0, 12.0))
    hi_A_shift = float(np.clip(stats["hi_A"] - mid_A, -12.0, 12.0))

    chroma = max(stats.get("chroma_mid", 10.0), 1e-3)
    sat_scale = float(np.clip(chroma / 18.0, 0.4, 2.0)) * saturation_scale

    return RegionDescriptor(
        warm_bias=round(warm_bias, 2),
        shadow_bias=round(sh_B_shift, 2),
        shadow_hue_rotate=round(sh_A_shift, 2),
        shadow_desat=round(stats.get("sh_desat", 0.0), 3),
        highlight_bias=round(hi_B_shift, 2),
        highlight_hue_rotate=round(hi_A_shift, 2),
        highlight_desat=round(stats.get("hi_desat", 0.0), 3),
        saturation_scale=round(sat_scale, 3),
        gradient=round(stats.get("gradient", 0.5), 3),
        contrast=round(float(np.clip(chroma / 14.0, 0.7, 1.8)), 3),
        hint_layers=3,
    )


def _dominant_palette_hex(color_bgr: np.ndarray, max_colors: int = 8) -> list[str]:
    """Extract visible dominant colours while ignoring paper, ink and gray pixels."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    valid = ((hsv[..., 1] > 28) & (hsv[..., 2] > 24) & (hsv[..., 2] < 248))
    pixels = color_bgr[valid]
    if len(pixels) < 16:
        return []
    # Bound K-Means cost for large reference pages.
    if len(pixels) > 30000:
        idx = np.linspace(0, len(pixels) - 1, 30000).astype(int)
        pixels = pixels[idx]
    k = max(1, min(max_colors, len(pixels) // 64))
    data = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    out = []
    for i in np.argsort(-counts):
        b, g, r = np.clip(centers[i], 0, 255).astype(np.uint8)
        value = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        if value not in out:
            out.append(value)
    return out


class StyleAnalyzer:
    """Analyzes color reference images and produces a StyleDescriptor.

    CLIP classifier is optional; without it we fall back to K-Means
    region grouping (still produces a meaningful StyleDescriptor, just
    without semantic region labelling).
    """

    def analyze(self, color_bgr: np.ndarray, name: str = "Extracted",
                classifier=None) -> StyleDescriptor:
        """Main entry point: one reference image -> StyleDescriptor."""
        global_stats = self._global_stats(color_bgr)
        per_region = {}

        if classifier is not None and classifier.available:
            per_region = self._semantic_region_stats(color_bgr, classifier)
        if not per_region:
            per_region = self._kmeans_region_stats(color_bgr)

        return self._build_descriptor(name, global_stats, per_region)

    def analyze_many(self, color_images: list, name: str = "Extracted",
                     classifier=None,
                     weights: list | None = None) -> StyleDescriptor:
        """Merge multiple references into one StyleDescriptor.

        Per-reference descriptors are linearly blended by weight.
        """
        if not color_images:
            raise ValueError("at least one image required")
        if weights is None:
            weights = [1.0] * len(color_images)

        descriptors = [self.analyze(img, name=f"{name}_{i}", classifier=classifier)
                       for i, img in enumerate(color_images)]
        total_w = float(sum(weights)) or 1.0

        def wblend_region(key: str) -> RegionDescriptor:
            fields_sum: dict = {}
            effective_total = 0.0
            for d, user_w in zip(descriptors, weights):
                samples = float(getattr(d, "region_samples", {}).get(key, 0))
                if samples <= 0:
                    continue
                # Pixel count contributes sub-linearly so one large background
                # does not overwhelm several clean close-up references.
                w = float(user_w) * max(1.0, np.sqrt(samples / 500.0))
                effective_total += w
                r = d.region(key)
                for f in RegionDescriptor.__dataclass_fields__:
                    v = getattr(r, f)
                    if isinstance(v, (int, float)):
                        fields_sum[f] = fields_sum.get(f, 0.0) + v * w
            if effective_total <= 0:
                return RegionDescriptor()
            return RegionDescriptor(**{
                f: round(v / effective_total, 3) for f, v in fields_sum.items()})

        def wmean(attr: str) -> float:
            return sum(getattr(d, attr, 0.0) * w
                       for d, w in zip(descriptors, weights)) / total_w

        temp_votes: dict = {}
        for d, w in zip(descriptors, weights):
            temp_votes[d.temperature] = temp_votes.get(d.temperature, 0.0) + w
        temperature = max(temp_votes, key=temp_votes.get)

        return StyleDescriptor(
            name=name,
            description=f"Merged from {len(color_images)} reference images",
            source="extracted",
            global_warm_cool=round(wmean("global_warm_cool"), 2),
            global_saturation=round(wmean("global_saturation"), 3),
            global_contrast=round(wmean("global_contrast"), 3),
            global_shadow_lift=round(wmean("global_shadow_lift"), 3),
            cel_flatten=round(wmean("cel_flatten"), 3),
            reference_lab_mean=[round(sum(
                (getattr(d, "reference_lab_mean", []) or [0.0, 128.0, 128.0])[i] * w
                for d, w in zip(descriptors, weights)) / total_w, 3) for i in range(3)],
            reference_lab_std=[round(sum(
                (getattr(d, "reference_lab_std", []) or [1.0, 12.0, 12.0])[i] * w
                for d, w in zip(descriptors, weights)) / total_w, 3) for i in range(3)],
            reference_palette=list(dict.fromkeys(
                color for d in descriptors for color in getattr(d, "reference_palette", [])))[:12],
            hair=wblend_region("hair"),
            eyes=wblend_region("eyes"),
            skin=wblend_region("skin"),
            sky=wblend_region("sky"),
            foliage=wblend_region("foliage"),
            clothing_primary=wblend_region("clothing_primary"),
            clothing_secondary=wblend_region("clothing_secondary"),
            clothing_accent=wblend_region("clothing_accent"),
            background=wblend_region("background"),
            metal=wblend_region("metal"),
            water=wblend_region("water"),
            fire=wblend_region("fire"),
            stone=wblend_region("stone"),
            wood=wblend_region("wood"),
            region_samples={
                key: int(sum(getattr(d, "region_samples", {}).get(key, 0)
                             for d in descriptors))
                for key in ("hair", "eyes", "skin", "sky", "foliage",
                            "clothing_primary", "clothing_secondary",
                            "clothing_accent", "background", "metal",
                            "water", "fire", "stone", "wood")
            },
            saturation=round(wmean("saturation"), 3),
            contrast=round(wmean("contrast"), 3),
            temperature=temperature,
            shadow_strength=round(wmean("shadow_strength"), 3),
            gradient=round(wmean("gradient"), 3),
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _global_stats(self, color_bgr: np.ndarray) -> dict:
        hsv  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        chromatic = ((hsv[..., 1] > 30) & (hsv[..., 2] > 25) &
                     (hsv[..., 2] < 245))
        if np.any(chromatic):
            saturation = float(np.clip(
                hsv[..., 1][chromatic].mean() / 255.0, 0.0, 1.0))
            hue = hsv[..., 0][chromatic]
            warm_frac = float(np.mean((hue < 30) | (hue > 150)))
            temperature = ("warm" if warm_frac > 0.58 else
                           "cool" if warm_frac < 0.42 else "neutral")
        else:
            saturation = 0.0
            temperature = "neutral"
        contrast = float(np.clip(gray.std() / 80.0, 0.0, 1.0))

        shadow_strength = float(np.clip(np.mean(gray < 90), 0.0, 1.0))

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        soft = lap[(lap > 2) & (lap < 40)]
        gradient = float(np.clip(soft.size / max(1, (lap > 2).sum()), 0.0, 1.0))

        # Global mid-tone B channel (for relative warm_bias calculation)
        B_channel = lab[..., 2].astype(np.float32) - 128.0
        L_channel = lab[..., 0]
        mid_mask  = (L_channel > _L_SH_MAX) & (L_channel < _L_HI_MIN)
        global_mid_B = float(B_channel[mid_mask].mean()) if mid_mask.any() else 0.0

        # Global shadow lift — how bright are the darkest 15% of pixels?
        dark_quantile = float(np.percentile(gray, 15))
        shadow_lift = float(np.clip(dark_quantile / 40.0, 0.0, 1.0))

        # Cel-flatness: low gradient -> flat fills
        cel_flatten = float(np.clip(1.0 - gradient * 1.5, 0.0, 0.85))

        # Reference-wide LAB signature, measured only on actual colour pixels.
        # This makes a saved reference style visibly effective even when the
        # semantic region classifier cannot reliably identify a cover image.
        if np.any(chromatic):
            ref_lab = lab[chromatic]
        else:
            nonpaper = (gray > 20) & (gray < 245)
            ref_lab = lab[nonpaper] if np.any(nonpaper) else lab.reshape(-1, 3)
        reference_lab_mean = ref_lab.mean(axis=0).astype(float).tolist()
        reference_lab_std = np.maximum(ref_lab.std(axis=0), [1.0, 4.0, 4.0]).astype(float).tolist()
        reference_palette = _dominant_palette_hex(color_bgr)

        return {
            "saturation": saturation, "contrast": contrast,
            "temperature": temperature, "shadow_strength": shadow_strength,
            "gradient": gradient, "global_mid_B": global_mid_B,
            "shadow_lift": shadow_lift, "cel_flatten": cel_flatten,
            "global_saturation": float(np.clip(saturation * 1.5, 0.4, 2.0)),
            "global_contrast":   float(np.clip(contrast   * 1.4, 0.5, 1.8)),
            "global_warm_cool":  round(global_mid_B * 0.3, 2),
            "reference_lab_mean": reference_lab_mean,
            "reference_lab_std": reference_lab_std,
            "reference_palette": reference_palette,
        }

    def _semantic_region_stats(self, color_bgr: np.ndarray, classifier) -> dict:
        """Use CLIP classifier to identify regions and extract LAB stats
        per semantic label."""
        from core.region_segmenter import segment_regions
        if not classifier.available:
            return {}
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        seg  = segment_regions(gray)
        if not seg.regions:
            return {}
        semantic_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        labels = classifier.classify(semantic_bgr, [r.bbox for r in seg.regions])
        if not labels:
            return {}

        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        clothing_keys = ("clothing_primary", "clothing_secondary", "clothing_accent")
        clothing_rank = 0
        pixel_buckets: dict[str, list] = {}

        for region, (label, conf) in zip(seg.regions, labels):
            if label == "bubble" or conf < 0.25:
                continue
            if label == "clothing":
                key = clothing_keys[clothing_rank % len(clothing_keys)]
                clothing_rank += 1
            elif label in DEFAULT_PALETTE:
                key = label
            else:
                continue

            m = seg.labels == region.label_id
            if not np.any(m):
                continue
            ys, xs = np.nonzero(m)
            ys = np.clip((ys / seg.scale).astype(int), 0, color_bgr.shape[0] - 1)
            xs = np.clip((xs / seg.scale).astype(int), 0, color_bgr.shape[1] - 1)
            pix = lab[ys, xs]          # N×3 float32
            pixel_buckets.setdefault(key, []).append(pix)

        result = {}
        for key, chunks in pixel_buckets.items():
            all_pix = np.concatenate(chunks, axis=0)
            result[key] = _lab_tier_stats(all_pix)
        return result

    def _kmeans_region_stats(self, color_bgr: np.ndarray, k: int = 6) -> dict:
        """No-CLIP fallback: do not invent semantic labels from colour clusters.

        K-Means can describe a palette but cannot know whether a cluster is hair,
        skin, sky or clothing.  Global statistics and ``reference_palette`` are
        already extracted by ``_global_stats``; returning an empty mapping keeps
        every semantic RegionDescriptor neutral instead of assigning arbitrary
        absolute colours to character parts.
        """
        return {}

    def _build_descriptor(self, name: str, global_stats: dict,
                          per_region: dict) -> StyleDescriptor:
        gmid_B = global_stats.get("global_mid_B", 0.0)
        global_sat = global_stats.get("global_saturation", 1.0)

        def rd(key: str) -> RegionDescriptor:
            stats = per_region.get(key, {})
            return _region_descriptor_from_stats(stats, gmid_B, global_sat)

        return StyleDescriptor(
            name=name,
            description=f"Analyzed from reference image",
            source="extracted",
            global_warm_cool=global_stats.get("global_warm_cool", 0.0),
            global_saturation=global_stats.get("global_saturation", 1.0),
            global_contrast=global_stats.get("global_contrast", 1.0),
            global_shadow_lift=global_stats.get("shadow_lift", 0.0),
            cel_flatten=global_stats.get("cel_flatten", 0.3),
            reference_lab_mean=global_stats.get("reference_lab_mean", []),
            reference_lab_std=global_stats.get("reference_lab_std", []),
            reference_palette=global_stats.get("reference_palette", []),
            hair=rd("hair"),
            eyes=rd("eyes"),
            skin=rd("skin"),
            sky=rd("sky"),
            foliage=rd("foliage"),
            clothing_primary=rd("clothing_primary"),
            clothing_secondary=rd("clothing_secondary"),
            clothing_accent=rd("clothing_accent"),
            background=rd("background"),
            metal=rd("metal"),
            water=rd("water"),
            fire=rd("fire"),
            stone=rd("stone"),
            wood=rd("wood"),
            region_samples={key: int(stats.get("count", 0))
                            for key, stats in per_region.items()},
            saturation=global_stats.get("saturation", 0.85),
            contrast=global_stats.get("contrast", 0.75),
            temperature=global_stats.get("temperature", "neutral"),
            shadow_strength=global_stats.get("shadow_strength", 0.6),
            gradient=global_stats.get("gradient", 0.4),
        )
