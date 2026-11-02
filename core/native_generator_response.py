"""Experimental native mc-v2 colour-hint response maps.

This module uses *the original frozen generator*, not GrabCut or a randomly
initialized segmentation head. It measures the effect of two alternative hint
colours at the *same positions* against an unhinted baseline. It does NOT
claim that this effect is a semantic hair-class probability.

The result is an explicitly REVIEW-REQUIRED candidate mask. It cannot grow
outside a user rough-selection and negative strokes always veto it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class NativeResponseDiagnostics:
    rough_pixels: int
    raw_response_pixels: int
    candidate_pixels: int
    removed_outside_rough: int
    removed_by_negative: int
    positive_coverage_pixels: int
    positive_pixels: int
    threshold_rgb: float
    confidence_mode: str
    method: str = 'frozen_mc_v2_rgb_hint_response_review_required'


def _binary(data: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if data is None:
        return np.zeros(shape, dtype=bool)
    a = np.asarray(data)
    if a.ndim == 3:
        a = np.max(a, axis=2)
    if a.shape != shape:
        raise ValueError(f'mask shape {a.shape} does not match image {shape}')
    return a > 0


def response_candidate_from_rgb(
    baseline_rgb: np.ndarray,
    cool_rgb: np.ndarray,
    warm_rgb: np.ndarray,
    rough_mask: np.ndarray,
    *,
    negative: np.ndarray | None = None,
    positive: np.ndarray | None = None,
    threshold: float = 10.0,
    mode: str = 'agreement',
    negative_collar_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, NativeResponseDiagnostics]:
    """Build conservative model-response candidate from consistent RGB arrays.

    `agreement` uses min(distance(cool,base), distance(warm,base)) and
    requires BOTH alternative prompts to affect a pixel. `cool` is a single
    prompt comparison for A/B only. No response signal creates new hair pixels
    outside the explicit user rough selection. Positive marks are reported,
    not forcibly added: this is model evidence rather than an authoritative
    user-edited semantic mask.
    """
    arrays = [np.asarray(x) for x in (baseline_rgb, cool_rgb, warm_rgb)]
    shape = arrays[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(x.shape != shape for x in arrays):
        raise ValueError('all generator results must be identically sized HxWx3 RGB images')
    if any(x.dtype != np.uint8 for x in arrays):
        raise ValueError('RGB model outputs must be uint8')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    if mode not in ('agreement', 'cool'):
        raise ValueError('mode must be agreement or cool')
    h, w = shape[:2]
    rough = _binary(rough_mask, (h,w))
    if not np.any(rough):
        raise ValueError('rough selection is empty')
    pos = _binary(positive, (h,w))
    neg = _binary(negative, (h,w))
    collar = max(0, min(16, int(negative_collar_px)))
    if collar and np.any(neg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*collar+1,2*collar+1))
        neg = cv2.dilate(neg.astype(np.uint8), kernel)>0
    # Convert to float BEFORE subtraction; uint8 arithmetic wraps at 0/255.
    b, c, a = (x.astype(np.float32) for x in arrays)
    dc = np.linalg.norm(c-b, axis=2)
    dw = np.linalg.norm(a-b, axis=2)
    confidence = np.minimum(dc,dw) if mode == 'agreement' else dc
    raw = confidence > float(threshold)
    within_rough = raw & rough
    selected = within_rough & ~neg
    info = NativeResponseDiagnostics(
        rough_pixels=int(rough.sum()),
        raw_response_pixels=int(raw.sum()),
        candidate_pixels=int(selected.sum()),
        removed_outside_rough=int((raw & ~rough).sum()),
        removed_by_negative=int((within_rough & neg).sum()),
        positive_coverage_pixels=int((selected & pos).sum()),
        positive_pixels=int(pos.sum()),
        threshold_rgb=float(threshold),
        confidence_mode=mode,
    )
    return (selected.astype(np.uint8)*255, confidence, info)


def native_hint_maps(
    source_rgb: np.ndarray,
    model_weights: str | Path,
    seeds: Sequence[tuple[int,int,int]],
    *,
    cool: tuple[int,int,int] = (55,145,235),
    warm: tuple[int,int,int] = (210,58,70),
    device: str = 'cpu',
) -> dict[str, np.ndarray]:
    """Run the original stock Generator 3 times using two alternative hints.

    All 3 images are RGB; callers must not subtract cv2 BGR images from them.
    Seeds are (x,y,radius) in source-image pixels and must be supplied by the
    user or from explicit positive hair strokes, not guessed semantic labels.
    """
    import torch
    from core.generator_control import GeneratorController
    src = np.asarray(source_rgb)
    if src.ndim != 3 or src.shape[2] != 3 or src.dtype != np.uint8:
        raise ValueError('source_rgb must be HxWx3 uint8 RGB')
    if not seeds:
        raise ValueError('at least one reviewed positive seed is required')
    h,w = src.shape[:2]
    # Padding matters: Controller._input pads the grayscale tensor with zeros,
    # which represents BLACK paper, not the WHITE canvas used by the original
    # application. White-pad the RGB source ourselves before inference, then
    # crop back to the requested size. This prevents a spurious black frame
    # from altering the generator response along page/panel boundaries.
    pad_h=(-h)%32
    pad_w=(-w)%32
    if pad_h or pad_w:
        model_src=cv2.copyMakeBorder(src,0,pad_h,0,pad_w,
                    cv2.BORDER_CONSTANT,value=(255,255,255))
    else:
        model_src=src
    checked=[]
    for seed in seeds:
        if len(seed)!=3:
            raise ValueError('each seed must be (x,y,radius)')
        x,y,r = map(int,seed)
        if not (0<=x<w and 0<=y<h and 1<=r<=64):
            raise ValueError('seed is outside image or its radius is invalid')
        checked.append((x,y,r))
    ctrl=GeneratorController(model_weights,device=device)
    results={}
    for name,color in [('baseline',None),('cool',cool),('warm',warm)]:
        hint = None
        if color is not None:
            if len(color)!=3 or any(int(v)<0 or int(v)>255 for v in color):
                raise ValueError('hint RGB must have three values in [0,255]')
            hint=np.zeros((*model_src.shape[:2],4),np.float32)
            for x,y,r in checked:
                cv2.circle(hint,(x,y),r,(*[int(v) for v in color],1.0),-1)
        tensor,shape=ctrl._input(model_src,hint)
        with torch.inference_mode():
            prediction,_=ctrl.generator(tensor)
        results[name]=ctrl._rgb_output(prediction,shape)[:h,:w]
    return results


def positive_scribble_seeds(positive: np.ndarray, *, radius: int=19,
                            min_area: int=12, max_seeds: int=12) -> list[tuple[int,int,int]]:
    """Derive safe model probe points from the user's *explicit* hair marks.

    Only one point per connected scribble; no automatic claim about unmarked
    areas or hair/skin classification.
    """
    a=np.asarray(positive)
    if a.ndim!=2: raise ValueError('positive must be a 2D mask')
    binary=(a>0).astype(np.uint8)
    count, labels, stats, _=cv2.connectedComponentsWithStats(binary,8)
    order=sorted(range(1,count),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)
    result=[]
    for i in order:
        if int(stats[i,cv2.CC_STAT_AREA])<min_area:continue
        dist=cv2.distanceTransform((labels==i).astype(np.uint8),cv2.DIST_L2,5)
        y,x=np.unravel_index(int(np.argmax(dist)), dist.shape)
        result.append((int(x),int(y),int(radius)))
        if len(result)>=max_seeds:break
    return result
