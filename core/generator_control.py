"""Controls and measurements around the frozen manga-colorization generator.

The controller never changes the stock generator checkpoint. It exposes the
actual encoder pyramid and runs decoder interventions from one cached encoder
pass. RGB values are uint8; masks are HxW binary arrays.
"""
from hashlib import sha256
from pathlib import Path
from threading import RLock
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vendor.manga_colorization_v2.networks.models import Generator


class GeneratorController:
    INJECTION_POINTS = ("to3", "tunnel4", "tunnel3", "tunnel2")
    INJECTION_MODES = ("residual", "film")

    def __init__(self, weights_path, device="cpu"):
        self.device = torch.device(device)
        self.generator = Generator().eval()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.generator.load_state_dict(state, strict=True)
        self.generator.to(self.device)
        self.generator.requires_grad_(False)
        self.weights_path = str(weights_path)
        self.forward_count = 0
        self._cache = None
        self._component_cache = None
        self._intervention_cache = None
        self._lock = RLock()

    @staticmethod
    def _image_key(x, shape):
        return shape, sha256(x.detach().cpu().numpy().tobytes()).digest()

    def _input(self, image, hints=None):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB uint8")
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        x = np.zeros((5, h, w), np.float32)
        x[0] = gray / 255.0
        if hints is not None:
            hints = np.asarray(hints, dtype=np.float32)
            if hints.shape != (h, w, 4) or not np.isfinite(hints).all():
                raise ValueError("hints must be HxWx4: RGB 0..255, alpha 0..1")
            if ((hints[:, :, :3] < 0).any() or (hints[:, :, :3] > 255).any()
                    or (hints[:, :, 3] < 0).any() or (hints[:, :, 3] > 1).any()):
                raise ValueError("hint values outside declared ranges")
            x[1:4] = ((hints[:, :, :3] / 127.5 - 1.0)
                      * hints[:, :, 3:4]).transpose(2, 0, 1)
            x[4] = hints[:, :, 3]
        x = np.pad(x, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="constant")
        return torch.from_numpy(x).unsqueeze(0).to(self.device), (h, w)

    @staticmethod
    def _rgb_output(output, shape):
        h, w = shape
        return ((output[0, :, :h, :w].detach().cpu().permute(1, 2, 0).numpy() + 1.0)
                * 127.5).clip(0, 255).astype(np.uint8)

    def _forward_with_features(self, x):
        features = {}

        def save(name):
            def hook(_module, _inputs, value):
                if name == "encoder":
                    features.update({f"x{i + 1}": v.detach().cpu().clone()
                                     for i, v in enumerate(value)})
                elif torch.is_tensor(value):
                    features[name] = value.detach().cpu().clone()
            return hook

        names = ("encoder", "to0", "to1", "to2", "to3", "tunnel4",
                 "tunnel3", "tunnel2")
        handles = [getattr(self.generator, name).register_forward_hook(save(name))
                   for name in names]
        try:
            with torch.inference_mode():
                output, _guide = self.generator(x)
        finally:
            for handle in handles:
                handle.remove()
        return output, features

    def _prepare_components(self, image):
        """Compute encoder and unmodified decoder stages exactly once."""
        x, shape = self._input(image)
        key = self._image_key(x, shape)
        if self._component_cache is not None and self._component_cache[0] == key:
            return self._component_cache[1]
        with torch.inference_mode():
            x0 = self.generator.to0(x)
            aux1 = self.generator.to1(x0)
            aux2 = self.generator.to2(aux1)
            aux3 = self.generator.to3(aux2)
            x1, x2, x3, x4 = self.generator.encoder(x[:, 0:1])
            base4 = self.generator.tunnel4(torch.cat([x4, aux3], 1))
            base3 = self.generator.tunnel3(torch.cat([base4, x3], 1))
            base2 = self.generator.tunnel2(torch.cat([base3, x2, x1], 1))
            output = torch.tanh(self.generator.exit(torch.cat([base2, x0], 1)))
        features = {
            "to0": x0.detach().cpu().clone(),
            "to1": aux1.detach().cpu().clone(),
            "to2": aux2.detach().cpu().clone(),
            "to3": aux3.detach().cpu().clone(),
            "x1": x1.detach().cpu().clone(),
            "x2": x2.detach().cpu().clone(),
            "x3": x3.detach().cpu().clone(),
            "x4": x4.detach().cpu().clone(),
            "tunnel4": base4.detach().cpu().clone(),
            "tunnel3": base3.detach().cpu().clone(),
            "tunnel2": base2.detach().cpu().clone(),
        }
        components = {
            "shape": shape, "key": key, "x0": x0, "aux3": aux3,
            "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            "base4": base4, "base3": base3, "base2": base2,
            "output": output, "features": features,
        }
        rgb = self._rgb_output(output, shape)
        self._component_cache = (key, components)
        self._cache = (key, rgb, features)
        self.forward_count += 1
        return components

    def extract_features(self, image, hints=None):
        with self._lock:
            if hints is None:
                components = self._prepare_components(image)
                return {
                    "base_color": self._rgb_output(components["output"], components["shape"]),
                    "features": {k: v.clone() for k, v in components["features"].items()},
                }
            x, shape = self._input(image, hints)
            with torch.inference_mode():
                output, features = self._forward_with_features(x)
            return {"base_color": self._rgb_output(output, shape), "features": features}

    @staticmethod
    def _mask(mask, shape):
        mask = np.asarray(mask)
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError("mask must match image HxW and be finite")
        return mask > 0

    @staticmethod
    def _validate_target(target_rgb):
        target = np.asarray(target_rgb, dtype=np.float32)
        if (target.shape != (3,) or not np.isfinite(target).all()
                or (target < 0).any() or (target > 255).any()):
            raise ValueError("target_rgb must contain three values in [0,255]")
        return target

    def _condition(self, feature, mask, target_rgb, alpha, mode, residual=None):
        """Apply a target-conditioned residual or FiLM transform."""
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(
            self.device, dtype=feature.dtype)
        mask_t = F.interpolate(mask_t, size=feature.shape[-2:], mode="bilinear",
                               align_corners=False).clamp(0.0, 1.0)
        target = torch.tensor(target_rgb / 127.5 - 1.0, device=self.device,
                              dtype=feature.dtype).view(1, 3, 1, 1)
        target = target.expand(feature.shape[0], 3, *feature.shape[-2:])
        if residual is None:
            channels = feature.shape[1]
            index = torch.arange(channels, device=self.device, dtype=feature.dtype)
            basis = torch.stack((
                torch.sin(index * 0.71 + 0.11),
                torch.cos(index * 0.53 + 0.37),
                torch.sin(index * 0.29 + 1.17),
            ), dim=1)
            basis = basis / torch.sqrt(torch.tensor(3.0, device=self.device,
                                                     dtype=feature.dtype))
            projected = torch.einsum("bchw,dc->bdhw", target, basis)
            projected = torch.tanh(projected) * 0.42
        else:
            projected = residual.to(device=self.device, dtype=feature.dtype)
        strength = float(np.clip(alpha, 0.0, 2.0))
        if mode == "residual":
            return feature + strength * projected * mask_t
        if mode == "film":
            gamma = torch.tanh(projected) * 0.55
            beta = projected * 0.75
            return feature * (1.0 + strength * gamma * mask_t) + strength * beta * mask_t
        raise ValueError(f"unsupported intervention mode: {mode}")

    def _hint_residuals(self, components, image, mask, target_rgb):
        """Use the stock hint branch to derive color residuals at each stage."""
        mask_digest = sha256(np.ascontiguousarray(mask).tobytes()).digest()
        key = (components["key"], mask_digest, tuple(float(v) for v in target_rgb))
        if self._intervention_cache is not None and self._intervention_cache[0] == key:
            return self._intervention_cache[1]
        h, w = components["shape"]
        hints = np.zeros((h, w, 4), np.float32)
        hints[mask, :3] = target_rgb
        hints[mask, 3] = 1.0
        controlled, _ = self._input(image, hints)
        with torch.inference_mode():
            ctrl0 = self.generator.to0(controlled)
            ctrl1 = self.generator.to1(ctrl0)
            ctrl2 = self.generator.to2(ctrl1)
            ctrl3 = self.generator.to3(ctrl2)
            ctrl4 = self.generator.tunnel4(torch.cat([components["x4"], ctrl3], 1))
            ctrl3_out = self.generator.tunnel3(torch.cat([ctrl4, components["x3"]], 1))
            ctrl2_out = self.generator.tunnel2(torch.cat([
                ctrl3_out, components["x2"], components["x1"]], 1))
        residuals = {
            "to3": ctrl3 - components["aux3"],
            "tunnel4": ctrl4 - components["base4"],
            "tunnel3": ctrl3_out - components["base3"],
            "tunnel2": ctrl2_out - components["base2"],
        }
        self._intervention_cache = (key, residuals)
        return residuals

    def _decode_intervened(self, components, mask, target_rgb, point, mode, alpha,
                           residuals=None):
        if point not in self.INJECTION_POINTS:
            raise ValueError(f"unsupported injection point: {point}")
        with torch.inference_mode():
            aux3 = components["aux3"]
            if point == "to3":
                aux3 = self._condition(aux3, mask, target_rgb, alpha, mode,
                                       None if residuals is None else residuals["to3"])
            out4 = self.generator.tunnel4(torch.cat([components["x4"], aux3], 1))
            if point == "tunnel4":
                out4 = self._condition(out4, mask, target_rgb, alpha, mode,
                                       None if residuals is None else residuals["tunnel4"])
            out3 = self.generator.tunnel3(torch.cat([out4, components["x3"]], 1))
            if point == "tunnel3":
                out3 = self._condition(out3, mask, target_rgb, alpha, mode,
                                       None if residuals is None else residuals["tunnel3"])
            out2 = self.generator.tunnel2(torch.cat([out3, components["x2"], components["x1"]], 1))
            if point == "tunnel2":
                out2 = self._condition(out2, mask, target_rgb, alpha, mode,
                                       None if residuals is None else residuals["tunnel2"])
            output = torch.tanh(self.generator.exit(torch.cat([out2, components["x0"]], 1)))
        return output

    @staticmethod
    def _lab_distance(image, target_rgb):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_px = np.array([[np.asarray(target_rgb, dtype=np.uint8)]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        return np.linalg.norm(lab - target_lab, axis=2)

    def _intervention_metrics(self, base, raw, mask, target_rgb, leakage_masks):
        raw_delta = np.max(np.abs(raw.astype(np.int16) - base.astype(np.int16)), axis=2)
        changed = raw_delta > 1
        base_distance = self._lab_distance(base, target_rgb)
        raw_distance = self._lab_distance(raw, target_rgb)
        improved = raw_distance + 0.5 < base_distance
        hair_pixels = int(mask.sum())
        result = {
            "hair_pixels": hair_pixels,
            "hair_coverage_percent": (float(100.0 * (improved & mask).sum() / hair_pixels)
                                       if hair_pixels else None),
            "hair_changed_percent": (float(100.0 * (changed & mask).sum() / hair_pixels)
                                      if hair_pixels else None),
            "hair_mean_delta_e_before": (float(base_distance[mask].mean())
                                          if hair_pixels else None),
            "hair_mean_delta_e_after": (float(raw_distance[mask].mean())
                                         if hair_pixels else None),
            "unrecolored_pixels_inside_hair": int((mask & ~changed).sum()),
            "raw_outside_changed_pixels": int((changed & ~mask).sum()),
            "raw_outside_changed_percent": float(100.0 * (changed & ~mask).sum()
                                                  / max(1, (~mask).sum())),
        }
        for name, candidate in dict(leakage_masks or {}).items():
            selected = np.asarray(candidate) > 0
            if selected.shape != mask.shape:
                raise ValueError(f"leakage mask shape mismatch: {name}")
            pixels = int(selected.sum())
            result[f"{name}_pixels"] = pixels
            result[f"{name}_leak_percent"] = (float(100.0 * (changed & selected).sum() / pixels)
                                               if pixels else None)
        return result

    def colorize_with_direct_generator_control(
            self, image, region_mask, target_rgb, *, strength=1.0,
            feature_gain=1.75, mask_feather_px=2, strict_output_mask=True):
        """Use the new control contract directly on ``Generator.forward``.

        This is intentionally distinct from ``colorize_with_feature_intervention``:
        the latter is the audit harness, while this method exercises the same
        public forward entry point that Colortina can use in production.
        """
        target = self._validate_target(target_rgb)
        image = np.asarray(image)
        mask = self._mask(region_mask, image.shape[:2])
        x, shape = self._input(image)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(
            self.device, dtype=x.dtype)
        with self._lock, torch.inference_mode():
            controlled, _guide, base = self.generator.forward_semantic_control(
                x, mask_t, tuple(float(v) for v in target),
                strength=float(strength),
                feature_gain=float(feature_gain),
                mask_feather_px=int(mask_feather_px),
                strict_output_mask=bool(strict_output_mask),
                return_baseline=True)
        base_rgb = self._rgb_output(base, shape)
        result = self._rgb_output(controlled, shape)
        changed = np.any(result != base_rgb, axis=2)
        return {
            "base": base_rgb,
            "result": result,
            "outside_changed_pixels": int(np.count_nonzero(changed & ~mask)),
            "inside_changed_pixels": int(np.count_nonzero(changed & mask)),
            "state_dict_keys": len(self.generator.state_dict()),
        }

    def colorize_with_feature_intervention(self, image, hair_mask, target_rgb,
                                           injection_point="tunnel3", mode="residual",
                                           alpha=1.0, strict_mask=True,
                                           leakage_masks=None):
        """Run one decoder intervention from one cached encoder computation."""
        target = self._validate_target(target_rgb)
        image = np.asarray(image)
        mask = self._mask(hair_mask, image.shape[:2])
        with self._lock:
            components = self._prepare_components(image)
            base = self._rgb_output(components["output"], components["shape"])
            residuals = self._hint_residuals(components, image, mask, target)
            raw = self._rgb_output(
                self._decode_intervened(components, mask, target, injection_point, mode,
                                        alpha, residuals=residuals),
                components["shape"])
            metrics = self._intervention_metrics(base, raw, mask, target, leakage_masks)
            final = base.copy()
            if strict_mask:
                final[mask] = raw[mask]
            final_changed = np.any(final != base, axis=2)
            metrics["outside_changed_pixels"] = int((final_changed & ~mask).sum())
            metrics["strict_mask_applied"] = bool(strict_mask)
            metrics["injection_point"] = injection_point
            metrics["mode"] = mode
            metrics["alpha"] = float(alpha)
            return {"base": base, "raw": raw, "final": final, "metrics": metrics,
                    "features": {k: v.clone() for k, v in components["features"].items()}}

    def audit_decoder_interventions(self, image, hair_mask, target_rgb,
                                    leakage_masks=None, output_dir=None, alpha=1.0,
                                    strict_mask=True):
        """Compare to3/tunnel4/tunnel3/tunnel2 with residual and FiLM modes."""
        target = self._validate_target(target_rgb)
        output_path = None if output_dir is None else Path(output_dir)
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            before = self.forward_count
            rows = []
            baseline = None
            for point in self.INJECTION_POINTS:
                for mode in self.INJECTION_MODES:
                    item = self.colorize_with_feature_intervention(
                        image, hair_mask, target, injection_point=point, mode=mode,
                        alpha=alpha, strict_mask=strict_mask,
                        leakage_masks=leakage_masks)
                    baseline = item["base"]
                    rows.append(dict(item["metrics"]))
                    if output_path is not None:
                        stem = f"{point}_{mode}"
                        cv2.imwrite(str(output_path / f"{stem}_raw.png"),
                                    cv2.cvtColor(item["raw"], cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(output_path / f"{stem}_final.png"),
                                    cv2.cvtColor(item["final"], cv2.COLOR_RGB2BGR))
                        delta = np.max(np.abs(item["raw"].astype(np.int16)
                                               - item["base"].astype(np.int16)), axis=2)
                        heat = cv2.applyColorMap(np.clip(delta * 8, 0, 255).astype(np.uint8),
                                                 cv2.COLORMAP_TURBO)
                        cv2.imwrite(str(output_path / f"{stem}_heatmap.png"), heat)
            if output_path is not None and baseline is not None:
                cv2.imwrite(str(output_path / "baseline.png"),
                            cv2.cvtColor(baseline, cv2.COLOR_RGB2BGR))
            return {"rows": rows, "forward_count_before": before,
                    "forward_count_after": self.forward_count,
                    "encoder_forward_delta": self.forward_count - before,
                    "injection_points": list(self.INJECTION_POINTS),
                    "modes": list(self.INJECTION_MODES),
                    "strict_mask": bool(strict_mask)}

    def colorize_with_region_hint(self, image, region_mask, target_rgb, hint_density=0.1,
                                  hint_strength=1.0, hint_placement_strategy="uniform",
                                  propagation_mode="masked", propagation_radius=None):
        if not 0 <= hint_density <= 1 or not 0 <= hint_strength <= 1:
            raise ValueError("density and strength must be in [0,1]")
        if propagation_mode not in {"masked", "unrestricted"}:
            raise ValueError("unsupported propagation mode")
        if hint_placement_strategy not in {"uniform", "center"}:
            raise ValueError("unsupported placement strategy")
        target = self._validate_target(target_rgb)
        with self._lock:
            baseline = self.extract_features(image)["base_color"]
            mask = self._mask(region_mask, image.shape[:2])
            coords = np.argwhere(mask)
            count = min(len(coords), int(np.ceil(len(coords) * hint_density)))
            if not count or hint_strength == 0:
                return baseline
            if hint_placement_strategy == "center":
                order = np.argsort(((coords - coords.mean(0)) ** 2).sum(1), kind="stable")[:count]
            else:
                order = np.linspace(0, len(coords) - 1, count, dtype=int)
            selected = coords[order]
            hints = np.zeros((*mask.shape, 4), np.float32)
            hints[selected[:, 0], selected[:, 1], :3] = target
            hints[selected[:, 0], selected[:, 1], 3] = hint_strength
            controlled = self.extract_features(image, hints)["base_color"]
            support = mask.copy() if propagation_mode == "masked" else np.ones_like(mask)
            if propagation_radius is not None:
                if not np.isfinite(propagation_radius) or propagation_radius < 0:
                    raise ValueError("radius must be finite and nonnegative")
                seeds = np.ones(mask.shape, np.uint8)
                seeds[selected[:, 0], selected[:, 1]] = 0
                support &= (cv2.distanceTransform(
                    seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= propagation_radius)
            result = baseline.copy()
            result[support] = controlled[support]
            return result

    def colorize_with_semantic_lock(self, image, *, hair_mask=None, iris_mask=None,
                                    clothes_mask=None, background_mask=None,
                                    target_colours=None):
        """Exact flat-colour mode using reviewed disjoint masks."""
        with self._lock:
            result = self.extract_features(image)["base_color"]
            used = np.zeros(image.shape[:2], bool)
            for name, mask in (("hair", hair_mask), ("iris", iris_mask),
                               ("clothes", clothes_mask), ("background", background_mask)):
                if mask is None or name not in (target_colours or {}):
                    continue
                selected = self._mask(mask, image.shape[:2])
                if np.any(used & selected):
                    raise ValueError("semantic masks overlap")
                colour = self._validate_target(target_colours[name])
                result[selected] = colour.astype(np.uint8)
                used |= selected
            return result

    def measure_hint_influence(self, image, hints):
        with self._lock:
            base = self.extract_features(image)
            changed = self.extract_features(image, hints)
            return {
                "pixel_delta": np.abs(changed["base_color"].astype(np.float32)
                                       - base["base_color"]).mean(2),
                "feature_delta": {
                    k: (v - base["features"][k]).abs().mean(1)
                    for k, v in changed["features"].items()
                },
            }

    def extract_generator_semantics(self, image):
        """Return unlabelled encoder embeddings, not semantic class masks."""
        return {k: v for k, v in self.extract_features(image)["features"].items()
                if k in {"x1", "x2", "x3", "x4"}}
