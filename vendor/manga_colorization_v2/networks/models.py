import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import math
from torch import Tensor
from torch.nn import Parameter

from .extractor import SEResNeXt_Origin, BottleneckX_Origin

'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class SelayerSpectr(nn.Module):
    def __init__(self, inplanes):
        super(SelayerSpectr, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1))
        self.conv2 = SpectralNorm(nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)

        x = self.shortcut.forward(x)
        return x + bottleneck


class SpectrResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(SpectrResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = SpectralNorm(nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_conv = SpectralNorm(nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False))
        self.conv_expand = SpectralNorm(nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

        self.selayer = SelayerSpectr(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)

        x = self.shortcut.forward(x)
        return x + bottleneck


class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(FeatureConv, self).__init__()

        no_bn = True

        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)


class MaskFiLM(nn.Module):
    """Optional trainable mask/colour adapter.

    It is created only after the stock checkpoint has loaded, so the original
    645-key generator remains strict-compatible. Zero initialisation makes an
    untrained adapter an exact no-op.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, channels * 2, 1),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, feature: Tensor, mask: Tensor, target_rgb: Tensor,
                strength: float = 1.0) -> Tensor:
        mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear",
                             align_corners=False).clamp(0.0, 1.0)
        target_rgb = F.interpolate(target_rgb, size=feature.shape[-2:],
                                   mode="bilinear", align_corners=False)
        gamma, beta = self.condition(torch.cat([target_rgb * mask, mask], 1)).chunk(2, 1)
        alpha = mask * float(max(0.0, min(1.0, strength)))
        return feature * (1.0 + alpha * gamma) + alpha * beta


class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()

        self.encoder = SEResNeXt_Origin(BottleneckX_Origin, [3, 4, 6, 3], num_classes=370, input_channels=1)

        self.to0 = self._make_encoder_block_first(5, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 92)
        self.to3 = self._make_encoder_block(92, 128)
        self.to4 = self._make_encoder_block(128, 256)

        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
        )

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)])

        self.tunnel4 = nn.Sequential(nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        depth = 2
        tunnel = [ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
                   ResNeXtBottleneck(256, 256, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
                   ResNeXtBottleneck(128, 128, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
                   ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Sequential(nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0))

        # Deliberately absent from the module tree while loading stock weights.
        self.region_adapters = None
        # Runtime diagnostic only; this is not a Parameter or state_dict entry.
        self.encoder_forward_count = 0

    def enable_region_adapters(self):
        if self.region_adapters is None:
            self.region_adapters = nn.ModuleDict({
                "deep": MaskFiLM(256),
                "middle": MaskFiLM(128),
                "shallow": MaskFiLM(64),
            })
        return self.region_adapters

    @staticmethod
    def _erode_mask(mask: Tensor, radius_px: int) -> Tensor:
        radius_px = max(0, int(radius_px))
        if radius_px <= 0:
            return mask
        kernel = radius_px * 2 + 1
        return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1,
                                  padding=radius_px)

    @classmethod
    def _feature_mask(cls, mask: Tensor, feature: Tensor, *, shrink_px: int = 0,
                      feather_px: int = 0) -> Tensor:
        """Build an inward-only soft mask at ``feature`` resolution.

        A normal blur expands a mask beyond its original boundary. That is
        undesirable for an explicit colour command because decoder features can
        then alter a face, garment or neighbouring panel just outside the user
        selection. We therefore blur for a soft *inner* transition and multiply
        it by a nearest-neighbour support mask afterwards.
        """
        source = mask.clamp(0.0, 1.0)
        work = cls._erode_mask(source, shrink_px)
        feather_px = max(0, int(feather_px))
        if feather_px > 0:
            k = feather_px * 2 + 1
            work = F.avg_pool2d(work, kernel_size=k, stride=1, padding=feather_px)
        resized = F.interpolate(work, size=feature.shape[-2:], mode="bilinear",
                                align_corners=False)
        support = F.interpolate(source, size=feature.shape[-2:], mode="nearest")
        return (resized * support).clamp(0.0, 1.0)

    @classmethod
    def _mix_region(cls, base: Tensor, controlled: Tensor, mask: Tensor,
                    strength: float, *, shrink_px: int = 0, feather_px: int = 0,
                    feature_gain: float = 1.0) -> Tensor:
        strength = float(max(0.0, min(1.0, strength)))
        if strength <= 0.0:
            return base
        alpha = cls._feature_mask(mask, base, shrink_px=shrink_px,
                                  feather_px=feather_px) * strength
        gain = float(max(0.5, min(4.0, feature_gain)))
        directed = base + (controlled - base) * gain
        return base * (1.0 - alpha) + directed * alpha

    def _hint_features(self, sketch: Tensor):
        x0 = self.to0(sketch)
        aux = self.to1(x0)
        aux = self.to2(aux)
        aux = self.to3(aux)
        return x0, aux

    def _encode_structure(self, sketch: Tensor):
        self.encoder_forward_count += 1
        return self.encoder(sketch)

    def forward_controlled(self, base_sketch: Tensor, controlled_sketch: Tensor,
                           region_mask: Tensor, *, target_rgb: Tensor | None = None,
                           deep_strength: float = 0.38,
                           middle_strength: float = 0.74,
                           shallow_strength: float = 0.94,
                           feature_gain: float = 1.75,
                           deep_shrink_px: int = 8,
                           middle_shrink_px: int = 3,
                           shallow_shrink_px: int = 0,
                           mask_feather_px: int = 3,
                           use_adapter: bool = False):
        """Dual-route in-generator region intervention without retraining.

        The monochrome semantic encoder runs once. A normal hint route and a
        command-colour hint route are decoded in parallel. Deep, middle and
        shallow controls are independent: each stage can influence the final
        output even when either neighbouring strength is zero.

        All feature masks are inward-only. The editor still composites only the
        selected pixels back into the page, which gives the user-facing result
        exact outside-selection preservation without paying for a third full
        decoder route.
        """
        deep_strength = float(max(0.0, min(1.0, deep_strength)))
        middle_strength = float(max(0.0, min(1.0, middle_strength)))
        shallow_strength = float(max(0.0, min(1.0, shallow_strength)))
        if (deep_strength <= 0.0 and middle_strength <= 0.0
                and shallow_strength <= 0.0):
            return self.forward(base_sketch)

        if base_sketch.shape != controlled_sketch.shape:
            raise ValueError("base and controlled mc-v2 inputs must have the same shape")
        if region_mask.ndim != 4 or region_mask.shape[1] != 1:
            raise ValueError("region_mask must be [B, 1, H, W]")
        if region_mask.shape[0] != base_sketch.shape[0]:
            raise ValueError("region_mask batch size does not match mc-v2 input")

        base_x0, base_aux = self._hint_features(base_sketch)
        ctrl_x0, ctrl_aux = self._hint_features(controlled_sketch)
        x1, x2, x3, x4 = self._encode_structure(base_sketch[:, 0:1])

        # Deep route. ``mixed4`` becomes the main route for the next stage,
        # while ``ctrl4`` remains the pure command route used by the independent
        # middle control.
        base4 = self.tunnel4(torch.cat([x4, base_aux], 1))
        ctrl4 = self.tunnel4(torch.cat([x4, ctrl_aux], 1))
        mixed4 = self._mix_region(
            base4, ctrl4, region_mask, deep_strength,
            shrink_px=deep_shrink_px, feather_px=mask_feather_px,
            feature_gain=feature_gain)
        if use_adapter and self.region_adapters is not None and target_rgb is not None:
            mixed4 = self.region_adapters["deep"](
                mixed4, region_mask, target_rgb, deep_strength)

        # Middle route. The main branch carries any deep intervention. The pure
        # command branch is decoded from ctrl4, so middle control remains useful
        # even if deep_strength is zero.
        main3 = self.tunnel3(torch.cat([mixed4, x3], 1))
        ctrl3 = self.tunnel3(torch.cat([ctrl4, x3], 1))
        mixed3 = self._mix_region(
            main3, ctrl3, region_mask, middle_strength,
            shrink_px=middle_shrink_px, feather_px=mask_feather_px,
            feature_gain=feature_gain)
        if use_adapter and self.region_adapters is not None and target_rgb is not None:
            mixed3 = self.region_adapters["middle"](
                mixed3, region_mask, target_rgb, middle_strength)

        # Shallow route. It likewise has an independent pure command branch.
        main2 = self.tunnel2(torch.cat([mixed3, x2, x1], 1))
        ctrl2 = self.tunnel2(torch.cat([ctrl3, x2, x1], 1))
        mixed2 = self._mix_region(
            main2, ctrl2, region_mask, shallow_strength,
            shrink_px=shallow_shrink_px, feather_px=mask_feather_px,
            feature_gain=feature_gain)
        if use_adapter and self.region_adapters is not None and target_rgb is not None:
            mixed2 = self.region_adapters["shallow"](
                mixed2, region_mask, target_rgb, shallow_strength)

        mixed_x0 = self._mix_region(
            base_x0, ctrl_x0, region_mask, shallow_strength,
            shrink_px=shallow_shrink_px, feather_px=mask_feather_px,
            feature_gain=feature_gain)

        output = torch.tanh(self.exit(torch.cat([mixed2, mixed_x0], 1)))
        guide = self.deconv_for_decoder(mixed4)
        return output, guide

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def extract_structure_features(self, sketch):
        """Return the pretrained line-art encoder pyramid without decoding RGB.

        The stock generator checkpoint already contains the encoder weights; this
        method exposes their implicit structural representation to Colortina's
        semantic-region fusion while leaving the normal forward() output and
        state_dict completely unchanged. ``sketch`` may be the normal 5-channel
        mc-v2 input or a 1-channel line-art tensor.
        """
        mono = sketch[:, 0:1] if sketch.shape[1] != 1 else sketch
        return self._encode_structure(mono)

    def _forward_stock(self, sketch: Tensor):
        """Original mc-v2 forward path, kept byte-for-byte in operation order.

        Keeping the stock path in a dedicated helper makes the compatibility
        contract explicit: ``forward(sketch)`` and
        ``forward(sketch, semantic_control=None)`` execute exactly this graph and
        therefore require no extra checkpoint keys.
        """
        x0 = self.to0(sketch)
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)

        x1, x2, x3, x4 = self._encode_structure(sketch[:, 0:1])

        out = self.tunnel4(torch.cat([x4, aux_out], 1))
        x = self.tunnel3(torch.cat([out, x3], 1))
        x = self.tunnel2(torch.cat([x, x2, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        decoder_output = self.deconv_for_decoder(out)
        return x, decoder_output

    @torch.inference_mode()
    def prepare_control_context(self, sketch: Tensor) -> dict:
        """Prepare one reusable encoder/base-decoder context for a page.

        The returned tensors are ordinary inference tensors on the generator's
        device.  A semantic head can read ``features`` and subsequent direct
        colour commands can pass this same dictionary as ``cached_features``;
        changing only target RGB then avoids another grayscale encoder pass.
        """
        if not torch.is_tensor(sketch) or sketch.ndim != 4 or sketch.shape[1] < 1:
            raise ValueError("sketch must be a BCHW tensor")
        x0, aux = self._hint_features(sketch)
        features = tuple(self._encode_structure(sketch[:, 0:1]))
        x1, x2, x3, x4 = features
        base4 = self.tunnel4(torch.cat([x4, aux], 1))
        base3 = self.tunnel3(torch.cat([base4, x3], 1))
        base2 = self.tunnel2(torch.cat([base3, x2, x1], 1))
        base_output = torch.tanh(self.exit(torch.cat([base2, x0], 1)))
        return {
            "sketch_shape": tuple(int(v) for v in sketch.shape),
            "x0": x0,
            "aux": aux,
            "features": features,
            "base4": base4,
            "base3": base3,
            "base2": base2,
            "base_output": base_output,
        }

    @staticmethod
    def _normalize_target_rgb(target_rgb, sketch: Tensor) -> Tensor:
        """Return target RGB as normalized ``[-1, 1]`` BCHW tensor.

        Tuples/lists are interpreted as ordinary uint8 RGB. Tensor inputs may
        already be normalized maps; tensors with values outside ``[-1.5, 1.5]``
        are interpreted as 0..255 RGB for convenience.
        """
        if torch.is_tensor(target_rgb):
            target = target_rgb.to(device=sketch.device, dtype=sketch.dtype)
            if target.ndim == 1 and target.numel() == 3:
                target = target.view(1, 3, 1, 1)
            elif target.ndim == 2 and target.shape[-1] == 3:
                target = target.view(target.shape[0], 3, 1, 1)
            elif target.ndim != 4 or target.shape[1] != 3:
                raise ValueError("target_rgb tensor must be [3], [B,3] or [B,3,H,W]")
            if float(target.detach().abs().max()) > 1.5:
                target = target / 127.5 - 1.0
        else:
            if target_rgb is None or len(target_rgb) != 3:
                raise ValueError("semantic_control target_rgb must contain three RGB values")
            target = torch.tensor(
                [float(target_rgb[0]), float(target_rgb[1]), float(target_rgb[2])],
                device=sketch.device, dtype=sketch.dtype).view(1, 3, 1, 1)
            target = target / 127.5 - 1.0
        if target.shape[0] == 1 and sketch.shape[0] > 1:
            target = target.expand(sketch.shape[0], -1, -1, -1)
        if target.shape[0] != sketch.shape[0]:
            raise ValueError("target_rgb batch size does not match sketch")
        return target

    @staticmethod
    def _normalize_control_mask(region_mask: Tensor, sketch: Tensor) -> Tensor:
        if not torch.is_tensor(region_mask):
            region_mask = torch.as_tensor(region_mask, device=sketch.device, dtype=sketch.dtype)
        else:
            region_mask = region_mask.to(device=sketch.device, dtype=sketch.dtype)
        if region_mask.ndim == 2:
            region_mask = region_mask.unsqueeze(0).unsqueeze(0)
        elif region_mask.ndim == 3:
            region_mask = region_mask.unsqueeze(1)
        if region_mask.ndim != 4 or region_mask.shape[1] != 1:
            raise ValueError("semantic_control mask must be [H,W], [B,H,W] or [B,1,H,W]")
        if region_mask.shape[0] == 1 and sketch.shape[0] > 1:
            region_mask = region_mask.expand(sketch.shape[0], -1, -1, -1)
        if region_mask.shape[0] != sketch.shape[0]:
            raise ValueError("semantic_control mask batch size does not match sketch")
        if tuple(region_mask.shape[-2:]) != tuple(sketch.shape[-2:]):
            region_mask = F.interpolate(region_mask, size=sketch.shape[-2:], mode="nearest")
        return region_mask.clamp(0.0, 1.0)

    def _command_sketch(self, sketch: Tensor, region_mask: Tensor, target_rgb: Tensor,
                        hint_opacity: float) -> Tensor:
        """Build the command-colour hint route inside the generator."""
        opacity = float(max(0.0, min(1.0, hint_opacity)))
        effective = region_mask * opacity
        target = target_rgb
        if tuple(target.shape[-2:]) != tuple(sketch.shape[-2:]):
            target = F.interpolate(target, size=sketch.shape[-2:], mode="bilinear",
                                   align_corners=False)
        if target.shape[-2:] == (1, 1):
            target = target.expand(-1, -1, sketch.shape[-2], sketch.shape[-1])
        command = sketch.clone()
        old_rgb = sketch[:, 1:4]
        old_alpha = sketch[:, 4:5]
        command[:, 1:4] = old_rgb * (1.0 - effective) + target * effective
        command[:, 4:5] = torch.maximum(old_alpha, effective)
        return command

    def forward_semantic_control(
            self, sketch: Tensor, region_mask: Tensor, target_rgb, *,
            command_sketch: Tensor | None = None, strength: float = 1.0,
            feature_gain: float = 1.75, mask_feather_px: int = 2,
            hint_opacity: float = 1.0, strict_output_mask: bool = True,
            cached_features=None, tunnel2_adapter=None,
            return_baseline: bool = False):
        """Direct tunnel2 semantic colour control inside the stock generator.

        The grayscale encoder runs **once**.  A second pass is made only through
        the lightweight hint/decoder branch to derive the generator's own colour
        residual.  That residual is injected at ``tunnel2`` because the formal
        intervention audit found it to have the best current coverage/leakage
        trade-off.  No trainable parameters are added and the original 645-key
        checkpoint stays strict-compatible.

        With ``strict_output_mask=True`` (the production default), the blend
        back to the stock output is performed *inside the generator forward*,
        guaranteeing exact preservation outside ``region_mask``.
        """
        mask = self._normalize_control_mask(region_mask, sketch)
        if not bool(torch.any(mask > 0)) or float(strength) <= 0.0:
            return self._forward_stock(sketch)
        target = self._normalize_target_rgb(target_rgb, sketch)
        if command_sketch is None:
            command_sketch = self._command_sketch(
                sketch, mask, target, hint_opacity=hint_opacity)
        else:
            if command_sketch.shape != sketch.shape:
                raise ValueError("command_sketch must have the same BCHW shape as sketch")
            command_sketch = command_sketch.to(device=sketch.device, dtype=sketch.dtype)

        # Stock/base route and shared grayscale semantics. A prepared context
        # carries both x1..x4 and the base decoder states, so semantic head and
        # direct tunnel2 control can consume the same encoder pass.
        if isinstance(cached_features, dict):
            try:
                cached_shape = tuple(int(v) for v in cached_features.get("sketch_shape", ()))
                if cached_shape and cached_shape != tuple(int(v) for v in sketch.shape):
                    raise ValueError(
                        "cached_features spatial shape does not match sketch; "
                        "prepare a new control context for this page")
                base_x0 = cached_features["x0"]
                base_aux = cached_features["aux"]
                x1, x2, x3, x4 = tuple(cached_features["features"])
                base4 = cached_features["base4"]
                base3 = cached_features["base3"]
                base2 = cached_features["base2"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("cached_features is not a valid control context") from exc
            expected = (base_x0, base_aux, x1, x2, x3, x4, base4, base3, base2)
            if any(int(value.shape[0]) != int(sketch.shape[0]) for value in expected):
                raise ValueError("cached_features batch size does not match sketch")
        else:
            base_x0, base_aux = self._hint_features(sketch)
            x1, x2, x3, x4 = self._encode_structure(sketch[:, 0:1])
            base4 = self.tunnel4(torch.cat([x4, base_aux], 1))
            base3 = self.tunnel3(torch.cat([base4, x3], 1))
            base2 = self.tunnel2(torch.cat([base3, x2, x1], 1))

        # Command route: no second encoder run.  It reuses x1..x4 and differs
        # only through the target-colour hint path.
        _ctrl_x0, ctrl_aux = self._hint_features(command_sketch)
        ctrl4 = self.tunnel4(torch.cat([x4, ctrl_aux], 1))
        ctrl3 = self.tunnel3(torch.cat([ctrl4, x3], 1))
        ctrl2 = self.tunnel2(torch.cat([ctrl3, x2, x1], 1))

        alpha = self._feature_mask(mask, base2, shrink_px=0,
                                   feather_px=max(0, int(mask_feather_px)))
        gain = float(max(0.0, min(4.0, feature_gain)))
        amount = float(max(0.0, min(2.0, strength)))
        controlled2 = base2 + (ctrl2 - base2) * alpha * gain * amount
        if tunnel2_adapter is not None:
            adapter_delta = tunnel2_adapter(
                base2, alpha, target, hint_residual=ctrl2 - base2,
                strength=amount)
            controlled2 = controlled2 + adapter_delta

        base_output = torch.tanh(self.exit(torch.cat([base2, base_x0], 1)))
        controlled_output = torch.tanh(self.exit(torch.cat([controlled2, base_x0], 1)))
        if strict_output_mask:
            output_mask = F.interpolate(mask, size=base_output.shape[-2:], mode="nearest")
            controlled_output = base_output * (1.0 - output_mask) + controlled_output * output_mask

        guide = self.deconv_for_decoder(base4)
        if return_baseline:
            return controlled_output, guide, base_output
        return controlled_output, guide

    def forward(self, sketch: Tensor, semantic_control: dict | None = None,
                semantic_controls: dict | None = None, cached_features=None):
        """Stock forward plus an optional direct semantic-control contract.

        ``semantic_control=None`` is guaranteed to execute the untouched stock
        graph.  A control dictionary accepts ``mask`` and ``target_rgb`` plus
        optional ``command_sketch``, ``strength``, ``feature_gain``,
        ``mask_feather_px``, ``hint_opacity``, ``strict_output_mask``,
        ``cached_features`` and an external ``tunnel2_adapter``. The plural
        ``semantic_controls`` alias accepts a ``hair`` control object.
        """
        if semantic_control is not None and semantic_controls is not None:
            raise ValueError("provide semantic_control or semantic_controls, not both")
        control = semantic_control if semantic_control is not None else semantic_controls
        if not control:
            return self._forward_stock(sketch)
        cfg = dict(control)
        if "mask" not in cfg and isinstance(cfg.get("hair"), dict):
            cfg = dict(cfg["hair"])
        if "mask" not in cfg or "target_rgb" not in cfg:
            raise ValueError("semantic_control requires mask and target_rgb")
        context = cfg.pop("cached_features", cached_features)
        adapter = cfg.pop("tunnel2_adapter", None)
        return self.forward_semantic_control(
            sketch, cfg.pop("mask"), cfg.pop("target_rgb"),
            cached_features=context, tunnel2_adapter=adapter, **cfg)


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.generator = Generator()

    def forward(self, x, extractor_grad=False):
        fake, guide = self.generator(x)
        return fake, guide
