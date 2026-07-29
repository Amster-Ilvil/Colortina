"""PyTorch network for Deep Extraction of Manga Structural Lines.

Adapted from the official MIT-licensed implementation:
https://github.com/ljsabc/MangaLineExtraction_PyTorch

Only the inference network is retained.  Layer and attribute names intentionally
match the upstream implementation so the official ``erika.pth`` state dict can
be loaded without key conversion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, fw: int, fh: int,
                 subsample: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_filters, nb_filters, (fw, fh), stride=subsample,
                padding=(fw // 2, fh // 2), padding_mode="zeros"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, fw: int, fh: int,
                 subsample: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_filters, nb_filters, (fw, fh), stride=subsample,
                padding=(fw // 2, fh // 2)),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _shortcut(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, subsample: int = 1):
        super().__init__()
        self.process = in_filters != nb_filters or subsample != 1
        self.model = None
        if self.process:
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x) + y if self.process else x + y


class _u_shortcut(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, subsample: int):
        super().__init__()
        self.process = in_filters != nb_filters
        self.model = None
        if self.process:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_filters, nb_filters, (1, 1), stride=subsample,
                    padding_mode="zeros"),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x) + y if self.process else x + y


class basic_block(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int,
                 init_subsample: int = 1):
        super().__init__()
        self.conv1 = _bn_relu_conv(
            in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(
            in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)


class _u_basic_block(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int,
                 init_subsample: int = 1):
        super().__init__()
        self.conv1 = _u_bn_relu_conv(
            in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(
            in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, repetitions: int,
                 is_first_layer: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(repetitions):
            init_subsample = 1
            if index == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            layers.append(basic_block(
                in_filters=in_filters if index == 0 else nb_filters,
                nb_filters=nb_filters,
                init_subsample=init_subsample))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters: int, nb_filters: int, repetitions: int):
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(repetitions):
            if index == 0:
                layer = _u_basic_block(
                    in_filters=in_filters, nb_filters=nb_filters)
            else:
                layer = basic_block(
                    in_filters=nb_filters, nb_filters=nb_filters)
            layers.append(layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class res_skip(nn.Module):
    """Official manga structural-line extraction network."""

    def __init__(self):
        super().__init__()
        self.block0 = _residual_block(
            in_filters=1, nb_filters=24, repetitions=2,
            is_first_layer=True)
        self.block1 = _residual_block(
            in_filters=24, nb_filters=48, repetitions=3)
        self.block2 = _residual_block(
            in_filters=48, nb_filters=96, repetitions=5)
        self.block3 = _residual_block(
            in_filters=96, nb_filters=192, repetitions=7)
        self.block4 = _residual_block(
            in_filters=192, nb_filters=384, repetitions=12)

        self.block5 = _upsampling_residual_block(
            in_filters=384, nb_filters=192, repetitions=7)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)

        self.block6 = _upsampling_residual_block(
            in_filters=192, nb_filters=96, repetitions=5)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)

        self.block7 = _upsampling_residual_block(
            in_filters=96, nb_filters=48, repetitions=3)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)

        self.block8 = _upsampling_residual_block(
            in_filters=48, nb_filters=24, repetitions=2)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)

        self.block9 = _residual_block(
            in_filters=24, nb_filters=16, repetitions=2,
            is_first_layer=True)
        self.conv15 = _bn_relu_conv(
            in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        x6 = self.block6(self.res1(x3, x5))
        x7 = self.block7(self.res2(x2, x6))
        x8 = self.block8(self.res3(x1, x7))
        x9 = self.block9(self.res4(x0, x8))
        return self.conv15(x9)
