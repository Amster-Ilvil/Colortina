"""MobileNetV2 U-Net used by siyeong0/Anime-Face-Segmentation.

Adapted from:
https://github.com/siyeong0/Anime-Face-Segmentation

MIT License, Copyright (c) 2025 siyeong.
The full license notice is included in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations


def build_anime_face_unet():
    import torch
    import torch.nn as nn
    import torchvision

    class UNet(nn.Module):
        NUM_SEG_CLASSES = 7  # background, hair, eye, mouth, face, skin, clothes

        def __init__(self):
            super().__init__()
            # The distributed checkpoint contains the MobileNet encoder weights,
            # so avoid an implicit ImageNet download in offline applications.
            mobilenet_v2 = torchvision.models.mobilenet_v2(weights=None)
            blocks = mobilenet_v2.features
            self.en_block0 = nn.Sequential(blocks[0], blocks[1])
            self.en_block1 = nn.Sequential(blocks[2], blocks[3])
            self.en_block2 = nn.Sequential(blocks[4], blocks[5], blocks[6])
            self.en_block3 = nn.Sequential(
                blocks[7], blocks[8], blocks[9], blocks[10],
                blocks[11], blocks[12], blocks[13])
            self.en_block4 = nn.Sequential(blocks[14], blocks[15], blocks[16])

            def decoder(in_channels: int, out_channels: int):
                return nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=0.2),
                )

            self.de_block4 = decoder(160, 96)
            self.de_block3 = decoder(96 * 2, 32)
            self.de_block2 = decoder(32 * 2, 24)
            self.de_block1 = decoder(24 * 2, 16)
            self.de_block0 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(16 * 2, self.NUM_SEG_CLASSES, kernel_size=3, padding=1),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            e0 = self.en_block0(x)
            e1 = self.en_block1(e0)
            e2 = self.en_block2(e1)
            e3 = self.en_block3(e2)
            e4 = self.en_block4(e3)
            d4 = self.de_block4(e4)
            d3 = self.de_block3(torch.cat((d4, e3), 1))
            d2 = self.de_block2(torch.cat((d3, e2), 1))
            d1 = self.de_block1(torch.cat((d2, e1), 1))
            return self.de_block0(torch.cat((d1, e0), 1))

    return UNet()
