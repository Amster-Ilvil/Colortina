"""BiSeNet-19 architecture used by StyleAnime/Danbooru-Parsing.

Adapted from zllrunning/face-parsing.PyTorch, which StyleAnime documents as
its anime parsing training implementation.

MIT License, Copyright (c) 2019 zll.
The full license notice is included in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations


def build_bisenet19(n_classes: int = 19):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        def __init__(self, in_chan, out_chan, stride=1):
            super().__init__()
            self.conv1 = conv3x3(in_chan, out_chan, stride)
            self.bn1 = nn.BatchNorm2d(out_chan)
            self.conv2 = conv3x3(out_chan, out_chan)
            self.bn2 = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if in_chan != out_chan or stride != 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_chan))

        def forward(self, x):
            residual = self.conv1(x)
            residual = F.relu(self.bn1(residual))
            residual = self.bn2(self.conv2(residual))
            shortcut = x if self.downsample is None else self.downsample(x)
            return self.relu(shortcut + residual)

    def make_layer(in_chan, out_chan, count, stride=1):
        layers = [BasicBlock(in_chan, out_chan, stride)]
        layers.extend(BasicBlock(out_chan, out_chan, 1) for _ in range(count - 1))
        return nn.Sequential(*layers)

    class Resnet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = make_layer(64, 64, 2, 1)
            self.layer2 = make_layer(64, 128, 2, 2)
            self.layer3 = make_layer(128, 256, 2, 2)
            self.layer4 = make_layer(256, 512, 2, 2)

        def forward(self, x):
            x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            feat8 = self.layer2(x)
            feat16 = self.layer3(feat8)
            feat32 = self.layer4(feat16)
            return feat8, feat16, feat32

    class ConvBNReLU(nn.Module):
        def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_chan, mid_chan, classes):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, mid_chan, 3, 1, 1)
            self.conv_out = nn.Conv2d(mid_chan, classes, 1, bias=False)

        def forward(self, x):
            return self.conv_out(self.conv(x))

    class AttentionRefinementModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, out_chan, 3, 1, 1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, 1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            feat = self.conv(x)
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = self.sigmoid(self.bn_atten(self.conv_atten(atten)))
            return feat * atten

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = Resnet18()
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128)
            self.conv_head16 = ConvBNReLU(128, 128)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)

        def forward(self, x):
            feat8, feat16, feat32 = self.resnet(x)
            avg = self.conv_avg(F.adaptive_avg_pool2d(feat32, 1))
            avg_up = F.interpolate(avg, feat32.shape[2:], mode='nearest')
            feat32_sum = self.arm32(feat32) + avg_up
            feat32_up = self.conv_head32(F.interpolate(
                feat32_sum, feat16.shape[2:], mode='nearest'))
            feat16_sum = self.arm16(feat16) + feat32_up
            feat16_up = self.conv_head16(F.interpolate(
                feat16_sum, feat8.shape[2:], mode='nearest'))
            return feat8, feat16_up, feat32_up

    class FeatureFusionModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, padding=0)
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, 1, bias=False)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, 1, bias=False)

        def forward(self, fsp, fcp):
            feat = self.convblk(torch.cat([fsp, fcp], dim=1))
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = torch.sigmoid(self.conv2(F.relu(self.conv1(atten), inplace=True)))
            return feat * atten + feat

    class BiSeNet(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.cp = ContextPath()
            self.ffm = FeatureFusionModule(256, 256)
            self.conv_out = BiSeNetOutput(256, 256, classes)
            self.conv_out16 = BiSeNetOutput(128, 64, classes)
            self.conv_out32 = BiSeNetOutput(128, 64, classes)

        def forward(self, x):
            h, w = x.shape[2:]
            feat_res8, feat_cp8, feat_cp16 = self.cp(x)
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            out = F.interpolate(self.conv_out(feat_fuse), (h, w),
                                mode='bilinear', align_corners=True)
            out16 = F.interpolate(self.conv_out16(feat_cp8), (h, w),
                                  mode='bilinear', align_corners=True)
            out32 = F.interpolate(self.conv_out32(feat_cp16), (h, w),
                                  mode='bilinear', align_corners=True)
            return out, out16, out32

    return BiSeNet(n_classes)
