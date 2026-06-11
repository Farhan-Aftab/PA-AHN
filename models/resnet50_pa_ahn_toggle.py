# ================================================================
# FILE 4: models/resnet50_pa_ahn_toggle.py
# ResNet50 with AHN (Bottleneck blocks)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from norms.prior_anchored_adaptive_hybrid_normalization_toggle import AdaptiveHybridNorm2d


# ================================================================
# Bottleneck Block (ResNet50 style)
# ================================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, temperature: float = 0.8, use_prior_anchor: bool = True):
        super().__init__()

        # 1x1 reduction
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = AdaptiveHybridNorm2d(planes, temperature=temperature, use_prior_anchor=use_prior_anchor)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.norm2 = AdaptiveHybridNorm2d(planes, temperature=temperature, use_prior_anchor=use_prior_anchor)

        # 1x1 expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = AdaptiveHybridNorm2d(planes * self.expansion, temperature=temperature, use_prior_anchor=use_prior_anchor)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                AdaptiveHybridNorm2d(planes * self.expansion, temperature=temperature, use_prior_anchor=use_prior_anchor),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ================================================================
# ResNet Core
# ================================================================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 100, temperature: float = 0.8, use_prior_anchor: bool = True):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = AdaptiveHybridNorm2d(64, temperature=temperature, use_prior_anchor=use_prior_anchor)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, temperature=temperature, use_prior_anchor=use_prior_anchor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, temperature=temperature, use_prior_anchor=use_prior_anchor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, temperature=temperature, use_prior_anchor=use_prior_anchor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, temperature=temperature, use_prior_anchor=use_prior_anchor)

        # IMPORTANT CHANGE for ResNet50
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, temperature: float, use_prior_anchor: bool):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s, temperature=temperature, use_prior_anchor=use_prior_anchor))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        logits = self.linear(out)
        return logits


# ================================================================
# Factory Function
# ================================================================
def ResNet50_AHN(num_classes: int = 100, temperature: float = 0.8, use_prior_anchor: bool = True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, temperature=temperature, use_prior_anchor=use_prior_anchor)
