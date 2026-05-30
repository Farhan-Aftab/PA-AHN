 ================================================================
# FILE 4: models/resnet50_sn.py
# ResNet50 with SN (Bottleneck blocks)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from norms.switchable_norm import SwitchableNorm2d


# ================================================================
# Bottleneck Block (ResNet50 style)
# ================================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        # 1x1 reduction
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = SwitchableNorm2d(planes)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.norm2 = SwitchableNorm2d(planes)

        # 1x1 expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = SwitchableNorm2d(planes * self.expansion)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                SwitchableNorm2d(planes * self.expansion),
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
    def __init__(self, block, num_blocks, num_classes: int = 100):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = SwitchableNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
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
def ResNet50_SN(num_classes: int = 100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
