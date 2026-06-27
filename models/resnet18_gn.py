# ================================================================
# FILE: models/resnet18_gn.py
# ResNet18 with Group Normalization baseline for CIFAR-10/CIFAR-100
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_num_groups(channels: int, requested_groups: int = 32) -> int:
    """Return the largest valid group count <= requested_groups that divides channels."""
    groups = min(requested_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def GN(channels: int, requested_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=valid_num_groups(channels, requested_groups), num_channels=channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, num_groups: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = GN(planes, num_groups)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = GN(planes, num_groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                GN(planes, num_groups),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 100, num_groups: int = 32):
        super().__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = GN(64, num_groups)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, num_groups=self.num_groups))
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


def ResNet18_GN(num_classes: int = 100, num_groups: int = 32):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_groups=num_groups)
