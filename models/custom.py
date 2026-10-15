# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.
#
# Adapted: the original file was an empty template (YourConvNet). It now holds
# SimpleCNN, a small plain CNN used as a CNN-JEPA backbone for quick experiments.

from typing import List, Sequence

import torch
import torch.nn as nn
from timm.models.registry import register_model


class SimpleCNN(nn.Module):
    """A plain CNN: a stack of [Conv 3x3 stride 2 -> BatchNorm -> ReLU] stages.

    Every stage halves the height and width, so with 3 stages a 128x128 image
    becomes a 16x16 feature map (downsample ratio 2**3 = 8).

    What CNN-JEPA needs from a backbone (see pretrain/train_ijepacnn.py):
      - forward(x) returns the last feature map (B, C, h, w) when global_pool=''
      - self.num_features == C (channels of the last feature map)
      - get_downsample_ratio() == how much the image shrinks
      - only layers the sparse conversion understands (Conv2d, BatchNorm2d, ReLU, ...)

    Args:
        dims: output channels of each stage; its length is the number of layers.
        kernel_size: kernel size of every conv (odd number).
        num_classes: >0 adds a linear classifier head (unused for pretraining, timm passes 0).
        global_pool: 'avg' pools the feature map to a vector; '' keeps the map (needed by CNN-JEPA).
    """

    def __init__(
        self,
        dims: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        in_chans: int = 3,
        num_classes: int = 0,
        global_pool: str = 'avg',
        **kwargs,
    ):
        super().__init__()
        self.dims = list(dims)
        self.num_features = self.dims[-1]
        self.num_classes = num_classes

        stages = []
        prev = in_chans
        for d in self.dims:
            stages.append(nn.Sequential(
                nn.Conv2d(prev, d, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            ))
            prev = d
        self.stages = nn.Sequential(*stages)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) if global_pool else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def get_downsample_ratio(self) -> int:
        return 2 ** len(self.dims)

    def get_feature_map_channels(self) -> List[int]:
        return self.dims

    def forward(self, x: torch.Tensor, hierarchical: bool = False):
        if hierarchical:
            feats = []
            for stage in self.stages:
                x = stage(x)
                feats.append(x)
            return feats
        x = self.stages(x)
        return self.head(self.pool(x))


@register_model
def simple_cnn(pretrained=False, **kwargs):
    """Name used in the config: backbone.name: "simple_cnn".
    Pass dims / kernel_size through backbone.kwargs to change the depth and width."""
    if pretrained:
        raise ValueError("simple_cnn has no pretrained weights")
    return SimpleCNN(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('simple_cnn', num_classes=0, global_pool='')
    B, C, H, W = 4, 3, 128, 128
    out = cnn(torch.rand(B, C, H, W))
    ratio = cnn.get_downsample_ratio()
    print('downsample ratio:', ratio, '| output:', tuple(out.shape), '| num_features:', cnn.num_features)
    assert out.shape == (B, cnn.num_features, H // ratio, W // ratio)
    print('parameters:', sum(p.numel() for p in cnn.parameters()))


if __name__ == '__main__':
    convnet_test()
