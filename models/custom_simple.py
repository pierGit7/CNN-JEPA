# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model


class BaseConv(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, num_classes=1000, dims=(32, 64, 128), in_chans=3, global_pool='avg', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.dims = list(dims)
        self.num_features = self.dims[-1]
        self.stages = nn.ModuleList()
        prev = in_chans
        for d in self.dims:
            self.stages.append(nn.Sequential(
                nn.Conv2d(prev, d, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            ))
            prev = d
        # global_pool='' and num_classes=0 (as passed by the pretraining code) make forward return the last feature map
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) if global_pool else nn.Identity()
        self.fc = nn.Linear(self.dims[-1], num_classes) if num_classes > 0 else nn.Identity()
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 2 ** len(self.stages)
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.dims
    
    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        x = inp_bchw
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        if hierarchical:
            return feats
        return self.fc(self.pool(x))

@register_model
def base_conv(pretrained=False, **kwargs):
    return BaseConv(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('base_conv')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
