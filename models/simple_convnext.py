"""A small ConvNeXt-style backbone for CNN-JEPA experiments.

Same skeleton as `simple_cnn` (models/custom.py): 3 stages, each halving the image,
channels 32 -> 64 -> 128, so a 224x224 image becomes a 28x28 grid (downsample ratio 8).
The difference is *what* each stage is made of — the ConvNeXt recipe:

    downsample: 2x2 conv, stride 2 (+ LayerNorm)
    block(s):   7x7 depthwise conv -> LayerNorm -> 1x1 expand (x4) -> GELU -> 1x1 project -> + residual

It is built from the masking-aware layers in models/sparse_encoder.py (SparseConvNeXtLayerNorm,
SparseConvNeXtBlock), the same ones the repo's full ConvNeXt (models/convnext.py) uses. This matters:
built from plain PyTorch layers, the LayerNorm / 1x1 layers would put non-zero values into the hidden
(masked) cells and the context encoder could "see" where the mask is.

The name starts with "convnext" on purpose: IJEPA_CNN then uses LayerNorm (not BatchNorm) in the predictor,
as it does for the authors' ConvNeXt.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from models.sparse_encoder import SparseConvNeXtBlock, SparseConvNeXtLayerNorm


class SimpleConvNeXt(nn.Module):
    """
    Args:
        dims: channels of each stage; its length is the number of stages (each stage halves H and W).
        depths: number of ConvNeXt blocks in each stage (same length as dims).
        kernel_size: kernel of the depthwise conv inside each block (ConvNeXt uses 7).
        layer_scale_init_value: initial scale of each block's residual branch (ConvNeXt default 1e-6).
        num_classes: >0 adds a linear classifier head (unused for pretraining, timm passes 0).
        global_pool: 'avg' pools the feature map to a vector; '' keeps the map (needed by CNN-JEPA).
        sparse: whether the masking-aware layers apply the mask. IJEPA_CNN sets this itself:
                True for the context encoder, False for the target (momentum) encoder.
    """

    def __init__(
        self,
        dims: Sequence[int] = (32, 64, 128),
        depths: Sequence[int] = (1, 1, 1),
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        in_chans: int = 3,
        num_classes: int = 0,
        global_pool: str = 'avg',
        sparse: bool = True,
        **kwargs,  # timm may pass extra keys (e.g. pretrained_cfg); they are ignored
    ):
        super().__init__()
        assert len(dims) == len(depths), "dims and depths must have the same length (one entry per stage)"
        self.dims: List[int] = list(dims)
        self.depths: List[int] = list(depths)
        self.num_features = self.dims[-1]
        self.num_classes = num_classes

        # Downsampling layers: 2x2 conv with stride 2 halves H and W.
        # First one ("stem") works on the image: conv then LayerNorm. The others: LayerNorm then conv.
        self.downsample_layers = nn.ModuleList()
        for i, d in enumerate(self.dims):
            if i == 0:
                layer = nn.Sequential(
                    nn.Conv2d(in_chans, d, kernel_size=2, stride=2),
                    SparseConvNeXtLayerNorm(d, eps=1e-6, data_format="channels_first", sparse=sparse),
                )
            else:
                layer = nn.Sequential(
                    SparseConvNeXtLayerNorm(self.dims[i - 1], eps=1e-6, data_format="channels_first", sparse=sparse),
                    nn.Conv2d(self.dims[i - 1], d, kernel_size=2, stride=2),
                )
            self.downsample_layers.append(layer)

        # Stages: `depth` ConvNeXt blocks each, keeping H, W and channels unchanged.
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                SparseConvNeXtBlock(dim=d, layer_scale_init_value=layer_scale_init_value, sparse=sparse, ks=kernel_size)
                for _ in range(depth)
            ])
            for d, depth in zip(self.dims, self.depths)
        ])

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) if global_pool else nn.Identity()
        if num_classes > 0:
            self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
            self.head = nn.Linear(self.num_features, num_classes)
        else:
            self.norm = nn.Identity()
            self.head = nn.Identity()

        self.apply(self._init_weights)
        self._sparse = sparse

    @staticmethod
    def _init_weights(m):
        # Same initialisation as the authors' ConvNeXt
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_downsample_ratio(self) -> int:
        return 2 ** len(self.dims)

    def get_feature_map_channels(self) -> List[int]:
        return self.dims

    def forward(self, x: torch.Tensor, hierarchical: bool = False):
        feats = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            feats.append(x)
        if hierarchical:
            return feats
        x = self.pool(x)
        if self.num_classes > 0:
            x = self.head(self.norm(x))
        return x

    # IJEPA_CNN checks for this attribute: it sets sparse=True on the context encoder
    # and sparse=False on the target (momentum) encoder, which sees the full image.
    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        self._sparse = value
        for m in self.modules():
            if isinstance(m, (SparseConvNeXtLayerNorm, SparseConvNeXtBlock)):
                m.sparse = value


@register_model
def convnext_simple(pretrained=False, **kwargs):
    """Name used in the config: backbone.name: "convnext_simple".
    Change dims / depths / kernel_size through backbone.kwargs."""
    if pretrained:
        raise ValueError("convnext_simple has no pretrained weights")
    return SimpleConvNeXt(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('convnext_simple', num_classes=0, global_pool='', sparse=False)
    B, C, H, W = 4, 3, 224, 224
    out = cnn(torch.rand(B, C, H, W))
    ratio = cnn.get_downsample_ratio()
    print('downsample ratio:', ratio, '| output:', tuple(out.shape), '| num_features:', cnn.num_features)
    assert out.shape == (B, cnn.num_features, H // ratio, W // ratio)
    print('parameters:', sum(p.numel() for p in cnn.parameters()))


if __name__ == '__main__':
    convnet_test()
