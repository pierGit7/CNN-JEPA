# Autoencoder baseline for CNN-JEPA, following SparK (Tian et al., ICLR 2023, "Designing BERT for
# Convolutional Networks: Sparse and Hierarchical Masked Modeling", https://github.com/keyu-tian/SparK).
#
# The same encoders as CNN-JEPA (e.g. simple_cnn, convnext_simple) are trained to reconstruct the PIXELS of
# hidden image cells, instead of predicting the FEATURES of a momentum target encoder:
#
#   1. Mask:    hide random cells of the feature grid (mask_ratio=0.6, as SparK).
#   2. Encode:  the sparse encoder (models/sparse_encoder.py, same layers as CNN-JEPA) processes only the visible
#               cells, so hidden cells stay zero and no information leaks into them.
#   3. Densify: at every encoder stage, fill hidden cells with a learned mask token, normalize, project.
#   4. Decode:  a light UNet-style decoder upsamples back to the image, adding the densified stage features.
#   5. Loss:    MSE to the original pixels on hidden cells only, with each patch normalized (mean 0, std 1).
#
# mask_ratio == 0 gives a plain autoencoder instead: no masking, the decoder only gets the last feature map
# (with skip connections from the shallow stages it could just copy the input), loss on all pixels.
#
# After training, the decoder is thrown away. The encoder is evaluated with the same online linear / kNN
# benchmark as the CNN-JEPA runs (val_metrics/lin_top1, ...), so the two can be compared directly.
#
# Run from the repo root:
#   PYTHONPATH=. python pretrain/train_autoencoder.py --config-name ae_imagenette_simplecnn.yaml

import math

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torchvision import transforms
from lightly.transforms.utils import IMAGENET_NORMALIZE

from timm.models.layers import trunc_normal_

from pretrain.trainer_common import LightlyModel, main_pretrain
from pretrain.ijepa_mask import MultiBlockMask

import models.sparse_encoder as sparse_encoder
from models.sparse_encoder import SparseBatchNorm2d, SparseConvNeXtLayerNorm
# import models.X registers the backbones in the timm model registry
import models.convnext
import models.custom  # simple_cnn
import models.simple_convnext  # convnext_simple


class DenseView(nn.Module):
    """Runs a sparse (masking-aware) encoder with every cell active, i.e. as a normal dense encoder.
    Used for the online classification benchmark of the masked autoencoder."""

    def __init__(self, encoder: nn.Module, downsample_ratio: int):
        super().__init__()
        self.encoder = encoder
        self.downsample_ratio = downsample_ratio
        self.num_features = encoder.num_features

    def forward(self, x):
        B, _, H, W = x.shape
        sparse_encoder._cur_active = torch.ones(
            B, 1, H // self.downsample_ratio, W // self.downsample_ratio, dtype=torch.bool, device=x.device
        )
        return self.encoder(x)


class UNetBlock(nn.Module):
    """SparK decoder block: 2x upsampling, then two 3x3 convs."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        return self.conv(self.up_sample(x))


class LightDecoder(nn.Module):
    """SparK light decoder: log2(downsample_ratio) UNet blocks, channels halving from `width`.
    Before block i, the densified encoder feature map of the same resolution (to_dec[i]) is added."""

    def __init__(self, downsample_ratio: int, width: int):
        super().__init__()
        n = round(math.log2(downsample_ratio))
        channels = [width // 2 ** i for i in range(n + 1)]
        self.dec = nn.ModuleList([UNetBlock(cin, cout) for cin, cout in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

    def forward(self, to_dec):
        x = 0
        for i, block in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = block(x)
        return self.proj(x)


class Autoencoder(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.masked = cfg.mask_ratio > 0
        self.downsample_ratio = self.backbone.get_downsample_ratio()

        if hasattr(self.backbone, 'sparse'):
            # convnext_simple: masking-aware layers only when masking
            self.backbone.sparse = self.masked
        if self.masked:
            self.backbone = sparse_encoder.dense_model_to_sparse(self.backbone)
            self.backbone_for_online_eval = DenseView(self.backbone, self.downsample_ratio)
        else:
            self.backbone_for_online_eval = self.backbone

        # Encoder stage channels, deepest first. Plain mode only uses the deepest stage.
        enc_channels = list(reversed(self.backbone.get_feature_map_channels()))
        if not self.masked:
            enc_channels = enc_channels[:1]
        is_convnext = cfg.backbone.name.lower().startswith('convnext')

        dec_width = cfg.decoder.get('width', None) or self.backbone.num_features
        self.decoder = LightDecoder(self.downsample_ratio, width=dec_width)

        # Densify each encoder stage (SparK): fill hidden cells with a mask token, normalize, project to the decoder width.
        self.mask_tokens = nn.ParameterList()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        d_width = dec_width
        for i, e_width in enumerate(enc_channels):
            token = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(token, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(token)
            # The norm runs before the mask tokens are filled in, so it only sees visible cells (sparse) when masking
            if is_convnext:
                norm = SparseConvNeXtLayerNorm(e_width, eps=1e-6, data_format="channels_first", sparse=self.masked)
            else:
                norm = SparseBatchNorm2d(e_width) if self.masked else nn.BatchNorm2d(e_width)
            self.densify_norms.append(norm)
            if i == 0 and e_width == d_width:
                proj = nn.Identity()
            else:
                kernel_size = 1 if i == 0 else 3
                proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
            self.densify_projs.append(proj)
            d_width //= 2

    def setup_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.fmap_h, self.fmap_w = self.input_size // self.downsample_ratio, self.input_size // self.downsample_ratio
        if self.masked:
            self.len_keep = round(self.fmap_h * self.fmap_w * (1 - self.cfg.mask_ratio))
            self.multi_block_mask = MultiBlockMask(
                input_size=self.input_size,
                patch_size=self.downsample_ratio,
                **self.cfg.mask.mutli_block_kwargs
            )

    def mask(self, B: int, device, generator=None):
        """Same masks as IJEPA_CNN.mask: returns (context_mask, target_mask), each (B, 1, h, w) bool."""
        if self.cfg.mask.strategy == "mixed":
            if torch.rand(1) < self.cfg.mask.mixed_mutli_block_ratio:
                strategy = "multi-block"
            else:
                strategy = "random"
        else:
            strategy = self.cfg.mask.strategy
        if strategy == "random":
            h, w = self.fmap_h, self.fmap_w
            idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
            idx = idx[:, :self.len_keep]  # (B, len_keep)
            # built on CPU: a bool scatter_ on MPS (torch 1.13) breaks the sparse BatchNorm downstream
            context_mask = torch.zeros(B, h * w, dtype=torch.bool).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w).to(device)
            target_mask = context_mask.logical_not()
            return context_mask, target_mask
        elif strategy == "multi-block":
            context_mask, target_mask = self.multi_block_mask(B)
            context_mask = context_mask.unsqueeze(1).to(device, dtype=torch.bool)
            target_mask = target_mask.unsqueeze(1).to(device, dtype=torch.bool)
            return context_mask, target_mask

    def to_pixels(self, mask_b1ff):
        return mask_b1ff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio, 3)

    def forward(self, x):
        """Returns the reconstruction (B, 3, H, W), and the target mask (B, 1, H, W) or None."""
        if self.masked:
            context_mask_b1ff, target_mask_b1ff = self.mask(x.shape[0], x.device)
            sparse_encoder._cur_active = context_mask_b1ff
            x = x * self.to_pixels(context_mask_b1ff)

        # Encoder feature maps of every stage, deepest first: (B, C_i, h * 2**i, w * 2**i)
        fea_bcffs = list(reversed(self.backbone(x, hierarchical=True)))[:len(self.mask_tokens)]

        to_dec = []
        cur_active = context_mask_b1ff if self.masked else None
        for i, bcff in enumerate(fea_bcffs):
            bcff = self.densify_norms[i](bcff)
            if self.masked:
                # fill hidden cells with this stage's [mask] token
                mask_tokens = self.mask_tokens[i].expand_as(bcff).to(bcff.dtype)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
                cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
            to_dec.append(self.densify_projs[i](bcff))

        reconstruction = self.decoder(to_dec)
        return reconstruction, (self.to_pixels(target_mask_b1ff) if self.masked else None)

    def reconstruction_target(self, x):
        if not self.cfg.decoder.get('norm_pix_loss', False):
            return x
        # MAE/SparK-style: normalize each (downsample_ratio x downsample_ratio) patch to zero mean, unit variance
        B, C, H, W = x.shape
        r = self.downsample_ratio
        patches = x.view(B, C, H // r, r, W // r, r)
        mean = patches.mean(dim=(1, 3, 5), keepdim=True)
        var = patches.var(dim=(1, 3, 5), keepdim=True)
        return ((patches - mean) / (var + 1e-6).sqrt()).view(B, C, H, W)

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x = batch[0]
        reconstruction, target_mask_b1hw = self.forward(x)
        loss = (reconstruction - self.reconstruction_target(x)).pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        if target_mask_b1hw is not None:
            loss = loss.mul(target_mask_b1hw).sum() / (target_mask_b1hw.sum() + 1e-8)  # loss only on hidden pixels
        else:
            loss = loss.mean()
        self.log(f"{metric_label}/ae_loss", loss, on_epoch=True)
        return loss

    def get_views_to_log_from_batch(self, batch):
        x = batch[0]
        with torch.no_grad():
            reconstruction, target_mask_b1hw = self.forward(x)
        # with norm_pix_loss the reconstruction is in per-patch normalized units, so colours look different from x
        if target_mask_b1hw is None:
            return [x, reconstruction]
        return [x, x * target_mask_b1hw.logical_not(), reconstruction]


@hydra.main(version_base="1.2", config_path="configs/", config_name="ae_imagenette_simplecnn.yaml")
def pretrain_autoencoder(cfg: DictConfig):
    main_pretrain(cfg, Autoencoder)

if __name__ == "__main__":
    pretrain_autoencoder()
