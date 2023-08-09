# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from functools import partial
from typing import Dict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from detectron2.config import CfgNode as CN, configurable
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY


def get_segformer_config(config_file):
    """
    Add default config for Segformer.
    """

    # Add parameters for the Segformer model.
    _C = get_cfg()
    _C.MODEL.SEGFORMER = CN()

    _C.MODEL.SEGFORMER.MODEL_TYPE = "mit_b0"
    _C.MODEL.SEGFORMER.LAYER_DIMENSIONS = (32, 64, 160, 256)
    _C.MODEL.SEGFORMER.ATTENTION_HEADS = (1, 2, 5, 8)
    _C.MODEL.SEGFORMER.FEED_FORWARD_EXPANSION_RATIOS = (8, 8, 4, 4)
    _C.MODEL.SEGFORMER.ATTENTION_REDUCTION_RATIOS = (8, 4, 2, 1)
    _C.MODEL.SEGFORMER.NUM_OF_EFFICIENT_SELF_ATTENTION_AND_MIX_FEED_FORWARD_LAYERS = (2, 2, 2, 2)
    _C.MODEL.SEGFORMER.NUM_OF_INPUT_CHANNELS = 3
    _C.MODEL.SEGFORMER.DECODER_DIMENSION = 256
    _C.MODEL.SEGFORMER.NUM_CLASSES = 4
    _C.MODEL.SEGFORMER.USE_PRETRAINED_BACKBONE = False

    _C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = -1

    # Add parameters for the AdamW optimizer
    _C.SOLVER.BETAS = (0.9, 0.999)
    _C.SOLVER.HEAD_LR_MULTIPLIER = 10.0

    # Add parameters for Polynomial LR scheduler
    _C.SOLVER.POLY_POWER = 1.0

    _C.SOLVER.BACKBONE_REGULAR_DECAY_MULTIPLIER = 1.0
    _C.SOLVER.BACKBONE_NORM_DECAY_MULTIPLIER = 0.0
    _C.SOLVER.HEAD_REGULAR_DECAY_MULTIPLIER = 1.0
    _C.SOLVER.HEAD_NORM_DECAY_MULTIPLIER = 0.0

    # _RAW_SOILCOVER_SPLITS = {
    #     "soilcover_train": (
    #         "/shared/data/SoilCoverGitter/train/img/remap/", "/shared/data/SoilCoverGitter/train/lbl/remap/"),
    #     "soilcover_val": ("/shared/data/SoilCoverGitter/valid/img/remap/", "/shared/data/SoilCoverGitter/valid/lbl/remap/"),
    #     "soilcover_test": ("/shared/data/SoilCoverGitter/test/img/remap/", "/shared/data/SoilCoverGitter/test/lbl/remap/"),
    # }

    _C.DATASETS.SOILCOVER = CN()
    _C.DATASETS.SOILCOVER.TRAIN_FILES = ("/shared/data/SoilCoverGitter/train/img/remap/",
                                         "/shared/data/SoilCoverGitter/train/lbl/remap/")
    _C.DATASETS.SOILCOVER.VALIDATION_FILES = ("/shared/data/SoilCoverGitter/valid/img/remap/",
                                              "/shared/data/SoilCoverGitter/valid/lbl/remap/")
    _C.DATASETS.SOILCOVER.TEST_FILES = ("/shared/data/SoilCoverGitter/test/img/remap/",
                                        "/shared/data/SoilCoverGitter/test/lbl/remap/")
    _C.DATASETS.SOILCOVER.TRAIN_MEAN_PIXELS = (0.0, 0.0, 0.0)
    _C.DATASETS.SOILCOVER.TRAIN_STD_PIXELS = (1.0, 1.0, 1.0)
    _C.DATASETS.SOILCOVER.VALIDATION_MEAN_PIXELS = (0.0, 0.0, 0.0)
    _C.DATASETS.SOILCOVER.VALIDATION_STD_PIXELS = (1.0, 1.0, 1.0)
    _C.DATASETS.SOILCOVER.TEST_MEAN_PIXELS = (0.0, 0.0, 0.0)
    _C.DATASETS.SOILCOVER.TEST_STD_PIXELS = (1.0, 1.0, 1.0)
    _C.DATASETS.SOILCOVER.CALCULATE_DATASET_STATISTICS = True
    _C.DATASETS.SOILCOVER.TRAIN_LABELS = (0, 1, 2, 3)
    _C.DATASETS.SOILCOVER.VALIDATION_LABELS = (0, 1, 2, 3)
    _C.DATASETS.SOILCOVER.TEST_LABELS = (0, 1, 2, 3)

    _C.INPUT.COLOR_AUG_SSD = False

    _C.merge_from_file(config_file)
    return _C


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(Backbone):
    # def __init__(self,
    #              in_chans=3,
    #              num_classes=1000,
    #              embed_dims=[64, 128, 256, 512],
    #              num_heads=[1, 2, 4, 8],
    #              mlp_ratios=[4, 4, 4, 4],
    #              qkv_bias=False,
    #              qk_scale=None,
    #              drop_rate=0.,
    #              attn_drop_rate=0.,
    #              drop_path_rate=0.,
    #              norm_layer=nn.LayerNorm,
    #              depths=[3, 4, 6, 3],
    #              sr_ratios=[8, 4, 2, 1]):
    def __init__(self,
                 in_chans,
                 num_classes,
                 embed_dims,
                 num_heads,
                 mlp_ratios,
                 qkv_bias,
                 qk_scale,
                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,
                 norm_layer,
                 depths,
                 sr_ratios):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self._out_features = ["mit_stage_1", "mit_stage_2", "mit_stage_3", "mit_stage_4"]

        # Run the model with an empty input to calculate the number of output channels
        images = torch.from_numpy(np.random.uniform(size=(8, 3, 512, 512))).to(torch.float)
        self.eval()
        outputs = self(images)
        for output_name, v in outputs.items():
            self._out_feature_channels[output_name] = v.shape[1]
            self._out_feature_strides[output_name] = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        outputs = {}
        layer_outputs = self.forward_features(x)

        for mit_stage_id, mit_stage_name in enumerate(self._out_features):
            outputs[mit_stage_name] = layer_outputs[mit_stage_id]

        return outputs

    @property
    def out_features(self):
        return self._out_features

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 32

    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        This property is a generalization of size_divisibility. Some backbones and training
        recipes require specific padding constraints, such as enforcing divisibility by a specific
        integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
        in :paper:vitdet). `padding_constraints` contains these optional items like:
        {
            "size_divisibility": int,
            "square_size": int,
            # Future options are possible
        }
        `size_divisibility` will read from here if presented and `square_size` indicates the
        square padding size if `square_size` > 0.

        TODO: use type of Dict[str, int] to avoid torchscipt issues. The type of padding_constraints
        could be generalized as TypedDict (Python 3.8+) to support more types in the future.
        """
        return {"size_divisibility": 32}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


@SEM_SEG_HEADS_REGISTRY.register()
class SegformerHead(nn.Module):
    @configurable
    def __init__(self, dims, decoder_dim, num_classes, ignore_value):
        super().__init__()
        self.dims = dims
        self.decoderDim = decoder_dim
        self.numClasses = num_classes
        self.ignore_value = ignore_value

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    # We don't need input_shape
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "dims": cfg.MODEL.SEGFORMER.LAYER_DIMENSIONS,
            "decoder_dim": cfg.MODEL.SEGFORMER.DECODER_DIMENSION,
            "num_classes": cfg.MODEL.SEGFORMER.NUM_CLASSES,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        }

    def losses(self, predictions, targets):
        targets = targets.to(torch.long)
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss}
        return losses

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        assert isinstance(features, dict) and len(features) == 4
        features_sorted = []
        for stage_id in range(len(features)):
            feature_name = "mit_stage_{0}".format(stage_id + 1)
            features_sorted.append(features[feature_name])

        fused = [to_fused(output) for output, to_fused in zip(features_sorted, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        final_map = self.to_segmentation(fused)

        # Rescale back to the original size
        predictions = final_map.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

        if self.training:
            return None, self.losses(predictions, targets)
        else:
            return predictions, {}


@BACKBONE_REGISTRY.register()
def build_segformer_backbone(cfg, input_shape):
    dims = cfg.MODEL.SEGFORMER.LAYER_DIMENSIONS
    heads = cfg.MODEL.SEGFORMER.ATTENTION_HEADS
    ff_expansion = cfg.MODEL.SEGFORMER.FEED_FORWARD_EXPANSION_RATIOS
    reduction_ratio = cfg.MODEL.SEGFORMER.ATTENTION_REDUCTION_RATIOS
    num_layers = cfg.MODEL.SEGFORMER.NUM_OF_EFFICIENT_SELF_ATTENTION_AND_MIX_FEED_FORWARD_LAYERS
    num_classes = cfg.MODEL.SEGFORMER.NUM_CLASSES

    segformer_backbone = MixVisionTransformer(in_chans=3,
                                              attn_drop_rate=0.0,
                                              depths=num_layers,
                                              drop_path_rate=0.1,
                                              drop_rate=0.0,
                                              embed_dims=dims,
                                              mlp_ratios=ff_expansion,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                              num_classes=num_classes,
                                              num_heads=heads,
                                              qk_scale=None,
                                              qkv_bias=True,
                                              sr_ratios=reduction_ratio)

    return segformer_backbone
