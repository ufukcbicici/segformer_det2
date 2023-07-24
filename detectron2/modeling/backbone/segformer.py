from math import sqrt
from functools import partial
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from detectron2.config import get_cfg, configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY
from detectron2.config import CfgNode as CN


def get_segformer_config(config_file):
    """
    Add default config for Segformer.
    """
    _C = get_cfg()
    _C.MODEL.SEGFORMER = CN()

    _C.MODEL.SEGFORMER.LAYER_DIMENSIONS = (32, 64, 160, 256)
    _C.MODEL.SEGFORMER.ATTENTION_HEADS = (1, 2, 5, 8)
    _C.MODEL.SEGFORMER.FEED_FORWARD_EXPANSION_RATIOS = (8, 8, 4, 4)
    _C.MODEL.SEGFORMER.ATTENTION_REDUCTION_RATIOS = (8, 4, 2, 1)
    _C.MODEL.SEGFORMER.NUM_OF_EFFICIENT_SELF_ATTENTION_AND_MIX_FEED_FORWARD_LAYERS = 2
    _C.MODEL.SEGFORMER.NUM_OF_INPUT_CHANNELS = 3
    _C.MODEL.SEGFORMER.DECODER_DIMENSION = 256
    _C.MODEL.SEGFORMER.NUM_CLASSES = 4

    _C.merge_from_file(config_file)
    return _C


# helpers

def exists(val):
    return val is not None


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads,
            reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
            self,
            *,
            channels,
            dims,
            heads,
            ff_expansion,
            reduction_ratio,
            num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        self.dimInDimOutList = []
        self.kernelStridePaddingList = []
        self.numLayersList = []
        self.ffExpansionList = []
        self.headsList = []
        self.reductionRatioList = []

        self._out_features = []

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
                dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):

            self.dimInDimOutList.append((dim_in, dim_out))
            self.kernelStridePaddingList.append((kernel, stride, padding))
            self.numLayersList.append(num_layers)
            self.ffExpansionList.append(ff_expansion)
            self.headsList.append(heads)
            self.reductionRatioList.append(reduction_ratio)

            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

        for stage_id in range(len(self.stages)):
            self._out_features.append("mit_stage_{0}".format(stage_id))

    @property
    def out_features(self):
        return self._out_features

    def forward(self, x):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = layer_outputs
        return ret


class SegformerBackbone(Backbone):
    def __init__(self,
                 dims,
                 heads,
                 ff_expansion,
                 reduction_ratio,
                 num_layers,
                 channels,
                 decoder_dim,
                 num_classes):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.ffExpansion = ff_expansion
        self.reductionRatio = reduction_ratio
        self.numLayers = num_layers
        self.channels = channels
        self.decoderDim = decoder_dim
        self.numClasses = num_classes

        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth=4), (
            dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio,
                                                 num_layers))]), \
            'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )

        self._out_features.extend(self.mit.out_features)

        # self.to_fused = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(dim, decoder_dim, 1),
        #     nn.Upsample(scale_factor=2 ** i)
        # ) for i, dim in enumerate(dims)])
        #
        # self.to_segmentation = nn.Sequential(
        #     nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
        #     nn.Conv2d(decoder_dim, num_classes, 1),
        # )

        # Run the model with an empty input to calculate the number of output channels
        images = torch.from_numpy(np.random.uniform(size=(8, 3, 512, 512))).to(torch.float)
        self.eval()
        outputs = self(images)
        for output_name, v in outputs.items():
            self._out_feature_channels[output_name] = v.shape[1]
            self._out_feature_strides[output_name] = None

    @property
    def out_features(self):
        return self._out_features

    def forward(self, x):
        outputs = {}
        layer_outputs = self.mit(x)

        # fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        # fused = torch.cat(fused, dim=1)
        # final_map = self.to_segmentation(fused)
        # outputs["segformer"] = final_map
        for mit_stage_id, mit_stage_name in enumerate(self.mit.out_features):
            outputs[mit_stage_name] = layer_outputs[mit_stage_id]

        return outputs

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


@SEM_SEG_HEADS_REGISTRY.register()
class SegformerHead(nn.Module):
    @configurable
    def __init__(self, dims, decoder_dim, num_classes):
        super().__init__()
        self.dims = dims
        self.decoderDim = decoder_dim
        self.numClasses = num_classes

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
            "num_classes": cfg.MODEL.SEGFORMER.NUM_CLASSES
        }

    #TODO: Convert this into Segformer compatible format.
    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
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
            feature_name = "mit_stage_{0}".format(stage_id)
            features_sorted.append(features[feature_name])

        fused = [to_fused(output) for output, to_fused in zip(features_sorted, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        final_map = self.to_segmentation(fused)

        if self.training:
            return None, self.losses(final_map, targets)
        else:
            return final_map, {}

#
#     def forward(self, features, targets=None):
#         """
#         Returns:
#             In training, returns (None, dict of losses)
#             In inference, returns (CxHxW logits, {})
#         """
#         x = self.layers(features)
#         if self.training:
#             return None, self.losses(x, targets)
#         else:
#             x = F.interpolate(
#                 x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
#             )
#             return x, {}
#
#     def layers(self, features):
#         for i, f in enumerate(self.in_features):
#             if i == 0:
#                 x = self.scale_heads[i](features[f])
#             else:
#                 x = x + self.scale_heads[i](features[f])
#         x = self.predictor(x)
#         return x
#
#     def losses(self, predictions, targets):
#         predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
#         predictions = F.interpolate(
#             predictions,
#             scale_factor=self.common_stride,
#             mode="bilinear",
#             align_corners=False,
#         )
#         loss = F.cross_entropy(
#             predictions, targets, reduction="mean", ignore_index=self.ignore_value
#         )
#         losses = {"loss_sem_seg": loss * self.loss_weight}
#         return losses
#

@BACKBONE_REGISTRY.register()
def build_segformer_backbone(cfg, input_shape):
    dims = cfg.MODEL.SEGFORMER.LAYER_DIMENSIONS
    heads = cfg.MODEL.SEGFORMER.ATTENTION_HEADS
    ff_expansion = cfg.MODEL.SEGFORMER.FEED_FORWARD_EXPANSION_RATIOS
    reduction_ratio = cfg.MODEL.SEGFORMER.ATTENTION_REDUCTION_RATIOS
    num_layers = cfg.MODEL.SEGFORMER.NUM_OF_EFFICIENT_SELF_ATTENTION_AND_MIX_FEED_FORWARD_LAYERS
    channels = cfg.MODEL.SEGFORMER.NUM_OF_INPUT_CHANNELS
    decoder_dim = cfg.MODEL.SEGFORMER.DECODER_DIMENSION
    num_classes = cfg.MODEL.SEGFORMER.NUM_CLASSES

    segformer_backbone = SegformerBackbone(dims=dims,
                                           heads=heads,
                                           ff_expansion=ff_expansion,
                                           reduction_ratio=reduction_ratio,
                                           num_layers=num_layers,
                                           channels=channels,
                                           decoder_dim=decoder_dim,
                                           num_classes=num_classes)
    return segformer_backbone
