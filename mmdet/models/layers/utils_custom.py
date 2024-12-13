import math
from functools import partial
from typing import Union, List, Tuple, Optional, Sequence

import numpy as np
import torch
from mmcv.cnn import ConvModule
from timm.layers import DropPath
from torch import nn, Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType
from .yolo_bricks import CSPLayerWithTwoConv, SPPFBottleneck, DarknetBottleneck
from mmdet.models.utils import yolo_make_divisible as make_divisible, make_round
from mmcv.cnn import (ConvModule, build_norm_layer)
from mmengine.model import BaseModule
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def get_act(act_name):
    dict_act = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'silu': nn.SiLU,
        'hs': nn.Hardswish,
        'gelu': nn.GELU
    }
    return dict_act[act_name]


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, groups=1, dilation=1, bias=True, norm_layer='none',
                 act_layer='none', padding=None, drop_path_rate=0.):
        super().__init__()
        assert stride in [1, 2], 'stride must 1 or 2'
        self.padding = autopad(kernel_size, None, dilation) if padding is None else padding
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=self.padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
