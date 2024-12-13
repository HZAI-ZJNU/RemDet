from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from ..layers import CSPLayerWithTwoConv, SPPFBottleneck
from ..utils import yolo_make_divisible as make_divisible, make_round
from .base_backbone import BaseBackbone

from mmdet.registry import MODELS
# from mmdet.models.layers import (GatedFFN, CED, MSPPF, RemDetStem)
from mmdet.models.layers.layers_remnet import *


@MODELS.register_module()
class RemNet(BaseBackbone):
    """
        remdet backbone
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp, is_first
    # the final out_channels will be set according to the param.
    arch_settings = {
        # in_channels, out_channels, num_blocks, add_identity, use_spp, is_first, expansion
        'P5': [[64, 128, 3, True, False, True, 2], [128, 256, 3, True, False, False, 1],
               [256, 512, 6, True, False, False, 1], [512, None, 3, True, True, False, 1]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 channel_expansion_ratio: int = 1,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.channel_expansion_ratio = channel_expansion_ratio
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp, is_first, expansion = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = CED(in_channels,
                         out_channels,
                         e=expansion,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg) if not is_first else ConvModule(in_channels,
                                                                               out_channels,
                                                                               kernel_size=3,
                                                                               stride=2,
                                                                               padding=1,
                                                                               norm_cfg=self.norm_cfg,
                                                                               act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = GatedFFN(
            out_channels,
            out_channels,
            n=num_blocks,
            shortcut=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
