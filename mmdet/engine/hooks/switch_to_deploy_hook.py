# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.models.layers import RepVGGBlock
from mmdet.registry import HOOKS


def switch_to_deploy(model):
    """Model switch to deploy status."""
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    print('Switch model to deploy modality.')

@HOOKS.register_module()
class SwitchToDeployHook(Hook):
    """Switch to deploy mode before testing.

    This hook converts the multi-channel structure of the training network
    (high performance) to the one-way structure of the testing network (fast
    speed and  memory saving).
    """

    def before_test_epoch(self, runner: Runner):
        """Switch to deploy mode before testing."""
        switch_to_deploy(runner.model)
