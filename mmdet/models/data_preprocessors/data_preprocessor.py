# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import is_list_of
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType
from mmengine.structures import BaseDataElement

try:
    import skimage
except ImportError:
    skimage = None

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


@MODELS.register_module()
class DetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


@MODELS.register_module()
class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 10,
                 size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor),
                                   round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(
        self, inputs: Tensor, data_samples: List[DetDataSample]
    ) -> Tuple[Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })
                data_sample.gt_instances.bboxes[
                    ...,
                    0::2] = data_sample.gt_instances.bboxes[...,
                                                            0::2] * scale_x
                data_sample.gt_instances.bboxes[
                    ...,
                    1::2] = data_sample.gt_instances.bboxes[...,
                                                            1::2] * scale_y
                if 'ignored_instances' in data_sample:
                    data_sample.ignored_instances.bboxes[
                        ..., 0::2] = data_sample.ignored_instances.bboxes[
                            ..., 0::2] * scale_x
                    data_sample.ignored_instances.bboxes[
                        ..., 1::2] = data_sample.ignored_instances.bboxes[
                            ..., 1::2] * scale_y
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, aspect_ratio: float,
                         device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size


@MODELS.register_module()
class BatchFixedSizePad(nn.Module):
    """Fixed size padding for batch images.

    Args:
        size (Tuple[int, int]): Fixed padding size. Expected padding
            shape (h, w). Defaults to None.
        img_pad_value (int): The padded pixel value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
    """

    def __init__(self,
                 size: Tuple[int, int],
                 img_pad_value: int = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255) -> None:
        super().__init__()
        self.size = size
        self.pad_mask = pad_mask
        self.pad_seg = pad_seg
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.seg_pad_value = seg_pad_value

    def forward(
        self,
        inputs: Tensor,
        data_samples: Optional[List[dict]] = None
    ) -> Tuple[Tensor, Optional[List[dict]]]:
        """Pad image, instance masks, segmantic segmentation maps."""
        src_h, src_w = inputs.shape[-2:]
        dst_h, dst_w = self.size

        if src_h >= dst_h and src_w >= dst_w:
            return inputs, data_samples

        inputs = F.pad(
            inputs,
            pad=(0, max(0, dst_w - src_w), 0, max(0, dst_h - src_h)),
            mode='constant',
            value=self.img_pad_value)

        if data_samples is not None:
            # update batch_input_shape
            for data_sample in data_samples:
                data_sample.set_metainfo({
                    'batch_input_shape': (dst_h, dst_w),
                    'pad_shape': (dst_h, dst_w)
                })

            if self.pad_mask:
                for data_sample in data_samples:
                    masks = data_sample.gt_instances.masks
                    data_sample.gt_instances.masks = masks.pad(
                        (dst_h, dst_w), pad_val=self.mask_pad_value)

            if self.pad_seg:
                for data_sample in data_samples:
                    gt_sem_seg = data_sample.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    gt_sem_seg = F.pad(
                        gt_sem_seg,
                        pad=(0, max(0, dst_w - w), 0, max(0, dst_h - h)),
                        mode='constant',
                        value=self.seg_pad_value)
                    data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

        return inputs, data_samples


@MODELS.register_module()
class MultiBranchDataPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor wrapper for multi-branch data.

    Take semi-supervised object detection as an example, assume that
    the ratio of labeled data and unlabeled data in a batch is 1:2,
    `sup` indicates the branch where the labeled data is augmented,
    `unsup_teacher` and `unsup_student` indicate the branches where
    the unlabeled data is augmented by different pipeline.

    The input format of multi-branch data is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor, None, None],
                    'unsup_teacher': [None, Tensor, Tensor],
                    'unsup_student': [None, Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample, None, None],
                    'unsup_teacher': [None, DetDataSample, DetDataSample],
                    'unsup_student': [NOne, DetDataSample, DetDataSample],
                }
        }

    The format of multi-branch data
    after filtering None is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor],
                    'unsup_teacher': [Tensor, Tensor],
                    'unsup_student': [Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample],
                    'unsup_teacher': [DetDataSample, DetDataSample],
                    'unsup_student': [DetDataSample, DetDataSample],
                }
        }

    In order to reuse `DetDataPreprocessor` for the data
    from different branches, the format of multi-branch data
    grouped by branch is as below :

    .. code-block:: none
        {
            'sup':
                {
                    'inputs': [Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
            'unsup_teacher':
                {
                    'inputs': [Tensor, Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
            'unsup_student':
                {
                    'inputs': [Tensor, Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
        }

    After preprocessing data from different branches,
    the multi-branch data needs to be reformatted as:

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor],
                    'unsup_teacher': [Tensor, Tensor],
                    'unsup_student': [Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample],
                    'unsup_teacher': [DetDataSample, DetDataSample],
                    'unsup_student': [DetDataSample, DetDataSample],
                }
        }

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict): Config of
            :class:`DetDataPreprocessor` to process the input data.
    """

    def __init__(self, data_preprocessor: ConfigType) -> None:
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor`` for multi-branch data.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict:

            - 'inputs' (Dict[str, obj:`torch.Tensor`]): The forward data of
                models from different branches.
            - 'data_sample' (Dict[str, obj:`DetDataSample`]): The annotation
                info of the sample from different branches.
        """

        if training is False:
            return self.data_preprocessor(data, training)

        # Filter out branches with a value of None
        for key in data.keys():
            for branch in data[key].keys():
                data[key][branch] = list(
                    filter(lambda x: x is not None, data[key][branch]))

        # Group data by branch
        multi_branch_data = {}
        for key in data.keys():
            for branch in data[key].keys():
                if multi_branch_data.get(branch, None) is None:
                    multi_branch_data[branch] = {key: data[key][branch]}
                elif multi_branch_data[branch].get(key, None) is None:
                    multi_branch_data[branch][key] = data[key][branch]
                else:
                    multi_branch_data[branch][key].append(data[key][branch])

        # Preprocess data from different branches
        for branch, _data in multi_branch_data.items():
            multi_branch_data[branch] = self.data_preprocessor(_data, training)

        # Format data by inputs and data_samples
        format_data = {}
        for branch in multi_branch_data.keys():
            for key in multi_branch_data[branch].keys():
                if format_data.get(key, None) is None:
                    format_data[key] = {branch: multi_branch_data[branch][key]}
                elif format_data[key].get(branch, None) is None:
                    format_data[key][branch] = multi_branch_data[branch][key]
                else:
                    format_data[key][branch].append(
                        multi_branch_data[branch][key])

        return format_data

    @property
    def device(self):
        return self.data_preprocessor.device

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Args:
            device (int or torch.device, optional): The desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.to(device, *args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cpu(*args, **kwargs)


@MODELS.register_module()
class BatchResize(nn.Module):
    """Batch resize during training. This implementation is modified from
    https://github.com/Purkialo/CrowdDet/blob/master/lib/data/CrowdHuman.py.

    It provides the data pre-processing as follows:
    - A batch of all images will pad to a uniform size and stack them into
      a torch.Tensor by `DetDataPreprocessor`.
    - `BatchFixShapeResize` resize all images to the target size.
    - Padding images to make sure the size of image can be divisible by
      ``pad_size_divisor``.

    Args:
        scale (tuple): Images scales for resizing.
        pad_size_divisor (int): Image size divisible factor.
            Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
    """

    def __init__(
        self,
        scale: tuple,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
    ) -> None:
        super().__init__()
        self.min_size = min(scale)
        self.max_size = max(scale)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(
        self, inputs: Tensor, data_samples: List[DetDataSample]
    ) -> Tuple[Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes."""

        batch_height, batch_width = inputs.shape[-2:]
        target_height, target_width, scale = self.get_target_size(
            batch_height, batch_width)

        inputs = F.interpolate(
            inputs,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False)

        inputs = self.get_padded_tensor(inputs, self.pad_value)

        if data_samples is not None:
            batch_input_shape = tuple(inputs.size()[-2:])
            for data_sample in data_samples:
                img_shape = [
                    int(scale * _) for _ in list(data_sample.img_shape)
                ]
                data_sample.set_metainfo({
                    'img_shape': tuple(img_shape),
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': batch_input_shape,
                    'scale_factor': (scale, scale)
                })

                data_sample.gt_instances.bboxes *= scale
                data_sample.ignored_instances.bboxes *= scale

        return inputs, data_samples

    def get_target_size(self, height: int,
                        width: int) -> Tuple[int, int, float]:
        """Get the target size of a batch of images based on data and scale."""
        im_size_min = np.min([height, width])
        im_size_max = np.max([height, width])
        scale = self.min_size / im_size_min
        if scale * im_size_max > self.max_size:
            scale = self.max_size / im_size_max
        target_height, target_width = int(round(height * scale)), int(
            round(width * scale))
        return target_height, target_width, scale

    def get_padded_tensor(self, tensor: Tensor, pad_value: int) -> Tensor:
        """Pad images according to pad_size_divisor."""
        assert tensor.ndim == 4
        target_height, target_width = tensor.shape[-2], tensor.shape[-1]
        divisor = self.pad_size_divisor
        padded_height = (target_height + divisor - 1) // divisor * divisor
        padded_width = (target_width + divisor - 1) // divisor * divisor
        padded_tensor = torch.ones([
            tensor.shape[0], tensor.shape[1], padded_height, padded_width
        ]) * pad_value
        padded_tensor = padded_tensor.type_as(tensor)
        padded_tensor[:, :, :target_height, :target_width] = tensor
        return padded_tensor


@MODELS.register_module()
class BoxInstDataPreprocessor(DetDataPreprocessor):
    """Pseudo mask pre-processor for BoxInst.

    Comparing with the :class:`mmdet.DetDataPreprocessor`,

    1. It generates masks using box annotations.
    2. It computes the images color similarity in LAB color space.

    Args:
        mask_stride (int): The mask output stride in boxinst. Defaults to 4.
        pairwise_size (int): The size of neighborhood for each pixel.
            Defaults to 3.
        pairwise_dilation (int): The dilation of neighborhood for each pixel.
            Defaults to 2.
        pairwise_color_thresh (float): The thresh of image color similarity.
            Defaults to 0.3.
        bottom_pixels_removed (int): The length of removed pixels in bottom.
            It is caused by the annotation error in coco dataset.
            Defaults to 10.
    """

    def __init__(self,
                 *arg,
                 mask_stride: int = 4,
                 pairwise_size: int = 3,
                 pairwise_dilation: int = 2,
                 pairwise_color_thresh: float = 0.3,
                 bottom_pixels_removed: int = 10,
                 **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.mask_stride = mask_stride
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.bottom_pixels_removed = bottom_pixels_removed

        if skimage is None:
            raise RuntimeError('skimage is not installed,\
                 please install it by: pip install scikit-image')

    def get_images_color_similarity(self, inputs: Tensor,
                                    image_masks: Tensor) -> Tensor:
        """Compute the image color similarity in LAB color space."""
        assert inputs.dim() == 4
        assert inputs.size(0) == 1

        unfolded_images = unfold_wo_center(
            inputs,
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)
        diff = inputs[:, :, None] - unfolded_images
        similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

        unfolded_weights = unfold_wo_center(
            image_masks[None, None],
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)
        unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

        return similarity * unfolded_weights

    def forward(self, data: dict, training: bool = False) -> dict:
        """Get pseudo mask labels using color similarity."""
        det_data = super().forward(data, training)
        inputs, data_samples = det_data['inputs'], det_data['data_samples']

        if training:
            # get image masks and remove bottom pixels
            b_img_h, b_img_w = data_samples[0].batch_input_shape
            img_masks = []
            for i in range(inputs.shape[0]):
                img_h, img_w = data_samples[i].img_shape
                img_mask = inputs.new_ones((img_h, img_w))
                pixels_removed = int(self.bottom_pixels_removed *
                                     float(img_h) / float(b_img_h))
                if pixels_removed > 0:
                    img_mask[-pixels_removed:, :] = 0
                pad_w = b_img_w - img_w
                pad_h = b_img_h - img_h
                img_mask = F.pad(img_mask, (0, pad_w, 0, pad_h), 'constant',
                                 0.)
                img_masks.append(img_mask)
            img_masks = torch.stack(img_masks, dim=0)
            start = int(self.mask_stride // 2)
            img_masks = img_masks[:, start::self.mask_stride,
                                  start::self.mask_stride]

            # Get origin rgb image for color similarity
            ori_imgs = inputs * self.std + self.mean
            downsampled_imgs = F.avg_pool2d(
                ori_imgs.float(),
                kernel_size=self.mask_stride,
                stride=self.mask_stride,
                padding=0)

            # Compute color similarity for pseudo mask generation
            for im_i, data_sample in enumerate(data_samples):
                # TODO: Support rgb2lab in mmengine?
                images_lab = skimage.color.rgb2lab(
                    downsampled_imgs[im_i].byte().permute(1, 2,
                                                          0).cpu().numpy())
                images_lab = torch.as_tensor(
                    images_lab, device=ori_imgs.device, dtype=torch.float32)
                images_lab = images_lab.permute(2, 0, 1)[None]
                images_color_similarity = self.get_images_color_similarity(
                    images_lab, img_masks[im_i])
                pairwise_mask = (images_color_similarity >=
                                 self.pairwise_color_thresh).float()

                per_im_bboxes = data_sample.gt_instances.bboxes
                if per_im_bboxes.shape[0] > 0:
                    per_im_masks = []
                    for per_box in per_im_bboxes:
                        mask_full = torch.zeros((b_img_h, b_img_w),
                                                device=self.device).float()
                        mask_full[int(per_box[1]):int(per_box[3] + 1),
                                  int(per_box[0]):int(per_box[2] + 1)] = 1.0
                        per_im_masks.append(mask_full)
                    per_im_masks = torch.stack(per_im_masks, dim=0)
                    pairwise_masks = torch.cat(
                        [pairwise_mask for _ in range(per_im_bboxes.shape[0])],
                        dim=0)
                else:
                    per_im_masks = torch.zeros((0, b_img_h, b_img_w))
                    pairwise_masks = torch.zeros(
                        (0, self.pairwise_size**2 - 1, b_img_h, b_img_w))

                # TODO: Support BitmapMasks with tensor?
                data_sample.gt_instances.masks = BitmapMasks(
                    per_im_masks.cpu().numpy(), b_img_h, b_img_w)
                data_sample.gt_instances.pairwise_masks = pairwise_masks
        return {'inputs': inputs, 'data_samples': data_samples}





@MODELS.register_module()
class YOLOXBatchSyncRandomResize(BatchSyncRandomResize):
    """YOLOX batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def forward(self, inputs: Tensor, data_samples: dict) -> Tensor and dict:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        inputs = inputs.float()
        assert isinstance(data_samples, dict)

        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)

            data_samples['bboxes_labels'][:, 2::2] *= scale_x
            data_samples['bboxes_labels'][:, 3::2] *= scale_y

            if 'keypoints' in data_samples:
                data_samples['keypoints'][..., 0] *= scale_x
                data_samples['keypoints'][..., 1] *= scale_y

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)

        return inputs, data_samples


@MODELS.register_module()
class YOLOv5DetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        if 'keypoints' in data_samples:
            data_samples_output['keypoints'] = data_samples['keypoints']
            data_samples_output['keypoints_visible'] = data_samples[
                'keypoints_visible']

        return {'inputs': inputs, 'data_samples': data_samples_output}


@MODELS.register_module()
class PPYOLOEDetDataPreprocessor(DetDataPreprocessor):
    """Image pre-processor for detection tasks.

    The main difference between PPYOLOEDetDataPreprocessor and
    DetDataPreprocessor is the normalization order. The official
    PPYOLOE resize image first, and then normalize image.
    In DetDataPreprocessor, the order is reversed.

    Note: It must be used together with
    `mmyolo.datasets.utils.yolov5_collate`
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``. This class use batch_augments first, and then
        normalize the image, which is different from the `DetDataPreprocessor`
        .

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        assert isinstance(data['inputs'], list) and is_list_of(
            data['inputs'], torch.Tensor), \
            '"inputs" should be a list of Tensor, but got ' \
            f'{type(data["inputs"])}. The possible reason for this ' \
            'is that you are not using it with ' \
            '"mmyolo.datasets.utils.yolov5_collate". Please refer to ' \
            '"cconfigs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py".'

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # Process data.
        batch_inputs = []
        for _input in inputs:
            # channel transform
            if self._channel_conversion:
                _input = _input[[2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _input = _input.float()
            batch_inputs.append(_input)

        # Batch random resize image.
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(batch_inputs, data_samples)

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }

        return {'inputs': inputs, 'data_samples': data_samples}


# TODO: No generality. Its input data format is different
#  mmdet's batch aug, and it must be compatible in the future.
@MODELS.register_module()
class PPYOLOEBatchRandomResize(BatchSyncRandomResize):
    """PPYOLOE batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
        random_interp (bool): Whether to choose interp_mode randomly.
            If set to True, the type of `interp_mode` must be list.
            If set to False, the type of `interp_mode` must be str.
            Defaults to True.
        interp_mode (Union[List, str]): The modes available for resizing
            are ('nearest', 'bilinear', 'bicubic', 'area').
        keep_ratio (bool): Whether to keep the aspect ratio when resizing
            the image. Now we only support keep_ratio=False.
            Defaults to False.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 1,
                 size_divisor: int = 32,
                 random_interp=True,
                 interp_mode: Union[List[str], str] = [
                     'nearest', 'bilinear', 'bicubic', 'area'
                 ],
                 keep_ratio: bool = False) -> None:
        super().__init__(random_size_range, interval, size_divisor)
        self.random_interp = random_interp
        self.keep_ratio = keep_ratio
        # TODO: need to support keep_ratio==True
        assert not self.keep_ratio, 'We do not yet support keep_ratio=True'

        if self.random_interp:
            assert isinstance(interp_mode, list) and len(interp_mode) > 1,\
                'While random_interp==True, the type of `interp_mode`' \
                ' must be list and len(interp_mode) must large than 1'
            self.interp_mode_list = interp_mode
            self.interp_mode = None
        else:
            assert isinstance(interp_mode, str),\
                'While random_interp==False, the type of ' \
                '`interp_mode` must be str'
            assert interp_mode in ['nearest', 'bilinear', 'bicubic', 'area']
            self.interp_mode_list = None
            self.interp_mode = interp_mode

    def forward(self, inputs: list,
                data_samples: dict) -> Tuple[Tensor, Tensor]:
        """Resize a batch of images and bboxes to shape ``self._input_size``.

        The inputs and data_samples should be list, and
        ``PPYOLOEBatchRandomResize`` must be used with
        ``PPYOLOEDetDataPreprocessor`` and ``yolov5_collate`` with
        ``use_ms_training == True``.
        """
        assert isinstance(inputs, list),\
            'The type of inputs must be list. The possible reason for this ' \
            'is that you are not using it with `PPYOLOEDetDataPreprocessor` ' \
            'and `yolov5_collate` with use_ms_training == True.'

        bboxes_labels = data_samples['bboxes_labels']

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            # get current input size
            self._input_size, interp_mode = self._get_random_size_and_interp()
            if self.random_interp:
                self.interp_mode = interp_mode

        # TODO: need to support type(inputs)==Tensor
        if isinstance(inputs, list):
            outputs = []
            for i in range(len(inputs)):
                _batch_input = inputs[i]
                h, w = _batch_input.shape[-2:]
                scale_y = self._input_size[0] / h
                scale_x = self._input_size[1] / w
                if scale_x != 1. or scale_y != 1.:
                    if self.interp_mode in ('nearest', 'area'):
                        align_corners = None
                    else:
                        align_corners = False
                    _batch_input = F.interpolate(
                        _batch_input.unsqueeze(0),
                        size=self._input_size,
                        mode=self.interp_mode,
                        align_corners=align_corners)

                    # rescale boxes
                    indexes = bboxes_labels[:, 0] == i
                    bboxes_labels[indexes, 2] *= scale_x
                    bboxes_labels[indexes, 3] *= scale_y
                    bboxes_labels[indexes, 4] *= scale_x
                    bboxes_labels[indexes, 5] *= scale_y

                    data_samples['bboxes_labels'] = bboxes_labels
                else:
                    _batch_input = _batch_input.unsqueeze(0)

                outputs.append(_batch_input)

            # convert to Tensor
            return torch.cat(outputs, dim=0), data_samples
        else:
            raise NotImplementedError('Not implemented yet!')

    def _get_random_size_and_interp(self) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and a
        interp_mode in interp_mode_list."""
        size = random.randint(*self._random_size_range)
        input_size = (self._size_divisor * size, self._size_divisor * size)

        if self.random_interp:
            interp_ind = random.randint(0, len(self.interp_mode_list) - 1)
            interp_mode = self.interp_mode_list[interp_ind]
        else:
            interp_mode = None
        return input_size, interp_mode
