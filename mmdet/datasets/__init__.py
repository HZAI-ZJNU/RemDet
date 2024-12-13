# Copyright (c) OpenMMLab. All rights reserved.
from .ade20k import (ADE20KInstanceDataset, ADE20KPanopticDataset,
                     ADE20KSegDataset)
from .base_det_dataset import BaseDetDataset
from .base_semseg_dataset import BaseSegDataset
from .base_video_dataset import BaseVideoDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_caption import CocoCaptionDataset
from .coco_panoptic import CocoPanopticDataset
from .coco_semantic import CocoSegDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import ConcatDataset, MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .dod import DODDataset
from .dsdl import DSDLDetDataset
from .flickr30k import Flickr30kDataset
from .isaid import iSAIDDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .mdetr_style_refcoco import MDETRStyleRefCocoDataset
from .mot_challenge_dataset import MOTChallengeDataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .odvg import ODVGDataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .refcoco import RefCocoDataset
from .reid_dataset import ReIDDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       CustomSampleSizeSampler, GroupMultiSourceSampler,
                       MultiSourceSampler, TrackAspectRatioBatchSampler,
                       TrackImgSampler)
from .utils import get_loading_pipeline, BatchShapePolicy, yolov5_collate
from .v3det import V3DetDataset
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .yolov5_coco import YOLOv5CocoDataset
from .youtube_vis_dataset import YouTubeVISDataset
from .yolov5_dota import YOLOv5DOTADataset
from .yolov5_voc import YOLOv5VOCDataset

__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset', 'DSDLDetDataset',
    'BaseVideoDataset', 'MOTChallengeDataset', 'TrackImgSampler',
    'ReIDDataset', 'YouTubeVISDataset', 'TrackAspectRatioBatchSampler',
    'ADE20KPanopticDataset', 'CocoCaptionDataset', 'RefCocoDataset',
    'BaseSegDataset', 'ADE20KSegDataset', 'CocoSegDataset',
    'ADE20KInstanceDataset', 'iSAIDDataset', 'V3DetDataset', 'ConcatDataset',
    'ODVGDataset', 'MDETRStyleRefCocoDataset', 'DODDataset',
    'CustomSampleSizeSampler', 'Flickr30kDataset', "YOLOv5CocoDataset", 'YOLOv5VOCDataset',
    'YOLOv5DOTADataset', "BatchShapePolicy", "yolov5_collate"
]
