# dataset config
dataset_type = 'CocoDataset'
data_root = '/mnt/datasets/UAVDT/'
backend_args = None
classes = ('car', 'truck', 'bus')
# training and testing settings
# train_cfg = dict(
#     assigner=dict(type='ATSSAssigner', topk=9),
#     allowed_border=-1,
#     pos_weight=-1,
#     debug=False)
# test_cfg = dict(
#     nms_pre=1000,
#     min_bbox_size=0,
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),
#     max_per_img=500)
img_scale = (1024, 540)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/UAV-benchmark-M-Train.json',
        data_prefix=dict(img='images/UAV-benchmark-M'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/UAV-benchmark-M-Val.json',
        data_prefix=dict(img='images/UAV-benchmark-M'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/UAV-benchmark-M-Val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator