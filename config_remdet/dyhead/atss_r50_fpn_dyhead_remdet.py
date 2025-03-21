_base_ = [
    '../datasets/visdrone2019.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]

classes = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
# train_dataloader = dict(batch_size=4)
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config  for ddp
# val_dataloader = dict(batch_size=8)
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone=dict(
        type='RemNet',
        arch='P5',
        last_stage_out_channels=1024,
        deepen_factor=1.0,
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        out_indices=[1, 2, 3, 4],
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(type='Pretrained',
                      checkpoint='/mnt/workspace/RemDet-mmdet/output_dir/pretrain_weights/epoch_300.pth',
                      prefix='backbone.')),
    neck=[
        dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
