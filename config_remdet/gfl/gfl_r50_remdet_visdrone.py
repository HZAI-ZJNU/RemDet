_base_ = [
    '../datasets/visdrone2019.py',
    '../_base_/schedule_2x.py', '../_base_/default_runtime.py'
]
classes = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config  for ddp
model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='RemNet',
        arch='pretrain',
        last_stage_out_channels=1024,
        deepen_factor=1.0,
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        out_indices=[1, 2, 3, 4],
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(type='Pretrained',
                      checkpoint='/mnt/workspace/RemDet-mmdet/output_dir/pretrain_weights/epoch_300.pth',
                      prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
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
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))