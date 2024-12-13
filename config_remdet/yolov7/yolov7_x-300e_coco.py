_base_ = './yolov7_l-300e_visdrone.py'

base_lr = 0.01
optim_wrapper = dict(optimizer=dict(lr=base_lr))
train_batch_size_per_gpu = 8
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
val_batch_size_per_gpu = 8
val_dataloader = dict(batch_size=val_batch_size_per_gpu)
model = dict(
    backbone=dict(arch='X'),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        use_repconv_outs=False),
    bbox_head=dict(head_module=dict(in_channels=[320, 640, 1280])))
