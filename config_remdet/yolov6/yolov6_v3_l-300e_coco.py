_base_ = './yolov6_v3_m-300e_coco.py'

base_lr = 0.01
optim_wrapper = dict(optimizer=dict(lr=base_lr))
train_batch_size_per_gpu = 8
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
val_batch_size_per_gpu = 8
val_dataloader = dict(batch_size=val_batch_size_per_gpu)
# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1
# The scaling factor that controls the width of the network structure
widen_factor = 1

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        block_act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
