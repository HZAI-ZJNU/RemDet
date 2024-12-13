_base_ = './yolov6_s-300e_visdrone.py'

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.25

# -----train val related-----
lr_factor = 0.02  # Learning rate scaling factor
train_batch_size_per_gpu = 64
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor),
        loss_bbox=dict(iou_mode='siou')))

default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))
