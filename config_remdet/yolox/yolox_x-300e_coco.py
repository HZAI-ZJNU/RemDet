_base_ = './yolox_s-300e_visdrone.py'

train_batch_size_per_gpu = 8
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
# ========================modified parameters======================
deepen_factor = 1.33
widen_factor = 1.25

# =======================Unmodified in most cases==================
# model settings
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
