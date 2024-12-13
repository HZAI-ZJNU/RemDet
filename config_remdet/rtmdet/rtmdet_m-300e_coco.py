_base_ = './rtmdet_l-300e_coco.py'

train_batch_size_per_gpu = 12
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
