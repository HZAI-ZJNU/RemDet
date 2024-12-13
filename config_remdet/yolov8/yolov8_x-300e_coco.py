_base_ = './yolov8_m-300e_coco.py'

base_lr = 0.01
optim_wrapper = dict(optimizer=dict(lr=base_lr))
train_batch_size_per_gpu = 8
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
val_batch_size_per_gpu = 8
val_dataloader = dict(batch_size=val_batch_size_per_gpu)
deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
