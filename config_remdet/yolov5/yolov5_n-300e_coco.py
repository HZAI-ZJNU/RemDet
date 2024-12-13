_base_ = './yolov5_s-300e_visdrone.py'

train_batch_size_per_gpu = 32
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
