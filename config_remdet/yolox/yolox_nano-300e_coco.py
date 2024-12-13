_base_ = './yolox_s-300e_visdrone.py'

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.25
use_depthwise = True

# =======================Unmodified in most cases==================
# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        use_depthwise=use_depthwise),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        use_depthwise=use_depthwise),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor, use_depthwise=use_depthwise)))
