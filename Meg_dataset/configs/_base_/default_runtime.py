checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,   # 多少批次 打印一次
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

seed = 0
deterministic = False
load_from = False
# load_from      = '/home/data/BEV_Detection/bevfusion-main_v1/pretrained/2022-11-03-test/epoch_16.pth'
resume_from = False
cudnn_benchmark = False
distributed = True
dist_params = dict(backend='nccl')