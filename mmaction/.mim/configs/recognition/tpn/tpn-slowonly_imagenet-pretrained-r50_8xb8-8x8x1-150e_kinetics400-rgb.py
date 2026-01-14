_base_ = [
    '../../_base_/models/tpn_slowonly_r50.py',
    '../../_base_/default_runtime.py'
]

dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=10) # max_epochs=150
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True), # 默认情况下0.01
    clip_grad=dict(max_norm=40, norm_type=2), # 默认
)

default_hooks = dict(checkpoint=dict(max_keep_ckpts=5)) # 默认5

# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW',  # 使用 AdamW 优化器
#         lr=0.01,       # 学习率 默认0.01
#         betas=(0.9, 0.999),  # 一阶和二阶矩估计的衰减系数
#         eps=1e-8,      # 防止除零错误的常数
#         weight_decay=1e-2,  # 权重衰减（L2正则化），默认1e-2
#         amsgrad=False,  # 是否使用 AMSGrad
#     ),
#     clip_grad=dict(
#         max_norm=40,  # 梯度裁剪的最大范数,默认40
#         norm_type=2   # 使用L2范数进行裁剪
#     ),
# )
#
#
# param_scheduler = [
#     dict(
#         type='LinearLR',  # 学习率预热
#         start_factor=0.1,  # 初始学习率为 lr * 0.1
#         end_factor=1.0,  # 逐步增加到 lr * 1.0
#         begin=0,  # 从第0个epoch开始
#         end=30,  # 到第10个epoch结束
#         by_epoch=True),
#     dict(
#         type='MultiStepLR',
#         begin=0, # 默认设置是0
#         end=150,
#         by_epoch=True,
#         milestones=[40, 125], # 默认[75, 125]
#         gamma=0.1)
# ]

