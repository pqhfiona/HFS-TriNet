_base_ = [
    '../tpn/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
]

# 使用标准BN的稳定模型配置
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',  # 使用标准ResNet3dSlowOnly，不使用WIN
        depth=50,
        pretrained='torchvision://resnet50',
        lateral=False,
        out_indices=(2, 3),
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    neck=dict(
        type='TPN',  # 使用标准TPN，不使用WIN版本
        in_channels=(1024, 2048),
        out_channels=1024,
        spatial_modulation_cfg=dict(
            in_channels=(1024, 2048), out_channels=2048),
        temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024, 1024, 1024, 1024),
            mid_channels=(1024, 1024, 1024, 1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=2, loss_weight=0.5)),  # 降低aux loss权重
    cls_head=dict(
        type='TPNHead',
        num_classes=2,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=dict(fcn_test=True))

dataset_type = 'RawframeDataset'
data_root = '/data/pqh/dataset/US_1343/all'
data_root_val = '/data/pqh/dataset/US_1343/all'
ann_file_train = '/data/pqh/dataset/US_1343/train_file_list_12.txt'
ann_file_val = '/data/pqh/dataset/US_1343/external_p27_n25.txt'
ann_file_test = ann_file_val

# 数据管道配置
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# 数据加载器配置 - 保守设置
train_dataloader = dict(
    batch_size=1,  # 每个GPU的批次大小
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

# 评估器配置
val_evaluator = dict(type='AccMetric',
                     metric_list=('accuracy', 'f1_score','AUC',
                           'sensitivity', 'precision', 'positive_accuracy', 'negative_accuracy',
                         'save_predictions'))
test_evaluator = val_evaluator

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 保守的优化器配置
optim_wrapper = dict(
    type='OptimWrapper',  # 不使用AMP，避免数值不稳定
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),  # 更低的学习率
    clip_grad=dict(max_norm=0.5, norm_type=2))  # 更严格的梯度裁剪

# 学习率调度器 - 保守的预热策略
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,  # 从更小的学习率开始
        end_factor=1.0,
        begin=0,
        end=10,  # 更长的预热期
        by_epoch=True),
    dict(
        type='MultiStepLR',
        begin=10,
        end=150,
        by_epoch=True,
        milestones=[75, 125],
        gamma=0.1)
]

# 默认运行时配置
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'))

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 日志配置
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=vis_backends, name='visualizer')

# 随机种子
randomness = dict(seed=0)
