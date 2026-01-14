_base_ = [
    '../../_base_/models/slowonly_r50.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(pretrained=None))

# dataset settings
dataset_type = 'RawframeDataset'

data_root = '/data/pqh/dataset/US_1343/all'
data_root_val = '/data/pqh/dataset/US_1343/all'
ann_file_train = '/data/pqh/dataset/US_1343/selected_train_file_list.txt' # 阴性数据训练340
ann_file_val = '/data/pqh/dataset/US_1343/val_file_list.txt'
ann_file_test = ann_file_val

file_client_args = dict(io_backend='disk')

train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=4, frame_interval=16, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric',
                     metric_list=('top_k_accuracy', 'f1_score', 'positive_accuracy', 'negative_accuracy', 'precision', 'AUC',
                                  'sensitivity' ))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=256, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=34),
    dict(
        type='CosineAnnealingLR',
        T_max=222,
        eta_min=0,
        by_epoch=True,
        begin=34,
        end=256)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.3, momentum=0.9, weight_decay=1e-4), # 原来lr=0.2
    clip_grad=dict(max_norm=40, norm_type=2))

# runtime settings
default_hooks = dict(checkpoint=dict(interval=4, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
