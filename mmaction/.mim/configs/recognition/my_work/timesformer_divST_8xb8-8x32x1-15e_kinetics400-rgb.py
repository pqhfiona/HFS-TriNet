_base_ = '../timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py'

model = dict(backbone=dict(attention_type='divided_space_time'))

dataset_type = 'RawframeDataset'

data_root = './data/US_1343/all'
data_root_val = './data/US_1343/all'
ann_file_train = './data/US_1343/train_file_list.txt'
ann_file_val = './data/US_1343/val_file_list.txt'
ann_file_test = ann_file_val

train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1), # mmaction/datasets/transforms/loading.py
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='RandomRescale', scale_range=(256, 320)), # mmaction/datasets/transforms/processing.py
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),  # mmaction/datasets/transforms/formatting.py
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
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
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    # batch_size=8,
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    # batch_size=8,
    batch_size=4,
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
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric',
                     metric_list=('top_k_accuracy', 'f1_score', 'positive_accuracy', 'negative_accuracy', 'precision', 'AUC',
                                  'sensitivity' ))
test_evaluator = val_evaluator