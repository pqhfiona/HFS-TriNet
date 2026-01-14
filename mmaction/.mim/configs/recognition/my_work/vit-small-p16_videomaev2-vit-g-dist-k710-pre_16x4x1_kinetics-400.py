_base_ = ['../videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py']

# model settings
model = dict(
    backbone=dict(embed_dims=384, depth=12, num_heads=6),
    cls_head=dict(in_channels=384))

dataset_type = 'RawframeDataset'
data_root_val = '/data/pqh/dataset/US_1343/all'
ann_file_test = '/data/pqh/dataset/US_1343/val_file_list.txt'


test_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1, #默认是1
    num_workers=8, #默认是8
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator= dict(type='AccMetric',
                     metric_list=('f1_score','AUC',
                         'top_k_accuracy', 'mean_class_accuracy', 'sensitivity', 'precision', 'positive_accuracy', 'negative_accuracy' ))