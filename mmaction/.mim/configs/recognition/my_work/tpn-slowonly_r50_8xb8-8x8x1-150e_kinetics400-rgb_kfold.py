import os
import os.path as osp

_base_ = [
    '../tpn/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
]

dataset_type = 'RawframeDataset'

data_root = os.getenv('TPN_DATA_ROOT', '/data/pqh/dataset/US_1343/all')
data_root_val = data_root
kfold_root = os.getenv('TPN_KFOLD_ROOT', '/data/pqh/env/MM/dataset/1083_easy/kfold')
num_folds = int(os.getenv('TPN_NUM_FOLDS', 5))
fold = int(os.getenv('TPN_FOLD', 0))

train_ann_file_template = osp.join(kfold_root, 'train_fold{fold}.txt')
val_ann_file_template = osp.join(kfold_root, 'val_fold{fold}.txt')

ann_file_train = train_ann_file_template.format(fold=fold)
ann_file_val = val_ann_file_template.format(fold=fold)
# 按你的设定，test 与 val 使用完全相同的文件
ann_file_test = ann_file_val

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

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
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
    batch_size=8,
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

val_evaluator = dict(
    type='AccMetric',
    metric_list=('accuracy', 'f1_score', 'AUC', 'sensitivity', 'precision',
                 'positive_accuracy', 'negative_accuracy', 'save_predictions'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 记录模板路径，供K折脚本读取
kfold_meta = dict(
    root=kfold_root,
    num_folds=num_folds,
    train_template=train_ann_file_template,
    val_template=val_ann_file_template,
    test_template=val_ann_file_template)  # test 和 val 使用相同的文件

