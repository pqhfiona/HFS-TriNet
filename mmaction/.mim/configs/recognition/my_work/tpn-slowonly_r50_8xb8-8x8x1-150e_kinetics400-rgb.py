
_base_ = [
    '../tpn/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
]

dataset_type = 'RawframeDataset' # 这里的数据类型本来是VideoDataset，定义为视频帧后记得要改后面的相应内容

data_root = '/data/pqh/dataset/US_1343/all'
data_root_val = '/data/pqh/dataset/US_1343/all'
# ann_file_train = '/data/pqh/env/MM/69_timeline_nodiffu/tempt.txt'
ann_file_train = '/data/pqh/dataset/US_1343/train_file_list_p340.txt' # 阴性数据训练340，HFS-TriNet
# ann_file_train = '/data/pqh/dataset/US_1343/train_file_list_p369.txt' # 阴性数据训练340
# ann_file_train = '/data_nas/pqh/dataset/US_+340_1/train_file_list.txt' # 阳性数据训练680
# ann_file_train = '/data/pqh/dataset/US_1343/train_file_list_12.txt' # 只有随机的12个训练数据
ann_file_val = '/data/pqh/dataset/US_1343/val_file_list.txt' # HFS-TriNet
# ann_file_val = '/data/pqh/dataset/US_1343/external_97.txt'
# ann_file_val = '/data/pqh/dataset/US_1343/external_p27_n25.txt'
ann_file_test = ann_file_val

train_pipeline = [
    # dict(type='DecordInit'), # 因为是处理的视频的相关数据
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),  # mmaction/datasets/transforms/loading_Tri.py
    # dict(type='DecordDecode'), # 因为是处理的视频的相关数据
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # mmaction/datasets/transforms/processing_Tri.py
    dict(type='Flip', flip_ratio=0.5),  # 图像旋转，数据增广。NPZ的维度不同
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),  # mmaction/datasets/transforms/formatting_Tri.py
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=10,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'), #
    dict(type='RawFrameDecode', io_backend='disk'),  # 需要单独加这一行，用来对应前面定义的数据类型是视频帧的
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=6,
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
    batch_size=8, #默认是8
    num_workers=8,#默认是8
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),  # 这里原本是video，这里现在改成img
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, #默认是8,30
    num_workers=8, #默认是8
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
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

val_evaluator = dict(type='AccMetric',
                     metric_list=('accuracy', 'f1_score','AUC',
                           'sensitivity', 'precision', 'positive_accuracy', 'negative_accuracy',
                         'save_predictions'))
test_evaluator = val_evaluator

# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=1) # max_epochs=150

#训练十次，测试一次
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=10) # max_epochs=150
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')