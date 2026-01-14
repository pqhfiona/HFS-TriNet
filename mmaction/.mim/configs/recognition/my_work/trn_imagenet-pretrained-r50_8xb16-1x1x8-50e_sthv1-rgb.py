_base_ = ['../trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py']

# model settings
model = dict(cls_head=dict(num_classes=174))

# dataset settings
dataset_type = 'RawframeDataset' # 这里的数据类型本来是VideoDataset，定义为视频帧后记得要改后面的相应内容

data_root = '/data/pqh/dataset/US_1343/all'
data_root_val = '/data/pqh/dataset/US_1343/all'
ann_file_train = '/data/pqh/dataset/US_1343/selected_train_file_list.txt'
ann_file_val = '/data/pqh/dataset/US_1343/val_file_list.txt'
ann_file_test = '/data/pqh/dataset/US_1343/val_file_list.txt'

file_client_args = dict(io_backend='disk')

sthv1_flip_label_map = {2: 4, 4: 2, 30: 41, 41: 30, 52: 66, 66: 52}
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv1_flip_label_map),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
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
        filename_tmpl='img_{:05}.jpg',  # 在这个之前增加了img_
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
        filename_tmpl='img_{:05}.jpg',  # 在这个之前增加了img_
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
        filename_tmpl='img_{:05}.jpg',  # 在这个之前增加了img_
        pipeline=test_pipeline,
        test_mode=True))
