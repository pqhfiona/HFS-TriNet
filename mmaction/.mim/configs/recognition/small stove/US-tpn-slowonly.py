_base_ = [
    '../../../_base_/models/tpn_slowonly_r50_us.py',
    '../../../_base_/default_runtime.py'
]


def adjust_learning_rate(original_lr, original_batch_size, new_batch_size):
    """
    根据线性缩放规则调整学习率。

    参数:
    - original_lr (float): 原始学习率。
    - original_batch_size (int): 原始批量大小。
    - new_batch_size (int): 新批量大小。

    返回:
    - new_lr (float): 调整后的新学习率。
    """
    # 计算批量大小的比例变化
    scale_factor = new_batch_size / original_batch_size

    # 调整学习率
    new_lr = original_lr * scale_factor

    return new_lr


model = dict(
    # type='Recognizer3D',
    # type='Recognizer3D_multimodal'
    # type='Recognizer3D_multimodal_SFT'
    # type='Recognizer3D_multimodal_2backbone_SFT',
    type='Recognizer3D_multimodal_1backbone_SFT',

    # backbone2=dict(
    #         type='ResNet3dSlowOnly',
    #         depth=50,
    #         pretrained='torchvision://resnet50',
    #         lateral=False,
    #         out_indices=(2, 3),
    #         conv1_kernel=(1, 7, 7),
    #         conv1_stride_t=1,
    #         pool1_stride_t=1,
    #         inflate=(0, 0, 1, 1),
    #         norm_eval=False)
)

# dataset settings 
dataset_type = 'RawframeDatasetRatio'
# dataset_type = 'RawframeDatasetModi'
# dataset_type = 'RawframeDataset'
ratio = 5  # 0 5 10
num = 640  # 1343 640
batch_size = 12
# data_root = './data/US_multimodal_notime_449/train'
# data_root_val = './data/US_multimodal_notime_449/val'
# ann_file_train = f'./data/US_multimodal_notime_449/train449_n269p180_ratio{ratio}_list.txt'
# ann_file_val = f'./data/US_multimodal_notime_449/val191_n166p75_ratio{ratio}_list.txt'
# ann_file_test = ann_file_val

data_root = './data/US_multimodal_notime_1343/all'
data_root_val = './data/US_multimodal_notime_1343/all'
ann_file_train = f'./data/US_multimodal_notime_1343/train_{num}file_ratio{ratio}_list.txt'
ann_file_val = f'./data/US_multimodal_notime_1343/val_{num}file_ratio{ratio}_list.txt'
ann_file_test = ann_file_val

train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),

    dict(type='RawFrameDecode', io_backend='disk'),
    # dict(type='DecordDecode'),
    # dict(type='RandomResizedCrop'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),

    # dict(type='Flip', flip_ratio=0.5),
    # dict(type='ColorJitter'),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),

    # dict(type='ColorJitter'),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='ThreeCrop', crop_size=256),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='img_{:05}.jpg',
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
        filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric',
                     metric_list=('f1_score', 'top_k_accuracy', 'positive_accuracy', 'negative_accuracy', 'sensitivity',
                                  'precision', 'AUC'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

###############################################################
original_lr = 0.01  # 原始tpn学习率为0.01
original_batch_size = 16  # 原始配置下的总批量大小（例如，8 GPUs x 2 videos/GPU）
# 计算新的学习率
new_lr = adjust_learning_rate(original_lr, original_batch_size, batch_size)
print("新的学习率应该设置为:", new_lr)

optim_wrapper = dict(
    # constructor='TSMOptimWrapperConstructor',
    # paramwise_cfg=dict(fc_lr5=True),

    optimizer=dict(
        type='SGD', lr=new_lr, momentum=0.9, weight_decay=0.0001, nesterov=True),
    # type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True),

    # optimizer = dict(
    #     type='Adam', lr=0.01, weight_decay=0.00001) , # this lr is used for 1 gpus

    clip_grad=dict(max_norm=40, norm_type=2),
)
randomness = dict(deterministic=False, diff_rank_seed=False, seed=470272517)

param_scheduler = [
    # Warmup
    # dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=20),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[40, 80],
        gamma=0.1)
]
###############################################################

default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))
work_dir = f"./work_dirs/test_modified_model/"

print(work_dir)
