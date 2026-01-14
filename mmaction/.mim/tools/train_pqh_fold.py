# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from mmengine.config import Config, DictAction
from mmengine.runner import Runner  # 用于训练和评估模型
import multiprocessing as mp

import torch, gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

new_working_directory = "/data/pqh/env/MM"
os.chdir(new_working_directory)
print("当前工作目录已更改为: ", os.getcwd())

def parse_args():
    model_index = 0  # 从0开始索引
    config_files = [
        # 'tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
        'tpn_kfold.py'
        # 'timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
        # 'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
    ]  # 可以继续添加其他配置文件
    default_config_path = os.path.join(
        os.getenv('MY_DEFAULT_CONFIG', './configs/recognition/my_work'), config_files[model_index])

    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument('config', nargs='?', default=default_config_path, help='train config file path')

    parser.add_argument('--work-dir',
                       help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
             'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=470272517, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                               osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    # 合并CLI参数到配置
    cfg = merge_args(cfg, args)

    # 确保 cfg.work_dir 存在
    if not hasattr(cfg, 'work_dir'):
        cfg.work_dir = './work_dirs'  # 设置默认值

    # 在 cfg.work_dir 下创建子目录
    kfold_results_dir = osp.join(cfg.work_dir, 'kfold_results')
    os.makedirs(kfold_results_dir, exist_ok=True)

    # 五折交叉验证的划分结果目录
    kfold_splits_dir = '/data/pqh/dataset/US_1343/kfold_splits'

    # 进行五折交叉验证训练
    for fold in range(1, 6):  # 遍历每一折
        print(f"Fold {fold}")

        # 加载当前折的训练集和验证集文件
        train_ann_file = osp.join(kfold_splits_dir, f'train_fold_{fold}.txt')
        val_ann_file = osp.join(kfold_splits_dir, f'val_fold_{fold}.txt')

        # 动态修改配置
        cfg.train_dataloader.dataset.ann_file = train_ann_file
        cfg.val_dataloader.dataset.ann_file = val_ann_file

        # 设置当前折的工作目录
        fold_work_dir = osp.join(kfold_results_dir, f'fold_{fold}')
        cfg.work_dir = fold_work_dir

        # 创建当前折的工作目录
        os.makedirs(fold_work_dir, exist_ok=True)

        # 初始化Runner并开始训练
        runner = Runner.from_cfg(cfg)
        runner.train()

        print(f"Fold {fold} 训练完成，结果保存在: {fold_work_dir}")


if __name__ == '__main__':
    main()