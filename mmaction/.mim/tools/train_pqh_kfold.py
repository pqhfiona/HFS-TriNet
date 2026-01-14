import argparse
import os
import os.path as osp
import random
import sys

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置工作目录，确保能找到 mmaction 模块
new_working_directory = "/data/pqh/env/MM"
os.chdir(new_working_directory)
if new_working_directory not in sys.path:
    sys.path.insert(0, new_working_directory)
print("当前工作目录已更改为: ", os.getcwd())
def parse_args():
    parser = argparse.ArgumentParser(description='K折训练动作识别模型')
    parser.add_argument(
        'config',
        nargs='?',
        default='./configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_kfold.py',
        help='配置文件路径')
    parser.add_argument('--work-dir', help='日志与模型的根目录，程序会在其下创建fold子目录')
    parser.add_argument('--resume', nargs='?', type=str, const='auto')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--seed', type=int, default=470272517)
    parser.add_argument('--diff-rank-seed', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--train-target-size',
        type=int,
        default=866,
        help='平衡抽样后训练集的期望总量，默认866（适配1083样本的5折）')
    parser.add_argument(
        '--balance-train',
        dest='balance_train',
        action='store_true',
        help='开启训练集1:1阳性/阴性平衡抽样')
    parser.add_argument(
        '--no-balance-train',
        dest='balance_train',
        action='store_false',
        help='关闭训练集平衡抽样')
    parser.set_defaults(balance_train=True)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='以key=value形式覆盖配置')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--num-folds', type=int, default=None, help='覆盖配置中的折数')
    parser.add_argument(
        '--fold-indices',
        type=int,
        nargs='+',
        help='仅运行指定折，例如: --fold-indices 0 2 4')
    return parser.parse_args()


def merge_args(cfg, args):
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper']
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def _load_ann_lines(txt_path):
    lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f'标注文件 {txt_path} 中存在格式异常的行: {raw}')
            lines.append(line)
    return lines


def balance_train_split(train_file, seed, target_total=866):
    """在不拆分病人的前提下，将训练集按1:1的正负比例下采样并打乱."""
    lines = _load_ann_lines(train_file)
    pos = [ln for ln in lines if int(ln.split()[2]) == 1]
    neg = [ln for ln in lines if int(ln.split()[2]) == 0]

    target_per_class = min(target_total // 2, len(pos), len(neg))
    if target_per_class == 0:
        raise ValueError(
            f'无法平衡训练集，阳性({len(pos)})或阴性({len(neg)})数量为0: {train_file}')

    rng = random.Random(seed)
    pos_sel = rng.sample(pos, target_per_class) if len(pos) > target_per_class else pos
    neg_sel = rng.sample(neg, target_per_class) if len(neg) > target_per_class else neg

    merged = pos_sel + neg_sel
    rng.shuffle(merged)

    base, ext = osp.splitext(train_file)
    balanced_path = f'{base}_balanced{ext or ".txt"}'
    with open(balanced_path, 'w', encoding='utf-8') as f:
        for line in merged:
            f.write(line + '\n')

    print(
        f'平衡训练集: 原始 {len(lines)} 条 -> '
        f'正 {len(pos_sel)}, 负 {len(neg_sel)}, 总 {len(merged)} | 保存 {balanced_path}'
    )
    return balanced_path


def apply_fold_paths(cfg, fold_idx):
    if 'kfold_meta' not in cfg:
        raise ValueError('配置中缺少kfold_meta，无法确定划分文件')
    meta = cfg.kfold_meta

    def format_path(template_key):
        template = meta.get(template_key)
        if template is None:
            return None
        return template.format(fold=fold_idx)

    train_file = format_path('train_template')
    val_file = format_path('val_template')
    # 如果没有单独的 test_template，就直接复用 val_file
    test_file = format_path('test_template') or val_file

    cfg.fold = fold_idx
    cfg.ann_file_train = train_file
    cfg.ann_file_val = val_file
    cfg.ann_file_test = test_file

    cfg.train_dataloader.dataset.ann_file = train_file
    if cfg.val_dataloader is not None:
        cfg.val_dataloader.dataset.ann_file = val_file
    if cfg.test_dataloader is not None:
        cfg.test_dataloader.dataset.ann_file = test_file


def main():
    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    kfold_meta = cfg.get('kfold_meta', None)
    if kfold_meta is None:
        raise ValueError('配置文件缺少kfold_meta字段，无法执行K折训练')

    total_folds = args.num_folds or kfold_meta.get('num_folds', 5)
    if args.fold_indices:
        fold_indices = args.fold_indices
    else:
        fold_indices = list(range(total_folds))

    base_work_dir = cfg.work_dir

    for fold in fold_indices:
        fold_cfg = cfg.copy()
        apply_fold_paths(fold_cfg, fold)
        fold_cfg.work_dir = osp.join(base_work_dir, f'fold_{fold}')

        if args.balance_train:
            balanced_train = balance_train_split(
                fold_cfg.ann_file_train,
                seed=args.seed + fold,  # 每折独立随机序列
                target_total=args.train_target_size)
            fold_cfg.ann_file_train = balanced_train
            fold_cfg.train_dataloader.dataset.ann_file = balanced_train

        print(f'开始训练 fold {fold}, 数据: '
              f"{fold_cfg.train_dataloader.dataset.ann_file}")

        runner = Runner.from_cfg(fold_cfg)

        # 记录模型参数量（总参数数与可训练参数数）和FLOPs，写入日志便于回溯
        model = runner.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        runner.logger.info(
            f'Model params: total={total_params:,}, trainable={trainable_params:,}'
        )

        # 计算FLOPs（浮点运算数）
        try:
            from mmengine.analysis import get_model_complexity_info
            
            # 从配置中获取输入形状
            # 对于视频模型，输入格式通常是 NCTHW: (batch, channels, time, height, width)
            train_pipeline = fold_cfg.get('train_pipeline', [])
            clip_len = 8  # 默认值
            resolution = 224  # 默认值
            
            for item in train_pipeline:
                if item.get('type') == 'SampleFrames':
                    clip_len = item.get('clip_len', 8)
                elif item.get('type') == 'Resize':
                    scale = item.get('scale', (224, 224))
                    if isinstance(scale, (list, tuple)) and len(scale) == 2:
                        resolution = scale[1] if scale[0] == -1 else scale[0]
                    elif isinstance(scale, int):
                        resolution = scale
            
            # 视频模型输入形状: (batch_size, channels, time, height, width)
            input_shape = (1, 3, clip_len, resolution, resolution)
            
            # 保存原始状态
            original_forward = model.forward
            was_training = model.training
            
            # 尝试使用 extract_feat 方法（适用于大多数识别模型）
            if hasattr(model, 'extract_feat'):
                # 创建一个包装函数，只提取特征，用于FLOPs计算
                # 对于TPN等需要两个输入的模型，只计算backbone部分以避免形状不匹配
                def extract_feat_wrapper(inputs, data_samples=None, **kwargs):
                    # 移除可能冲突的参数
                    kwargs.pop('mode', None)
                    kwargs.pop('stage', None)
                    # 对于复杂模型，只计算backbone以避免neck的形状不匹配问题
                    try:
                        return model.extract_feat(inputs, data_samples=data_samples, stage='backbone')
                    except Exception:
                        # 如果backbone也失败，尝试neck（可能会失败，但至少尝试了）
                        return model.extract_feat(inputs, data_samples=data_samples, stage='neck')
                
                model.forward = extract_feat_wrapper
            elif hasattr(model, '_forward'):
                # 如果没有 extract_feat，尝试使用 _forward
                def forward_wrapper(inputs, data_samples=None, **kwargs):
                    kwargs.pop('mode', None)
                    # 对于复杂模型，只计算backbone
                    try:
                        return model._forward(inputs, data_samples=data_samples, stage='backbone')
                    except Exception:
                        return model._forward(inputs, data_samples=data_samples, stage='neck')
                model.forward = forward_wrapper
            else:
                # 恢复原始状态后再抛出异常
                model.forward = original_forward
                model.train(was_training)
                raise NotImplementedError(
                    'Model does not have extract_feat or _forward method for FLOPs calculation')
            
            try:
                model.eval()
                analysis_results = get_model_complexity_info(model, input_shape)
            finally:
                # 确保无论成功还是失败都恢复原始状态
                model.forward = original_forward
                model.train(was_training)
            
            flops_str = analysis_results['flops_str']
            # 提取数值（如果格式是 "XX.XX GFLOPs" 或 "XX.XX MFLOPs"）
            flops_value = analysis_results.get('flops', 0)
            if flops_value > 0:
                gflops = flops_value / 1e9
                runner.logger.info(
                    f'Model FLOPs: {flops_str} ({gflops:.2f} GFLOPs) - '
                    'Note: For TPN models, this may only include backbone, not neck and head')
            else:
                runner.logger.info(f'Model FLOPs: {flops_str}')
                
        except ImportError:
            runner.logger.warning(
                'mmengine.analysis.get_model_complexity_info not available, '
                'skipping FLOPs calculation')
        except NotImplementedError as e:
            runner.logger.warning(
                f'FLOPs calculation not supported for this model: {e}')
        except Exception as e:
            runner.logger.warning(
                f'Failed to calculate FLOPs: {str(e)[:200]}, skipping FLOPs calculation. '
                'This is normal for some complex models like TPN.')

        runner.train()

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()

