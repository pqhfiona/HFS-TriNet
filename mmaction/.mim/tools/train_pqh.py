# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys

# 分布式训练GPU设置 - 使用两张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mmengine.config import Config, DictAction
from mmengine.runner import Runner # 用于训练和评估模型
import multiprocessing as mp

import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 设置工作目录，确保能找到 mmaction 模块
new_working_directory = "/data/pqh/env/MM"
# new_working_directory = "/data/pqh/env/MAction/mar_scripts/manet/mmaction2"
os.chdir(new_working_directory)
if new_working_directory not in sys.path:
    sys.path.insert(0, new_working_directory)
print("当前工作目录已更改为: ", os.getcwd())

def parse_args():
    model_index = 0    # 从0开始索引
    config_files = [
    # 'manet.py' # 2024
    'tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
    # 'i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb.py'
    # 'mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.py'
    # 'slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.py'
    # 'slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py'
    # 'swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'
    # 'tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py'
    # 'trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py'
    # 'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
    # 'tpn-slowonly_r50_WIN_8xb8-8x8x1-150e_kinetics400-rgb.py'

    ] # 可以继续添加其他配置文件
    # default_config_path = os.path.join(
    #     os.getenv('MY_DEFAULT_CONFIG', './configs/recognition/my_work'), config_files[model_index])
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
    parser.add_argument('--seed', type=int, default=470272517, help='random seed') # 470272517
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
    parser.add_argument(
        '--robustness_eval',
        action='store_true',
        help='启用鲁棒性评估实验，测试模型在不同speckle噪声强度下的性能')
    parser.add_argument(
        '--robustness_work_dir',
        default='./robustness_eval_results',
        help='鲁棒性评估结果保存目录')
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

    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # 记录模型参数量（总参数与可训练参数）和FLOPs，方便在日志中回溯
    model = runner.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    runner.logger.info(
        f'Model params: total={total_params:,}, trainable={trainable_params:,}')

    # 计算FLOPs（浮点运算数）
    try:
        from mmengine.analysis import get_model_complexity_info
        
        # 从配置中获取输入形状
        # 对于视频模型，输入格式通常是 NCTHW: (batch, channels, time, height, width)
        train_pipeline = cfg.get('train_pipeline', [])
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

    # Add debugging code to print data
    def debug_data(data):
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Data keys: {data.keys()}")
        else:
            print(f"Data content: {data}")

    # start training
    runner.train()

    # 如果启用鲁棒性评估，在训练完成后进行评估
    if args.robustness_eval:
        print("\n=== 开始鲁棒性评估实验 ===")
        try:
            # 导入鲁棒性评估模块
            import robustness_eval
            import sys
            from types import SimpleNamespace

            # 创建鲁棒性评估的参数对象
            robustness_args = SimpleNamespace(
                config=args.config,
                checkpoint=cfg.get('load_from', ''),
                work_dir=args.robustness_work_dir,
                noise_configs=['clean', 'low', 'medium', 'high'],
                repeats=3,
                baseline_types=['full', 'no_wtcr', 'no_sam']
            )

            # 运行鲁棒性评估
            print(f"运行鲁棒性评估，配置: config={args.config}, checkpoint={cfg.get('load_from', '')}")
            robustness_eval.run_robustness_eval(robustness_args)

        except ImportError as e:
            print(f"无法导入鲁棒性评估模块: {e}")
            print("请确保 robustness_eval.py 在同一目录下")
        except Exception as e:
            print(f"鲁棒性评估失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
