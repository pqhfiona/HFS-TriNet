# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
sys.path.append("/data/pqh/env/MM/mmaction2")
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config',
                        default='/data/pqh/env/MM/work_dirs/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/20250305_095247/vis_data/config.py',
                        help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
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
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
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
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()

####################################################################################################################################
    outputs = runner.test()
    dataset = runner.test_dataloader.dataset

    print("📦 Dataset 类型：", type(dataset))
    print("📦 Dataset 的属性列表：", dir(dataset))
    video_infos = dataset.data_list

    print(f"video_infos 长度: {len(video_infos)}")

    video_names = [osp.basename(info['frame_dir']) for info in video_infos]

    from collections import defaultdict
    import numpy as np

    results = list(zip(video_names, outputs))

    patient_scores = defaultdict(list)
    for video_name, score in results:
        patient_id = video_name[:4]
        patient_scores[patient_id].append(score)

    fused_scores = {}
    for patient_id, score_list in patient_scores.items():
        score_array = np.stack(score_list, axis=0)
        fused_score = np.mean(score_array, axis=0)
        fused_scores[patient_id] = fused_score

    # 输出病人级结果
    print("\n=== 病人级预测结果 ===")
    print(f"\n共有 {len(fused_scores)} 位病人")
    for pid, score in fused_scores.items():
        # 若是二分类，score 为 float，表示正类概率
        if isinstance(score, (float, np.floating)):
            pred = int(score >= 0.5)
            print(f'病人ID: {pid}, 分数: {score:.4f}, 预测标签: {pred}')
        else:
            # 多分类，score 是 array
            pred = int(np.argmax(score))
            print(f'病人ID: {pid}, 分数: {score}, 预测标签: {pred}')

if __name__ == '__main__':
    main()