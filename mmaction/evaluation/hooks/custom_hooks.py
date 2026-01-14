from mmengine.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmengine.runner import Runner
from ..metrics.acc_metric import AccMetric # 确保路径正确，这是为了类型检查

@HOOKS.register_module()
class PredictionSaveHook(Hook):
    """
    这个 Hook 在每次验证循环开始前运行。
    它的作用是找到 AccMetric，并将当前的 epoch 和 work_dir 注入进去。
    """
    def before_val(self, runner: Runner) -> None:
        # 从 runner 中获取 evaluator
        evaluator = runner.val_loop.evaluator
        # 遍历 evaluator 中的所有 metric
        for metric in evaluator.metrics:
            # 检查这个 metric是不是我们想要操作的 AccMetric
            if isinstance(metric, AccMetric):
                # 将 runner 的当前 epoch 和 work_dir 设置为 AccMetric 的属性
                # 这样 AccMetric 内部就可以直接使用了
                metric.current_epoch = runner.epoch
                metric.work_dir = runner.work_dir
                print(f"Hook 已运行：为 AccMetric 注入 epoch: {runner.epoch}")