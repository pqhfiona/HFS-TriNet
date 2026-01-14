"""
2025.2.28 对日志中的grad_norm、loss、top1_acc、loss_cls、loss_aux进行可视化，并生成csv文件，可以给AI读取并理解
"""
import re
import matplotlib.pyplot as plt
import pandas as pd
import os


def parse_log_file(log_file_path):
    # 正则表达式匹配日志中的关键信息
    pattern = re.compile(
        r"Epoch\(train\)\s+\[\d+\]\[\s*\d+/\d+\].*grad_norm: ([\d.]+).*loss: ([\d.]+).*top1_acc: ([\d.]+).*loss_cls: ([\d.]+).*loss_aux: ([\d.]+)"
    )

    data = {
        "grad_norm": [],
        "loss": [],
        "top1_acc": [],
        "loss_cls": [],
        "loss_aux": [],
    }

    with open(log_file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                data["grad_norm"].append(float(match.group(1)))
                data["loss"].append(float(match.group(2)))
                data["top1_acc"].append(float(match.group(3)))
                data["loss_cls"].append(float(match.group(4)))
                data["loss_aux"].append(float(match.group(5)))

    return data


def plot_metrics(data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制 grad_norm 曲线
    plt.figure()
    plt.plot(data["grad_norm"], label="Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm over Training Steps")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "grad_norm.png"))
    plt.close()

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(data["loss"], label="Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Total Loss over Training Steps")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # 绘制 top1_acc 曲线
    plt.figure()
    plt.plot(data["top1_acc"], label="Top-1 Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Top-1 Accuracy over Training Steps")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "top1_acc.png"))
    plt.close()

    # 绘制 loss_cls 曲线
    plt.figure()
    plt.plot(data["loss_cls"], label="Classification Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Classification Loss over Training Steps")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_cls.png"))
    plt.close()

    # 绘制 loss_aux 曲线
    plt.figure()
    plt.plot(data["loss_aux"], label="Auxiliary Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Auxiliary Loss over Training Steps")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_aux.png"))
    plt.close()


def save_data_to_csv(data, save_path):
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def main():
    # 日志文件路径
    log_file_path = "/data/pqh/env/MM/work_dirs/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/20250312_093756/20250312_093756.log"  # 替换为你的日志文件路径
    match = re.search(r'(\d{8}_\d{6})\.log$', log_file_path)
    # 图像保存目录
    if match:
        # 提取到的时间戳
        timestamp = match.group(1)
        # 图像保存目录
        save_dir = os.path.join('/data/pqh/env/MM/plots/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb', timestamp)
        # 如果目录不存在，创建目录
        os.makedirs(save_dir, exist_ok=True)
    else:
        print("未找到合适的时间戳格式，无法生成保存路径")
        save_dir = '/data/pqh/env/MM/plots/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb' # 替换为你想保存图像的目录
    # 数据保存路径
    data_save_path = "/data/pqh/env/MM/plots/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/20250312_093756/training_data.csv"  # 替换为你想保存数据的路径

    # 解析日志文件
    data = parse_log_file(log_file_path)

    # 绘制并保存曲线图像
    plot_metrics(data, save_dir)

    # 保存数据为CSV文件
    save_data_to_csv(data, data_save_path)

    print(f"曲线图像已保存到: {save_dir}")
    print(f"训练数据已保存到: {data_save_path}")


if __name__ == "__main__":
    main()