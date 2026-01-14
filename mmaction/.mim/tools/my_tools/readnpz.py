import numpy as np


def read_npz_and_print_dimensions(file_path):
    """
    读取指定路径下的 .npz 文件，并打印每个数组的维度信息。

    :param file_path: .npz 文件的路径
    """
    try:
        # 加载 .npz 文件
        npz_file = np.load(file_path)

        # 遍历文件中的所有数组
        for array_name in npz_file.files:
            # 获取数组
            array = npz_file[array_name]

            # 打印数组的名称和维度
            print(f"数组名称: {array_name}") # img_embeddings
            print(f"维度: {array.shape}") # (256, 64, 64)
            print(f"数据类型: {array.dtype}") # float32
            print(f"数组内容:\n{array}\n")

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except OSError as e:
        print(f"读取文件时发生错误: {e}")
    finally:
        # 关闭 .npz 文件
        npz_file.close()


if __name__ == "__main__":
    # 指定 .npz 文件的路径
    file_path = input("请输入 .npz 文件的路径: ")

    # 调用函数读取 .npz 文件并打印维度信息
    read_npz_and_print_dimensions(file_path)