import os
import random

def split_dataset(label_dir, image_sets_dir, split_ratio=0.2):
    """
    将 labels 目录下的文件按指定比例随机划分为训练集和验证集。

    Args:
        label_dir (str): labels 文件夹的路径。
        image_sets_dir (str): ImageSets 文件夹的路径。
        split_ratio (float): 验证集所占比例，默认为 0.2。
    """
    # 确保 ImageSets 文件夹存在
    os.makedirs(image_sets_dir, exist_ok=True)

    # 获取 labels 目录下的所有文件
    label_files = os.listdir(label_dir)
    # 过滤掉非文件项
    label_files = [f for f in label_files if os.path.isfile(os.path.join(label_dir, f))]
    # 去除文件后缀
    file_names = [os.path.splitext(f)[0] for f in label_files]

    # 随机打乱文件名列表
    random.shuffle(file_names)

    # 计算验证集的大小
    val_size = int(len(file_names) * split_ratio)

    # 划分训练集和验证集
    val_files = file_names[:val_size]
    train_files = file_names[val_size:]

    # 定义 train.txt 和 val.txt 的路径
    train_txt_path = os.path.join(image_sets_dir, 'train.txt')
    val_txt_path = os.path.join(image_sets_dir, 'val.txt')

    # 写入 train.txt
    with open(train_txt_path, 'w') as f:
        for name in train_files:
            f.write(name + '\n')

    # 写入 val.txt
    with open(val_txt_path, 'w') as f:
        for name in val_files:
            f.write(name + '\n')

if __name__ == "__main__":
    # 定义 labels 文件夹和 ImageSets 文件夹的路径
    label_dir = 'data/20240617-720/labels'
    image_sets_dir = 'data/20240617-720/ImageSets'
    # 设置划分比例
    split_ratio = 0.0

    split_dataset(label_dir, image_sets_dir, split_ratio)