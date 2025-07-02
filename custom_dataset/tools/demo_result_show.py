import argparse
import copy
import os
import glob
import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from custom_dataset.mmdet3d.core.visualize import visualize_lidar
from mmdet3d.models import build_model


def load_points_from_bin(bin_path):
    """
    从bin文件加载点云数据
    
    Args:
        bin_path (str): bin文件路径
        
    Returns:
        np.ndarray: 点云数据，形状为(N, 4)，包含x, y, z, intensity
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 4)  # x, y, z, intensity
    return points


def save_results_to_txt(save_path, name, bboxes, scores, labels, classes):
    """
    将推理结果保存到txt文件中
    
    Args:
        save_path (str): 保存路径
        name (str): 文件名（不含扩展名）
        bboxes (np.ndarray): 边界框数据，形状为 (N, 7)，包含 x, y, z, dx, dy, dz, yaw
        scores (np.ndarray): 置信度分数，形状为 (N,)
        labels (np.ndarray): 类别标签，形状为 (N,)
        classes (list): 类别名称列表
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{name}.txt")
    
    with open(file_path, 'w') as f:
        if bboxes is not None and len(bboxes) > 0:
            for i in range(len(bboxes)):
                # 获取类别名称
                # class_name = classes[labels[i]]
                # 获取边界框参数
                box = bboxes[i]
                # 写入格式：类别 置信度 x y z dx dy dz yaw
                f.write(f" {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} "
                        f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} {labels[i]} {scores[i]:.6f}\n")


def prepare_input_data(points, point_cloud_range, voxel_size):
    """
    准备模型输入数据
    
    Args:
        points (np.ndarray): 点云数据，形状为(N, 4)
        point_cloud_range (list): 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size (list): 体素大小 [x_size, y_size, z_size]
        
    Returns:
        dict: 模型输入数据
    """
    # 过滤点云范围
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )
    points = points[mask]
    
    # 转换为torch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32).cuda()
    points_tensor = points_tensor.view(1, -1, points.shape[1])  # [1, N, 4]
    
    # 构建输入数据字典 - 修复 metas 结构
    data = {
        "points": points_tensor,
        "metas": [
            {
                "box_type_3d": LiDARInstance3DBoxes,
                "point_cloud_range": point_cloud_range,
                "voxel_size": voxel_size,
            }
        ]
    }
    
    return data


def process_bin_file(bin_file, model, cfg, txt_output_dir, viz_output_dir, bbox_score):
    """
    处理单个bin文件，进行推理并保存结果
    
    Args:
        bin_file (str): bin文件路径
        model: 推理模型
        cfg: 配置对象
        txt_output_dir (str): 文本结果保存目录
        viz_output_dir (str): 可视化结果保存目录
        bbox_score (float): 边界框置信度阈值
        
    Returns:
        tuple: (检测到的目标数量, 文件名)
    """
    # 获取文件名（不含路径和扩展名）
    base_name = os.path.splitext(os.path.basename(bin_file))[0]
    
    # 加载点云数据
    points = load_points_from_bin(bin_file)
    
    # 准备输入数据
    data = prepare_input_data(
        points, 
        cfg.point_cloud_range, 
        cfg.voxel_size if hasattr(cfg, 'voxel_size') else [0.05, 0.05, 0.1]
    )
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**data)
    
    # 处理推理结果
    if "boxes_3d" in outputs[0]:
        bboxes = outputs[0]["boxes_3d"].tensor.cpu().numpy()
        scores = outputs[0]["scores_3d"].cpu().numpy()
        labels = outputs[0]["labels_3d"].cpu().numpy()
        
        # 应用置信度阈值
        indices = scores >= bbox_score
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
        
        # 创建LiDARInstance3DBoxes对象用于可视化
        bboxes_3d = LiDARInstance3DBoxes(bboxes, box_dim=7)
        
        # 保存结果到txt文件
        save_results_to_txt(
            txt_output_dir,
            base_name,
            bboxes,
            scores,
            labels,
            cfg.object_classes
        )
        
        # 可视化结果
        visualize_lidar(
            os.path.join(viz_output_dir, f"{base_name}.png"),
            points,
            bboxes=bboxes_3d,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.object_classes,
        )
        
        return len(bboxes), base_name
    else:
        # 如果没有检测到目标，仍然保存空的txt文件
        save_results_to_txt(
            txt_output_dir,
            base_name,
            np.zeros((0, 7)),
            np.array([]),
            np.array([]),
            cfg.object_classes
        )
        
        # 仅可视化点云
        visualize_lidar(
            os.path.join(viz_output_dir, f"{base_name}.png"),
            points,
            bboxes=None,
            labels=None,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.object_classes,
        )
        
        return 0, base_name


def main():
    parser = argparse.ArgumentParser(description="增强版批量推理点云文件并保存结果")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--checkpoint", help="模型检查点文件路径", required=True)
    parser.add_argument("--input-dir", help="输入点云文件夹路径", required=True)
    parser.add_argument("--output-dir", help="输出结果保存路径", default="data/enhanced_inference")
    parser.add_argument("--bbox-score", type=float, default=0.3, help="边界框置信度阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--recursive", action="store_true", help="是否递归搜索子文件夹中的bin文件")
    parser.add_argument("--summary", action="store_true", help="是否生成推理结果摘要")
    args = parser.parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 创建输出目录
    txt_output_dir = os.path.join(args.output_dir, "txt")
    viz_output_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    
    # 构建模型并加载检查点
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[device.index])
    model.eval()
    
    # 获取点云文件列表
    if args.recursive:
        bin_files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.bin"), recursive=True))
    else:
        bin_files = sorted(glob.glob(os.path.join(args.input_dir, "*.bin")))
    
    if not bin_files:
        print(f"在 {args.input_dir} 中未找到.bin文件")
        return
    
    print(f"找到 {len(bin_files)} 个点云文件，开始处理...")
    
    # 处理每个点云文件
    results = []
    for bin_file in tqdm(bin_files):
        result = process_bin_file(
            bin_file, 
            model, 
            cfg, 
            txt_output_dir, 
            viz_output_dir, 
            args.bbox_score
        )
        results.append(result)
    
    # 生成摘要信息
    if args.summary:
        summary_path = os.path.join(args.output_dir, "inference_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"推理摘要\n")
            f.write(f"配置文件: {args.config}\n")
            f.write(f"检查点: {args.checkpoint}\n")
            f.write(f"输入目录: {args.input_dir}\n")
            f.write(f"置信度阈值: {args.bbox_score}\n")
            f.write(f"处理文件总数: {len(bin_files)}\n\n")
            
            # 按检测目标数量排序
            sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
            
            f.write(f"检测到目标最多的文件:\n")
            for i, (num_objects, filename) in enumerate(sorted_results[:10]):
                f.write(f"{i+1}. {filename}: {num_objects} 个目标\n")
            
            f.write(f"\n检测到目标最少的文件:\n")
            for i, (num_objects, filename) in enumerate(sorted_results[-10:]):
                f.write(f"{i+1}. {filename}: {num_objects} 个目标\n")
            
            # 统计信息
            total_objects = sum(num for num, _ in results)
            avg_objects = total_objects / len(results) if results else 0
            f.write(f"\n总检测目标数: {total_objects}\n")
            f.write(f"平均每帧检测目标数: {avg_objects:.2f}\n")
            
            # 统计没有检测到目标的文件数量
            no_detection_files = sum(1 for num, _ in results if num == 0)
            f.write(f"没有检测到目标的文件数: {no_detection_files} ({no_detection_files/len(results)*100:.2f}%)\n")
        
        print(f"摘要信息已保存到: {summary_path}")
    
    print(f"处理完成！结果已保存到：")
    print(f"- 文本结果：{txt_output_dir}")
    print(f"- 可视化结果：{viz_output_dir}")


if __name__ == "__main__":
    main()