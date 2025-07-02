import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from custom_dataset.mmdet3d.core.visualize import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


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
                class_name = classes[labels[i]]
                # 获取边界框参数
                box = bboxes[i]
                # 写入格式：类别 置信度 x y z dx dy dz yaw
                f.write(f"{class_name} {scores[i]:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} "
                        f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f}\n")


def main() -> None:
    # dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.2)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--save-dir", type=str, default="data/inference_results", 
                        help="保存推理结果的目录")
    args, opts = parser.parse_known_args()

    cfg = Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDataParallel(model, device_ids=[0])
        model.eval()

    # 创建保存结果的目录
    save_path = os.path.join(args.save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}".format(metas["timestamp"])
        
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)
                
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            scores = np.ones_like(labels, dtype=np.float32)  # 对于GT，分数设为1

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]
                scores = scores[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
            bboxes_tensor = bboxes.tensor
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
            bboxes_tensor = bboxes.tensor
        else:
            bboxes = None
            scores = None
            labels = None
            bboxes_tensor = None

        # 保存推理结果到txt文件
        if bboxes_tensor is not None:
            # 获取原始点云文件名（不含路径和扩展名）
            if "pts_filename" in metas:
                # 从pts_filename中提取文件名（不含路径和扩展名）
                pts_filename = metas["pts_filename"]
                base_name = os.path.splitext(os.path.basename(pts_filename))[0]
            else:
                # 如果没有pts_filename，使用timestamp作为文件名
                base_name = name
                
            # 保存结果到txt文件
            save_results_to_txt(
                save_path,
                base_name,
                bboxes_tensor,
                scores,
                labels,
                cfg.object_classes
            )

        # 可视化部分（与原始代码相同）
        if "img" in data:           
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if "masks_bev" in outputs[0] if args.mode == "pred" else False:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()