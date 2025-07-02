

import os
import numpy as np
from shapely.geometry import Polygon

# 参数配置
IOU_THRESHOLD = 0.5  # IoU阈值
USE_BEV = True       # 使用鸟瞰图IoU计算


# 类别映射
class_mapping = {
    0: 'Car',
    1: 'Truck',
    #2: 'Bridge',
    #3: 'Conetank',
    #4: 'Lockbox',
    2: 'Lockstation',
    # 4: 'ForkLift',
    # 7: 'Pedestrian',
    # 8: 'IGV',
}

# 反向类别映射（用于从字符串到整数的映射）
reverse_class_mapping = {v.lower(): k for k, v in class_mapping.items()}

def parse_pred_line(line):
    """解析预测"""
    parts = list(map(float, line.strip().split()))
    return {
        'x': parts[0], 'y': parts[1], 'z': parts[2],
        'w': parts[3], 'l': parts[4], 'h': parts[5],
        'angle': np.deg2rad(parts[6]),  # 假设角度为度数转换为弧度
        'class': int(parts[7]),
        'score': parts[8]
    }

def parse_gt_line(line):
    """解析真值"""
    parts = line.strip().split()
    parts1 = list(map(float, parts[:7]))
    class_type_str = parts[7].lower()  # 转换为小写
    class_type = reverse_class_mapping.get(class_type_str, -1)  # 获取类别索引
    return {
        'x': parts1[0], 'y': parts1[1], 'z': parts1[2],
        'w': parts1[3], 'l': parts1[4], 'h': parts1[5],
        'angle': np.deg2rad(parts1[6]),  # 假设角度为度数转换为弧度
        'class': class_type
    }

def compute_bev_iou(box1, box2):
    """计算鸟瞰图下的IoU"""
    def get_vertices(x, y, w, l, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        half_w = w / 2
        half_l = l / 2
        corners = np.array([
            [-half_w, -half_l],
            [half_w, -half_l],
            [half_w, half_l],
            [-half_w, half_l]
        ])
        rotated = np.dot(corners, np.array([[cos_a, -sin_a], [sin_a, cos_a]]))
        rotated[:, 0] += x
        rotated[:, 1] += y
        return rotated
    
    try:
        poly1 = Polygon(get_vertices(box1['x'], box1['y'], box1['w'], box1['l'], box1['angle']))
        poly2 = Polygon(get_vertices(box2['x'], box2['y'], box2['w'], box2['l'], box2['angle']))
    except Exception as e:
        print(f"Error computing IoU for boxes: {box1}, {box2}. Error: {e}")
        return 0.0

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union != 0 else 0.0

def compute_ap(recall, precision):
    """计算AP PASCAL VOC 11点法 """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

# 初始化统计字典 - 使用所有预定义类别
class_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0} for c in class_mapping.keys()}
all_preds = []

# 遍历预测和真值文件
pred_dir = 'data/enhanced_inference_results/txt'
gt_dir = 'data/custom_0610_3class/labels'

for pred_file in sorted(os.listdir(pred_dir)):
    # 获取对应的真值文件
    gt_file = pred_file  
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, gt_file)

    # 检查文件是否存在
    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth file {gt_path} does not exist. Skipping...")
        continue

    # 读取数据
    try:
        with open(pred_path, 'r') as f:
            pred_boxes = [parse_pred_line(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading prediction file {pred_path}: {e}")
        continue
    
    try:
        with open(gt_path, 'r') as f:
            gt_boxes = [parse_gt_line(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading ground truth file {gt_path}: {e}")
        continue

    # 按类别组织数据
    gt_dict = {}
    for gt in gt_boxes:
        c = gt['class']
        if c != -1:  # 跳过无效类别
            gt_dict.setdefault(c, []).append(gt)
    
    pred_dict = {}
    for pred in pred_boxes:
        c = pred['class']
        if c != -1:  # 跳过无效类别
            pred_dict.setdefault(c, []).append(pred)

    # 处理每个类别
    for c in class_mapping.keys():  # 使用预定义的类别顺序
        # 获取当前类别的预测和真值
        class_preds = sorted(pred_dict.get(c, []), key=lambda x: x['score'], reverse=True)
        class_gts = gt_dict.get(c, [])
        
        # 更新总真值数
        class_stats[c]['total_gt'] += len(class_gts)
        
        # 初始化匹配状态
        matched = [False] * len(class_gts)
        
        # 处理每个预测
        for pred in class_preds:
            best_iou = 0.0
            best_idx = -1
            
            # 寻找最佳匹配
            for i, gt in enumerate(class_gts):
                if not matched[i]:
                    iou = compute_bev_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
            
            # 判断TP/FP
            if best_iou >= IOU_THRESHOLD and best_idx != -1:
                matched[best_idx] = True
                all_preds.append({'class': c, 'score': pred['score'], 'is_tp': True})
                class_stats[c]['tp'] += 1
            else:
                all_preds.append({'class': c, 'score': pred['score'], 'is_tp': False})
                class_stats[c]['fp'] += 1
        
        # 统计FN
        class_stats[c]['fn'] += len(class_gts) - sum(matched)

# TP FP FN
total_tp = sum(s['tp'] for s in class_stats.values())
total_fp = sum(s['fp'] for s in class_stats.values())
total_fn = sum(s['fn'] for s in class_stats.values())

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# 计算AP和mAP，按预定义类别顺序输出
aps = []
print("类别AP结果：")
for c in class_mapping.keys():
    class_name = class_mapping[c]
    
    # 获取当前类别的所有预测
    class_preds = [p for p in all_preds if p['class'] == c]
    class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
    
    # 检查是否有预测结果和真值
    if len(class_preds) == 0 or class_stats[c]['total_gt'] == 0:
        ap = 0.0
        print(f'Class {class_name:<12} AP: {ap:.4f} (无预测结果或无真值)')
    else:
        # 提取TP/FP标记
        tp_list = np.array([p['is_tp'] for p in class_preds])
        fp_list = np.logical_not(tp_list)
        
        # 计算累积TP/FP
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        
        # 计算precision和recall
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        recalls = tp_cum / class_stats[c]['total_gt']
        
        # 计算AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        print(f'Class {class_name:<12} AP: {ap:.4f}')

mAP = np.mean(aps) if aps else 0

# 输出结果
print('\n')
print(f'{"指标":<20} {"结果":>6}')
print('-' * 32)
print(f'{"Precision":<20} {precision:>10.4f}')
print(f'{"Recall":<20} {recall:>10.4f}')
print(f'{"F1 Score":<20} {f1:>10.4f}')
print(f'{"mAP":<20} {mAP:>10.4f}')

