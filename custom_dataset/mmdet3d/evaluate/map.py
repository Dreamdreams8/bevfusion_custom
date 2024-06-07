import numpy as np


def calculate_iou(box1, box2):
    # 计算两个3D框之间的IoU（Intersection over Union）
    # 这里假设box1和box2都是(x, y, z, w, h, d)形式的框，分别表示中心坐标和宽高深度
    # 返回IoU值

    # 计算两个框的边界坐标
    box1_min = (box1[0] - box1[3] / 2, box1[1] - box1[4] / 2, box1[2] - box1[5] / 2)
    box1_max = (box1[0] + box1[3] / 2, box1[1] + box1[4] / 2, box1[2] + box1[5] / 2)
    box2_min = (box2[0] - box2[3] / 2, box2[1] - box2[4] / 2, box2[2] - box2[5] / 2)
    box2_max = (box2[0] + box2[3] / 2, box2[1] + box2[4] / 2, box2[2] + box2[5] / 2)

    # 计算两个框的相交区域的体积
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    intersect_size = np.maximum(0, intersect_max - intersect_min)

    # 计算两个框的并集区域的体积
    # box1_size = np.maximum(0, box1_max - box1_min)
    box1_size = (
        np.maximum(0, box1_max[0] - box1_min[0]),
        np.maximum(0, box1_max[1] - box1_min[1]),
        np.maximum(0, box1_max[2] - box1_min[2]))

    # box2_size = np.maximum(0, box2_max - box2_min)
    box2_size = (
        np.maximum(0, box2_max[0] - box2_min[0]),
        np.maximum(0, box2_max[1] - box2_min[1]),
        np.maximum(0, box2_max[2] - box2_min[2]))

    # union_size = box1_size + box2_size - intersect_size
    union_size = (
        box1_size[0] + box2_size[0] - intersect_size[0],
        box1_size[1] + box2_size[1] - intersect_size[1],
        box1_size[2] + box2_size[2] - intersect_size[2])

    # 计算IoU值
    iou = np.prod(intersect_size) / np.prod(union_size)
    return iou


def calculate_ap(recall, precision):
    # 计算单类别的平均精度（AP）
    # 输入为召回率（recall）和精确率（precision）数组
    # 返回AP值

    recall = np.array(recall).flatten()
    precision = np.array(precision).flatten()

    # 将召回率和精确率数组进行插值，以确保召回率是单调递增的
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.interp(recall_interp, recall, precision)

    # 计算AP值
    ap = np.mean(precision_interp)
    return ap


def calculate_map(gt_boxes, pred_boxes, iou_threshold=0.5):
    """

    gt_boxes = [
        [(0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6)],  # 类别0的真实框
        [(0, 0, 0, 0, 0, 0, 0), (2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8)],   # 类别1的真实框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]

    pred_boxes = [
        [(0, 0, 0, 0, 0, 0, 0), (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.9), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.8)],  # 类别0的预测框
        [(0, 0, 0, 0, 0, 0, 0), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.7), (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.6)],   # 类别1的预测框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]
    """
    # 计算3D目标检测的平均精度（mAP）
    # 输入为真实框（gt_boxes）和预测框（pred_boxes）列表，每个框都是(x, y, z, w, h, d)形式的元组
    # iou_threshold是IoU阈值，用于判断预测框和真实框之间的匹配关系
    # 返回mAP值

    num_classes = len(gt_boxes)  # 类别数
    aps = []  # 每个类别的AP值列表

    for class_idx in range(num_classes):
        gt_class_boxes = gt_boxes[class_idx]  # 当前类别的真实框列表
        pred_class_boxes = pred_boxes[class_idx]  # 当前类别的预测框列表

        num_gt_boxes = len(gt_class_boxes)
        num_pred_boxes = len(pred_class_boxes)

        # 初始化匹配矩阵
        match_matrix = np.zeros((num_pred_boxes, num_gt_boxes))

        # 对每个预测框和真实框计算IoU，并进行匹配
        for pred_idx, pred_box in enumerate(pred_class_boxes):
            for gt_idx, gt_box in enumerate(gt_class_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > iou_threshold:
                    match_matrix[pred_idx, gt_idx] = 1

        # 计算每个预测框的置信度得分
        scores = [box[6] for box in pred_class_boxes]

        # 根据置信度得分对匹配矩阵进行排序
        sorted_indices = np.argsort(scores)[::-1]
        match_matrix = match_matrix[sorted_indices, :]

        # 计算召回率和精确率
        true_positives = np.cumsum(match_matrix, axis=0)
        false_positives = np.cumsum(1 - match_matrix, axis=0)
        recall = true_positives / num_gt_boxes
        precision = true_positives / (true_positives + false_positives)

        # 计算AP值
        ap = calculate_ap(recall, precision)
        aps.append(ap)

    aps_filter_0 = list(filter(lambda x: x != 0, aps))  # 去掉 0
    # 计算mAP值
    if len(aps_filter_0) == 0:
        return 0.0
    else:
        mAP = np.mean(aps_filter_0)
        return mAP


def main():
    # 生成示例数据
    gt_boxes = [
        [(1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6)],  # 类别0的真实框
        [(2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8)],   # 类别1的真实框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]

    pred_boxes = [
        [(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.9), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.8)],  # 类别0的预测框
        [(2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.7), (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.6)],   # 类别1的预测框
        [(0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)]
    ]

    # 计算mAP
    mAP = calculate_map(gt_boxes, pred_boxes)

    # 输出结果
    print("mAP:", mAP)


if __name__ == '__main__':
    main()