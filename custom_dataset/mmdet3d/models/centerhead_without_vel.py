import torch
from mmcv.runner import force_fp32
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.bbox.centerpoint import CenterHead, clip_sigmoid


@HEADS.register_module()
class CenterHeadWithoutVel(CenterHead):
    """
    注释掉 CenterHead 的 vx,vy,vel 等速度信息
    """

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])

        feature_map_size = torch.div(
            grid_size[:2],
            self.train_cfg["out_size_factor"],
            rounding_mode="trunc",
        )

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append(
                [
                    torch.where(gt_labels_3d == class_name.index(i) + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0])
            )

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius = max(self.train_cfg["min_radius"], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = (
                        task_boxes[idx][k][0],
                        task_boxes[idx][k][1],
                        task_boxes[idx][k][2],
                    )

                    coor_x = (
                        (x - pc_range[0])
                        / voxel_size[0]
                        / self.train_cfg["out_size_factor"]
                    )
                    coor_y = (
                        (y - pc_range[1])
                        / voxel_size[1]
                        / self.train_cfg["out_size_factor"]
                    )

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)
                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (
                        x * feature_map_size[1] + y
                        < feature_map_size[0] * feature_map_size[1]
                    )

                    ind[new_idx] = x * feature_map_size[1] + y

                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    # vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat(
                        [
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            # vx.unsqueeze(0),
                            # vy.unsqueeze(0),
                        ]
                    )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]["heatmap"] = clip_sigmoid(preds_dict[0]["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]["heatmap"], heatmaps[task_id], avg_factor=max(num_pos, 1)
            )
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]["anno_box"] = torch.cat(
                (
                    preds_dict[0]["reg"],
                    preds_dict[0]["height"],
                    preds_dict[0]["dim"],
                    preds_dict[0]["rot"],
                    # preds_dict[0]["vel"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)
            )
            loss_dict[f"heatmap/task{task_id}"] = loss_heatmap
            loss_dict[f"bbox/task{task_id}"] = loss_bbox
        return loss_dict