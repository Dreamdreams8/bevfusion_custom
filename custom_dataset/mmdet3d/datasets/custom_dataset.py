import tempfile
from os import path as osp
from typing import Any, Dict

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from ..evaluate.map import calculate_map


@DATASETS.register_module()
class MyCustomDataset(Custom3DDataset):
    NameMapping = {
        "Car": "car",
        "Truck":"truck",
        "Pedestrian":"pedestrian",
        "Cone":"cone",
        "Lock":"lock"
    }

    CLASSES = (
        "car",
        "truck",
        # "trailer",
        # "bus",
        # "construction_vehicle",
        # "bicycle",
        # "motorcycle",
        # "pedestrian",
        # "traffic_cone",
        # "barrier",
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=False,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.map_classes = map_classes

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info["gt_names"])
        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        #     gt_names = set(info["gt_names"][mask])
        # else:
        #     gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        # print(info)
        data = dict(
            lidar_path=info["lidar_path"],
            timestamp=info["timestamp"],
        )
        data["sample_idx"] = index
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        # print("lidar2ego_rotation",  info["lidar2ego_rotation"])
        # print("lidar2ego_translation",  info["lidar2ego_translation"])
        # print("lidar2ego",  lidar2ego[:3, 3])
        lidar2ego[:3, :3] = info["lidar2ego_rotation"]
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego
        # print("use_camera+++++++++++",self.modality["use_camera"])
        self.modality["use_camera"] = True  # 单激光时也默认打开 # add by why
        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            for _, camera_info in info["cams"].items():
                # print("camera_info:  ",camera_info["data_path"])
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                # print("sensor2lidar_rotation",camera_info["sensor2lidar_rotation"])
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                # camera2lidar[:3, :3] = camera_info["camera_extrinsic_r"]   # del by why
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        # print("+++++++++++-_______----------data:  ",data)
        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]

        gt_bboxes_3d = info["gt_boxes"]
        gt_names_3d = info["gt_names"]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        # print("gt_labels_3d:  ",gt_labels_3d)

        # if self.with_velocity:
        #     gt_velocity = info["gt_velocity"][mask]
        #     nan_mask = np.isnan(gt_velocity[:, 0])
        #     gt_velocity[nan_mask] = [0.0, 0.0]
        #     gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

# add why
    def evaluate(
        self,
        results,
        metric="bbox",
        **kwargs
    ):
        metrics_dict = self.calc_metrics(results)  # 计算评价指标
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_json = osp.join(tmp_dir.name, "metrics_summary.json")
        mmcv.dump(metrics_dict, tmp_json)

        # self.metric_table(tmp_json)  # 表格形式输出评价指标
        # self.metric_dict(tmp_json)  # 字典形式输出评价指标

        tmp_dir.cleanup()
        return metrics_dict

    def calc_metrics(self, results, score_thr=0.5):
        #  results[0]: dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])
        mAP_list = []  # 存放每一帧的 mAP
        for frame_i, (frame_gt, frame_pred) in enumerate(zip(self.data_infos, results)):
            gt_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
            pred_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
            for gt_box, gt_label in zip(frame_gt['gt_boxes'], frame_gt['gt_names']):
                if str(gt_label) != 'masked_area':  # 过滤掉对象车道蒙板
                    gt_label_idx = self.CLASSES.index(str(gt_label))
                    gt_boxes_list[gt_label_idx].append(gt_box)

            for pred_box, pred_score, pred_label_idx in zip(frame_pred['boxes_3d'], frame_pred['scores_3d'], frame_pred['labels_3d']):
                if pred_score >= score_thr:
                    pred_boxes_list[int(pred_label_idx)].append(pred_box)

            # 计算单帧 mAP
            mAP = calculate_map(gt_boxes_list, pred_boxes_list, iou_threshold=0.5)
            print("frame_{} mAP is {}:".format(frame_i, mAP))
            mAP_list.append(mAP)

        mAP_list_filter_0 = list(filter(lambda x: x != 0, mAP_list))  # 去掉 0
        mAP = np.mean(mAP_list_filter_0)
        metrics_summary = {
            'mAP': mAP,
        }

        return metrics_summary
