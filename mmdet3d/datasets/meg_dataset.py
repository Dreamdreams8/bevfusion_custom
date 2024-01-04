# projects/Meg_dataset/mmdet3d/datasets/meg_dataset.py
from typing import Any, Dict
import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
import os.path as osp
import tempfile
from scipy.spatial.transform import Rotation as R
# from ...mmdet3d.evaluate.map import calculate_map


def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

@DATASETS.register_module()
class MegDataset(Custom3DDataset):
    NameMapping = {
        "小汽车": "car",
        "汽车": "car",
        "货车": "truck",
        "工程车": "construction_vehicle",
        "巴士": "bus",
        "摩托车": "motorcycle",
        "自行车": "bicycle",
        "三轮车": "tricycle",
        "骑车人": "cyclist",
        "骑行的人": "cyclist",
        "人": "pedestrian",
        "行人": "pedestrian",
        "其它": "other",
        "残影": "ghost",
        "蒙版": "masked_area",
        "其他": "other",
        "拖挂": "other",
        "锥桶": "traffic_cone",
        "防撞柱": "traffic_cone"
    }

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    CLASSES = (
        "car",
        "truck"
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        with_velocity=False,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        data_config=None,
        test_mode=False,
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

        self.with_velocity = with_velocity
        self.data_config = data_config

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
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
        data_infos = list(sorted(data["infos"], key=lambda e: e["frame_id"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['frame_id'],
            lidar_path=info["lidar_path"],
        )

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        input_dict["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2cameras = []
            lidar2images = []
            camera2egos = []
            camera_intrinsics = []
            camera2lidars = []                        
            # input_dict["image_paths"] = []
            # input_dict["lidar2camera"] = []
            # input_dict["lidar2image"] = []
            # input_dict["camera2ego"] = []
            # input_dict["camera_intrinsics"] = []
            # input_dict["camera2lidar"] = []

            # info["cams"]:['cam_front', 'cam_left', 'cam_right']
            for camera_type, camera_info in info["cams"].items():
                image_path = camera_info["camera_path"]
                image_paths.append(image_path)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = euler2rot(
                    camera_info["camera_extrinsic_r"][0]
                )
                camera2lidar[:3, 3] = camera_info["camera_extrinsic_t"][0]
                camera2lidars.append(camera2lidar)

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera2lidar[:3, :3])
                lidar2camera_t = (
                    camera2lidar[:3, 3] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                lidar2cameras.append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsic = np.eye(4).astype(np.float32)
                camera_intrinsic[:3, :3] = camera_info["camera_intrinsics"]
                camera_intrinsics.append(camera_intrinsic)

                # lidar to image transform
                lidar2image = camera_intrinsic @ lidar2camera_rt.T
                lidar2images.append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = euler2rot(
                    camera_info["camera_extrinsic_r"][0]
                )
                camera2ego[:3, 3] = camera_info["camera_extrinsic_t"][0]
                camera2egos.append(camera2ego)         
            input_dict.update(
                dict(
                    image_paths=image_paths,
                    camera2ego=camera2egos,
                    camera_intrinsics=camera_intrinsics,
                    lidar2camera=lidar2cameras,
                    lidar2image=lidar2images,
                    camera2lidar=camera2lidars,
                )
            )

        # if not self.test_mode:
        # TODO (Haotian): test set submission.
        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos

        return input_dict

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
        # 此处根据点云数量，类别进行过滤，无相关需求，暂时屏蔽相关处理        
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
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

        # # filter out bbox containing no points
        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        # else:
        #     mask = info["num_lidar_pts"] > 0
        # gt_bboxes_3d = info["gt_boxes"][mask]
        # gt_names_3d = info["gt_names"][mask]
        # gt_labels_3d = []
        # for cat in gt_names_3d:
        #     if cat in self.CLASSES:
        #         gt_labels_3d.append(self.CLASSES.index(cat))
        #     else:
        #         gt_labels_3d.append(-1)

        # gt_labels_3d = np.array(gt_labels_3d)
        # label_mask = gt_labels_3d >= 0
        # gt_labels_3d = gt_labels_3d[label_mask]
        # gt_bboxes_3d = gt_bboxes_3d[label_mask]  # TODO 过滤非指定类别的信息

        # if self.with_velocity:
        #     gt_velocity = info["gt_velocity"][mask]
        #     nan_mask = np.isnan(gt_velocity[:, 0])
        #     gt_velocity[nan_mask] = [0.0, 0.0]
        #     gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # # the same as KITTI (0.5, 0.5, 0)
        # # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        # ).convert_to(self.box_mode_3d)

        # anns_results = dict(
        #     gt_bboxes_3d=gt_bboxes_3d,
        #     gt_labels_3d=gt_labels_3d,
        #     gt_names=gt_names_3d,
        # )
        # return anns_results

    # def evaluate(
    #     self,
    #     results,
    #     metric="bbox",
    #     **kwargs
    # ):
    #     metrics_dict = self.calc_metrics(results)  # 计算评价指标
    #     tmp_dir = tempfile.TemporaryDirectory()
    #     tmp_json = osp.join(tmp_dir.name, "metrics_summary.json")
    #     mmcv.dump(metrics_dict, tmp_json)

    #     # self.metric_table(tmp_json)  # 表格形式输出评价指标
    #     # self.metric_dict(tmp_json)  # 字典形式输出评价指标

    #     tmp_dir.cleanup()
    #     return metrics_dict

    # def calc_metrics(self, results, score_thr=0.5):
    #     #  results[0]: dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])
    #     mAP_list = []  # 存放每一帧的 mAP
    #     for frame_i, (frame_gt, frame_pred) in enumerate(zip(self.data_infos, results)):
    #         gt_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
    #         pred_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
    #         for gt_box, gt_label in zip(frame_gt['gt_boxes'], frame_gt['gt_names']):
    #             if str(gt_label) != 'masked_area':  # 过滤掉对象车道蒙板
    #                 gt_label_idx = self.CLASSES.index(str(gt_label))
    #                 gt_boxes_list[gt_label_idx].append(gt_box)

    #         for pred_box, pred_score, pred_label_idx in zip(frame_pred['boxes_3d'], frame_pred['scores_3d'], frame_pred['labels_3d']):
    #             if pred_score >= score_thr:
    #                 pred_boxes_list[int(pred_label_idx)].append(pred_box)

    #         # 计算单帧 mAP
    #         mAP = calculate_map(gt_boxes_list, pred_boxes_list, iou_threshold=0.5)
    #         print("frame_{} mAP is {}:".format(frame_i, mAP))
    #         mAP_list.append(mAP)

    #     mAP_list_filter_0 = list(filter(lambda x: x != 0, mAP_list))  # 去掉 0
    #     mAP = np.mean(mAP_list_filter_0)
    #     metrics_summary = {
    #         'mAP': mAP,
    #     }

    #     return metrics_summary