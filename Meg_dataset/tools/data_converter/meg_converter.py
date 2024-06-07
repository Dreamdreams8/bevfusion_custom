import os
from os import path as osp
import mmcv
import numpy as np
import json
from pyquaternion import Quaternion
from projects.Meg_dataset.mmdet3d.datasets.meg_dataset import MegDataset
import math


def get_train_val_scenes(root_path):
    """
    划分训练集和测试集
    """
    p = osp.join(root_path, 'jsons')
    all_scenes = os.listdir(p)
    all_scenes = [scenes.split('.')[0] for scenes in all_scenes]

    test_num = math.floor(len(all_scenes) / 10)  # 取 1/10 场景为测试
    train_num = len(all_scenes) - test_num
    train_scenes = all_scenes[:train_num]
    val_scenes = all_scenes[train_num:]

    return train_scenes, val_scenes  # ['0', '1', '2', ...]


def create_meg_infos(
    root_path, info_prefix
):
    train_scenes, val_scenes = get_train_val_scenes(root_path)

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version="v1.0-mini")

    print(
        "train sample: {}, val sample: {}".format(
            len(train_nusc_infos), len(val_nusc_infos)
        )
    )
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_nusc_infos
    info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_trainval_infos(root_path, train_scenes, val_scenes, test=False):

    train_nusc_infos = []
    val_nusc_infos = []

    available_scene_names = train_scenes + val_scenes

    for sid, scenes_json in enumerate(available_scene_names):
        # dataset json
        with open(os.path.join(root_path, 'jsons', scenes_json + '.json'), 'r') as jsonread:
            jsondata = json.load(jsonread)

        for sample in jsondata['frames']:
            if sample['is_key_frame']:
                print(scenes_json, '---', sample['frame_id'], ' is key frame')
                frame_id = sample['frame_id'] + sid * 10000
                assert not root_path.endswith('/'), "root_path should not end with an '/' sign"
                lidar_path = osp.join(root_path, sample['sensor_data']['fuser_lidar']['file_path'])
                # lidar_path = sample['sensor_data']['front_lidar']['file_path']
                timestamp = eval(sample['sensor_data']['fuser_lidar']['timestamp']) * int(1000) * int(1000)
                lidar2ego_info = jsondata["calibrated_sensors"]['lidar_ego']['extrinsic']['transform']
                lidar2ego_translation = np.array([lidar2ego_info['translation']['x'],
                                                  lidar2ego_info['translation']['y'],
                                                  lidar2ego_info['translation']['z']])
                lidar2ego_rotation = np.array(Quaternion([lidar2ego_info['rotation']['w'],
                                                          lidar2ego_info['rotation']['x'],
                                                          lidar2ego_info['rotation']['y'],
                                                          lidar2ego_info['rotation']['z']]
                                                         ).rotation_matrix)
                # dataset infos
                info = {
                    "frame_id": frame_id,
                    "lidar_path": lidar_path,
                    "sweeps": [],
                    "cams": dict(),
                    "lidar2ego_translation": lidar2ego_translation,
                    "lidar2ego_rotation": lidar2ego_rotation,
                    "timestamp": timestamp,
                }

                # camera-obtain 6 image's information per frame
                camera_types = [
                    "cam_back_120",
                    "cam_back_left_120",
                    "cam_back_right_120",
                    # "cam_front_30",
                    "cam_front_70_left",
                    # "cam_front_70_right",
                    "cam_front_left_120",
                    "cam_front_right_120"
                ]
                for cam in camera_types:
                    cam_path = osp.join(root_path, sample['sensor_data'][cam]['file_path'])
                    cam_intrinsics = np.array(
                        jsondata["calibrated_sensors"][cam]["intrinsic"]["K"],
                    )

                    lidar2cam_translation = np.array([
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['x'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['y'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['z']]
                    )
                    lidar2cam_rotation = np.array(Quaternion([
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['w'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['x'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['y'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['z']]
                    ).rotation_matrix)
                    cam2lidar_rotation = np.linalg.inv(lidar2cam_rotation)
                    cam2lidar_translation = np.dot(cam2lidar_rotation, -lidar2cam_translation.T)

                    T_lidar_to_pixel = np.array(jsondata["calibrated_sensors"][cam]
                                                ['T_lidar_to_pixel'], dtype=np.float32)
                    lidar2img = np.eye(4).astype(np.float32)
                    lidar2img[:3, :3] = T_lidar_to_pixel[:3, :3]
                    lidar2img[:3, 3] = T_lidar_to_pixel[:3, 3:].T
                    cam_info = dict(
                        camera_path=cam_path,
                        lidar2img=lidar2img,
                        camera_intrinsics=cam_intrinsics,
                        # camera2lidar_rotation=cam2lidar_rotation,
                        # camera2lidar_translation=cam2lidar_translation,
                        lidar2cam_rotation=lidar2cam_rotation,
                        lidar2cam_translation=lidar2cam_translation
                    )
                    info["cams"].update({cam: cam_info})

                # obtain annotation
                if not test:
                    annotations = sample['labels']
                    locs = np.array([[box['xyz_lidar']['x'],
                                      box['xyz_lidar']['y'],
                                      box['xyz_lidar']['z']] for box in annotations]
                                    ).reshape(-1, 3)
                    dims = np.array([[box['lwh']['w'],
                                      box['lwh']['l'],
                                      box['lwh']['h']] for box in annotations]
                                    ).reshape(-1, 3)
                    locs[:, 2] = locs[:, 2] - dims[:, 2] / 2.0  # mmdet3d 中以底边中心为中心点
                    rots = np.array([Quaternion([box['angle_lidar']['w'],
                                                 box['angle_lidar']['x'],
                                                 box['angle_lidar']['y'],
                                                 box['angle_lidar']['z']]
                                                ).yaw_pitch_roll[0] for box in annotations]
                                    ).reshape(-1, 1)

                    names = [box['category'] for box in annotations]
                    for i in range(len(names)):
                        if names[i] in MegDataset.NameMapping:
                            names[i] = MegDataset.NameMapping[names[i]]
                    names = np.array(names)

                    # we need to convert rot to SECOND format.
                    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                    assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
                    info["gt_boxes"] = gt_boxes
                    info["gt_names"] = names
                    info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                    info["is_2d_visible"] = np.array([a["is_2d_visible"] for a in annotations])

                if jsondata['scene_id'].strip('.json') in train_scenes:
                    train_nusc_infos.append(info)
                if jsondata['scene_id'].strip('.json') in val_scenes:
                    val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos