# tools/data_convert/custom_converter.py
import os
from os import path as osp
import mmcv
import numpy as np
import json
from pyquaternion import Quaternion
from custom_dataset.mmdet3d.datasets.custom_dataset import MyCustomDataset
from mmdet3d.datasets import NuScenesDataset
import math

# 左边为标注标签                                                         Bridge                                                            Forklift
class_id = {"Truck" : "truck","Car" : "car","Lockstation" : "lockstation",
        "Bridge" : "bridge","Forklift" : "forklift","Conetank" : "conetank",
        "Lockbox" : "lockbox",
        "ForkLift" : "forkLift","Pedestrian" : "pedestrian"}                                   

def _read_imageset_file(path):
    try:
        result = []
        with open(path, 'r') as f:
            for line in f:
                # 去除行尾的换行符
                clean_line = line.rstrip('\n')
                result.append(clean_line)
        return result
    except FileNotFoundError:
        print(f"文件 {path} 未找到，请检查路径。")
        return []
    except ValueError:
        print(f"文件 {path} 中的内容存在问题，请检查文件内容。")
        return []

def get_train_val_scenes(root_path):
    """
    划分训练集和测试集float
    """
    imageset_folder = osp.join(root_path, 'ImageSets')
    train_img_ids = _read_imageset_file(str(imageset_folder + '/train.txt'))
    # print("train_scenes:  ",train_img_ids)
    val_img_ids = _read_imageset_file(str(imageset_folder + '/val.txt'))
    # test_img_ids = _read_imageset_file(str(imageset_folder + '/test.txt'))    
    return  train_img_ids,val_img_ids


def create_custom_infos(
    root_path, info_prefix
):
    train_scenes, val_scenes= get_train_val_scenes(root_path)
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version="custom")

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


    cam_intrinsic = np.array([625.30933437, 0.0, 961.13004221,
                            0.0, 623.64759937, 546.09541553,
                            0,0,1]).reshape((3,3))
    cam_front_extrinsic_r = np.array([0.703439089347274, -0.7103943323318405, -0.020487775159276703, -0.009674256462361884])   # w,x,y,z
    cam_front_extrinsic_t = np.array([-0.0, 5.1654, 0.891921])
    # cam_front_extrinsic_r,cam_front_extrinsic_t = correct_camera_coord(cam_front_extrinsic_r,cam_front_extrinsic_t)

    cam_left_extrinsic_r = np.array([0.3532152326674449,-0.3298521727440406,-0.6083741220437878,0.6295375057732218])   # w,x,y,z
    cam_left_extrinsic_t = np.array([-1.3472,5.0299,1.51004])
    # cam_left_extrinsic_r,cam_left_extrinsic_t = correct_camera_coord(cam_left_extrinsic_r,cam_left_extrinsic_t)

    cam_right_extrinsic_r = np.array([-0.35289060105342246,0.3575524195433318,-0.6372609494402228,0.5843996691352665])   # w,x,y,z
    cam_right_extrinsic_t = np.array([1.3472,5.0299,1.51004]) 
    # cam_right_extrinsic_r,cam_right_extrinsic_t = correct_camera_coord(cam_right_extrinsic_r,cam_right_extrinsic_t) 
    cam_extrinsic_r = {}
    cam_extrinsic_t = {}
    cam_extrinsic_r ['cam_front'] = cam_front_extrinsic_r
    cam_extrinsic_t['cam_front'] = cam_front_extrinsic_t
    cam_extrinsic_r ['cam_left'] = cam_left_extrinsic_r
    cam_extrinsic_t['cam_left'] = cam_left_extrinsic_t
    cam_extrinsic_r ['cam_right'] = cam_right_extrinsic_r
    cam_extrinsic_t['cam_right'] = cam_right_extrinsic_t


    train_kitti_infos = []
    val_kitti_infos = []

    available_scene_names = train_scenes + val_scenes
    for sid, scenes_id in enumerate(available_scene_names):  
        frame_id = scenes_id
        lidar_path =  osp.abspath(osp.join(root_path,"points",str(scenes_id) + ".bin"))
        label_path = osp.abspath(osp.join(root_path,"labels",str(scenes_id)+ ".txt"))
        # print("label_path:  ",label_path)
        # dataset infos
        # lidar2ego_rotation_matrix = Quaternion(w=1, x=0, y=0, z=0)
        lidar2ego_rotation_matrix = np.eye(3).astype(np.float32)
        # lidar2ego_rotation_matrix = np.zeros((1,3)).T  # 欧拉角
        lidar2ego_translation = np.zeros(3).T
        info = {
            "frame_id": frame_id,
            'lidar_path': lidar_path,
            'token': '',
            'sweeps': [],
            'cams': dict(),
            'radars': dict(), 
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation_matrix,
            'timestamp':scenes_id,
        }

        camera_types = [
            "cam_front",
            "cam_left",
            "cam_right",
        ]        

        for cam in camera_types:
            cam_path =  osp.abspath(osp.join(root_path,"camera",cam,str(scenes_id)+ ".jpg"))
            cam_info = {
                'data_path': cam_path,
                'type': cam,
                'sensor2ego_translation': cam_extrinsic_t[cam],
                'sensor2ego_rotation': cam_extrinsic_r[cam],
                'sensor2lidar_translation': cam_extrinsic_t[cam],
                'sensor2lidar_rotation':Quaternion(cam_extrinsic_r[cam]).rotation_matrix,
                'cam_intrinsic':cam_intrinsic,
            }
            info["cams"].update({cam: cam_info})            

        gt_boxes = []
        gt_names = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        # print("+++++++++++++++++: ",label_path)    
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(np.array(line_list[:-1],dtype = np.float32))
            # class_id.index(line_list[-1])   # 这里要注意是存id还是字符
            gt_names.append(class_id[line_list[-1]])  # 字符
            # gt_names.append(class_id.index(line_list[-1]))
        # print("gt_names+++++++++++++++++++++++++++++++:  ",gt_names)
        info["gt_boxes"] = np.array(gt_boxes)
        info["gt_names"] = np.array(gt_names)
        info['gt_velocity'] = np.array([0,0] * len(gt_names)).reshape(-1, 2)  # 没有速度，只是为了跟nuscences对齐
        # 暂无该信息
        # info['num_lidar_pts']  
        info['valid_flag'] = np.array(True * len(gt_names)).reshape(-1)
        info["lidar_path"] = lidar_path
        cal_path = osp.join(root_path,"training","calib",str(scenes_id)+ ".txt")
        if scenes_id   in train_scenes:
            train_kitti_infos.append(info)
        if scenes_id   in val_scenes:
            val_kitti_infos.append(info)
    return   train_kitti_infos ,val_kitti_infos