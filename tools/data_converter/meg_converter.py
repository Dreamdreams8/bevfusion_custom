# projects/tools/data_convert/meg_converter.py
import os
from os import path as osp
import mmcv
import numpy as np
import json
from pyquaternion import Quaternion
from mmdet3d.datasets.meg_dataset import MegDataset
import math

class_id = {"Truck" : "truck","Car" : "car"}

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()  
    return [str(int(line)).zfill(len(line)-1) for line in lines]
    # return [int(line) for line in lines]

def get_train_val_scenes(root_path):
    """
    划分训练集和测试集
    """
    imageset_folder = osp.join(root_path, 'ImageSets')
    train_img_ids = _read_imageset_file(str(imageset_folder + '/train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder + '/val.txt'))
    # test_img_ids = _read_imageset_file(str(imageset_folder + '/test.txt'))    
    return  train_img_ids,val_img_ids


def create_meg_infos(
    root_path, info_prefix
):
    train_scenes, val_scenes= get_train_val_scenes(root_path)

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version="meg")

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

    train_kitti_infos = []
    val_kitti_infos = []

    available_scene_names = train_scenes + val_scenes
    for sid, scenes_id in enumerate(available_scene_names):  
        frame_id = scenes_id
        lidar_path =  osp.abspath(osp.join(root_path,"training","velodyne",str(scenes_id) + ".npy"))
        label_path = osp.abspath(osp.join(root_path,"training","label_2",str(scenes_id)+ ".txt"))
        # print("label_path:  ",label_path)
        # dataset infos
        info = {
            "frame_id": frame_id,
            "lidar_path": lidar_path,
            "cams": dict(),
        }

        # camera-obtain 6 image's information per frame
        camera_types = [
            "cam_front",
            "cam_left",
            "cam_right",
        ]        
               
        gt_boxes = []
        gt_names = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(np.array(line_list[:-1],dtype = np.float32))
            # class_id.index(line_list[-1])   # 这里要注意是存id还是字符
            gt_names.append(class_id[line_list[-1]])
        info["gt_boxes"] = np.array(gt_boxes)
        info["gt_names"] = np.array(gt_names)
        info["frame_id"] = frame_id
        info["lidar_path"] = lidar_path
        cal_path = osp.join(root_path,"training","calib",str(scenes_id)+ ".txt")
        # print("cal_path:   ",cal_path)
        with open(cal_path, 'r') as cal_f:
            cal_lines = cal_f.readlines()        
            cam0_intrinsic = np.array( [float(info) for info in cal_lines[0].strip().split(' ')[1:10]]).reshape([3, 3])
            cam1_intrinsic = np.array([float(info) for info in cal_lines[1].strip().split(' ')[1:10]
                                    ]).reshape([3, 3])
            cam2_intrinsic = np.array([float(info) for info in cal_lines[2].strip().split(' ')[1:10]
                                    ]).reshape([3, 3])        
            cam0_extrinsic_r = np.array([float(info) for info in cal_lines[3].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])
            cam1_extrinsic_r = np.array([float(info) for info in cal_lines[4].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])
            cam2_extrinsic_r = np.array([float(info) for info in cal_lines[5].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])      
            cam0_extrinsic_t = np.array([float(info) for info in cal_lines[6].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])
            cam1_extrinsic_t = np.array([float(info) for info in cal_lines[7].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])
            cam2_extrinsic_t = np.array([float(info) for info in cal_lines[8].strip().split(' ')[1:4]
                                    ]).reshape([-1, 3])                              
        intrinsic = dict()
        extrinsic_r = dict()
        extrinsic_t = dict()
        # print("cam0_intrinsic:    ",cam0_intrinsic)
        intrinsic['cam_front']  =  cam0_intrinsic
        intrinsic['cam_left']  =  cam1_intrinsic
        intrinsic['cam_right']  =  cam2_intrinsic
        extrinsic_r['cam_front']  =  cam0_extrinsic_r
        extrinsic_r['cam_left']  =  cam1_extrinsic_r
        extrinsic_r['cam_right']  =  cam2_extrinsic_r
        extrinsic_t['cam_front']  =  cam0_extrinsic_t
        extrinsic_t['cam_left']  =  cam1_extrinsic_t
        extrinsic_t['cam_right']  =  cam2_extrinsic_t                
        for cam in camera_types:
            cam_path =  osp.abspath(osp.join(root_path,"training","image_2",cam,str(scenes_id)+ ".png"))
            cam_info = dict(
                camera_path = cam_path,
                camera_intrinsics = intrinsic[cam],
                camera_extrinsic_r = extrinsic_r[cam],
                camera_extrinsic_t = extrinsic_t[cam]
            )
            info["cams"].update({cam: cam_info})
        if scenes_id   in train_scenes:
            train_kitti_infos.append(info)
        if scenes_id   in val_scenes:
            val_kitti_infos.append(info)
    return   train_kitti_infos ,val_kitti_infos