# projects/Meg_dataset/tools/data_convert/create_gt_database.py
import pickle
from os import path as osp
import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
import torch
import os
from pathlib import Path

def create_groundtruth_database( 
        data_path,
        root_path,        
        info_prefix, 
        info_path=None,    
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None):

        if database_save_path is None:
            database_save_path = osp.join(root_path, f"{info_prefix}_gt_database")
        if db_info_save_path is None:
            db_info_save_path = osp.join(root_path, f"{info_prefix}_dbinfos_train.pkl")

        if not os.path.exists(database_save_path):  
            os.makedirs(database_save_path)
        all_db_infos = {}
        info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        # print("info_path:  ",infos["infos"])
        for k in range(len(infos["infos"])):
            info = infos["infos"][k]
            sample_idx = info['lidar_path']
            frame_id = info["frame_id"]
            # print(" info['lidar_path']:    ----------------", info['lidar_path'])
            assert os.path.exists(os.path.realpath(str(sample_idx)))
            
            points = np.load(sample_idx)
            names = info["gt_names"]
            # difficulty = annos['difficulty']    # add by why
            gt_boxes = info["gt_boxes"] 

            num_obj = gt_boxes.shape[0]
            point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
            for i in range(num_obj):         
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = osp.join(database_save_path ,filename)
                gt_points = points[point_indices[:, i]]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                      
                    db_path = str(Path(filepath).relative_to(root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_groundtruth_database2(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,

    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,

):

    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )

    if dataset_class_name == 'MegDataset':
        dataset_cfg.update(
            use_valid_flag=False,
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=4,
                    use_dim=4,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True
                ),
            ],
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
