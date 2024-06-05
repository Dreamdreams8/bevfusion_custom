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
        dataset_class_name,
        root_path,        
        info_prefix, 
        info_path=None,    
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None,
        with_mask=False):

    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=root_path, ann_file=info_path
    )
    dataset_cfg.update(
        use_valid_flag=True,
        pipeline=[
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
            ),
            # dict(
            #     type="LoadPointsFromMultiSweeps",
            #     sweeps_num=10,
            #     use_dim=[0, 1, 2, 3, 4],
            #     pad_empty_sweeps=True,
            #     remove_close=True,
            # ),
            dict(
                type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
            ),
        ],
    )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(root_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(root_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    
    all_db_infos = dict()
    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        # print("input_dict:  ",input_dict)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        # print("annos:  ",annos)
        image_idx = example["sample_idx"]
        # print("image_idx:  ",image_idx)
        points = example["points"].tensor.numpy()
        # print("pointsshape:  ",points.shape)
        # break        
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        # print("gt_boxes_3d:  ",gt_boxes_3d)
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
        # print("num_obj:  ",num_obj)
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
                if with_mask:
                    db_info.update({"box2d_camera": gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

