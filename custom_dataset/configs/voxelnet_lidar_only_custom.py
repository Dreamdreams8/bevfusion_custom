_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'custom_dataset.mmdet3d.datasets.custom_dataset',
        'custom_dataset.mmdet3d.models.bevfusion_simple',
        'custom_dataset.mmdet3d.models.centerhead_without_vel',
    ],
    allow_failed_imports=False)

root_path = '/home/bevfusion/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'MyCustomDataset'
dataset_root = root_path + 'data/custom_data/'

gt_paste_stop_epoch = 15
reduce_beams = 32
load_dim = 4
use_dim = 4
load_augmented = False
max_epochs = 450

# 使用与voxelnet_0p075.yaml相同的参数
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

augment2d = {
    'resize': [[0.38, 0.55], [0.48, 0.48]],
    'rotate': [-5.4, 5.4],
    'gridmask': dict(prob=0.0, fixed_prob=True)
}
augment3d = {
    'scale': [0.95, 1.05],
    'rotate': [-0.3925, 0.3925],
    'translate': 0.5
}

object_classes = [
    'car', 'truck'
]

model = dict(
    type='BEVFusionSimple',
    depth_gt=False,
    encoders=dict(
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=dict(
                type='SparseEncoder',
                in_channels=5,  # 4 + 1 (intensity + xyz + 1)
                sparse_shape=[1440, 1440, 41],  # 根据voxel_size和point_cloud_range计算
                output_channels=128,
                order=['conv', 'norm', 'act'],
                encoder_channels=[
                    [16, 16, 32],
                    [32, 32, 64],
                    [64, 64, 128],
                    [128, 128]
                ],
                encoder_paddings=[
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, [1, 1, 0]],
                    [0, 0]
                ],
                block_type='basicblock',
            ),
        ),
    ),
    fuser=None,
    heads=dict(
        object=dict(
            type='TransFusionHead',
            num_proposals=200,
            auxiliary=True,
            in_channels=384,
            hidden_channel=128,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",
            train_cfg=dict(
                dataset="MyCustomDataset",
                point_cloud_range=point_cloud_range,
                grid_size=[1440, 1440, 41],  # 根据voxelnet_0p075.yaml设置
                voxel_size=voxel_size,
                out_size_factor=4,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                assigner=dict(
                    type="HungarianAssigner3D",
                    iou_calculator=dict(
                        type="BboxOverlaps3D",
                        coordinate="lidar",
                    ),
                    cls_cost=dict(
                        type="FocalLossCost",
                        gamma=2.0,
                        alpha=0.25,
                        weight=0.15,
                    ),
                    reg_cost=dict(
                        type="BBoxBEVL1Cost",
                        weight=0.25,
                    ),
                    iou_cost=dict(
                        type="IoU3DCost",
                        weight=0.25,
                    ),
                )
            ),
            test_cfg=dict(
                dataset="MyCustomDataset",
                point_cloud_range=point_cloud_range,
                grid_size=[1440, 1440, 41],  # 根据voxelnet_0p075.yaml设置
                voxel_size=voxel_size[:2],
                out_size_factor=4,
                pc_range=point_cloud_range[:2],
                nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
                gaussian_overlap=0.1,
                pre_maxsize=1000,
                post_maxsize=83,
                nms_thr=0.2,
                nms_scale=[[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]],
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.05,
            ),
            common_heads=dict(
                center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.05,
                out_size_factor=4,
                voxel_size=voxel_size[:2],
                code_size=8
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0,
            ),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight=1.0,
            ),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        ),
    ),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=128,  # 与SparseEncoder的output_channels匹配
            out_channels=[128, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
        ),
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
        )
    ),
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=augment3d['scale'],
        rot_lim=augment3d['rotate'],
        trans_lim=augment3d['translate'],
        is_train=True
    ),
    dict(type='RandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=object_classes),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=["lidar2ego", "lidar_aug_matrix"])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=["lidar2ego", "lidar_aug_matrix"])
]

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + 'custom_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True
    )
)

# Optimizer
optimizer = dict(type='AdamW', lr=1.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='cyclic')
momentum_config = dict(policy='cyclic')

runner = dict(type='CustomEpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=24, pipeline=test_pipeline)