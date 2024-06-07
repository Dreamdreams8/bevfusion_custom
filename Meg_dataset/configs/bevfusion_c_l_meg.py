_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'projects.Meg_dataset.mmdet3d.datasets.meg_dataset',
        'projects.Meg_dataset.mmdet3d.models.lss_transform_simple',
        'projects.Meg_dataset.mmdet3d.models.bevfusion_simple',
        'projects.Meg_dataset.mmdet3d.models.pillar_encoder',
        'projects.Meg_dataset.mmdet3d.models.centerhead_without_vel',
    ],
    allow_failed_imports=False)


data_config = {
    'cams': ['cam_front_left_120', 'cam_front_70_left', 'cam_front_right_120',
             'cam_back_left_120', 'cam_back_120', 'cam_back_right_120']
}

root_path = '/home/daxiongpro/code/bevfusion/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'MegDataset'
dataset_root = root_path + 'data/meg_data/new_custom_data/'

gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = False
max_epochs = 24

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
image_size = [256, 704]

augment2d = {
    'resize': [[0.38, 0.55], [0.48, 0.48]],
    'rotate': [-5.4, 5.4],
    'gridmask': dict(prob=0.0, fixed_prob=True)
}
augment3d = {
    'scale': [0.95, 1.05],
    'rotate': [-0.3925, 0.3925],
    'translate': 0.0
}

object_classes = [
    'car', 'truck', 'bus', 'motorcycle', 'pedestrian', 'bicycle', 'cyclist',
    'tricycle'
]

model = dict(
    type='BEVFusionSimple',
    depth_gt=False,
    encoders=dict(
        lidar=dict(
            sparse_shape=[512, 512, 1],
            voxelize_reduce=False,
            voxelize=dict(
                max_num_points=20,
                point_cloud_range=point_cloud_range,
                voxel_size=[0.2, 0.2, 8],
                max_voxels=[30000, 60000],
            ),
            backbone=dict(
                type='PointPillarsEncoderWithConv4x',
                pts_conv_encoder=dict(
                    type='ConvTest',
                ),
                pts_voxel_encoder=dict(
                    type='PillarFeatureNet',
                    in_channels=5,
                    feat_channels=[64, 64],
                    with_distance=False,
                    point_cloud_range=point_cloud_range,
                    voxel_size=[0.2, 0.2, 8],
                    norm_cfg=dict(
                        type='BN1d',
                        eps=1.0e-3,
                        momentum=0.01,
                    ),
                ),
                pts_middle_encoder=dict(
                    type='PointPillarsScatter',
                    in_channels=64,
                    output_shape=[512, 512],
                    sparse_shape=[512, 512, 1],
                ),
            ),
        ),
        camera=dict(
            backbone=dict(
                pretrained=pretrained_path + 'resnet50-0676ba61.pth',
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=[2, 3],
                frozen_stages=-1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                with_cp=True,
                style='pytorch'
            ),
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[1024, 2048],
                out_channels=512,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False)
            ),
            vtransform=dict(
                type='LSSTransformSimple',
                in_channels=512,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 16, image_size[1] // 16],
                xbound=[-51.2, 51.2, 0.8],
                ybound=[-51.2, 51.2, 0.8],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 1.0],
                downsample=1
            )
        )
    ),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256,
    ),
    heads=dict(
        object=dict(
            type='CenterHeadWithoutVel',
            in_channels=256,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[1024, 1024, 1],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ),
            test_cfg=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                # voxel_size=voxel_size[:2],
                nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
                nms_scale=[[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]]
            ),
            tasks=[
                ["car"], ["truck"], ["bus"], ["motorcycle", "bicycle", "cyclist"],
                ["pedestrian"], ["tricycle"]
            ],
            common_heads=dict(
                reg=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]
            ),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=7
            ),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3
            ),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True
        ),
        map=None
    ),
    decoder=dict(
        backbone=dict(
            type='GeneralizedResNet',
            in_channels=256,
            blocks=[[2, 128, 2], [2, 256, 2], [2, 512, 1]],
        ),
        neck=dict(
            type='LSSFPN', in_indices=[-1, 0], in_channels=[512, 128],
            out_channels=256, scale_factor=2
        )
    ),
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
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
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=augment2d['resize'][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=True,
        is_train=True
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
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(
        type='GridMask', use_h=True, use_w=True, max_epoch=max_epochs,
        rotate=1, offset=False, ratio=0.5, mode=1,
        prob=augment2d['gridmask']['prob'],
        fixed_prob=augment2d['gridmask']['fixed_prob']
    ),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
                    'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
         )
]


test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
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
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
                    'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
         )
]


input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + 'meg_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            data_config=data_config,
            test_mode=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "meg_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        box_type_3d='LiDAR',
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "meg_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        box_type_3d='LiDAR',
        test_mode=True
    )
)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1.0e-5)
runner = dict(type='CustomEpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=24, pipeline=test_pipeline)