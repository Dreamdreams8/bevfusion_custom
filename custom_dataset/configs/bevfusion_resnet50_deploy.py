_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'custom_dataset.mmdet3d.datasets.custom_dataset',
        'custom_dataset.mmdet3d.models.lss_transform_simple',
        'custom_dataset.mmdet3d.models.depth_lss_transform_simple',
        'custom_dataset.mmdet3d.models.bevfusion_simple',
    ],
    allow_failed_imports=False)

# 基础配置参数
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -1.0, 54.0, 54.0, 7.0]
image_size = [256, 704]  # 模型输入图像大小

# NuScenes数据集配置
data_config = {
    'cams': ['cam_front', 'cam_left', 'cam_right'],
    'Ncams': 3,  # 修正相机数量为3个
    'input_size': image_size,  # 输入模型的图像大小 [256, 704]
    'src_size': [1080, 1920],  # 原始图像大小
    'resize': [0.94, 1.06],  # 修正resize范围，避免图像尺寸为0
    'rot': [-5.4, 5.4],
    'flip': True,
    'crop_h': [0.0, 0.0],
    'resize_test': 1.0,  # 修正测试时的resize比例，避免图像尺寸为0
}

root_path = '/home/why/mnt/why/demo/bevfusion/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'MyCustomDataset'
dataset_root = root_path + 'data/20240617-720/'

gt_paste_stop_epoch = -1    # change by why
reduce_beams = 32
load_dim = 4
use_dim = 4
load_augmented = False
max_epochs = 480 


# 数据增强配置
augment2d = {
    'resize': [[0.38, 0.55], [0.48, 0.48]],  # 修正resize范围，与data_config保持一致
    'rotate': [-5.4, 5.4],
    'gridmask': dict(prob=0.0, fixed_prob=True)
}
augment3d = {
    'scale': [0.95, 1.05],
    'rotate': [-0.3925, 0.3925],
    'translate': 0.5
}

# NuScenes目标类别
object_classes = [
    'truck'
]

# 模型配置
model = dict(
    type='BEVFusion',
    encoders=dict(
        camera=dict(
            backbone=dict(
                type='ResNet',
                depth=50,
                out_indices=[1, 2, 3],
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://download.pytorch.org/models/resnet50-0676ba61.pth'
                ),
                norm_cfg=dict(type='BN2d', requires_grad=True),
                norm_eval=False,
                style='pytorch'
            ),
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[512, 1024, 2048],  # ResNet50 out_indices=[1,2,3]对应的通道数
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False)
            ),
            vtransform=dict(
                type='DepthLSSTransform',
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
                add_depth_features=False
            )
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[120000, 160000]
            ),
            backbone=dict(
                type='SparseEncoder',
                in_channels=4,
                sparse_shape=[1440, 1440, 41],
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
                block_type='basicblock'
            )
        )
    ),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256
    ),
    heads=dict(
        object=dict(
            type='TransFusionHead',
            num_proposals=200,
            auxiliary=True,
            in_channels=256,
            hidden_channel=128,
            num_classes=len(object_classes),
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation='relu',
            train_cfg=dict(
                dataset='MyCustomDataset',
                point_cloud_range=point_cloud_range,
                grid_size=[1440, 1440, 41],
                voxel_size=voxel_size,
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # 不包含速度
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
                ),
            ),
            test_cfg=dict(
                dataset="MyCustomDataset",
                grid_size=[1440, 1440, 41],
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2],
                nms_type=None
            ),
            common_heads=dict(
                center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]     # 不包含速度
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=8    # 不包含速度
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0
            ),
            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=0.25
            ),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight=1.0
            )
        )
    ),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)
        ),
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[128, 128],      # 通道设置问题待优化
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True
        )
    ),
)

# 训练配置
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
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
        type='ObjectPaste',
        stop_epoch=gt_paste_stop_epoch,
        db_sampler=dict(
            dataset_root=dataset_root,
            info_path=dataset_root + "custom_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    # car=5,
                    truck=5,
                    # pedestrian=5
                )
            ),
            classes=object_classes,
            sample_groups=dict(
                # car=2,
                truck=3,
            ),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=load_dim,
                use_dim=use_dim,
                reduce_beams=reduce_beams
            )
        )
    ),
    dict(
        type='ImageAug3D',
        final_dim=data_config['input_size'],
        resize_lim=data_config['resize'],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=data_config['rot'],
        rand_flip=data_config['flip'],
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
    # dict(type='GTDepth'),
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
    dict(
        type='Collect3D',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
            'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'
        ]
    ),
    dict(type='GTDepth',keyframe_only=True),
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
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
        final_dim=data_config['input_size'],
        resize_lim=data_config['resize_test'],
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
    # dict(type='GTDepth'),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(
        type='Collect3D',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
            'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'
        ]
    ),
    dict(type='GTDepth',keyframe_only=True),
]

# 数据集配置
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + 'custom_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            data_config=data_config,
            test_mode=False,
            use_valid_flag=False,
            box_type_3d='LiDAR'
        )
    ),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        test_mode=False,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

# 优化器配置
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率配置
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1.0e-3
)

# 运行配置
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=400, pipeline=test_pipeline)

# 检查点配置
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# 其他配置
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "output/0725_camera_lidar_result/epoch_34.pth"
resume_from = "output/0725_camera_lidar_result/epoch_34.pth"
workflow = [('train', 1)]