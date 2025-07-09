_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'custom_dataset.mmdet3d.datasets.custom_dataset',
        'custom_dataset.mmdet3d.models.lss_transform_simple',
        'custom_dataset.mmdet3d.models.bevfusion_simple',
        'custom_dataset.mmdet3d.models.pillar_encoder',
        'custom_dataset.mmdet3d.models.centerhead_without_vel',
    ],
    allow_failed_imports=False)


data_config = {
    'cams': ['cam_front', 'cam_left', 'cam_right']
}

root_path = '/home/bevfusion_custom/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'MyCustomDataset'
dataset_root = root_path + 'data/20240617-720/'

gt_paste_stop_epoch = -1    
reduce_beams = 32
load_dim = 4
use_dim = 4
load_augmented = False
max_epochs = 24         

point_cloud_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 7.0]
voxel_size = [0.2, 0.2, 8]
image_size = [256, 704]   # 370\1224

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
    'truck'
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
                type='PointPillarsEncoder',
                pts_voxel_encoder=dict(
                    type='PillarFeatureNet',
                    in_channels=4,
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
    ),
    fuser=None,
    heads=dict(
        object=dict(
            type='TransFusionHead',
            num_proposals=200,
            auxiliary=True,
            in_channels=384 ,  
            hidden_channel=128,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=8 ,   
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",   
            train_cfg=dict(
                dataset="MyCustomDataset",   
                point_cloud_range=point_cloud_range,
                grid_size=[512, 512, 1],           # 注意此处代表的意义
                voxel_size=voxel_size,
                out_size_factor=4,                  # 注意此处代表的意义
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                assigner=dict(
                    type ="HungarianAssigner3D",
                    iou_calculator = dict(
                        type="BboxOverlaps3D",
                        coordinate="lidar",
                    ),
                    cls_cost=dict(
                        type="FocalLossCost",
                        gamma=2.0,
                        alpha= 0.25,
                        weight= 0.15,                    
                    ),
                    reg_cost =dict(
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
                grid_size=[512, 512, 1],
                voxel_size=voxel_size[:2],
                out_size_factor=4,
                pc_range=point_cloud_range[:2],
                nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
                gaussian_overlap=0.1,      
                # pre_max_size=1000,
                pre_maxsize=1000,
                post_maxsize=83,
                nms_thr=0.2,
                nms_scale=[[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]],       
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.05,                         
            ),
            # tasks=[
            #     ["car"], ["truck"]
            # ],
            common_heads=dict(
                center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],     
                # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.05,      
                out_size_factor=4,
                voxel_size=voxel_size[:2],
                code_size=8   
            ),
            loss_cls=dict(
                type='FocalLoss', 
                use_sigmoid=True,
                gamma=2.0,
                alpha= 0.25,
                reduction='mean',
                loss_weight=1.0,
                ),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight= 1.0,      
            ),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        ),
        # map=None
    ),
    decoder=dict(
        backbone=dict(
            type='SECOND',
            in_channels=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
        ),
        neck=dict(
            type='SECONDFPN',     # 此处待确认
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
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
        type='GridMask', use_h=True, use_w=True, max_epoch=max_epochs,
        rotate=1, offset=False, ratio=0.5, mode=1,
        prob=augment2d['gridmask']['prob'],
        fixed_prob=augment2d['gridmask']['fixed_prob']
    ),
    dict(type="ImageNormalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    # dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
    #      meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
    #                 'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
    #      )
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego','camera2lidar', 'lidar2image', 
                    'lidar2camera',"lidar2ego","lidar_aug_matrix","img_aug_matrix"]
         ),         
    dict(type="GTDepth",keyframe_only=True)
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
        # resize_lim=augment2d['resize'][1],
        resize_lim=augment2d['resize'][0],             
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
    # dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
    #      meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
    #                 'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
    #      )
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego','camera2lidar', 'lidar2image', 
                    'lidar2camera',"lidar2ego","lidar_aug_matrix","img_aug_matrix"]
         ),                  
    dict(type="GTDepth",keyframe_only=True)     
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
            # data_config=data_config,
            test_mode=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        # data_config=data_config,
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
        # data_config=data_config,
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
evaluation = dict(interval=100, pipeline=test_pipeline)