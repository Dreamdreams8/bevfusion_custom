from mmcv.runner import force_fp32
from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.vtransforms.depth_lss import DepthLSSTransform
import torch
import logging

__all__ = ["DepthLSSTransformSimple"]


@VTRANSFORMS.register_module()
class DepthLSSTransformSimple(DepthLSSTransform):
    """
    原版的 DepthLSSTransform 继承 BaseDepthTransform，BaseDepthTransform 在 forward 里传入了一堆没用的参数。
    这里将其重写，只保留有用的参数，与 BEVFusion 中的 extract_camera_features 方法兼容。
    """

    @force_fp32()
    def forward(
        self,
        img,
        camera2lidar,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        points=None,
        radar=None,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        metas=None,
        depth_loss=False,
        gt_depths=None,
        **kwargs
    ):
        # 检查 camera_intrinsics 是否为 None，如果是，记录错误并尝试从其他参数中获取
        if camera_intrinsics is None:
            logging.error("camera_intrinsics is None in DepthLSSTransformSimple.forward")
            # 记录更多调试信息
            if metas is not None:
                logging.info(f"metas type: {type(metas)}, length: {len(metas) if isinstance(metas, list) else 'not a list'}")
                if isinstance(metas, list) and len(metas) > 0:
                    logging.info(f"metas[0] keys: {metas[0].keys() if isinstance(metas[0], dict) else 'not a dict'}")
            
            # 尝试从 metas 中获取 camera_intrinsics
            if metas is not None and isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], dict):
                if 'camera_intrinsics' in metas[0]:
                    logging.info("Retrieving camera_intrinsics from metas['camera_intrinsics']")
                    try:
                        camera_intrinsics = [meta['camera_intrinsics'] for meta in metas]
                        camera_intrinsics = torch.stack([torch.tensor(ci, dtype=torch.float32, device=img.device) 
                                                       if not isinstance(ci, torch.Tensor) else ci for ci in camera_intrinsics], dim=0)
                        logging.info(f"Successfully retrieved camera_intrinsics from metas, shape: {camera_intrinsics.shape}")
                    except Exception as e:
                        logging.error(f"Error processing camera_intrinsics from metas: {str(e)}")
                        # 继续尝试其他方法
                        camera_intrinsics = None
                
                elif 'cam_intrinsic' in metas[0]:
                    logging.info("Retrieving camera_intrinsics from metas['cam_intrinsic']")
                    try:
                        # 构建完整的相机内参矩阵
                        batch_size = img.shape[0]
                        num_cams = img.shape[1]
                        camera_intrinsics = []
                        
                        for b in range(batch_size):
                            cam_intrinsics_list = []
                            for c in range(num_cams):
                                # 获取相机内参
                                if 'cam_intrinsic' in metas[b] and c < len(metas[b]['cam_intrinsic']):
                                    cam_intrinsic = metas[b]['cam_intrinsic'][c]
                                    # 构建 4x4 矩阵
                                    intrinsic_mat = torch.eye(4, dtype=torch.float32, device=img.device)
                                    intrinsic_mat[:3, :3] = torch.tensor(cam_intrinsic, dtype=torch.float32, device=img.device)
                                    cam_intrinsics_list.append(intrinsic_mat)
                                else:
                                    # 如果相机数量不匹配，使用默认值
                                    logging.warning(f"Camera intrinsic not found for camera {c} in batch {b}")
                                    intrinsic_mat = torch.eye(4, dtype=torch.float32, device=img.device)
                                    cam_intrinsics_list.append(intrinsic_mat)
                            camera_intrinsics.append(torch.stack(cam_intrinsics_list))
                        
                        camera_intrinsics = torch.stack(camera_intrinsics)
                        logging.info(f"Successfully constructed camera_intrinsics from cam_intrinsic, shape: {camera_intrinsics.shape}")
                    except Exception as e:
                        logging.error(f"Error processing cam_intrinsic from metas: {str(e)}")
                        # 继续尝试其他方法
                        camera_intrinsics = None
                
                # 尝试从其他可能的字段获取相机内参
                elif any(key in metas[0] for key in ['lidar2image', 'lidar2camera', 'camera2ego']):
                    logging.info("Attempting to derive camera_intrinsics from transformation matrices")
                    try:
                        # 这里可以添加从其他变换矩阵推导相机内参的逻辑
                        # 例如，从 lidar2image 和 lidar2camera 可以推导出相机内参
                        # 这需要根据具体的数据格式和变换关系来实现
                        pass
                    except Exception as e:
                        logging.error(f"Error deriving camera_intrinsics from transformation matrices: {str(e)}")
            
            # 如果仍然无法获取，抛出错误
            if camera_intrinsics is None:
                error_msg = "camera_intrinsics cannot be None and could not be retrieved from metas. "
                if metas is not None and isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], dict):
                    error_msg += f"Available keys in metas[0]: {list(metas[0].keys())}"
                raise ValueError(error_msg + " Please check the data pipeline.")
            
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        return x
