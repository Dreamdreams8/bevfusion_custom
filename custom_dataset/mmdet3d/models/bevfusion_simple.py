import torch
from mmcv.runner import auto_fp16, force_fp32
from torch.nn import functional as F
import logging
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class BEVFusionSimple(BEVFusion):
    """
    原版的 BevFusion 在 forward 里传入了一堆没用的参数。这里将其删除，只保留有用的参数。
    """

    def extract_camera_features(
            self,
            x,
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        logging.info("ViewTransformer is LSSTransform_Img")
        x = self.encoders["camera"]["vtransform"](
            x,
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix
        )
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points,sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    # 融合模块推理
    # @auto_fp16(apply_to=("img"))
    # def forward(
    #         self,
    #         img,
    #         points,
    #         camera2ego,
    #         camera2lidar,
    #         camera_intrinsics,
    #         img_aug_matrix,
    #         lidar_aug_matrix,
    #         metas,
    #         gt_masks_bev=None,
    #         gt_bboxes_3d=None,
    #         gt_labels_3d=None,
    #         **kwargs,
    # ):
    #     features = []
    #     # for sensor in self.encoders:
    #     for sensor in (
    #             self.encoders if self.training else list(self.encoders.keys())[::-1]
    #     ):
    #         if sensor == "camera":
    #             feature = self.extract_camera_features(
    #                 img,
    #                 camera2lidar,
    #                 camera_intrinsics,
    #                 img_aug_matrix,
    #                 lidar_aug_matrix,
    #             )
    #         elif sensor == "lidar":
    #             # feature = self.extract_lidar_features(points)
    #             feature = self.extract_features(points,sensor)   # change by why
    #         else:
    #             raise ValueError(f"unsupported sensor: {sensor}")
    #         features.append(feature)

    #     if not self.training:
    #         # avoid OOM
    #         features = features[::-1]

    #     if self.fuser is not None:
    #         x = self.fuser(features)
    #     else:
    #         assert len(features) == 1, features
    #         x = features[0]

    #     batch_size = x.shape[0]

    #     x = self.decoder["backbone"](x)
    #     x = self.decoder["neck"](x)
    #     if self.training:
    #         outputs = {}
    #         for type, head in self.heads.items():
    #             if type == "object":
    #                 pred_dict = head(x, metas)
    #                 losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
    #             elif type == "map":
    #                 losses = head(x, gt_masks_bev)
    #             else:
    #                 raise ValueError(f"unsupported head: {type}")
    #             for name, val in losses.items():
    #                 if val.requires_grad:
    #                     outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
    #                 else:
    #                     outputs[f"stats/{type}/{name}"] = val
    #         return outputs
    #     else:
    #         outputs = [{} for _ in range(batch_size)]
    #         for type, head in self.heads.items():
    #             if type == "object":
    #                 pred_dict = head(x, metas)          
    #                 bboxes = head.get_bboxes(pred_dict, metas)
    #                 for k, (boxes, scores, labels) in enumerate(bboxes):                     
    #                     outputs[k].update(
    #                         {
    #                             "boxes_3d": boxes.to("cpu"),
    #                             "scores_3d": scores.cpu(),
    #                             "labels_3d": labels.cpu(),
    #                         }
    #                     )
    #             elif type == "map":
    #                 logits = head(x)
    #                 for k in range(batch_size):
    #                     outputs[k].update(
    #                         {
    #                             "masks_bev": logits[k].cpu(),
    #                             "gt_masks_bev": gt_masks_bev[k].cpu(),
    #                         }
    #                     )
    #             else:
    #                 raise ValueError(f"unsupported head: {type}")
    #         return outputs

    # 单激光推理
    @auto_fp16(apply_to=("img", "points"))
    def forward(
            self,
            points,
            metas,
            img=None,
            camera2ego=None,
            camera2lidar=None,
            camera_intrinsics=None,
            img_aug_matrix=None,
            lidar_aug_matrix=None,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        features = []
        # for sensor in self.encoders:
        for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                if img is None:
                    continue  # 如果没有图像数据，跳过相机特征提取
                feature = self.extract_camera_features(
                    img,
                    camera2lidar,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                )
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)   # change by why
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

