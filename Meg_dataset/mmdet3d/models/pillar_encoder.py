from typing import Any, Dict
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_backbone
from mmdet.models import BACKBONES
from mmdet3d.models.backbones.pillar_encoder import PointPillarsEncoder

__all__ = ["PointPillarsEncoderWithConv4x", "ConvTest"]


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


@BACKBONES.register_module()
class ConvTest(nn.Module):
    def __init__(self):
        super(ConvTest, self).__init__()

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


@BACKBONES.register_module()
class PointPillarsEncoderWithConv4x(PointPillarsEncoder):
    """
    PointPillarsEncoder 的基础上增加 4 倍下采样，
    使得与 camera 模块输出的 feature map 匹配。
    """
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_conv_encoder: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(pts_voxel_encoder, pts_middle_encoder, **kwargs,)
        self.pts_conv_encoder = build_backbone(pts_conv_encoder)

    def forward(self, feats, coords, batch_size, sizes):
        x = super().forward(feats, coords, batch_size, sizes)
        x = self.pts_conv_encoder(x)

        return x