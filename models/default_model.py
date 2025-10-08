import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


class ResNetDetector(nn.Module):
    def __init__(self, num_classes=4, backbone='resnet50', pretrained=True):
        """
        基于ResNet的目标检测模型
        Args:
            num_classes: 类别数（不包括背景）
            backbone: 主干网络类型 ['resnet18', 'resnet34', 'resnet50', 'resnet101']
            pretrained: 是否使用预训练权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.stride = 16  # ResNet最终下采样倍数

        # 加载预训练ResNet
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_channels = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_channels = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_channels = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 移除最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # FPN特征金字塔网络（可选，但能提升多尺度检测）
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[backbone_channels],
            out_channels=256
        )

        # 检测头 - 热力图分支
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        # 检测头 - 宽高分支
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)  # 输出宽高 (w, h)
        )

        # 检测头 - 偏移分支
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)  # 输出偏移 (dx, dy)
        )

        # 初始化检测头权重
        self._init_weights()

        print(f"✅ 初始化 {backbone} 检测模型，输出特征图尺寸: 1024/{self.stride} = 64x64")

    def _init_weights(self):
        """初始化检测头权重"""
        for m in self.heatmap_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 1024, 1024] 输入图像
        Returns:
            dict: {
                'heatmap': [B, num_classes, 64, 64],
                'wh': [B, 2, 64, 64],
                'offset': [B, 2, 64, 64]
            }
        """
        # 主干网络特征提取
        features = self.backbone(x)  # [B, C, 64, 64]

        # FPN特征融合（简化版，实际可以多尺度）
        fpn_features = self.fpn({'0': features})
        fused_features = fpn_features['0']  # [B, 256, 64, 64]

        # 三个检测头
        heatmap = self.heatmap_head(fused_features)
        wh = self.wh_head(fused_features)
        offset = self.offset_head(fused_features)

        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset
        }


class SimpleResNetDetector(nn.Module):
    """
    简化版ResNet检测器，不使用FPN，更快更轻量
    """

    def __init__(self, num_classes=4, backbone='resnet18', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.stride = 16

        # 加载预训练ResNet
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_channels = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_channels = 512
        else:
            raise ValueError("简化版只支持resnet18/resnet34")

        # 移除最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # 通道调整卷积
        self.channel_adjust = nn.Conv2d(in_channels, 256, 1)

        # 上采样到64x64
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

        # 共享特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 添加dropout
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 添加dropout
        )

        # 三个检测头
        self.heatmap_head = nn.Conv2d(256, num_classes, 1)
        self.wh_head = nn.Conv2d(256, 2, 1)
        self.offset_head = nn.Conv2d(256, 2, 1)

        self._init_weights()
        print(f"✅ 初始化简化版 {backbone} 检测模型")

    def _init_weights(self):
        # for m in [self.heatmap_head, self.wh_head, self.offset_head]:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        # 热力图头 - 鼓励更高激活
        nn.init.normal_(self.heatmap_head.weight, std=0.1)
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.0)

        # 宽高头 - 关键修复：使用更大的初始化来匹配目标范围
        nn.init.normal_(self.wh_head.weight, std=0.01)  # 增加标准差
        if self.wh_head.bias is not None:
            # 根据诊断信息，初始化到目标均值附近
            nn.init.constant_(self.wh_head.bias, 2.0)  # 初始化为3.0，接近目标均值

        # 偏移头
        nn.init.normal_(self.offset_head.weight, std=0.01)
        if self.offset_head.bias is not None:
            nn.init.constant_(self.offset_head.bias, 0.0)

    def forward(self, x):
        features = self.backbone(x)  # [B, 512, 32, 32]
        features = self.channel_adjust(features)  # [B, 256, 32, 32]
        features = self.upsample(features)  # [B, 256, 64, 64]  ← 关键修复！
        shared_features = self.shared_conv(features)  # [B, 256, 64, 64]

        heatmap = self.heatmap_head(shared_features)
        wh = self.wh_head(shared_features)
        offset = self.offset_head(shared_features)

        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset
        }


# 为了兼容你的现有代码，保留DummyModel名称
class DummyModel(SimpleResNetDetector):
    """为了兼容性，将SimpleResNetDetector重命名为DummyModel"""

    def __init__(self, num_classes=4):
        super().__init__(num_classes=num_classes, backbone='resnet18', pretrained=True)


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = DummyModel(num_classes=4).to(device)

    # 测试输入
    dummy_input = torch.randn(2, 3, 1024, 1024).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(dummy_input)

    print("模型输出:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")