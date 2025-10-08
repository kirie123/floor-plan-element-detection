# sta_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detector_head import DetectionHead


class SpatialPriorModule(nn.Module):
    """简化版 STA，移除 SyncBatchNorm，适配单卡训练"""

    def __init__(self, inplanes=16):
        super().__init__()
        # 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
        )
        # 1/16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )
        # 1/32
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32
        return c2, c3, c4


class DINOv3WithSTA(nn.Module):
    def __init__(
            self,
            model_path: str = "F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m",
            use_sta: bool = True,
            conv_inplane: int = 16,
            hidden_dim: int = 256
    ):
        super().__init__()
        self.use_sta = use_sta

        # 1. 加载 DINOv3（冻结）
        from transformers import AutoModel
        self.dinov3 = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True
        )
        for param in self.dinov3.parameters():
            param.requires_grad = False

        # 2. STA 模块
        if use_sta:
            self.sta = SpatialPriorModule(inplanes=conv_inplane)
            sta_dims = [conv_inplane * 2, conv_inplane * 4, conv_inplane * 4]  # c2,c3,c4 通道数
        else:
            sta_dims = [0, 0, 0]

        # 3. 融合卷积（对应 1/8, 1/16, 1/32）
        dinov3_dim = 1024  # DINOv3-L 的 embed_dim
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(dinov3_dim + sta_dims[0], hidden_dim, 1),
            nn.Conv2d(dinov3_dim + sta_dims[1], hidden_dim, 1),
            nn.Conv2d(dinov3_dim + sta_dims[2], hidden_dim, 1)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim)
        ])

    def forward(self, pixel_values: torch.Tensor):
        """
        Returns:
            features: (c2, c3, c4) at scales 1/8, 1/16, 1/32
        """
        B, C, H_img, W_img = pixel_values.shape

        # 1. 提取 DINOv3 多层特征（Block5,8,11）
        with torch.no_grad():
            outputs = self.dinov3(pixel_values, output_hidden_states=True)
            # 取 Block5,8,11（索引 5,8,11）
            selected_blocks = [outputs.hidden_states[i] for i in [5, 8, 11]]

        # 2. 转换为特征图 [B, D, H, W]
        dinov3_feats = []
        for feat in selected_blocks:
            # feat: [B, N, 1024], N = (H/16 * W/16) + 1
            patch_feat = feat[:, 1:, :]  # 去掉 class token
            H = W = int(patch_feat.shape[1] ** 0.5)
            feat_map = patch_feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, 1024, H, W]
            dinov3_feats.append(feat_map)

        # 3. 上采样到多尺度（1/8, 1/16, 1/32）
        target_sizes = [
            (H_img // 8, W_img // 8),
            (H_img // 16, W_img // 16),
            (H_img // 32, W_img // 32)
        ]
        resized_dinov3 = [
            F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            for feat, size in zip(dinov3_feats, target_sizes)
        ]

        # 4. 融合 STA 细节（如果启用）
        if self.use_sta:
            sta_feats = self.sta(pixel_values)  # (c2, c3, c4)
            fused = [
                torch.cat([dinov3_f, sta_f], dim=1)
                for dinov3_f, sta_f in zip(resized_dinov3, sta_feats)
            ]
        else:
            fused = resized_dinov3

        # 5. 融合卷积 + Norm
        outputs = []
        for i in range(3):
            out = self.norms[i](self.fuse_convs[i](fused[i]))
            outputs.append(out)

        return outputs[0], outputs[1], outputs[2]  # c2(1/8), c3(1/16), c4(1/32)


class DINOv3STADetector(nn.Module):
    def __init__(self, num_classes=26, model_path="F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m"):
        super().__init__()
        self.backbone = DINOv3WithSTA(model_path=model_path, use_sta=True)
        self.head = DetectionHead(in_dim=256, num_classes=num_classes)

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        return self.head(features)
