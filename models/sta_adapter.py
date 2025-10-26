# sta_adapter.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



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
            nn.Conv2d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * inplanes),
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
            self.sta_dims = [conv_inplane * 2, conv_inplane * 4, conv_inplane * 8]  # c2,c3,c4 通道数
        else:
            self.sta_dims = [0, 0, 0]

        # 3. 融合卷积（对应 1/8, 1/16, 1/32）
        dinov3_dim = 1024  # DINOv3-L 的 embed_dim
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(dinov3_dim + self.sta_dims[0], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(dinov3_dim + self.sta_dims[1], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(dinov3_dim + self.sta_dims[2], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
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
        # 假设 patch_size = 16
        h_patches = H_img // 16
        w_patches = W_img // 16
        num_patches = h_patches * w_patches  # 4096 for 1024x1024
        #patch_tokens = patch_tokens.transpose(1, 2).view(bs, -1, h_patches, w_patches)
        dinov3_feats = []
        for feat in selected_blocks:
            # feat: [B, N, 1024], N = (H/16 * W/16) + 1
            patch_feat = feat[:, :num_patches, :]  # 去掉 class token 以及 register token
            feat_map = patch_feat.reshape(B, h_patches, w_patches, -1).permute(0, 3, 1, 2)  # [B, 1024, H, W]
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

class DetectionHead(nn.Module):
    def __init__(self, in_dim=256, num_classes=26):
        super().__init__()
        self.num_classes = num_classes
        # 分类头（每类一个 heatmap）
        self.cls_head = nn.Conv2d(in_dim, num_classes, 1)
        # 回归头（x1,y1,x2,y2 归一化坐标）
        self.reg_head = nn.Conv2d(in_dim, 4, 1)

    def forward(self, features):
        """
        features: (c2, c3, c4) at 1/8, 1/16, 1/32
        We use the 1/16 scale (c3) for simplicity
        """
        c3 = features[1]  # 1/16 scale
        cls_logits = self.cls_head(c3)
        reg_outputs = self.reg_head(c3)
        return cls_logits, reg_outputs

class DINOv3STADetector(nn.Module):
    def __init__(self, num_classes=26,
                 model_path="F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m",
                 use_sta: bool = True,
                 conv_inplane: int = 16,
                 hidden_dim: int = 256
                 ):
        super().__init__()
        #self.backbone = DINOv3WithSTA(model_path=model_path, use_sta=True)
        self.dropout = 0.3
        self.use_sta = use_sta
        # 1. 加载 DINOv3（冻结）
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True
        )
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # 2. STA 模块
        if self.use_sta:
            self.sta = SpatialPriorModule(inplanes=conv_inplane)
            self.sta_dims = [conv_inplane * 2, conv_inplane * 4, conv_inplane * 8]  # c2,c3,c4 通道数
        else:
            self.sta_dims = [0, 0, 0]

        # 3. 融合卷积（对应 1/8, 1/16, 1/32）
        dinov3_dim = 1024  # DINOv3-L 的 embed_dim
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(dinov3_dim + self.sta_dims[0], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(dinov3_dim + self.sta_dims[1], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(dinov3_dim + self.sta_dims[2], hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.BatchNorm2d(hidden_dim)
        ])

        #self.head = DetectionHead(in_dim=256, num_classes=num_classes)
        # 共享特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),  # 添加dropout
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),  # 添加dropout
        )
        # 三个检测头
        self.heatmap_head = nn.Conv2d(256, num_classes, 1)
        self.wh_head = nn.Conv2d(256, 2, 1)
        self.offset_head = nn.Conv2d(256, 2, 1)
        self._init_weights()

    def _init_weights(self):
        # 热力图头 - 使用更合理的初始化
        nn.init.normal_(self.heatmap_head.weight, std=0.01)
        # 计算基于类别不平衡的偏置
        prior_prob = 0.01  # 假设1%的位置有目标
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap_head.bias, bias_value)

        # 宽高和偏移头 - 使用标准初始化
        for m in [self.wh_head, self.offset_head]:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)
        # 共享卷积层使用Kaiming初始化
        for m in self.shared_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, pixel_values):
        #features = self.backbone(pixel_values)
        B, C, H_img, W_img = pixel_values.shape

        # 1. 提取 DINOv3 多层特征（Block5,8,11）
        with torch.no_grad():
            outputs = self.backbone(pixel_values, output_hidden_states=True)
            # 取 Block5,8,11（索引 5,8,11）
            selected_blocks = [outputs.hidden_states[i] for i in [5, 8, 11]]
        # 2. 转换为特征图 [B, D, H, W]
        # 假设 patch_size = 16
        h_patches = H_img // 16
        w_patches = W_img // 16
        num_patches = h_patches * w_patches  # 4096 for 1024x1024
        # patch_tokens = patch_tokens.transpose(1, 2).view(bs, -1, h_patches, w_patches)
        dinov3_feats = []
        for feat in selected_blocks:
            # feat: [B, N, 1024], N = (H/16 * W/16) + 1
            patch_feat = feat[:, :num_patches, :]  # 去掉 class token 以及 register token
            feat_map = patch_feat.reshape(B, h_patches, w_patches, -1).permute(0, 3, 1, 2)  # [B, 1024, H, W]
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
        features = []
        for i in range(3):
            out = self.norms[i](self.fuse_convs[i](fused[i]))
            features.append(out)
        # features: (p2, p3, p4) = ([B,256,128,128], [B,256,64,64], [B,256,32,32])
        p2, p3, p4 = features
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')  # [B,256,64,64]
        p2_down = F.interpolate(p2, scale_factor=0.5, mode='bilinear')
        p3_fused = p2_down + p3 + p4_up  # 或 torch.cat + 1x1 conv（更重）

        shared_features = self.shared_conv(p3_fused)  # [B, 256, 64, 64]
        heatmap = self.heatmap_head(shared_features)  # [B, 4, 64, 64]
        wh = self.wh_head(shared_features)  # [B, 2, 64, 64]
        offset = self.offset_head(shared_features)  # [B, 2, 64, 64]

        return {
            "heatmap": heatmap,
            "wh": wh,
            "offset": offset
        }

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = DINOv3STADetector(num_classes=4).to(device)

    # 测试输入
    dummy_input = torch.randn(2, 3, 1024, 1024).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(dummy_input)


    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")