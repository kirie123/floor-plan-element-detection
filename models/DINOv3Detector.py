# detector/dinov3_detector.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import List, Tuple, Optional
from transformers.image_utils import load_image

class DINOv3Detector(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            selected_layers: List[int] = [3, 6, 9, 12],
            hidden_dim: int = 256,
            model_path: str = "F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.selected_layers = selected_layers

        # 加载 DINOv3（冻结）
        self.backbone = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.processor = AutoImageProcessor.from_pretrained(model_path)

        # 融合多层特征 → 统一通道
        in_channels = len(selected_layers) * 1024  # DINOv3-L 输出 dim=1024
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_classes, 1)  # 每个类别一个 heatmap
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            heatmaps: [B, num_classes, H_out, W_out] (e.g., 64x64)
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (layer0, ..., layer12)

        # 提取并重组多层 patch features
        features = []
        for i in self.selected_layers:
            feat = hidden_states[i]  # [B, N, 1024]
            B, N, C = feat.shape
            H = W = int((N - 1) ** 0.5)
            patch_feat = feat[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            features.append(patch_feat)

        # 上采样到统一尺寸（取最大尺寸）
        target_h, target_w = features[0].shape[-2:]
        resized = [
            torch.nn.functional.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            for f in features
        ]
        fused = torch.cat(resized, dim=1)  # [B, 4*1024, H, W]

        # 轻量解码
        heatmaps = self.fusion(fused)  # [B, 4, H, W]
        return heatmaps

    def preprocess(self, image_paths: List[str]) -> torch.Tensor:
        images = [load_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs.pixel_values.to(self.backbone.device)