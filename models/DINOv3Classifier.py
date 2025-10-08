# classifier/dinov3_classifier.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import List, Optional
import os
from transformers.image_utils import load_image



# classifier/label_map.py
CLASS_NAMES = [
    "wall1", "wall2", "wall3", "wall4", "wall5", "wall6", "wall7", "wall8", "wall9", "wall10",
    "wall11", "wall12", "wall13", "wall14", "wall15", "wall16", "wall17", "wall18", "wall19", "wall20",
    "wall21", "wall22", "wall23",
    "door", "window", "column"
]

LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

class DINOv3Classifier(nn.Module):
    def __init__(
            self,
            num_classes: int = 26,
            selected_layers: List[int] = [6, 9, 12],  # 选中中间+深层
            hidden_dim: int = 512,
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

        # 分类头：融合多层特征 → GAP → MLP
        in_dim = len(selected_layers) * 1024  # 3 * 1024 = 3072
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取多层 patch features 并融合"""
        with torch.no_grad():
            outputs = self.backbone(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (layer0, ..., layer12)

        # 提取选中层的 patch tokens（去掉 class token）
        fused_list = []
        for i in self.selected_layers:
            feat = hidden_states[i]  # [B, N, 1024]
            patch_feat = feat[:, 1:, :]  # [B, N-1, 1024]
            # 全局平均池化（GAP）→ [B, 1024]
            gap_feat = patch_feat.mean(dim=1)
            fused_list.append(gap_feat)

        # 拼接 → [B, 3072]
        fused = torch.cat(fused_list, dim=1)
        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values)
        logits = self.classifier(features)
        return logits

    def preprocess(self, image_paths: List[str]) -> torch.Tensor:
        images = [load_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs.pixel_values.to(self.backbone.device)