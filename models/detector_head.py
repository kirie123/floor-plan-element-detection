# detector_head.py
import torch
import torch.nn as nn

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