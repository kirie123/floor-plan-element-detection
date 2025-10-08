# wall_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import pandas as pd



class WallClassifier(nn.Module):
    """带有Dropout的墙类型分类器"""

    def __init__(self, num_classes=23, backbone='resnet18', pretrained=True, dropout_rate=0.5):
        super().__init__()

        self.dropout_rate = dropout_rate

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            # 移除原来的全连接层
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

            # 添加带dropout的分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # 移除原分类器

            # 添加带dropout的分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),  # EfficientNet使用SiLU
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        print(f"✅ 初始化带Dropout的墙分类器 - {backbone}, 类别数: {num_classes}, Dropout: {dropout_rate}")

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

class WallClassificationDataset(Dataset):
    """墙分类数据集"""

    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # 获取所有类别
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # 收集样本
        for class_dir in class_dirs:
            class_name = class_dir.name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

        print(f"✅ 加载 {len(self.samples)} 个墙分类样本, 类别数: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224):
    """获取数据增强变换"""
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform