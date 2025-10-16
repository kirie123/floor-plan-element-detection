
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def convert_class(cls_name):
    if 'WALL'in cls_name:
        return 'wall'
    return cls_name.lower()


class DetectionDataset(Dataset):
    def __init__(self, img_dir: str, csv_dir: str, class_names: list, training=True, img_size=1024):
        """
        Args:
            img_dir: 切片图像目录
            csv_dir: 对应CSV目录
            class_names: 类别列表
            training: 训练/测试模式
        """
        self.img_dir = Path(img_dir)
        self.csv_dir = Path(csv_dir)
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.training = training
        self.img_size = img_size

        # 收集样本
        self.samples = []
        for img_path in self.img_dir.glob("*.jpg"):
            csv_path = self.csv_dir / (img_path.stem + ".csv")
            if csv_path.exists():
                self.samples.append((img_path, csv_path))

        # 使用albumentations进行数据增强（同步处理图像和边界框）
        if training:
            self.transform = A.Compose([
                # 颜色增强
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

                # 几何变换 - 这些会同步调整边界框
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # 轻微平移
                    scale_limit=0.1,  # 轻微缩放
                    rotate_limit=5,  # 小角度旋转
                    p=0.5
                ),

                # 针对建筑平面图的增强
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.2),

                # 标准化并转换为Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # 使用[x_min, y_min, x_max, y_max]格式
                label_fields=['labels']
            ))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))

        print(f"✅ 加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]

        # 使用OpenCV加载图像（albumentations需要numpy数组）
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = image.shape[:2]  # (H, W)

        # 加载标注
        df = pd.read_csv(csv_path)
        boxes = []
        labels = []

        for _, row in df.iterrows():
            # 使用绝对坐标（不是归一化的）
            # albumentations会在变换后自动调整边界框
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            # 确保边界框有效
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[convert_class(row['class'])])

        if len(boxes) == 0:
            # 如果没有有效边界框，创建空目标
            boxes = [[0, 0, 1, 1]]  # 虚拟小框
            labels = [0]  # 默认类别

        # 应用数据增强（同步处理图像和边界框）
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']  # 已经是Tensor [3, H, W]
            boxes = transformed['bboxes']
            labels = transformed['labels']

        # 转换边界框为Tensor并归一化
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # 归一化到 [0, 1]
            boxes[:, 0] = boxes[:, 0] / self.img_size  # xmin
            boxes[:, 1] = boxes[:, 1] / self.img_size  # ymin
            boxes[:, 2] = boxes[:, 2] / self.img_size  # xmax
            boxes[:, 3] = boxes[:, 3] / self.img_size  # ymax
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor(orig_size)
        }

        return image, target