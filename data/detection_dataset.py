
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from pathlib import Path
import torch
from torch.utils.data import Dataset

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

        # 数据增强（仅在训练时）
        if training:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        print(f"✅ 加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # (W, H)
        image = self.transform(image)  # [3, 1024, 1024]

        # 加载标注
        df = pd.read_csv(csv_path)
        boxes = []
        labels = []
        for _, row in df.iterrows():
            #boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            # 归一化到 [0, 1]
            xmin_norm = row['xmin'] / self.img_size
            ymin_norm = row['ymin'] / self.img_size
            xmax_norm = row['xmax'] / self.img_size
            ymax_norm = row['ymax'] / self.img_size

            boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
            labels.append(self.class_to_idx[convert_class(row['class'])])

        boxes = torch.tensor(boxes, dtype=torch.float32)  # [N, 4]
        labels = torch.tensor(labels, dtype=torch.int64)  # [N]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor(orig_size)
        }

        return image, target