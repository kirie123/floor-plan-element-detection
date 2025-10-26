# create_wall_classification_dataset.py
import pandas as pd
from PIL import Image
import os
import shutil
from pathlib import Path
import numpy as np


def create_wall_classification_dataset(img_dir, csv_dir, output_dir, min_size=1):
    """
    从目标检测数据集中提取墙的子图像用于分类任务

    Args:
        img_dir: 原始图像目录
        csv_dir: CSV标注目录
        output_dir: 输出目录
        min_size: 最小子图尺寸，避免太小的墙片段
    """
    img_dir = Path(img_dir)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    # 检查目录是否存在
    if not img_dir.exists():
        raise ValueError(f"图像目录不存在: {img_dir}")
    if not csv_dir.exists():
        raise ValueError(f"CSV目录不存在: {csv_dir}")
    # 创建输出目录
    wall_output_dir = output_dir / "wall_classification"
    wall_output_dir.mkdir(parents=True, exist_ok=True)

    # 统计每个类别的样本数
    class_counts = {}

    # 处理每个图像文件
    for img_path in img_dir.glob("*.jpg"):
        csv_path = csv_dir / (img_path.stem + ".csv")

        if not csv_path.exists():
            continue

        print(f"处理图像: {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # 读取CSV标注
        df = pd.read_csv(csv_path)

        # 筛选出墙的标注
        wall_annotations = df[df['class'].str.contains('WALL', case=False, na=False)]

        for _, row in wall_annotations.iterrows():
            cls_name = row['class']

            # 跳过已经统一的'wall'标签，只处理具体的WALL类型
            if cls_name.lower() == 'wall':
                continue

            # 获取边界框
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            # 检查尺寸是否足够大
            if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
                continue

            # 确保边界框在图像范围内
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img_width, int(xmax))
            ymax = min(img_height, int(ymax))

            # 提取子图像
            try:
                patch = image.crop((xmin, ymin, xmax, ymax))

                # 创建类别目录
                class_dir = wall_output_dir / cls_name
                class_dir.mkdir(exist_ok=True)

                # 保存子图像
                patch_filename = f"{img_path.stem}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
                patch_path = class_dir / patch_filename
                patch.save(patch_path)

                # 更新统计
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            except Exception as e:
                print(f"处理 {img_path.name} 的边界框 ({xmin},{ymin},{xmax},{ymax}) 时出错: {e}")
                continue

    # 输出统计信息
    print("\n=== 数据集统计 ===")
    total_samples = 0
    for cls_name, count in class_counts.items():
        print(f"{cls_name}: {count} 个样本")
        total_samples += count

    print(f"总样本数: {total_samples}")
    print(f"输出目录: {wall_output_dir}")

    return wall_output_dir, class_counts


def analyze_dataset_balance(dataset_dir):
    """分析数据集的类别平衡情况"""
    dataset_dir = Path(dataset_dir)
    class_counts = {}

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            num_samples = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = num_samples

    print("\n=== 数据集平衡分析 ===")
    for cls_name, count in class_counts.items():
        print(f"{cls_name}: {count} 样本")

    return class_counts


if __name__ == "__main__":
    # 配置路径
    img_dir = "cvt2_images"  # 你的目标检测图像目录
    csv_dir = "cvt2_images"  # 你的CSV标注目录
    output_dir = "classification_data"

    # 创建数据集
    dataset_dir, stats = create_wall_classification_dataset(img_dir, csv_dir, output_dir)

    # 分析数据集平衡
    analyze_dataset_balance(dataset_dir)