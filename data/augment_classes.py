# enhance_dataset.py
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import random
import shutil


def enhance_dataset(input_dir, output_dir, min_samples_per_class=15):
    """
    增强数据集，确保每个类别至少有min_samples_per_class个样本
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 数据集增强 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标最小样本数: {min_samples_per_class}")

    # 定义多种增强变换
    augment_transforms = [
        # 基础增强
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        T.RandomRotation(degrees=10),
        T.RandomRotation(degrees=12),

        # 颜色增强
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
        T.ColorJitter(contrast=0.3),

        # 几何变换
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        T.RandomAffine(degrees=0, scale=(1.1, 0.9)),
    ]

    # 统计原始数据
    class_stats = {}
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            samples = list(class_dir.glob("*.jpg"))
            class_stats[class_name] = len(samples)

    print("\n原始数据统计:")
    for cls, count in sorted(class_stats.items()):
        print(f"  {cls}: {count} 样本")

    # 处理每个类别
    total_augmented = 0
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        original_samples = list(class_dir.glob("*.jpg"))
        current_count = len(original_samples)

        # 创建输出目录
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(exist_ok=True)

        # 复制所有原始样本
        for img_path in original_samples:
            shutil.copy2(img_path, output_class_dir / img_path.name)

        # 如果需要增强
        if current_count < min_samples_per_class:
            needed = min_samples_per_class - current_count
            print(f"\n增强类别 {class_name}: {current_count} -> {min_samples_per_class} (+{needed})")

            augmented_count = 0
            attempt_count = 0
            max_attempts = needed * 3  # 防止无限循环

            while augmented_count < needed and attempt_count < max_attempts:
                attempt_count += 1

                # 随机选择原始图像和增强方式
                original_path = random.choice(original_samples)
                transform = random.choice(augment_transforms)

                try:
                    img = Image.open(original_path)
                    augmented_img = transform(img)

                    # 保存增强图像
                    aug_name = f"aug_{augmented_count:03d}_{original_path.stem}.jpg"
                    aug_path = output_class_dir / aug_name
                    augmented_img.save(aug_path)

                    augmented_count += 1
                    total_augmented += 1

                except Exception as e:
                    continue

    # 统计增强后数据
    enhanced_stats = {}
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            samples = list(class_dir.glob("*.jpg"))
            enhanced_stats[class_name] = len(samples)

    print(f"\n=== 增强完成 ===")
    print(f"总共生成 {total_augmented} 个增强样本")
    print(f"\n增强后数据统计:")
    for cls, count in sorted(enhanced_stats.items()):
        print(f"  {cls}: {count} 样本")

    return output_dir


def verify_dataset_split(dataset_dir, train_ratio=0.8):
    """
    验证数据集分割，确保每个类别在训练集和验证集中都有样本
    """
    dataset_dir = Path(dataset_dir)
    class_stats = {}

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            samples = list(class_dir.glob("*.jpg"))
            class_stats[class_name] = len(samples)

    print("\n=== 数据集分割验证 ===")
    print("假设训练集比例:", train_ratio)

    all_classes_have_samples = True
    for cls, count in class_stats.items():
        expected_train = max(1, int(count * train_ratio))
        expected_val = count - expected_train

        print(f"{cls}: 总共{count} -> 训练约{expected_train}, 验证约{expected_val}")

        if expected_val == 0:
            print(f"  ⚠️  警告: {cls} 可能在验证集中没有样本!")
            all_classes_have_samples = False

    if all_classes_have_samples:
        print("✅ 所有类别在训练集和验证集中都应该有样本")
    else:
        print("❌ 有些类别可能在验证集中缺失样本")

    return all_classes_have_samples


if __name__ == "__main__":
    input_dir = "classification_data/wall_classification"
    output_dir = "classification_data_enhanced/wall_classification"

    # 增强数据集
    enhanced_dir = enhance_dataset(input_dir, output_dir, min_samples_per_class=10)

    # 验证分割
    verify_dataset_split(enhanced_dir)