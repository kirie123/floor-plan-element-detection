import os

import pandas as pd
import torchvision.transforms as T
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps



def convert_class(cls_name):
    if 'WALL'in cls_name:
        return 'wall'
    return cls_name.lower()


class ColorAugmentation:
    """专门针对CAD图纸的颜色增强"""

    @staticmethod
    def invert_background(image, prob=0.5):
        """背景反转：白色<->黑色"""
        if random.random() < prob:
            # 反转图像
            inverted = cv2.bitwise_not(image)
            return inverted
        return image

    @staticmethod
    def change_background_color(image, prob=0.3):
        """改变背景颜色"""
        if random.random() < prob:
            h, w, c = image.shape
            # 随机背景颜色（常见CAD背景色）
            bg_colors = [
                [255, 255, 255],  # 白色
                [0, 0, 0],  # 黑色
                [240, 240, 240],  # 浅灰色
                [30, 30, 30],  # 深灰色
                [255, 250, 240],  # 米白色
                [245, 245, 245],  # 烟白色
            ]

            bg_color = random.choice(bg_colors)
            # 创建新背景
            new_bg = np.full_like(image, bg_color)

            # 基于阈值分离前景和背景
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 自适应阈值，适应不同对比度
            if random.random() < 0.5:
                _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

            # 对mask进行形态学操作，消除噪声
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 应用mask
            foreground = cv2.bitwise_and(image, image, mask=mask)
            background = cv2.bitwise_and(new_bg, new_bg, mask=cv2.bitwise_not(mask))
            result = cv2.add(foreground, background)

            return result
        return image

    @staticmethod
    def adjust_line_color(image, prob=0.4):
        """调整线条颜色"""
        if random.random() < prob:
            # 随机线条颜色变化
            line_colors = [
                lambda img: img,  # 保持原色
                lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB),  # 灰度
                lambda img: ColorAugmentation.apply_sepia(img),  # 棕褐色调
                lambda img: ColorAugmentation.apply_blueprint(img),  # 蓝图风格
                lambda img: ColorAugmentation.apply_red_lines(img),  # 红色线条
                lambda img: ColorAugmentation.apply_green_lines(img),  # 绿色线条
            ]

            augment_func = random.choice(line_colors)
            return augment_func(image)
        return image

    @staticmethod
    def apply_sepia(image):
        """应用棕褐色调"""
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(image, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    @staticmethod
    def apply_blueprint(image):
        """蓝图风格（蓝色背景，白色/黄色线条）"""
        h, w, c = image.shape
        # 创建蓝色背景
        blueprint_bg = np.array([30, 60, 120])  # 深蓝色

        # 转换为灰度并二值化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 创建蓝图风格
        result = np.full_like(image, blueprint_bg)

        # 线条可以是白色或黄色
        line_color = random.choice([np.array([255, 255, 255]), np.array([255, 255, 0])])
        result[binary > 127] = line_color

        return result

    @staticmethod
    def apply_red_lines(image):
        """红色线条风格"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        result = np.full_like(image, [255, 255, 255])  # 白色背景
        result[binary > 127] = [255, 0, 0]  # 红色线条

        return result

    @staticmethod
    def apply_green_lines(image):
        """绿色线条风格"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        result = np.full_like(image, [0, 0, 0])  # 黑色背景
        result[binary > 127] = [0, 255, 0]  # 绿色线条

        return result

    @staticmethod
    def random_color_shift(image, prob=0.3):
        """随机颜色偏移"""
        if random.random() < prob:
            h, w, c = image.shape
            # 对每个通道应用随机偏移
            for i in range(3):
                shift = random.randint(-20, 20)
                image[:, :, i] = np.clip(image[:, :, i].astype(np.int32) + shift, 0, 255)
        return image

    @staticmethod
    def adjust_contrast_brightness(image, prob=0.5):
        """调整对比度和亮度"""
        if random.random() < prob:
            # 随机对比度 (0.7 ~ 1.5)
            contrast = random.uniform(0.7, 1.5)
            # 随机亮度 (-30 ~ 30)
            brightness = random.randint(-30, 30)

            image = image.astype(np.float32)
            image = image * contrast + brightness
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def apply_all_color_augmentations(image):
        """应用所有颜色增强（按顺序）"""
        aug_image = image.copy()

        # 1. 背景颜色变化
        aug_image = ColorAugmentation.change_background_color(aug_image, prob=0.3)

        # 2. 背景反转
        aug_image = ColorAugmentation.invert_background(aug_image, prob=0.3)

        # 3. 线条颜色调整
        aug_image = ColorAugmentation.adjust_line_color(aug_image, prob=0.4)

        # 4. 对比度亮度调整
        aug_image = ColorAugmentation.adjust_contrast_brightness(aug_image, prob=0.5)

        # 5. 随机颜色偏移
        aug_image = ColorAugmentation.random_color_shift(aug_image, prob=0.3)

        return aug_image

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
        self.use_color_aug = True
        # 收集样本
        self.samples = []
        for img_path in self.img_dir.glob("*.jpg"):
            csv_path = self.csv_dir / (img_path.stem + ".csv")
            if csv_path.exists():
                self.samples.append((img_path, csv_path))

        # 使用albumentations进行数据增强（同步处理图像和边界框）
        if training:
            self.transform = self.get_simple_transform()

            #self.transform = self.get_enhance_transform()
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))

        print(f"✅ 加载 {len(self.samples)} 个样本")

    def resize_image_and_boxes(self, image, boxes, target_size):
        """
        手动调整图像尺寸和对应的边界框

        Args:
            image: 原始图像 (H, W, C)
            boxes: 边界框列表 [[xmin, ymin, xmax, ymax], ...]
            target_size: 目标尺寸

        Returns:
            resized_image: 调整后的图像
            resized_boxes: 调整后的边界框
        """
        # 获取原始尺寸
        orig_height, orig_width = image.shape[:2]

        # 调整图像尺寸
        resized_image = cv2.resize(image, (target_size, target_size))

        # 计算缩放比例
        scale_x = target_size / orig_width
        scale_y = target_size / orig_height

        # 调整边界框坐标
        resized_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # 应用缩放
            new_xmin = xmin * scale_x
            new_ymin = ymin * scale_y
            new_xmax = xmax * scale_x
            new_ymax = ymax * scale_y

            # 确保边界框在图像范围内
            new_xmin = max(0, min(new_xmin, target_size - 1))
            new_ymin = max(0, min(new_ymin, target_size - 1))
            new_xmax = max(0, min(new_xmax, target_size - 1))
            new_ymax = max(0, min(new_ymax, target_size - 1))

            # 只保留有效的边界框
            if new_xmin < new_xmax and new_ymin < new_ymax:
                resized_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

        return resized_image, resized_boxes

    def get_simple_transform(self):
        """针对建筑平面图的简化增强"""
        return A.Compose([
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
                # 专门的颜色增强
                A.Lambda(name='ColorAugmentation',
                         image=lambda image, **kwargs: self.apply_color_augmentation(image),
                         p=0.8),  # 80%概率应用颜色增强
                # 针对建筑平面图的增强
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.2),

                # 调整尺寸到img_size
                A.Resize(self.img_size, self.img_size, always_apply=True),

                # 标准化并转换为Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # 使用[x_min, y_min, x_max, y_max]格式
                label_fields=['labels']
            ))

    def get_enhance_transform(self):
        return A.Compose([
                # 基础几何变换
                A.OneOf([
                    A.ShiftScaleRotate(
                        shift_limit=0.1,  # 增加平移幅度
                        scale_limit=0.2,  # 增加缩放范围
                        rotate_limit=15,  # 增加旋转角度
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,  # 用黑色填充
                        p=0.7
                    ),
                    A.Affine(
                        scale=(0.8, 1.2),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-15, 15),
                        shear=(-5, 5),  # 添加剪切变换
                        p=0.3
                    )
                ], p=0.8),

                # 镜像翻转
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),

                # 透视变换 - 模拟不同视角
                A.Perspective(
                    scale=(0.05, 0.1),  # 轻微透视
                    keep_size=True,
                    p=0.3
                ),

                # 颜色增强 - 更丰富的颜色变化
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.3,  # 增加亮度变化
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=1.0
                    ),
                    A.RGBShift(
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=1.0
                    )
                ], p=0.7),

                # 噪声和模糊 - 模拟不同图像质量
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                ], p=0.4),

                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.3),

                # 针对建筑图纸的特殊增强
                A.OneOf([
                    # 模拟图纸质量变化
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    ),
                ], p=0.4),

                # 模拟遮挡和缺失
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=0.5
                    ),
                    A.GridDropout(
                        unit_size_min=16,
                        unit_size_max=64,
                        holes_number_x=4,
                        holes_number_y=4,
                        shift_x=0,
                        shift_y=0,
                        random_offset=True,
                        fill_value=0,
                        p=0.3
                    )
                ], p=0.3),

                # 图像质量退化
                A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),

                # 标准化
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_area=1,  # 确保小目标不被过滤
                min_visibility=0.1  # 允许部分可见的目标
            ))

    def apply_color_augmentation(self, image, **kwargs):
        """应用自定义颜色增强"""
        if not self.training or not self.use_color_aug:
            return image

        # 随机选择颜色增强策略
        strategies = [
            lambda img: ColorAugmentation.apply_all_color_augmentations(img),
            lambda img: ColorAugmentation.change_background_color(img, prob=0.5),
            lambda img: ColorAugmentation.invert_background(img, prob=0.5),
            lambda img: ColorAugmentation.adjust_line_color(img, prob=0.6),
            lambda img: ColorAugmentation.apply_blueprint(img),
            lambda img: img  # 保持原图
        ]

        strategy = random.choice(strategies)
        augmented_image = strategy(image)

        return augmented_image
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

    def tensor_to_numpy(self, tensor_image):
        """
        将标准化后的tensor图像转换回numpy格式用于保存
        """
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = tensor_image * std + mean

        # 转换为numpy并调整范围到0-255
        image = image.mul(255).byte()
        image = image.numpy().transpose(1, 2, 0)  # CHW -> HWC

        return image

    def get_absolute_boxes(self, normalized_boxes):
        """
        将归一化的边界框坐标转换回绝对坐标
        """
        if len(normalized_boxes) == 0:
            return []

        boxes_abs = normalized_boxes.clone()
        boxes_abs[:, 0] = boxes_abs[:, 0] * self.img_size  # xmin
        boxes_abs[:, 1] = boxes_abs[:, 1] * self.img_size  # ymin
        boxes_abs[:, 2] = boxes_abs[:, 2] * self.img_size  # xmax
        boxes_abs[:, 3] = boxes_abs[:, 3] * self.img_size  # ymax

        return boxes_abs.numpy().astype(int)

    def save_boxes_to_csv(self, boxes, labels, csv_path, img_filename):
        """
        将边界框和标签保存为CSV文件
        """
        data = []
        for i, (box, label_idx) in enumerate(zip(boxes, labels)):
            class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else "unknown"
            data.append({
                'filename': img_filename,
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'class': class_name
            })

        df = pd.DataFrame(data)
        # 如果没有有效的边界框，创建一个空的DataFrame但保持列结构
        if df.empty:
            df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
        df.to_csv(csv_path, index=False)

    def save_augmented_samples(self,save_augmented_dir,  num_samples=50, start_idx=0):
        """
        保存经过数据增强后的样本到指定文件夹

        Args:
            num_samples: 要保存的样本数量
            start_idx: 起始索引
        """
        if not save_augmented_dir:
            print("❌ 未设置保存目录，请初始化时指定 save_augmented_dir")
            return

        saved_count = 0
        idx = start_idx

        while saved_count < num_samples and idx < len(self.samples):
            try:
                # 获取增强后的样本
                image_tensor, target = self.__getitem__(idx)

                # 转换图像为可保存格式
                image_np = self.tensor_to_numpy(image_tensor)

                # 获取边界框的绝对坐标
                boxes_abs = self.get_absolute_boxes(target["boxes"])
                labels = target["labels"].numpy()

                # 生成文件名
                orig_name = self.samples[idx][0].stem
                img_filename = f"aug_{orig_name}_{saved_count:04d}.jpg"
                csv_filename = f"aug_{orig_name}_{saved_count:04d}.csv"
                img_path = os.path.join(save_augmented_dir, f"{img_filename}")
                csv_path = os.path.join(save_augmented_dir, f"{csv_filename}")

                # 保存图像
                cv2.imwrite(str(img_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

                # 保存CSV标注
                self.save_boxes_to_csv(boxes_abs, labels, csv_path, img_filename)

                saved_count += 1
                print(f"✅ 保存增强样本 {saved_count}/{num_samples}: {img_filename}")

            except Exception as e:
                print(f"❌ 保存样本 {idx} 失败: {e}")

            idx += 1

        print(f"🎉 成功保存 {saved_count} 个增强样本到 {save_augmented_dir}")

if __name__ == "__main__":
    class_names = ["wall", "door", "window", "column"]
    dataset = DetectionDataset(
        img_dir="cvt_images",
        csv_dir="cvt_images",
        class_names=class_names,
        training=True
    )
    save_augmented_dir = "vis_images"
    dataset.save_augmented_samples(save_augmented_dir=save_augmented_dir, num_samples=100)