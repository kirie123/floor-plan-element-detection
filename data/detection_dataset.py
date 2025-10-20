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



class GridMask:
    """GridMask数据增强 - 网格状遮挡"""
    
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=0.7):
        """
        Args:
            use_h (bool): 是否在高度方向使用网格
            use_w (bool): 是否在宽度方向使用网格
            rotate (float): 旋转角度范围
            offset (bool): 是否使用偏移
            ratio (float): 网格密度比例
            mode (int): 模式，0: 随机，1: 固定
            prob (float): 应用概率
        """
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        
    def __call__(self, img, boxes=None, labels=None):
        """
        Args:
            img: 输入图像 [H, W, C]
            boxes: 边界框列表
            labels: 标签列表
        
        Returns:
            增强后的图像和标注
        """
        if random.random() > self.prob:
            return img, boxes, labels
            
        h, w, c = img.shape
        
        # 生成网格掩码
        mask = self.generate_grid_mask(h, w)
        
        # 应用掩码
        if self.mode == 1:
            # 模式1: 直接应用网格
            img = img * mask
        else:
            # 模式0: 随机选择网格单元进行遮挡
            img = self.apply_random_grid_mask(img, mask)
        
        return img, boxes, labels
    
    def generate_grid_mask(self, h, w):
        """生成网格掩码"""
        # 计算网格尺寸
        grid_h = int(h * self.ratio)
        grid_w = int(w * self.ratio)
        
        # 确保网格尺寸合理
        grid_h = max(8, min(grid_h, h // 4))
        grid_w = max(8, min(grid_w, w // 4))
        
        # 创建基础网格
        mask = np.ones((h, w), dtype=np.float32)
        
        if self.use_h:
            # 在高度方向添加网格线
            for i in range(0, h, grid_h):
                mask[i:min(i+2, h), :] = 0  # 2像素宽的网格线
        
        if self.use_w:
            # 在宽度方向添加网格线
            for j in range(0, w, grid_w):
                mask[:, j:min(j+2, w)] = 0
        
        # 随机旋转
        if self.rotate > 0:
            angle = random.uniform(-self.rotate, self.rotate)
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        # 随机偏移
        if self.offset:
            dx = random.randint(-grid_w//2, grid_w//2)
            dy = random.randint(-grid_h//2, grid_h//2)
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            mask = cv2.warpAffine(mask, translation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        return np.expand_dims(mask, axis=2)  # [H, W, 1]
    
    def apply_random_grid_mask(self, img, mask):
        """随机应用网格掩码"""
        h, w, c = img.shape
        
        # 创建随机选择掩码
        random_mask = np.random.random((h, w, 1)) > 0.5
        
        # 组合掩码
        final_mask = np.where(random_mask, mask, 1.0)
        
        # 应用掩码
        img = img * final_mask
        
        return img.astype(np.uint8)


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
        self.use_mosaic = training and True
        self.use_color_aug = training and True
        self.use_mixup = training and False
        self.mixup_alpha = 0.5  # MixUp参数
        self.mosaic_prob = 0.75 # 75%概率使用Mosaic，可以根据效果调整
        self.use_gridmask = training and True
        self.gridmask = GridMask(
            use_h=True,
            use_w=True,
            rotate=15,  # 旋转角度范围
            offset=True,  # 使用偏移
            ratio=0.6,  # 网格密度
            mode=0,  # 随机模式
            prob=0.99  # 应用概率
        )
        # 收集样本
        self.samples = []
        for img_path in self.img_dir.glob("*.jpg"):
            csv_path = self.csv_dir / (img_path.stem + ".csv")
            if csv_path.exists():
                self.samples.append((img_path, csv_path))

        # 使用albumentations进行数据增强（同步处理图像和边界框）
        self.default_transform = self.get_default_trasform()
        if training:
            self.transform = self.get_simple_transform()
            #self.transform = self.get_enhance_transform()
        else:
            self.transform = self.get_default_trasform()

        #self.transform = self.get_default_trasform()
        print(f"✅ 加载 {len(self.samples)} 个样本")



    def apply_gridmask(self, image):
        """应用GridMask增强"""
        if self.training and self.use_gridmask:
            # 确保图像是uint8类型
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            # 应用GridMask
            image, _, _ = self.gridmask(image)
            # 确保返回正确的数据类型
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
        return image

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
    def get_test_transform(self):
        return A.Compose([

                # 4. 弹性变换（模拟图纸变形）
                A.ElasticTransform(
                    alpha=30,
                    sigma=5,
                    alpha_affine=8,
                    p=0.2
                ),
                # 调整尺寸到img_size
                A.Resize(self.img_size, self.img_size, always_apply=True),

                # 标准化并转换为Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # 使用[x_min, y_min, x_max, y_max]格式
                label_fields=['labels']
            ))
    def get_default_trasform(self):
        return A.Compose([
            # 调整尺寸到img_size
            A.Resize(self.img_size, self.img_size, always_apply=True),
            # 标准化并转换为Tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # 使用[x_min, y_min, x_max, y_max]格式
            label_fields=['labels']
        ))
    def get_simple_transform(self):
        """针对建筑平面图的简化增强"""
        return A.Compose([
                # 颜色增强
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.OneOf([
                    # 轻微的非均匀缩放
                    A.Affine(
                        scale={"x": (0.8, 1.8), "y": (0.8, 1.8)},  # 横纵独立缩放
                        translate_percent=(-0.05, 0.05),
                        rotate=(-10, 10),
                        shear=(-3, 3),
                        cval=255,  # 关键：设置填充为白色
                        p=0.5
                    ),
                    # 均匀缩放
                    A.Affine(
                        scale=(0.8, 1.2),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-10, 10),
                        cval=255,  # 关键：设置填充为白色
                        p=0.5
                    )
                ], p=0.3),  # 30%概率应用缩放

                # 弹性变换（模拟图纸变形）
                A.ElasticTransform(
                    alpha=15,
                    sigma=3,
                    alpha_affine=5,
                    fill_value=255,  # 关键：设置填充为白色
                    p=0.2
                ),
                # 几何变换 - 这些会同步调整边界框
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # 轻微平移
                    scale_limit=0.1,  # 轻微缩放
                    rotate_limit=5,  # 小角度旋转
                    border_mode=cv2.BORDER_CONSTANT,  # 使用常数填充
                    value=255,  # 关键：填充白色
                    p=0.5
                ),
                # 专门的颜色增强
                A.Lambda(name='ColorAugmentation',
                         image=lambda image, **kwargs: self.apply_color_augmentation(image),
                         p=0.4),  # 80%概率应用颜色增强

                # 针对建筑平面图的增强
                A.GaussNoise(var_limit=(2.0, 10.0), p=0.2),
                #A.Blur(blur_limit=1, p=0.5),
                # 在标准化之前添加GridMask
                A.Lambda(
                    image=lambda image, **kwargs: self.apply_gridmask(image),
                    p=0.5  # GridMask的应用概率
                ),
                # 调整尺寸到img_size
                A.Resize(self.img_size, self.img_size, always_apply=True),
                # 标准化并转换为Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # 使用[x_min, y_min, x_max, y_max]格式
                label_fields=['labels']
            ))

    def apply_color_augmentation(self, image, **kwargs):
        """应用自定义颜色增强"""
        if not self.training or not self.use_color_aug:
            return image

        # 随机选择颜色增强策略
        strategies = [
            #lambda img: ColorAugmentation.apply_all_color_augmentations(img),
            lambda img: ColorAugmentation.change_background_color(img, prob=0.5),
            lambda img: ColorAugmentation.invert_background(img, prob=0.5),
            lambda img: ColorAugmentation.adjust_line_color(img, prob=0.6),
            lambda img: ColorAugmentation.apply_blueprint(img),
            lambda img: img  # 保持原图
        ]

        strategy = random.choice(strategies)
        augmented_image = strategy(image)

        return augmented_image

    def get_mixup_item(self, idx):
        """MixUp数据增强 - 两张图像分别增强后混合"""
        img_path1, csv_path1 = self.samples[idx]
        if "example" in str(img_path1):
            print(f"⚠️  警告：原始图像包含'example'，跳过MixUp: {img_path1.name}")
            return self.load_single_sample(idx=idx,
                                           img_path=img_path1,
                                           csv_path=csv_path1, apply_transform=True)
        # 随机选择另一张图像
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            idx2 = random.randint(0, len(self.samples) - 1)
            if idx2 == idx:  # 确保不是同一张图像
                continue
            img_path2, csv_path2 = self.samples[idx2]
            # 检查第二张图像路径是否包含"example"
            if "example" not in str(img_path2):
                break  # 找到合适的图像，跳出循环
            attempts += 1
        # 如果尝试多次后仍未找到合适的图像，放弃MixUp
        if attempts >= max_attempts:
            print(f"⚠️  警告：无法找到不包含'example'的图像进行MixUp，返回单张图像")
            return self.load_single_sample(idx=idx,
                                           img_path=img_path1,
                                           csv_path=csv_path1, apply_transform=True)
        # 加载两张图像并分别应用完整的数据增强
        img_path1, csv_path1 = self.samples[idx]
        img_path2, csv_path2 = self.samples[idx2]
        # 分别应用完整的数据增强
        image1, target1 = self.load_single_sample(idx=idx,
                                                  img_path=img_path1,
                                                  csv_path=csv_path1, apply_transform=False)
        image2, target2 = self.load_single_sample(idx=idx,
                                                  img_path=img_path2,
                                                  csv_path=csv_path2, apply_transform=False)
        # 生成混合系数
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # 混合图像（现在图像已经是tensor格式）
        mixed_image = lam * image1 + (1 - lam) * image2
        # 合并标签 - 对于目标检测，我们合并两张图像的所有边界框
        mixed_boxes = torch.cat([target1["boxes"], target2["boxes"]], dim=0)
        mixed_labels = torch.cat([target1["labels"], target2["labels"]], dim=0)
        # 如果没有边界框，创建虚拟框
        if len(mixed_boxes) == 0:
            mixed_boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            mixed_labels = torch.tensor([0], dtype=torch.int64)
        target = {
            "boxes": mixed_boxes,
            "labels": mixed_labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([self.img_size, self.img_size]),
            "mixup_lambda": torch.tensor([lam])  # 记录混合系数
        }

        return mixed_image, target
    def get_valid_mixup_idx(self, idx):
        img_path, csv_path = self.samples[idx]
        if "example" in str(img_path):
            return False
        return True

    def get_cutmix_item_strict(self, idx):
        """CutMix增强 - 严格处理边界框重叠"""
        img_path1, csv_path1 = self.samples[idx]
        idx2 = self.get_valid_mixup_idx(idx)
        img_path2, csv_path2 = self.samples[idx2]

        image1, target1 = self.load_single_sample(idx, img_path1, csv_path1, apply_transform=True)
        image2, target2 = self.load_single_sample(idx2, img_path2, csv_path2, apply_transform=True)

        # 生成裁剪区域
        h, w = image1.shape[1], image1.shape[2]
        lam = np.random.beta(0.3, 0.3)

        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # 执行CutMix
        mixed_image = image1.clone()
        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

        # 处理图像1的边界框 - 裁剪掉与裁剪区域重叠的部分
        boxes1 = target1["boxes"]
        labels1 = target1["labels"]

        if len(boxes1) > 0:
            boxes1_pixel = boxes1 * self.img_size
            cropped_boxes1, cropped_labels1 = self.clip_boxes_outside_region(boxes1_pixel, labels1, x1, y1, x2, y2)
            cropped_boxes1 = cropped_boxes1 / self.img_size
        else:
            cropped_boxes1 = boxes1
            cropped_labels1 = labels1

        # 处理图像2的边界框 - 只保留在裁剪区域内的部分
        boxes2 = target2["boxes"]
        labels2 = target2["labels"]

        if len(boxes2) > 0:
            boxes2_pixel = boxes2 * self.img_size
            clipped_boxes2, clipped_labels2 = self.clip_boxes_inside_region(boxes2_pixel, labels2, x1, y1, x2, y2)

            if len(clipped_boxes2) > 0:
                # 调整坐标到混合图像中的位置
                clipped_boxes2[:, 0] = clipped_boxes2[:, 0] - x1 + x1
                clipped_boxes2[:, 1] = clipped_boxes2[:, 1] - y1 + y1
                clipped_boxes2[:, 2] = clipped_boxes2[:, 2] - x1 + x1
                clipped_boxes2[:, 3] = clipped_boxes2[:, 3] - y1 + y1
                clipped_boxes2 = clipped_boxes2 / self.img_size
        else:
            clipped_boxes2 = boxes2
            clipped_labels2 = labels2

        # 合并边界框
        mixed_boxes = torch.cat([cropped_boxes1, clipped_boxes2], dim=0)
        mixed_labels = torch.cat([cropped_labels1, clipped_labels2], dim=0)

        if len(mixed_boxes) == 0:
            mixed_boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            mixed_labels = torch.tensor([0], dtype=torch.int64)

        target = {
            "boxes": mixed_boxes,
            "labels": mixed_labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([self.img_size, self.img_size]),
            "cutmix_lambda": torch.tensor([lam])
        }

        return mixed_image, target

    def clip_boxes_outside_region(self, boxes, labels, x1, y1, x2, y2, min_area_ratio=0.3):
        """裁剪边界框，移除与指定区域重叠的部分"""
        if len(boxes) == 0:
            return boxes, labels

        clipped_boxes = []
        clipped_labels = []

        for i, box in enumerate(boxes):
            box_x1, box_y1, box_x2, box_y2 = box

            # 计算与裁剪区域的交集
            inter_x1 = max(box_x1, x1)
            inter_y1 = max(box_y1, y1)
            inter_x2 = min(box_x2, x2)
            inter_y2 = min(box_y2, y2)

            # 如果没有交集，保留整个边界框
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                clipped_boxes.append(box.unsqueeze(0))
                clipped_labels.append(labels[i].unsqueeze(0))
                continue

            # 计算原始面积和交集面积
            orig_area = (box_x2 - box_x1) * (box_y2 - box_y1)
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

            # 如果重叠面积很小，保留整个边界框
            if inter_area / orig_area < 0.1:
                clipped_boxes.append(box.unsqueeze(0))
                clipped_labels.append(labels[i].unsqueeze(0))
                continue

            # 如果重叠面积很大，丢弃整个边界框
            if inter_area / orig_area > min_area_ratio:
                continue

            # 否则，裁剪边界框（这里简化处理，实际应该分割为多个边界框）
            # 我们选择保留较大的部分
            left_area = (x1 - box_x1) * (box_y2 - box_y1) if x1 > box_x1 else 0
            right_area = (box_x2 - x2) * (box_y2 - box_y1) if box_x2 > x2 else 0
            top_area = (box_x2 - box_x1) * (y1 - box_y1) if y1 > box_y1 else 0
            bottom_area = (box_x2 - box_x1) * (box_y2 - y2) if box_y2 > y2 else 0

            # 选择面积最大的部分
            areas = [left_area, right_area, top_area, bottom_area]
            max_area_idx = areas.index(max(areas))

            if max_area_idx == 0 and left_area > 0:  # 左侧部分
                new_box = torch.tensor([box_x1, box_y1, x1, box_y2], dtype=torch.float32)
            elif max_area_idx == 1 and right_area > 0:  # 右侧部分
                new_box = torch.tensor([x2, box_y1, box_x2, box_y2], dtype=torch.float32)
            elif max_area_idx == 2 and top_area > 0:  # 顶部部分
                new_box = torch.tensor([box_x1, box_y1, box_x2, y1], dtype=torch.float32)
            elif max_area_idx == 3 and bottom_area > 0:  # 底部部分
                new_box = torch.tensor([box_x1, y2, box_x2, box_y2], dtype=torch.float32)
            else:
                continue  # 没有合适的部分，丢弃

            clipped_boxes.append(new_box.unsqueeze(0))
            clipped_labels.append(labels[i].unsqueeze(0))

        if len(clipped_boxes) > 0:
            return torch.cat(clipped_boxes, dim=0), torch.cat(clipped_labels, dim=0)
        else:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    def clip_boxes_inside_region(self, boxes, labels, x1, y1, x2, y2):
        """裁剪边界框，只保留在指定区域内的部分"""
        if len(boxes) == 0:
            return boxes, labels

        clipped_boxes = []
        clipped_labels = []

        for i, box in enumerate(boxes):
            box_x1, box_y1, box_x2, box_y2 = box

            # 计算与裁剪区域的交集
            inter_x1 = max(box_x1, x1)
            inter_y1 = max(box_y1, y1)
            inter_x2 = min(box_x2, x2)
            inter_y2 = min(box_y2, y2)

            # 如果没有交集，跳过
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue

            # 使用交集作为新的边界框
            new_box = torch.tensor([inter_x1, inter_y1, inter_x2, inter_y2], dtype=torch.float32)

            # 检查新边界框是否有效
            if (inter_x2 - inter_x1) > 2 and (inter_y2 - inter_y1) > 2:  # 最小尺寸限制
                clipped_boxes.append(new_box.unsqueeze(0))
                clipped_labels.append(labels[i].unsqueeze(0))

        if len(clipped_boxes) > 0:
            return torch.cat(clipped_boxes, dim=0), torch.cat(clipped_labels, dim=0)
        else:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    def __len__(self):
        return len(self.samples)

    def load_single_sample(self, idx, img_path, csv_path, apply_transform=True):
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
            if apply_transform:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
            else:
                transformed = self.default_transform(
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

    def __getitem__(self, idx):
        # 如果是训练模式且使用MixUp，随机选择另一张图像
        if self.training and self.use_mixup and random.random() < 0.2:
            return self.get_cutmix_item_strict(idx)

        # 否则返回单张图像（应用完整transform）
        img_path, csv_path = self.samples[idx]
        return self.load_single_sample(idx, img_path, csv_path, apply_transform=True)

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
    dataset.save_augmented_samples(save_augmented_dir=save_augmented_dir, num_samples=1000)