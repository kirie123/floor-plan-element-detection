# detector.py
import os

import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from typing import List, Dict, Tuple
import json

from data.wall_classifier import WallClassifier, get_transforms
from models.default_model import DummyModel
from utils.shape import Shape, ImageAnnotation, save_image_annotations_to_jsonl
from validate import decode_outputs


class IntegratedDetector:
    """集成了目标检测和墙分类的检测器"""

    def __init__(self,
                 detection_model_path: str = "training_output/best_model.pth",
                 classification_model_path: str = "best_wall_classifier.pth",
                 device: str = None):

        self.device = torch.device(device if device else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

        # 加载目标检测模型
        self.detection_model = DummyModel(num_classes=4).to(self.device)
        self.detection_model.load_state_dict(torch.load(detection_model_path, map_location=self.device))
        self.detection_model.eval()

        # 加载墙分类模型
        self.classification_model = WallClassifier(num_classes=23, backbone='efficientnet_b0')
        self.classification_model.load_state_dict(torch.load(classification_model_path, map_location=self.device))
        self.classification_model.to(self.device)
        self.classification_model.eval()

        # 类别名称
        self.detection_class_names = ["wall", "door", "window", "column"]
        self.wall_class_names = [f"WALL{i}" for i in range(1, 24)]

        # 图像变换
        self.detection_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classification_transform, _ = get_transforms(224)

        print(f"✅ 集成检测器初始化完成")
        print(f"   目标检测模型: {detection_model_path}")
        print(f"   墙分类模型: {classification_model_path}")
        print(f"   设备: {self.device}")


    def slice_image(self, image_path: str, slice_size: int = 1024, overlap: float = 0.1) -> List[Dict]:
        """将大图切片成小图用于检测"""
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size

        step = int(slice_size * (1 - overlap))
        slices = []
        slice_idx = 0

        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                # 计算裁剪区域
                x2 = x + slice_size
                y2 = y + slice_size
                crop_x2 = min(x2, img_w)
                crop_y2 = min(y2, img_h)

                # 裁剪图像
                patch = image.crop((x, y, crop_x2, crop_y2))

                # 如果尺寸不足，进行padding
                pad_right = slice_size - patch.width
                pad_bottom = slice_size - patch.height

                if pad_right > 0 or pad_bottom > 0:
                    padded_patch = Image.new('RGB', (slice_size, slice_size), (254, 254, 254))
                    padded_patch.paste(patch, (0, 0))
                    patch = padded_patch

                # 保存切片信息
                slices.append({
                    'slice_idx': slice_idx,
                    'patch': patch,
                    'original_coords': (x, y, crop_x2, crop_y2),
                    'slice_coords': (0, 0, slice_size, slice_size),
                    'padding': (pad_right, pad_bottom)
                })

                slice_idx += 1
                x += step
            y += step

        print(f"✅ 图像切片完成: {len(slices)} 个切片")
        return slices

    def detect_in_slice(self, patch, confidence_threshold=0.3) -> List[Dict]:
        """在单个切片中进行目标检测"""
        # 预处理
        input_tensor = self.detection_transform(patch).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.detection_model(input_tensor)

        # 解码输出
        detections = decode_outputs(outputs, confidence_threshold)
        batch_detections = detections[0]  # 取第一个batch

        results = []
        for detection in batch_detections:
            xmin, ymin, xmax, ymax, confidence, class_id = detection
            class_name = self.detection_class_names[class_id]

            results.append({
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'class': class_name,
                'confidence': confidence,
                'class_id': class_id
            })

        return results

    def classify_wall(self, image_patch) -> Dict:
        """对墙图像块进行分类"""
        # 预处理
        input_tensor = self.classification_transform(image_patch).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.classification_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_prob, pred_class = torch.max(probabilities, 1)

        # 获取原始分类结果
        original_class_idx = pred_class.item()
        class_name = self.wall_class_names[original_class_idx]
        # 重映射特定类别到 WALL1
        if class_name in ["WALL13", "WALL20", "WALL23"]:
            class_name = "WALL1"
            original_class_idx = 0  # WALL1 的索引
        return {
            'class_name': class_name,
            'class_idx': original_class_idx,
            'confidence': pred_prob.item()
        }

    def merge_detections(self, slices_detections: List[Dict], original_size: Tuple) -> List[Dict]:
        """合并所有切片的检测结果到原图坐标"""
        original_w, original_h = original_size
        all_detections = []

        for slice_info, detections in slices_detections:
            slice_x, slice_y, crop_x2, crop_y2 = slice_info['original_coords']

            for detection in detections:
                # 转换坐标到原图
                xmin_orig = detection['xmin'] + slice_x
                ymin_orig = detection['ymin'] + slice_y
                xmax_orig = detection['xmax'] + slice_x
                ymax_orig = detection['ymax'] + slice_y

                # 确保坐标在图像范围内
                xmin_orig = max(0, min(xmin_orig, original_w))
                ymin_orig = max(0, min(ymin_orig, original_h))
                xmax_orig = max(0, min(xmax_orig, original_w))
                ymax_orig = max(0, min(ymax_orig, original_h))

                # 只保留有效的检测
                if xmax_orig > xmin_orig and ymax_orig > ymin_orig:
                    detection_orig = detection.copy()
                    detection_orig.update({
                        'xmin': int(xmin_orig),
                        'ymin': int(ymin_orig),
                        'xmax': int(xmax_orig),
                        'ymax': int(ymax_orig)
                    })
                    all_detections.append(detection_orig)

        # 简单的非极大值抑制去重
        filtered_detections = self.category_aware_nms(all_detections, iou_threshold=0.4)
        #filtered_detections = all_detections
        return filtered_detections

    def category_aware_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """按类别感知的非极大值抑制，对column类别进行合并"""
        if not detections:
            return []

        # 按类别分组
        detections_by_class = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append(det)

        filtered_detections = []

        for class_id, class_detections in detections_by_class.items():
            class_name = self.detection_class_names[class_id]

            if class_name == "column":
                # 对column类别进行合并
                merged_detections = self.merge_columns(class_detections, iou_threshold)
                filtered_detections.extend(merged_detections)
            else:
                # 对其他类别进行标准NMS
                nms_detections = self.nms_for_class(class_detections, iou_threshold)
                filtered_detections.extend(nms_detections)

        return filtered_detections

    def nms_for_class(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """对单个类别进行标准非极大值抑制"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            # 取置信度最高的检测
            best = detections.pop(0)
            keep.append(best)

            # 移除与当前检测重叠度高的检测
            detections = [det for det in detections
                          if self.calculate_iou(best, det) < iou_threshold]

        return keep

    def merge_columns(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """合并重叠的column检测框"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        merged_groups = []

        while detections:
            # 取当前最高置信度的检测作为种子
            seed = detections.pop(0)
            current_group = [seed]

            # 寻找与种子重叠的所有检测
            remaining = []
            for det in detections:
                iou = self.calculate_iou(seed, det)
                if iou > iou_threshold:
                    current_group.append(det)
                else:
                    remaining.append(det)

            # 合并当前组中的所有检测框
            if len(current_group) > 1:
                # 计算合并后的包围框（取所有框的并集）
                xmin = min(det['xmin'] for det in current_group)
                ymin = min(det['ymin'] for det in current_group)
                xmax = max(det['xmax'] for det in current_group)
                ymax = max(det['ymax'] for det in current_group)

                # 取组内最高置信度
                confidence = max(det['confidence'] for det in current_group)

                # 创建合并后的检测结果
                merged_detection = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'class': 'column',
                    'confidence': confidence,
                    'class_id': current_group[0]['class_id']  # 保持相同的class_id
                }
                merged_groups.append(merged_detection)
            else:
                # 如果没有重叠的，直接保留原检测
                merged_groups.append(seed)

            detections = remaining

        return merged_groups

    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """计算两个包围框的交并比"""
        # 计算交集区域
        x1 = max(box1['xmin'], box2['xmin'])
        y1 = max(box1['ymin'], box2['ymin'])
        x2 = min(box1['xmax'], box2['xmax'])
        y2 = min(box1['ymax'], box2['ymax'])

        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算各自面积
        area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
        area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

        # 计算并集面积
        union = area1 + area2 - intersection

        # 避免除以零
        if union == 0:
            return 0.0

        return intersection / union

    def detect(self, image_path: str, confidence_threshold: float = 0.3) -> List[Dict]:
        """完整的检测流程"""
        print(f"🔍 开始处理图像: {image_path}")

        # 1. 加载原图并获取尺寸
        original_image = Image.open(image_path).convert('RGB')
        original_w, original_h = original_image.size
        print(f"   原图尺寸: {original_w} x {original_h}")

        # 2. 切片
        slices = self.slice_image(image_path, slice_size=1024, overlap=0.1)

        # 3. 对每个切片进行目标检测
        slices_detections = []
        for slice_info in slices:
            patch = slice_info['patch']
            slice_detections = self.detect_in_slice(patch, confidence_threshold)
            slices_detections.append((slice_info, slice_detections))


        # 4. 合并检测结果到原图坐标
        merged_detections = self.merge_detections(slices_detections, (original_w, original_h))
        print(f"   合并后检测到 {len(merged_detections)} 个目标")

        # 5. 对墙目标进行分类
        final_detections = []
        wall_count = 0

        for detection in merged_detections:
            if detection['class'] == 'wall':
                # 从原图中裁剪墙区域
                xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']

                # 确保坐标有效
                if xmax > xmin and ymax > ymin:
                    wall_patch = original_image.crop((xmin, ymin, xmax, ymax))

                    # 分类墙类型
                    classification_result = self.classify_wall(wall_patch)

                    # 更新检测结果
                    detection['class'] = classification_result['class_name'].lower()
                    #detection['class'] = 'wall1'
                    detection['wall_confidence'] = classification_result['confidence']
                    wall_count += 1

            final_detections.append(detection)

        print(f"   对 {wall_count} 个墙目标进行了分类")
        print(f"✅ 处理完成: 总共 {len(final_detections)} 个检测目标")

        return final_detections


# 为了兼容你的现有代码，保留DummyDetector名称
class DummyDetector(IntegratedDetector):
    """为了兼容性，将IntegratedDetector重命名为DummyDetector"""

    def __init__(self,
                 detection_model_path: str = "training_output/best_model.pth",
                 classification_model_path: str = "best_wall_classifier.pth"):
        super().__init__(detection_model_path, classification_model_path)

    def detect(self, image_path: str) -> List:
        """
        检测接口，返回Shape对象列表
        为了兼容你的服务框架
        """
        detections = super().detect(image_path, confidence_threshold=0.5)

        # 转换为Shape对象 - 根据Shape类定义修正格式
        shapes = []
        for detection in detections:
            shape = Shape(
                label=detection['class'],
                points=[
                    [float(detection['xmin']), float(detection['ymin'])],  # [[x_min, y_min]
                    [float(detection['xmax']), float(detection['ymax'])]  # [x_max, y_max]]
                ],
                confidence_score=float(detection.get('wall_confidence', detection['confidence'])),
                shape_type="rectangle",
                group_id=None,
                flags={}
            )
            shapes.append(shape)

        return shapes

if __name__ == '__main__':
    from pathlib import Path
    detector = DummyDetector()

    # 1. 在这里加载您的模型
    print("模型加载中...")
    # 2. 读取输入数据
    # 假设输入是一个名为 data.csv 的文件
    input_file = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\验证集图片"
    output_path = "output"
    print(f"正在读取输入文件: {input_file}")
    # 获取待处理图片
    drawing_files = []
    img_dir = Path(input_file)
    for img_path in img_dir.glob("*.jpg"):
        drawing_files.append(img_path)

    all_annotations = []
    for drawing_file in drawing_files:
        drawing_path = drawing_file
        img = cv2.imread(str(drawing_path))  # 确保使用字符串路径
        h, w = (img.shape[0], img.shape[1]) if img is not None else (3118, 4414)
        # 3. 进行模型预测
        # 调用检测器
        print(f"正在处理图像: {drawing_file.name}")
        shapes = detector.detect(str(drawing_path))  # 确保使用字符串路径
        print(f"模型预测完成！检测到 {len(shapes)} 个目标")

        # 还需要从shapes中过滤掉图例中没有的类别

        # 创建 ImageAnnotation 对象
        annotation = ImageAnnotation(
            image_path=str(drawing_file.name),  # 转换为字符串
            image_height=h,
            image_width=w,
            shapes=shapes
        )
        all_annotations.append(annotation)

    # 保存为 .jsonl
    output_file = os.path.join(output_path, 'results.jsonl')
    save_image_annotations_to_jsonl(all_annotations, output_file)