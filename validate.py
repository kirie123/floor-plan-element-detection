import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from loss import detection_loss
from models.default_model import DummyModel


def decode_outputs(outputs, confidence_threshold=0.3, nms_threshold=0.5, stride=16, img_size=1024):
    """
    将模型输出解码为边界框（处理归一化坐标）
    Args:
        outputs: 模型输出字典 {'heatmap', 'wh', 'offset'}
        confidence_threshold: 置信度阈值
        nms_threshold: 非极大值抑制阈值
        stride: 下采样倍数
        img_size: 图像尺寸（用于反归一化）
    Returns:
        list: 每个元素的格式为 [xmin, ymin, xmax, ymax, confidence, class_id]
    """
    heatmap = torch.sigmoid(outputs['heatmap'])
    wh = outputs['wh']
    offset = outputs['offset']
    # 在 decode_outputs 函数开头添加
    #print(f"Max heatmap value: {heatmap.max().item():.4f}")
    #print(f"Mean heatmap value: {heatmap.mean().item():.4f}")
    batch_size, num_classes, H, W = heatmap.shape
    detections = []

    for b in range(batch_size):
        batch_detections = []

        for cls_id in range(num_classes):
            # 获取当前类别的热力图
            cls_heatmap = heatmap[b, cls_id]

            # 找到峰值点（使用最大池化实现简单的NMS）
            pooled = torch.nn.functional.max_pool2d(
                cls_heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1
            )[0]

            # 找到局部最大值
            peak_mask = (cls_heatmap == pooled) & (cls_heatmap > confidence_threshold)
            peak_indices = torch.nonzero(peak_mask)

            for idx in peak_indices:
                y, x = idx[0].item(), idx[1].item()
                confidence = cls_heatmap[y, x].item()

                # 解码边界框 - 注意：现在wh是在特征图尺度上的
                w_feat = torch.exp(wh[b, 0, y, x]).item()
                h_feat = torch.exp(wh[b, 1, y, x]).item()
                dx = offset[b, 0, y, x].item()
                dy = offset[b, 1, y, x].item()
                # 计算中心点坐标（特征图尺度）
                center_x_feat = x + dx
                center_y_feat = y + dy
                # 转换到原图尺度
                center_x = center_x_feat * stride
                center_y = center_y_feat * stride
                w = w_feat * stride
                h = h_feat * stride

                # 计算边界框坐标（像素坐标）
                xmin = max(0, center_x - w / 2)
                ymin = max(0, center_y - h / 2)
                xmax = min(img_size, center_x + w / 2)
                ymax = min(img_size, center_y + h / 2)

                batch_detections.append([xmin, ymin, xmax, ymax, confidence, cls_id])

        # 应用NMS
        if batch_detections:
            batch_detections = apply_nms(batch_detections, nms_threshold)
        detections.append(batch_detections)

    return detections


def apply_nms(detections, iou_threshold=0.5):
    """
    应用非极大值抑制
    """
    if not detections:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []

    while detections:
        current = detections.pop(0)
        keep.append(current)

        detections = [det for det in detections if
                      iou(current[:4], det[:4]) < iou_threshold or
                      current[5] != det[5]]  # 不同类别不抑制

    return keep


def iou(box1, box2):
    """
    计算两个边界框的IoU
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集区域
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算并集区域
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def predict_image(model, image_path, class_names, device, confidence_threshold=0.3, img_size=1024):
    """
    对单张图像进行预测并生成CSV格式结果
    Args:
        model: 训练好的模型
        image_path: 图像路径
        class_names: 类别名称列表
        device: 设备
        confidence_threshold: 置信度阈值
        img_size: 图像尺寸
    Returns:
        pd.DataFrame: 包含预测结果的DataFrame，格式与训练CSV相同
    """
    # 图像预处理（与训练时相同）
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_image = transform(image).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(original_image)

    # 解码输出
    detections = decode_outputs(outputs, confidence_threshold=confidence_threshold, img_size=img_size, stride = 16)
    batch_detections = detections[0]  # 取第一个batch

    # 转换为DataFrame
    results = []
    for detection in batch_detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection
        class_name = class_names[class_id]

        results.append({
            'filename': Path(image_path).name,
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax),
            'class': class_name,
            'confidence': confidence
        })

    return pd.DataFrame(results)


def predict_and_save_csv(model, image_path, output_csv_path, class_names, device, confidence_threshold=0.3, img_size=1024):
    """
    预测图像并保存结果为CSV
    """
    df = predict_image(model, image_path, class_names, device, confidence_threshold, img_size)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 预测结果保存至: {output_csv_path}")
    return df


# 批量预测函数
def predict_batch(model, image_dir, output_dir, class_names, device, confidence_threshold=0.3, img_size=1024):
    """
    批量预测目录中的所有图像
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    for image_path in image_paths:
        output_csv_path = output_dir / f"{image_path.stem}_pred.csv"
        predict_and_save_csv(model, image_path, output_csv_path, class_names, device, confidence_threshold, img_size)

    print(f"🎉 批量预测完成! 共处理 {len(image_paths)} 张图像")




def validate_model(model, val_loader, device, num_classes, img_size=1024, stride = 16):
    """
    在验证集上评估模型
    """
    model.eval()
    total_val_loss = 0
    hm_val_loss = 0
    wh_val_loss = 0
    off_val_loss = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)

            outputs = model(images)
            loss, loss_dict = detection_loss(outputs, targets, num_classes=num_classes,stride = stride, img_size=img_size)

            total_val_loss += loss_dict["total"].item()
            hm_val_loss += loss_dict["hm"].item()
            wh_val_loss += loss_dict["wh"].item()
            off_val_loss += loss_dict["off"].item()

    num_batches = len(val_loader)
    avg_loss = total_val_loss / num_batches
    avg_hm = hm_val_loss / num_batches
    avg_wh = wh_val_loss / num_batches
    avg_off = off_val_loss / num_batches

    return {
        "total": avg_loss,
        "hm": avg_hm,
        "wh": avg_wh,
        "off": avg_off
    }


def calculate_metrics(model, val_loader, device, class_names,stride = 16, iou_threshold=0.5, img_size=1024):
    """
    计算更详细的评估指标（mAP等）
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)

            # 解码预测
            predictions = decode_outputs(outputs, confidence_threshold=0.1, img_size=img_size, stride = stride)

            # 收集预测和目标
            for i, (preds, target) in enumerate(zip(predictions, targets)):
                image_id = target['image_id'].item()

                # 处理预测
                for pred in preds:
                    xmin, ymin, xmax, ymax, confidence, class_id = pred
                    all_predictions.append({
                        'image_id': image_id,
                        'bbox': [xmin, ymin, xmax, ymax],
                        'score': confidence,
                        'category_id': class_id
                    })

                # 处理真实标注 - 需要反归一化到像素坐标
                boxes = target['boxes'].cpu().numpy() * img_size  # 反归一化
                labels = target['labels'].cpu().numpy()

                for box, label in zip(boxes, labels):
                    all_targets.append({
                        'image_id': image_id,
                        'bbox': box.tolist(),
                        'category_id': label
                    })

    # 计算AP（简化版）
    # 在实际应用中，你可能想使用更成熟的评估库如torchmetrics或pycocotools
    aps = []
    for class_id in range(len(class_names)):
        class_predictions = [p for p in all_predictions if p['category_id'] == class_id]
        class_targets = [t for t in all_targets if t['category_id'] == class_id]

        if not class_targets:
            aps.append(0.0)
            continue

        # 简化的AP计算（实际应该计算PR曲线下面积）
        matched = 0
        for pred in sorted(class_predictions, key=lambda x: x['score'], reverse=True):
            for target in class_targets:
                if target['image_id'] == pred['image_id'] and \
                        iou(pred['bbox'], target['bbox']) > iou_threshold:
                    matched += 1
                    break

        precision = matched / len(class_predictions) if class_predictions else 0
        recall = matched / len(class_targets) if class_targets else 0

        # 简化的AP（实际应该更复杂）
        ap = 2 * precision * recall / (precision + recall + 1e-6)  # F1分数作为简化AP
        aps.append(ap)

    map_score = sum(aps) / len(aps) if aps else 0

    metrics = {
        'mAP': map_score,
        'AP_per_class': {class_names[i]: ap for i, ap in enumerate(aps)},
        'num_predictions': len(all_predictions),
        'num_targets': len(all_targets)
    }

    return metrics

# ---------------------------推理----------------------------------

def predict_single_image(image_path, output_csv_path, model_path=None, class_names=None, device=None,
                         confidence_threshold=0.3):
    """
    对单张图像进行预测并保存结果为CSV

    Args:
        image_path (str): 输入图像路径
        output_csv_path (str): 输出CSV文件路径
        model_path (str, optional): 模型权重路径，如果为None则使用默认路径
        class_names (list, optional): 类别名称列表，如果为None则使用默认值
        device (torch.device, optional): 设备，如果为None则自动选择
        confidence_threshold (float): 置信度阈值

    Returns:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    # 设置默认值
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if class_names is None:
        class_names = ["wall", "door", "window", "column"]

    if model_path is None:
        model_path = "training_output/best_model.pth"

    # 检查文件是否存在
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return None

    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    # 加载模型
    model = DummyModel(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 创建输出目录（如果需要）
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进行预测
    model.eval()
    df = predict_and_save_csv(
        model=model,
        image_path=image_path,
        output_csv_path=output_csv_path,
        class_names=class_names,
        device=device,
        confidence_threshold=confidence_threshold
    )

    print(f"✅ 预测完成! 共检测到 {len(df)} 个目标")
    return df


def predict_multiple_images(image_dir, output_dir, model_path=None, class_names=None, device=None,
                            confidence_threshold=0.3):
    """
    批量预测目录中的所有图像

    Args:
        image_dir (str): 输入图像目录
        output_dir (str): 输出目录
        model_path (str, optional): 模型权重路径
        class_names (list, optional): 类别名称列表
        device (torch.device, optional): 设备
        confidence_threshold (float): 置信度阈值

    Returns:
        dict: 每个图像的预测结果DataFrame字典
    """
    # 设置默认值
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if class_names is None:
        class_names = ["wall", "door", "window", "column"]

    if model_path is None:
        model_path = "training_output/best_model.pth"

    # 检查目录是否存在
    if not Path(image_dir).exists():
        print(f"❌ 图像目录不存在: {image_dir}")
        return {}

    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return {}

    # 加载模型
    model = DummyModel(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return {}

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(ext))
        image_paths.extend(Path(image_dir).glob(ext.upper()))

    if not image_paths:
        print(f"❌ 在目录 {image_dir} 中未找到图像文件")
        return {}

    print(f"🔍 找到 {len(image_paths)} 张图像，开始批量预测...")

    # 批量预测
    results = {}
    for i, image_path in enumerate(image_paths):
        output_csv_path = output_dir / f"{image_path.stem}_pred.csv"

        try:
            df = predict_and_save_csv(
                model=model,
                image_path=image_path,
                output_csv_path=output_csv_path,
                class_names=class_names,
                device=device,
                confidence_threshold=confidence_threshold
            )
            results[image_path.name] = df
            print(f"[{i + 1}/{len(image_paths)}] {image_path.name} -> 检测到 {len(df)} 个目标")
        except Exception as e:
            print(f"❌ 预测 {image_path.name} 失败: {e}")
            results[image_path.name] = None

    successful_predictions = sum(1 for v in results.values() if v is not None)
    print(f"🎉 批量预测完成! 成功处理 {successful_predictions}/{len(image_paths)} 张图像")

    return results


# 使用示例
if __name__ == "__main__":
    # 单张图像预测
    image_path = "data/pred_images/aaa.png"
    output_csv = "data/pred_images/aaa.csv"

    result_df = predict_single_image(
        image_path=image_path,
        output_csv_path=output_csv,
        model_path="training_output/best_model.pth",  # 可选，默认使用 training_output/best_model.pth
        class_names=["wall", "door", "window", "column"],  # 可选，默认使用这个列表
        confidence_threshold=0.3  # 可选，默认0.3
    )

    if result_df is not None:
        print("预测结果预览:")
        print(result_df.head())

    # 批量预测
    image_dir = "test_images/"
    output_dir = "batch_predictions/"

    batch_results = predict_multiple_images(
        image_dir=image_dir,
        output_dir=output_dir,
        model_path="training_output/best_model.pth",
        confidence_threshold=0.3
    )

    # 统计批量预测结果
    total_detections = 0
    for image_name, df in batch_results.items():
        if df is not None:
            total_detections += len(df)

    print(f"批量预测总计检测到 {total_detections} 个目标")