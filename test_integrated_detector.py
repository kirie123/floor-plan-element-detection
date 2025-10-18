# test_integrated_detector.py
from detector import IntegratedDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def visualize_detections(image_path, detections, output_path=None):
    """可视化检测结果 - 简化版，只显示边界框"""
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = {
        'wall': 'red',
        'door': 'blue',
        'window': 'green',
        'column': 'orange'
    }

    # 统计各类别的检测数量
    class_counts = {}

    for detection in detections:
        xmin, ymin = detection['xmin'], detection['ymin']
        width = detection['xmax'] - xmin
        height = detection['ymax'] - ymin

        # 获取颜色
        class_name = detection['class']
        base_class = class_name.split('_')[0].lower() if 'WALL' in class_name else class_name
        color = colors.get(base_class, 'purple')

        # 更新类别计数
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 绘制边界框 - 只画框，不添加标签
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

    # 创建图例
    legend_elements = []
    for class_name, color in colors.items():
        count = sum(1 for det in detections if det['class'].lower().startswith(class_name))
        if count > 0:
            legend_elements.append(
                patches.Patch(facecolor='none', edgecolor=color, label=f'{class_name} ({count})')
            )

    # 添加图例
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

    ax.set_title(f'检测结果 - {len(detections)} 个目标')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果保存至: {output_path}")

    plt.show()

def test_single_image():
    """测试单张图像"""
    detector = IntegratedDetector()

    # 测试图像路径
    image_path = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\补充图片\\bc1.jpg"  # 替换为你的测试图像路径
    image_path = "data/pred_images/dx9_slice_0000_1536.jpg"
    #image_path = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\图例.jpg"
    image_path = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\验证集图片\\dx18.jpg"
    # 进行检测
    detections = detector.detect(image_path, confidence_threshold=0.3)

    # 打印结果
    print("\n=== 检测结果 ===")
    for i, detection in enumerate(detections):
        print(f"{i + 1}. {detection['class']}: ({detection['xmin']}, {detection['ymin']}) - "
              f"({detection['xmax']}, {detection['ymax']}), 置信度: {detection['confidence']:.3f}")

    # 可视化
    visualize_detections(image_path, detections, "detection_result.png")


if __name__ == "__main__":
    test_single_image()