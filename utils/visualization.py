# utils/visualization.py
import cv2
import os
from typing import List

from utils.shape import Shape


def draw_shapes(image, shapes: List[Shape], font_scale=0.5, thickness=2):
    """
    在图像上绘制所有 Shape（矩形框 + 标签）
    """
    for shape in shapes:
        # 绘制矩形框
        pt1 = (int(shape.x_min), int(shape.y_min))
        pt2 = (int(shape.x_max), int(shape.y_max))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness)

        # 绘制标签文本
        label = f"{shape.label} {shape.confidence_score:.2f}"
        cv2.putText(
            image, label,
            (int(shape.x_min), int(shape.y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 0), thickness
        )
    return image


def visualize_result(image_path: str, shapes: List[Shape], output_path: str):
    """
    读取图像，绘制 shapes，保存可视化结果
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    image = draw_shapes(image, shapes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"✅ 可视化结果已保存至: {output_path}")



if __name__ == "__main__":
    shapes = [
        Shape("door", [[100, 200], [300, 400]], 0.95),
        Shape("window", [[500, 100], [600, 250]], 0.88)
    ]
    visualize_result("dx9.jpg", shapes, "output/vis_dx9.jpg")