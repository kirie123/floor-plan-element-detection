# utils/shape.py
import json
import os
from typing import List, Optional, Any


class Shape:
    """
    表示一个标注形状，完全对应比赛 JSON 中的 shape 字段。
    """
    def __init__(
        self,
        label: str,
        points: List[List[float]],  # [[x_min, y_min], [x_max, y_max]]
        confidence_score: float = 1.0,
        shape_type: str = "rectangle",
        group_id: Optional[int] = None,
        flags: dict = None
    ):
        self.label = label
        self.points = points
        self.confidence_score = confidence_score
        self.shape_type = shape_type
        self.group_id = group_id
        self.flags = flags if flags is not None else {}

    def to_dict(self) -> dict:
        """转换为字典，用于 JSON 序列化"""
        return {
            "label": self.label,
            "points": self.points,
            "confidence_score": self.confidence_score,
            "group_id": self.group_id,
            "shape_type": self.shape_type,
            "flags": self.flags
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典反序列化（可用于加载真值）"""
        return cls(
            label=data["label"],
            points=data["points"],
            confidence_score=data.get("confidence_score", 1.0),
            shape_type=data.get("shape_type", "rectangle"),
            group_id=data.get("group_id"),
            flags=data.get("flags", {})
        )

    @property
    def x_min(self) -> float:
        return self.points[0][0]

    @property
    def y_min(self) -> float:
        return self.points[0][1]

    @property
    def x_max(self) -> float:
        return self.points[1][0]

    @property
    def y_max(self) -> float:
        return self.points[1][1]

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def __repr__(self):
        return f"Shape(label='{self.label}', bbox=({self.x_min:.1f}, {self.y_min:.1f}, {self.x_max:.1f}, {self.y_max:.1f}), conf={self.confidence_score:.2f})"



class ImageAnnotation:
    """一张图纸的完整标注信息"""
    def __init__(
        self,
        image_path: str,
        image_height: int,
        image_width: int,
        shapes: List[Shape],
        image_data: Any = None  # 通常为 None
    ):
        self.image_path = image_path
        self.image_height = image_height
        self.image_width = image_width
        self.shapes = shapes
        self.image_data = image_data

    def to_dict(self) -> dict:
        """转换为比赛要求的 JSON 对象"""
        return {
            "shapes": [shape.to_dict() for shape in self.shapes],
            "imagePath": self.image_path,
            "imageData": self.image_data,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width
        }

    @classmethod
    def from_dict(cls, data: dict):
        shapes = [Shape.from_dict(s) for s in data["shapes"]]
        return cls(
            image_path=data["imagePath"],
            image_height=data["imageHeight"],
            image_width=data["imageWidth"],
            shapes=shapes,
            image_data=data.get("imageData")
        )

    def __repr__(self):
        return f"ImageAnnotation(image='{self.image_path}', num_shapes={len(self.shapes)})"


def save_image_annotations_to_jsonl(
    annotations: List[ImageAnnotation],
    output_file: str,
    encoding: str = 'utf-8'
):
    """
    将 ImageAnnotation 列表保存为比赛要求的 .jsonl 格式
    每行一个 JSON 对象，严格匹配样例。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding=encoding) as f:
        for ann in annotations:
            json_line = json.dumps(ann.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"✅ 成功保存 {len(annotations)} 张图像的标注到 {output_file}")


import cv2

if __name__ == '__main__':
    input_path = ""
    output_path = ""
    legends_dir = os.path.join(input_path, 'legends')
    drawings_dir = os.path.join(input_path, 'drawings')
    drawing_files = [f for f in os.listdir(drawings_dir) if f.lower().endswith('.jpg')]

    all_annotations = []
    for drawing_file in drawing_files:
        drawing_path = os.path.join(drawings_dir, drawing_file)
        img = cv2.imread(drawing_path)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (3118, 4414)

        # 创建 Shape 列表
        shapes = [
            Shape(
                label="door",
                points=[[0, 0], [1, 1]],
                confidence_score=0.9,
                shape_type="rectangle"
            )
        ]

        # 创建 ImageAnnotation 对象
        annotation = ImageAnnotation(
            image_path=drawing_file,
            image_height=h,
            image_width=w,
            shapes=shapes
        )
        all_annotations.append(annotation)

    # 保存为 .jsonl
    output_file = os.path.join(output_path, 'results.jsonl')
    save_image_annotations_to_jsonl(all_annotations, output_file)