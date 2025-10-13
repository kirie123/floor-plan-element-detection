import pandas as pd
import json
from pathlib import Path
from typing import List
import os

from PIL import Image
from utils.shape import Shape, ImageAnnotation, save_image_annotations_to_jsonl


def csv_to_jsonl(input_dir: str, output_file: str):
    """
    将目录中的CSV标注文件转换为JSONL格式

    Args:
        input_dir: 输入目录，包含jpg图片和对应的csv文件
        output_file: 输出的JSONL文件路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = list(input_path.glob("*.jpg"))
    print(f"找到 {len(image_files)} 个图像文件")

    all_annotations = []

    for image_file in image_files:
        # 对应的CSV文件
        csv_file = input_path / f"{image_file.stem}.csv"

        if not csv_file.exists():
            print(f"⚠️  跳过 {image_file.name}: 对应的CSV文件不存在")
            continue

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"处理 {image_file.name}: 找到 {len(df)} 个标注")

            # 验证必需的列
            required_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'class']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ {csv_file.name} 缺少必需的列，需要: {required_cols}")
                continue

            # 创建Shape对象列表
            shapes = []
            for _, row in df.iterrows():
                class_label = str(row['class']).lower()
                shape = Shape(
                    label=class_label,
                    points=[
                        [float(row['xmin']), float(row['ymin'])],
                        [float(row['xmax']), float(row['ymax'])]
                    ],
                    confidence_score=0.89,  # 真值标注的置信度为1.0
                    shape_type="rectangle",
                    group_id=None,
                    flags={}
                )
                shapes.append(shape)

            # 获取图像尺寸
            try:
                with Image.open(image_file) as img:
                    w, h = img.size
            except Exception as e:
                print(f"⚠️  无法读取图像尺寸: {e}，使用默认值")
                h, w = 1024, 1024  # 默认尺寸

            # 创建ImageAnnotation对象
            annotation = ImageAnnotation(
                image_path=image_file.name,
                image_height=h,
                image_width=w,
                shapes=shapes
            )

            all_annotations.append(annotation)
            print(f"✅ 成功处理 {image_file.name}")

        except Exception as e:
            print(f"❌ 处理 {image_file.name} 时出错: {e}")
            continue

    # 保存为JSONL
    save_image_annotations_to_jsonl(all_annotations, output_file)
    print(f"🎉 转换完成! 总共处理 {len(all_annotations)} 个图像，输出保存至: {output_file}")

    # 输出统计信息
    print("\n📊 统计信息:")
    total_shapes = sum(len(ann.shapes) for ann in all_annotations)
    print(f"总标注数量: {total_shapes}")

    # 类别统计
    class_counts = {}
    for ann in all_annotations:
        for shape in ann.shapes:
            class_counts[shape.label] = class_counts.get(shape.label, 0) + 1

    print("各类别数量:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")


def validate_jsonl_format(jsonl_file: str):
    """
    验证JSONL文件的格式是否正确
    """
    print(f"\n🔍 验证JSONL文件格式: {jsonl_file}")

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"文件包含 {len(lines)} 行")

        valid_count = 0
        errors = []

        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())

                # 检查必需字段
                required_fields = ['image_path', 'image_height', 'image_width', 'shapes']
                if not all(field in data for field in required_fields):
                    errors.append(f"第 {i + 1} 行缺少必需字段")
                    continue

                # 检查shapes字段
                shapes = data['shapes']
                for j, shape in enumerate(shapes):
                    shape_required = ['label', 'points', 'confidence_score', 'shape_type']
                    if not all(field in shape for field in shape_required):
                        errors.append(f"第 {i + 1} 行第 {j + 1} 个shape缺少必需字段")
                        continue

                    # 检查points格式
                    points = shape['points']
                    if not isinstance(points, list) or len(points) != 2:
                        errors.append(f"第 {i + 1} 行第 {j + 1} 个shape的points格式错误")
                        continue

                    if not all(isinstance(point, list) and len(point) == 2 for point in points):
                        errors.append(f"第 {i + 1} 行第 {j + 1} 个shape的points坐标格式错误")
                        continue

                valid_count += 1

            except json.JSONDecodeError as e:
                errors.append(f"第 {i + 1} 行JSON解析错误: {e}")
            except Exception as e:
                errors.append(f"第 {i + 1} 行验证错误: {e}")

        print(f"✅ 有效行数: {valid_count}/{len(lines)}")

        if errors:
            print(f"❌ 发现 {len(errors)} 个错误:")
            for error in errors[:10]:  # 只显示前10个错误
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... 还有 {len(errors) - 10} 个错误")
        else:
            print("🎉 所有行格式正确!")

    except Exception as e:
        print(f"❌ 验证失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 你的CSV数据目录
    input_directory = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\验证集图片_Karle_7"
    output_jsonl = "output/converted_annotations.jsonl"

    # 转换CSV到JSONL
    csv_to_jsonl(input_directory, output_jsonl)

    # 验证生成的JSONL格式
    validate_jsonl_format(output_jsonl)

    print("\n💡 提示: 将这个JSONL文件提交测试，看看分数是否正常")
    print("如果分数正常，说明标注数据格式正确，问题可能在模型")
    print("如果分数很低，说明标注数据格式有问题")