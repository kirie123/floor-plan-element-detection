import argparse
import os

import cv2

from detector import DummyDetector
from utils.shape import Shape, ImageAnnotation, save_image_annotations_to_jsonl


def predict(input_path, output_path):
    # 初始化检测器（后续可替换为 YOLODetector 等）
    detector = DummyDetector(
        detection_model_path="weights/best_detect_model.pth",
        classification_model_path="weights/best_wall_classifier.pth"
    )

    print("模型加载中...")

    # 修正：正确获取输入目录中的所有 .jpg 文件
    drawing_files = []
    for filename in os.listdir(input_path):
        if filename.lower().endswith('.jpg'):
            drawing_files.append(os.path.join(input_path, filename))

    print(f"找到 {len(drawing_files)} 个图片文件")
    if not drawing_files:
        print("警告：未找到任何 .jpg 文件")
        return
    all_annotations = []
    for drawing_file in drawing_files:
        drawing_path = drawing_file
        img = cv2.imread(drawing_path)
        if img is None:
            print(f"警告：无法读取图片 {drawing_path}")
            continue
        h, w = img.shape[0], img.shape[1]
        print(f"图片分辨率：{h}x{w}")
        # 进行模型预测
        shapes = detector.detect(drawing_path)
        print(f"检测到 {len(shapes)} 个形状")

        # 还需要从shapes中过滤掉图例中没有的类别

        # 创建 ImageAnnotation 对象
        annotation = ImageAnnotation(
            image_path=os.path.basename(drawing_path),  # 只保存文件名
            image_height=h,
            image_width=w,
            shapes=shapes
        )
        all_annotations.append(annotation)

    # 保存为 .jsonl
    output_file = os.path.join(output_path, 'results.jsonl')
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_image_annotations_to_jsonl(all_annotations, output_file)
    print(f"结果已保存到: {output_file}")
    print("处理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 接收 run.sh 传入的参数
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prediction results')
    args = parser.parse_args()
    predict(args.input_path, args.output_path)