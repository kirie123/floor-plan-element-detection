import argparse
import os

import cv2

from detector import DummyDetector
from utils.shape import Shape, ImageAnnotation, save_image_annotations_to_jsonl


def predict(input_path, output_path):
    # 初始化检测器（后续可替换为 YOLODetector 等）
    detector = DummyDetector()

    # 1. 在这里加载您的模型
    print("模型加载中...")
    # 2. 读取输入数据
    # 假设输入是一个名为 data.csv 的文件
    input_file = os.path.join(input_path, 'data.csv')
    print(f"正在读取输入文件: {input_file}")
    #获取待处理图片
    drawing_files = []
    all_annotations = []
    for drawing_file in drawing_files:
        drawing_path = drawing_file
        img = cv2.imread(drawing_path)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (3118, 4414)
        # 3. 进行模型预测
        # 调用检测器
        shapes = detector.detect(drawing_path)
        print("模型预测完成！")

        # 还需要从shapes中过滤掉图例中没有的类别

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



    # 4. 将结果写入指定的输出文件
    # 假设输出要求为 result.json
    output_file = os.path.join(output_path, 'result.json')
    print(f"正在写入结果文件: {output_file}")
    # 模拟写入一个文件
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_file, 'w') as f:
        f.write('{"prediction": "success"}')
    print("处理完成！")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 接收 run.sh 传入的参数
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prediction results')
    args = parser.parse_args()
    predict(args.input_path, args.output_path)