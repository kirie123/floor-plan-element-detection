# visualize_annotations.py
import os
import cv2
import pandas as pd
from PIL import Image


def visualize_csv_annotations(image_dir: str, csv_dir: str, output_dir: str = "vis_output"):
    """
    可视化 CSV 标注：在原图上绘制 bbox 和 label

    Args:
        image_dir: 存放图像的文件夹（如 "data/images"）
        csv_dir: 存放 CSV 标注的文件夹（如 "data/csv_labels"）
        output_dir: 可视化结果保存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有 CSV 文件
    for csv_file in os.listdir(csv_dir):
        if not csv_file.endswith('.csv'):
            continue

        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"⚠️ 跳过空文件: {csv_file}")
            continue

        # 获取图像名（假设 CSV 中 filename 列一致）
        image_name = df['filename'].iloc[0]
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            continue

        # 读取图像（OpenCV 读取为 BGR）
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            continue

        # 绘制每个 bbox
        for _, row in df.iterrows():
            x1, y1 = int(row['xmin']), int(row['ymin'])
            x2, y2 = int(row['xmax']), int(row['ymax'])
            label = str(row['class'])

            # 绘制矩形框（绿色）
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            # 绘制标签背景（黑色）
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 0, 0), -1)

            # 绘制标签文字（白色）
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存结果
        output_path = os.path.join(output_dir, f"vis_{image_name}")
        cv2.imwrite(output_path, image)
        print(f"✅ 已保存可视化结果: {output_path}")


if __name__ == "__main__":
    # ====== 修改以下路径为你自己的目录 ======
    IMAGE_DIR = "vis_images"  # 图像文件夹
    CSV_DIR = "vis_images"  # CSV 标注文件夹
    OUTPUT_DIR = "vis_output"  # 输出文件夹
    # ======================================

    visualize_csv_annotations(IMAGE_DIR, CSV_DIR, OUTPUT_DIR)