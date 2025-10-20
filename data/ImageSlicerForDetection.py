import os
import csv
import math
from pathlib import Path
from PIL import Image
import pandas as pd


class ImageSlicerForDetection:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        slice_size: int = 1024,
        overlap: float = 0.1,
        min_area_ratio: float = 0.5,
        image_exts: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        csv_ext: str = '.csv'
    ):
        """
        Args:
            input_dir: 输入目录，包含图像和同名CSV
            output_dir: 输出目录，保存切片图像和新CSV
            slice_size: 切片边长（正方形）
            overlap: 切片间重叠比例（0.0 ~ 0.5）
            min_area_ratio: 保留目标的最小面积保留率（0.0 ~ 1.0）
            image_exts: 支持的图像扩展名
            csv_ext: 标注文件扩展名
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.slice_size = slice_size
        self.overlap = overlap
        self.min_area_ratio = min_area_ratio
        self.image_exts = tuple(ext.lower() for ext in image_exts)
        self.csv_ext = csv_ext.lower()

        assert 0.0 <= overlap < 0.5, "overlap should be in [0, 0.5)"
        assert 0.0 < min_area_ratio <= 1.0, "min_area_ratio should be in (0, 1]"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_annotations(self, csv_path: Path):
        """加载CSV标注，返回list[dict]"""
        if not csv_path.exists():
            print(f"⚠️  跳过：{csv_path} 不存在")
            return None

        try:
            df = pd.read_csv(csv_path)
            required_cols = {'xmin', 'ymin', 'xmax', 'ymax', 'class'}
            if not required_cols.issubset(df.columns):
                print(f"❌ {csv_path} 缺少必要列: {required_cols}")
                return None

            annotations = []
            for _, row in df.iterrows():
                annotations.append({
                    'xmin': int(row['xmin']),
                    'ymin': int(row['ymin']),
                    'xmax': int(row['xmax']),
                    'ymax': int(row['ymax']),
                    'class': str(row['class'])
                })
            return annotations
        except Exception as e:
            print(f"❌ 读取 {csv_path} 失败: {e}")
            return None

    def _slice_single_image(self, img_path: Path, annotations: list):
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size

        step = int(self.slice_size * (1 - self.overlap))
        slice_idx = 0

        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                # 计算目标裁剪区域（可能超出边界）
                x2 = x + self.slice_size
                y2 = y + self.slice_size

                # 实际裁剪区域（不越界）
                crop_x2 = min(x2, img_w)
                crop_y2 = min(y2, img_h)

                # 裁剪原始图像
                patch = image.crop((x, y, crop_x2, crop_y2))

                # 如果尺寸不足，进行 padding 到 1024x1024
                pad_right = self.slice_size - patch.width
                pad_bottom = self.slice_size - patch.height

                if pad_right > 0 or pad_bottom > 0:
                    # 使用黑色填充（也可以用 edge / reflect）
                    patch = Image.new('RGB', (self.slice_size, self.slice_size), (254, 254, 254))
                    patch.paste(image.crop((x, y, crop_x2, crop_y2)), (0, 0))

                # 转换标注（注意：只考虑原始图像内的区域，padding 区域无目标）
                patch_annos = []
                for ann in annotations:
                    inter_x1 = max(ann['xmin'], x)
                    inter_y1 = max(ann['ymin'], y)
                    inter_x2 = min(ann['xmax'], x2)  # 注意：这里用 x2（含 padding 区域边界）
                    inter_y2 = min(ann['ymax'], y2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        orig_area = (ann['xmax'] - ann['xmin']) * (ann['ymax'] - ann['ymin'])
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        if inter_area / orig_area >= self.min_area_ratio:
                            # 转为 patch 局部坐标（padding 区域不会产生新目标）
                            patch_annos.append({
                                'xmin': inter_x1 - x,
                                'ymin': inter_y1 - y,
                                'xmax': inter_x2 - x,
                                'ymax': inter_y2 - y,
                                'class': ann['class']
                            })

                if patch_annos:
                    base_name = img_path.stem
                    new_img_name = f"{base_name}_slice_{slice_idx:04d}_{self.slice_size}{img_path.suffix}"
                    new_csv_name = f"{base_name}_slice_{slice_idx:04d}_{self.slice_size}.csv"

                    new_img_path = self.output_dir / new_img_name
                    new_csv_path = self.output_dir / new_csv_name

                    patch.save(new_img_path)

                    df_out = pd.DataFrame(patch_annos)
                    df_out.insert(0, 'filename', new_img_name)
                    df_out.to_csv(new_csv_path, index=False)

                    slice_idx += 1

                x += step
            y += step

        print(f"✅ {img_path.name} → 生成 {slice_idx} 个 1024x1024 切片")

    def run(self):
        """主执行函数"""
        image_files = [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_exts
        ]

        print(f"🔍 找到 {len(image_files)} 张图像，开始切片...")

        for img_path in sorted(image_files):
            csv_path = img_path.with_suffix(self.csv_ext)
            annotations = self._load_annotations(csv_path)

            if annotations is None:
                continue

            self._slice_single_image(img_path, annotations)

        print(f"🎉 切片完成！结果保存至: {self.output_dir.absolute()}")


def create_multi_scale_slices(original_images_dir, output_base_dir, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """
    为原始高分辨率图像创建多尺度切片
    """
    for scale in scales:
        #output_dir = Path(output_base_dir) / f"scale_{scale}"
        output_dir = Path(output_base_dir) / f"scale_all"
        output_dir.mkdir(parents=True, exist_ok=True)

        slicer = ImageSlicerForDetection(
            input_dir=original_images_dir,
            output_dir=str(output_dir),
            slice_size=int(1024 * scale),  # 根据缩放调整切片尺寸
            overlap=0.3,
            min_area_ratio=0.1
        )
        slicer.run()

if __name__ == "__main__":
    # slicer = ImageSlicerForDetection(
    #     input_dir="images/",          # 含 wall1.jpg, wall1.csv, ...
    #     output_dir="cvt_images/",
    #     slice_size=1024,
    #     overlap=0.4,
    #     min_area_ratio=0.2
    # )
    #slicer.run()
    create_multi_scale_slices(
        original_images_dir="images/",
        output_base_dir="cvt2_images/",
    )
