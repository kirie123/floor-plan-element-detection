# 建筑平面图元素检测系统

一个用于在建筑平面图图像中检测和分类建筑元素的深度学习系统。支持检测墙体、门、窗和柱子，并提供细粒度的墙体类型分类功能。

## 功能特性

- **多类别目标检测**：检测平面图中的墙体、门、窗和柱子
- **细粒度墙体分类**：将墙体分为23个子类别（WALL1-WALL23）
- **灵活的骨干网络**：支持 ResNet50+FPN 和 DINOv3+STA 两种架构
- **大图像处理**：支持图像切片处理高分辨率平面图
- **统一CLI**：简洁的命令行界面，支持所有操作

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（用于GPU训练）

## 安装

### 安装依赖

```bash
pip install -r requirements.txt
```

使用DINOv3骨干网络可能需要单独下载预训练权重。

## 快速开始

### 使用统一CLI

```bash
# 训练检测模型
python cli.py train --train_dir data/cvt_images --val_dir data/val_images --epochs 300

# 运行推理
python cli.py predict --input_path data/pred_images --output_path output/results

# 训练墙体分类器
python cli.py train_classifier --data_dir data/wall_images --epochs 50

# 验证模型
python cli.py validate --model_path weights/best_detect_model.pth --val_dir data/val_images
```

### 使用独立脚本

```bash
# 直接训练
python train.py

# 运行预测
python main.py --input_path data/pred_images --output_path output/results

# 训练墙体分类器
python train_classifier.py

# 验证模型
python validate.py
```

## 项目结构

```
floor-plan-element-detection/
├── cli.py                 # 统一命令行入口
├── main.py                # 推理主入口
├── train.py               # 检测模型训练脚本
├── train_classifier.py    # 墙体分类器训练脚本
├── validate.py            # 模型验证脚本
├── detector.py            # 集成检测器（检测+分类）
├── config.py              # 模型配置
├── loss.py                # 损失函数
├── requirements.txt       # Python依赖
├── Dockerfile             # Docker配置
├── run.sh                 # 评测系统执行脚本
│
├── data/                  # 数据处理模块
│   ├── detection_dataset.py
│   ├── wall_classifier.py
│   ├── ImageSlicerForDetection.py
│   ├── create_wall_classification_dataset.py
│   ├── visualize_annotations.py
│   └── augment_classes.py
│
├── models/                # 模型架构
│   ├── default_model.py   # ResNet50 + FPN 检测器
│   ├── sta_adapter.py     # DINOv3 + STA 检测器
│   ├── DINOv3Classifier.py
│   ├── DINOv3Detector.py
│   └── convert.py
│
└── utils/                 # 工具函数
    ├── data_loader.py
    ├── shape.py
    ├── similarity.py
    └── visualization.py
```

## CLI 命令详解

### 训练检测模型

```bash
python cli.py train \
    --train_dir data/cvt_images \
    --val_dir data/val_images \
    --epochs 300 \
    --batch_size 4 \
    --lr 1e-4 \
    --backbone resnet50 \
    --output_dir training_output
```

参数说明：
- `--train_dir`：训练数据目录
- `--val_dir`：验证数据目录
- `--epochs`：训练轮数（默认：300）
- `--batch_size`：批大小（默认：4）
- `--lr`：学习率（默认：1e-4）
- `--backbone`：骨干网络（resnet50, resnet101, dinov3）
- `--output_dir`：模型输出目录

### 运行推理

```bash
python cli.py predict \
    --input_path data/pred_images \
    --output_path output/results \
    --detection_model weights/best_detect_model.pth \
    --classifier_model weights/best_wall_classifier.pth \
    --slice_size 1024 \
    --confidence 0.3
```

参数说明：
- `--input_path`：输入图像目录
- `--output_path`：输出结果目录
- `--detection_model`：检测模型权重路径
- `--classifier_model`：墙体分类器权重路径
- `--slice_size`：大图像切片尺寸（默认：1024）
- `--confidence`：置信度阈值（默认：0.3）

### 训练墙体分类器

```bash
python cli.py train_classifier \
    --data_dir data/wall_images \
    --epochs 50 \
    --batch_size 32 \
    --backbone efficientnet_b0 \
    --num_classes 23
```

参数说明：
- `--data_dir`：训练数据目录
- `--epochs`：训练轮数（默认：50）
- `--batch_size`：批大小（默认：32）
- `--backbone`：骨干网络（efficientnet_b0, resnet50, dinov3）
- `--num_classes`：墙体类别数（默认：23）

### 验证模型

```bash
python cli.py validate \
    --model_path weights/best_detect_model.pth \
    --val_dir data/val_images \
    --batch_size 2
```

参数说明：
- `--model_path`：模型权重路径
- `--val_dir`：验证数据目录
- `--batch_size`：批大小（默认：2）

## 数据格式

### 检测数据集

训练数据应按以下结构组织：

```
data/
├── cvt_images/           # 训练图像
│   ├── image_001.jpg
│   ├── image_001.csv
│   ├── image_002.jpg
│   ├── image_002.csv
│   └── ...
└── val_images/           # 验证图像
    ├── image_001.jpg
    ├── image_001.csv
    └── ...
```

CSV格式：
```
filename,xmin,ymin,xmax,ymax,class
image_001.jpg,10,20,100,80,wall
image_001.jpg,150,30,200,90,door
...
```

### 墙体分类数据集

```
data/wall_classification/
├── WALL1/
│   ├── wall_001.jpg
│   └── ...
├── WALL2/
└── ...
```

## 模型架构

### 检测模型

基于CenterNet架构：
- **骨干网络**：ResNet50 + FPN 或 DINOv3 + STA
- **检测头**：热力图头、宽高头、偏置头
- **类别**：wall, door, window, column（4类）

### 墙体分类器

- **骨干网络**：EfficientNet-B0（默认）、ResNet50 或 DINOv3
- **类别**：23种墙体类型（WALL1-WALL23）

## 输出格式

推理结果以JSONL格式保存：

```json
{"image_path": "image_001.jpg", "image_height": 1024, "image_width": 1024, "shapes": [{"type": "rectangle", "label": "wall", "coordinates": [xmin, ymin, xmax, ymax], "confidence": 0.95}, ...]}
```

## Docker 部署

使用Docker构建和运行：

```bash
docker build -t floor-plan-detector .
docker run -v /path/to/data:/data floor-plan-detector /data/input /data/output
```

## 许可证

MIT License
