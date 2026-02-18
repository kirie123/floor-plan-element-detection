# Floor Plan Element Detection

A deep learning system for detecting and classifying architectural elements in floor plan images. Supports detection of walls, doors, windows, and columns, with fine-grained wall type classification.

## Features

- **Multi-class Object Detection**: Detect walls, doors, windows, and columns in floor plans
- **Fine-grained Wall Classification**: Classify walls into 23 sub-categories (WALL1-WALL23)
- **Flexible Backbone**: Support for both ResNet50+FPN and DINOv3+STA architectures
- **Large Image Support**: Image slicing for handling high-resolution floor plans
- **Unified CLI**: Clean command-line interface for all operations

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For DINOv3 support, you may need to download the pretrained weights separately.

## Quick Start

### Using the Unified CLI

```bash
# Train detection model
python cli.py train --train_dir data/cvt_images --val_dir data/val_images --epochs 300

# Run inference
python cli.py predict --input_path data/pred_images --output_path output/results

# Train wall classifier
python cli.py train_classifier --data_dir data/wall_images --epochs 50

# Validate model
python cli.py validate --model_path weights/best_detect_model.pth --val_dir data/val_images
```

### Using Individual Scripts

```bash
# Direct training
python train.py

# Run prediction
python main.py --input_path data/pred_images --output_path output/results

# Train wall classifier
python train_classifier.py

# Validate
python validate.py
```

## Project Structure

```
floor-plan-element-detection/
├── cli.py                 # Unified command-line interface
├── main.py                # Main entry point for inference
├── train.py               # Detection model training script
├── train_classifier.py    # Wall classifier training script
├── validate.py            # Model validation script
├── detector.py            # Integrated detector (detection + classification)
├── config.py              # Model configuration
├── loss.py                # Loss functions
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── run.sh                 # Execution script for evaluation system
│
├── data/                  # Data processing modules
│   ├── detection_dataset.py
│   ├── wall_classifier.py
│   ├── ImageSlicerForDetection.py
│   ├── create_wall_classification_dataset.py
│   ├── visualize_annotations.py
│   └── augment_classes.py
│
├── models/                # Model architectures
│   ├── default_model.py   # ResNet50 + FPN detector
│   ├── sta_adapter.py     # DINOv3 + STA detector
│   ├── DINOv3Classifier.py
│   ├── DINOv3Detector.py
│   └── convert.py
│
└── utils/                 # Utility functions
    ├── data_loader.py
    ├── shape.py
    ├── similarity.py
    └── visualization.py
```

## CLI Commands

### Train Detection Model

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

Arguments:
- `--train_dir`: Training data directory
- `--val_dir`: Validation data directory
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--backbone`: Backbone network (resnet50, resnet101, dinov3)
- `--output_dir`: Output directory for checkpoints

### Run Inference

```bash
python cli.py predict \
    --input_path data/pred_images \
    --output_path output/results \
    --detection_model weights/best_detect_model.pth \
    --classifier_model weights/best_wall_classifier.pth \
    --slice_size 1024 \
    --confidence 0.3
```

Arguments:
- `--input_path`: Input image directory
- `--output_path`: Output directory for results
- `--detection_model`: Path to detection model checkpoint
- `--classifier_model`: Path to wall classifier checkpoint
- `--slice_size`: Image slice size for large images (default: 1024)
- `--confidence`: Confidence threshold (default: 0.3)

### Train Wall Classifier

```bash
python cli.py train_classifier \
    --data_dir data/wall_images \
    --epochs 50 \
    --batch_size 32 \
    --backbone efficientnet_b0 \
    --num_classes 23
```

Arguments:
- `--data_dir`: Training data directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--backbone`: Backbone network (efficientnet_b0, resnet50, dinov3)
- `--num_classes`: Number of wall classes (default: 23)

### Validate Model

```bash
python cli.py validate \
    --model_path weights/best_detect_model.pth \
    --val_dir data/val_images \
    --batch_size 2
```

Arguments:
- `--model_path`: Path to model checkpoint
- `--val_dir`: Validation data directory
- `--batch_size`: Batch size (default: 2)

## Data Format

### Detection Dataset

The training data should be organized as follows:

```
data/
├── cvt_images/           # Training images
│   ├── image_001.jpg
│   ├── image_001.csv
│   ├── image_002.jpg
│   ├── image_002.csv
│   └── ...
└── val_images/           # Validation images
    ├── image_001.jpg
    ├── image_001.csv
    └── ...
```

CSV format:
```
filename,xmin,ymin,xmax,ymax,class
image_001.jpg,10,20,100,80,wall
image_001.jpg,150,30,200,90,door
...
```

### Wall Classification Dataset

```
data/wall_classification/
├── WALL1/
│   ├── wall_001.jpg
│   └── ...
├── WALL2/
└── ...
```

## Model Architecture

### Detection Model

Based on CenterNet architecture:
- **Backbone**: ResNet50 with FPN or DINOv3 with STA adapter
- **Heads**: Heatmap head, Width/Height head, Offset head
- **Classes**: wall, door, window, column (4 classes)

### Wall Classifier

- **Backbone**: EfficientNet-B0 (default), ResNet50, or DINOv3
- **Classes**: 23 wall types (WALL1-WALL23)

## Output Format

Inference results are saved in JSONL format:

```json
{"image_path": "image_001.jpg", "image_height": 1024, "image_width": 1024, "shapes": [{"type": "rectangle", "label": "wall", "coordinates": [xmin, ymin, xmax, ymax], "confidence": 0.95}, ...]}
```

## Docker

Build and run with Docker:

```bash
docker build -t floor-plan-detector .
docker run -v /path/to/data:/data floor-plan-detector /data/input /data/output
```

## License

MIT License
