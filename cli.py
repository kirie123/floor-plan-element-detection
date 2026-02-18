#!/usr/bin/env python3
"""
Floor Plan Element Detection System
A unified CLI for training, inference, and validation of architectural element detection.
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Floor Plan Element Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train detection model
  python cli.py train --train_dir data/cvt_images --val_dir data/val_images --epochs 300

  # Run inference
  python cli.py predict --input_path data/pred_images --output_path output/results

  # Train wall classifier
  python cli.py train_classifier --data_dir data/wall_images --epochs 50

  # Validate model
  python cli.py validate --model_path weights/best_detect_model.pth --val_dir data/val_images
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train detection model')
    train_parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    train_parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--backbone', type=str, default='resnet50', 
                              choices=['resnet50', 'resnet101', 'dinov3'], help='Backbone network')
    train_parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('--input_path', type=str, required=True, help='Input image directory')
    predict_parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    predict_parser.add_argument('--detection_model', type=str, default='weights/best_detect_model.pth',
                                help='Detection model path')
    predict_parser.add_argument('--classifier_model', type=str, default='weights/best_wall_classifier.pth',
                                help='Wall classifier model path')
    predict_parser.add_argument('--slice_size', type=int, default=1024, help='Image slice size for large images')
    predict_parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    predict_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Train classifier command
    classifier_parser = subparsers.add_parser('train_classifier', help='Train wall type classifier')
    classifier_parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    classifier_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    classifier_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    classifier_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    classifier_parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                                    choices=['efficientnet_b0', 'resnet50', 'dinov3'], help='Backbone network')
    classifier_parser.add_argument('--num_classes', type=int, default=23, help='Number of wall classes')
    classifier_parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    classifier_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate detection model')
    validate_parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    validate_parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    validate_parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    validate_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        from train import main as train_main
        sys.argv = [
            'train.py',
            '--train_dir', args.train_dir,
            '--val_dir', args.val_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--backbone', args.backbone,
            '--output_dir', args.output_dir,
            '--device', args.device
        ]
        train_main()
        
    elif args.command == 'predict':
        from main import predict as predict_main
        sys.argv = [
            'main.py',
            '--input_path', args.input_path,
            '--output_path', args.output_path
        ]
        predict_main(args.input_path, args.output_path)
        
    elif args.command == 'train_classifier':
        from train_classifier import main as classifier_main
        sys.argv = [
            'train_classifier.py',
            '--data_dir', args.data_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--backbone', args.backbone,
            '--num_classes', str(args.num_classes),
            '--output_dir', args.output_dir
        ]
        classifier_main()
        
    elif args.command == 'validate':
        import argparse as _argparse
        val_args = _argparse.Namespace(
            model_path=args.model_path,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # 调用validate逻辑
        import torch
        from pathlib import Path
        from torch.utils.data import DataLoader
        from data.detection_dataset import DetectionDataset, custom_collate_fn
        from models.default_model import DummyModel
        from validate import validate_model, calculate_metrics
        from config import ModelConfig as config
        
        device = torch.device(val_args.device if torch.device(val_args.device).type == 'cuda' else "cuda" if torch.cuda.is_available() else "cpu")
        class_names = ["wall", "door", "window", "column"]
        
        val_dataset = DetectionDataset(
            img_dir=val_args.val_dir,
            csv_dir=val_args.val_dir,
            class_names=class_names,
            training=False
        )
        val_loader = DataLoader(val_dataset, batch_size=val_args.batch_size, shuffle=False, 
                                num_workers=0, collate_fn=custom_collate_fn)
        
        stride = config.stride
        model = DummyModel(num_classes=4, stride=stride).to(device)
        model.load_state_dict(torch.load(val_args.model_path, map_location=device))
        model.eval()
        
        print(f"Loading model from: {val_args.model_path}")
        print(f"Validation data: {val_args.val_dir}")
        
        val_losses = validate_model(model, val_loader, device, len(class_names), stride=stride)
        print(f"Validation Loss - Total: {val_losses['total']:.4f}, "
              f"Heatmap: {val_losses['hm']:.4f}, "
              f"Width/Height: {val_losses['wh']:.4f}, "
              f"Offset: {val_losses['off']:.4f}")
        
        metrics = calculate_metrics(model, val_loader, device, class_names, stride=stride)
        print(f"mAP: {metrics['mAP']:.4f}")
        for class_name, ap in metrics['AP_per_class'].items():
            print(f"  {class_name} AP: {ap:.4f}")

if __name__ == '__main__':
    main()
