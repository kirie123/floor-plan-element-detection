# train.py
import torch
from data.detection_dataset import DetectionDataset, custom_collate_fn
from loss import detection_loss
from models.default_model import DummyModel
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from pathlib import Path

from models.sta_adapter import DINOv3STADetector
from validate import validate_model, calculate_metrics
from torch.utils.tensorboard import SummaryWriter  # 新增：用于可视化
from config import ModelConfig as config

def train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq=10, stride = 16):
    model.train()
    total_loss = 0
    hm_loss_total = 0
    wh_loss_total = 0
    off_loss_total = 0
    class_names = ["wall", "door", "window", "column"]
    for batch_idx, batch  in enumerate(dataloader):
        # 解包batch
        images, targets = batch

        # 将images移动到设备 - 现在images是张量
        images = images.to(device)
        # targets已经是字典列表，不需要额外处理

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_dict = detection_loss(outputs, targets, num_classes=len(class_names), stride = stride)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total"].item()
        hm_loss_total += loss_dict["hm"].item()
        wh_loss_total += loss_dict["wh"].item()
        off_loss_total += loss_dict["off"].item()

        if batch_idx % print_freq == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss_dict['total'].item():.4f} "
                  f"(hm: {loss_dict['hm'].item():.4f}, "
                  f"wh: {loss_dict['wh'].item():.4f}, "
                  f"off: {loss_dict['off'].item():.4f})")

    avg_loss = total_loss / len(dataloader)
    avg_hm = hm_loss_total / len(dataloader)
    avg_wh = wh_loss_total / len(dataloader)
    avg_off = off_loss_total / len(dataloader)

    return {
        "total": avg_loss,
        "hm": avg_hm,
        "wh": avg_wh,
        "off": avg_off
    }


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, path)
    print(f"✅ 检查点保存: {path}")



def freeze_backbone(model):
    """冻结主干网络"""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            print(f"冻结层: {name}")

def unfreeze_backbone(model):
    """解冻主干网络"""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = True
            print(f"解冻层: {name}")

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["wall", "door", "window", "column"]
    num_classes = len(class_names)

    # 创建输出目录
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)
    best_model_dir = Path("training_output")
    best_model_dir.mkdir(exist_ok=True)
    # 新增：TensorBoard日志记录器
    writer = SummaryWriter(log_dir=output_dir / "tensorboard_logs")
    # 数据集
    train_dataset = DetectionDataset(
        img_dir="data/cvt_images",
        csv_dir="data/cvt_images",
        class_names=class_names,
        training=True
    )
    has_validation = True
    # 验证集（如果有的话）
    val_dataset = DetectionDataset(
        img_dir="data/val_images",  # 你需要准备验证集
        csv_dir="data/val_images",
        class_names=class_names,
        training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0,
                            collate_fn=custom_collate_fn)

    # 模型（先用你的DummyModel测试，后续替换为DINOv3）
    stride = config.stride
    # resnet backbone
    #model = DummyModel(num_classes=num_classes, stride = stride).to(device)
    # dinov3 backbone
    model =DINOv3STADetector(num_classes=4).to(device)
    # 打印总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    is_only_val = False
    if is_only_val:
        model.load_state_dict(torch.load("training_output/best_detect_model.pth", map_location=device))
        model.eval()
        print("开始验证...")
        val_losses = validate_model(model, val_loader, device, num_classes, stride=stride)
        print(f"验证损失 - 总: {val_losses['total']:.4f}, "
              f"热力图: {val_losses['hm']:.4f}, "
              f"宽高: {val_losses['wh']:.4f}, "
              f"偏移: {val_losses['off']:.4f}")
        # 计算指标
        metrics = calculate_metrics(model, val_loader, device, class_names, stride=stride)
        print(f"验证指标 - mAP: {metrics['mAP']:.4f}")
        for class_name, ap in metrics['AP_per_class'].items():
            print(f"  {class_name} AP: {ap:.4f}")

    # # 冻结主干网络
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # 优化器
    #optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # optimizer = torch.optim.AdamW([
    #     {'params': model.backbone.parameters(), 'lr': 1e-5},  # 极低的学习率
    #     {'params': model.heatmap_head.parameters(), 'lr': 1e-4},
    #     {'params': model.wh_head.parameters(), 'lr': 1e-4},
    #     {'params': model.offset_head.parameters(), 'lr': 1e-4},
    # ], weight_decay=1e-3)
    max_epoch = 300
    phase1_epochs = 0  # 阶段1：冻结主干，训练头
    phase2_epochs = 300  # 阶段2：解冻主干，小学习率
    #scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    # 阶段1：冻结主干网络
    print("=== 阶段1：冻结主干网络，只训练检测头 ===")
    freeze_backbone(model)

    # 优化器 - 阶段1只优化检测头
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' not in name and param.requires_grad:
            head_params.append(param)

    optimizer = optim.AdamW(head_params, lr=1e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-6)
    # 训练循环
    best_loss = float('inf')
    train_history = []
    val_interval = 5  # 每5个epoch验证一次
    for epoch in range(max_epoch):
        print(f"\n--- Epoch {epoch + 1}/{max_epoch} ---")
        # 阶段切换逻辑
        if epoch == phase1_epochs:
            print("=== 阶段2：解冻主干网络，完整微调 ===")
            unfreeze_backbone(model)
            # 重新定义优化器，包含所有参数但使用不同学习率
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': 1e-4},  # 主干网络用小学习率
                {'params': head_params, 'lr': 1e-4}  # 检测头用较大学习率
            ], weight_decay=1e-3)
            scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
            # 带热重启的余弦退火
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=10,  # 10个epoch后重启
            #     T_mult=2,
            #     eta_min=1e-6
            # )
        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch + 1, stride = stride)
        print(f"训练损失 - 总: {train_losses['total']:.4f}, "
              f"热力图: {train_losses['hm']:.4f}, "
              f"宽高: {train_losses['wh']:.4f}, "
              f"偏移: {train_losses['off']:.4f}")
        # 新增：记录训练损失到TensorBoard
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train_hm', train_losses['hm'], epoch)
        writer.add_scalar('Loss/train_wh', train_losses['wh'], epoch)
        writer.add_scalar('Loss/train_off', train_losses['off'], epoch)
        # 记录学习率
        current_lr = scheduler.get_last_lr()[0] if epoch < phase1_epochs else optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch)
        # 验证
        if has_validation and (epoch + 1) % val_interval == 0:
            print("开始验证...")
            val_losses = validate_model(model, val_loader, device, num_classes, stride = stride)
            print(f"验证损失 - 总: {val_losses['total']:.4f}, "
                  f"热力图: {val_losses['hm']:.4f}, "
                  f"宽高: {val_losses['wh']:.4f}, "
                  f"偏移: {val_losses['off']:.4f}")
            # 新增：记录验证损失到TensorBoard
            writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
            writer.add_scalar('Loss/val_hm', val_losses['hm'], epoch)
            writer.add_scalar('Loss/val_wh', val_losses['wh'], epoch)
            writer.add_scalar('Loss/val_off', val_losses['off'], epoch)
            # 计算指标
            metrics = calculate_metrics(model, val_loader, device, class_names, stride = stride)
            print(f"验证指标 - mAP: {metrics['mAP']:.4f}")
            writer.add_scalar('Metrics/mAP', metrics['mAP'], epoch)
            for class_name, ap in metrics['AP_per_class'].items():
                print(f"  {class_name} AP: {ap:.4f}")
                writer.add_scalar(f'Metrics/AP_{class_name}', ap, epoch)
        # 学习率调度
        scheduler.step()

        # 记录历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses,
            'lr': scheduler.get_last_lr()[0]
        })

        # 保存检查点
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses,
            }, checkpoint_path)
            print(f"✅ 检查点保存: {checkpoint_path}")

        # 保存最佳模型
        current_loss = train_losses['total']
        if has_validation and (epoch + 1) % val_interval == 0:
            current_loss = val_losses['total']  # 使用验证损失选择最佳模型

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = best_model_dir / "best_detect_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 新的最佳模型! Loss: {best_loss:.4f}")

        # 保存训练历史
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(train_history, f, indent=2)
    # 新增：关闭TensorBoard写入器
    writer.close()
    print("训练完成！使用 'tensorboard --logdir training_output/tensorboard_logs' 查看训练曲线")

if __name__ == "__main__":
    import sys
    import torch
    import torchvision
    import numpy as np
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("Torchvision版本:", torchvision.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    print("CUDA版本:", torch.version.cuda)
    print("numpy版本:", np.__version__)
    print("设备:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    main()