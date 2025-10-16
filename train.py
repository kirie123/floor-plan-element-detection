# train.py
import torch
from data.detection_dataset import DetectionDataset
from loss import detection_loss

from models.default_model import DummyModel


from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from pathlib import Path

from validate import validate_model, calculate_metrics


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

def custom_collate_fn(batch):
    """
    简单的整理函数，将图像堆叠，目标保持为列表
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # 堆叠图像
    images = torch.stack(images, dim=0)

    return images, targets
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["wall", "door", "window", "column"]
    num_classes = len(class_names)

    # 创建输出目录
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)

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
    stride = 16
    model = DummyModel(num_classes=num_classes, stride = stride).to(device)
    # # 冻结主干网络
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # optimizer = torch.optim.AdamW([
    #     {'params': model.backbone.parameters(), 'lr': 1e-5},  # 极低的学习率
    #     {'params': model.heatmap_head.parameters(), 'lr': 1e-4},
    #     {'params': model.wh_head.parameters(), 'lr': 1e-4},
    #     {'params': model.offset_head.parameters(), 'lr': 1e-4},
    # ], weight_decay=1e-3)
    max_epoch = 200
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

    # 训练循环
    best_loss = float('inf')
    train_history = []
    val_interval = 5  # 每5个epoch验证一次
    for epoch in range(max_epoch):
        print(f"\n--- Epoch {epoch + 1}/{max_epoch} ---")

        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch + 1, stride = stride)
        print(f"训练损失 - 总: {train_losses['total']:.4f}, "
              f"热力图: {train_losses['hm']:.4f}, "
              f"宽高: {train_losses['wh']:.4f}, "
              f"偏移: {train_losses['off']:.4f}")

        # 验证
        if has_validation and (epoch + 1) % val_interval == 0:
            print("开始验证...")
            val_losses = validate_model(model, val_loader, device, num_classes, stride = stride)
            print(f"验证损失 - 总: {val_losses['total']:.4f}, "
                  f"热力图: {val_losses['hm']:.4f}, "
                  f"宽高: {val_losses['wh']:.4f}, "
                  f"偏移: {val_losses['off']:.4f}")

            # 计算指标
            metrics = calculate_metrics(model, val_loader, device, class_names, stride = stride)
            print(f"验证指标 - mAP: {metrics['mAP']:.4f}")
            for class_name, ap in metrics['AP_per_class'].items():
                print(f"  {class_name} AP: {ap:.4f}")
        # 学习率调度
        scheduler.step()

        # 记录历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_losses,
            'lr': scheduler.get_last_lr()[0]
        })

        # 保存检查点
        if epoch % 5 == 0:
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
            best_model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 新的最佳模型! Loss: {best_loss:.4f}")

        # 保存训练历史
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(train_history, f, indent=2)


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