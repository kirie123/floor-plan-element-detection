# train_wall_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm

from data.wall_classifier import get_transforms, WallClassificationDataset, WallClassifier


def train_classifier(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    """训练分类器"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 更新历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_wall_classifier.pth')
            print(f'  🎉 新的最佳模型! 准确率: {best_acc:.2f}%')

        scheduler.step()

    return history, best_acc


def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/classification_data_enhanced/wall_classification"
    batch_size = 16
    num_epochs = 50

    # 数据加载
    train_transform, val_transform = get_transforms(224)
    dataset = WallClassificationDataset(data_dir, transform=train_transform)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 为验证集设置不同的transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 模型
    num_classes = len(dataset.classes)
    model = WallClassifier(num_classes=num_classes, backbone='efficientnet_b0').to(device)

    # 训练
    history, best_acc = train_classifier(
        model, train_loader, val_loader, device, num_epochs
    )

    # 保存训练历史
    with open('classification_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"类别映射: {dataset.class_to_idx}")


if __name__ == "__main__":
    main()