# predict_wall_type.py
import torch
from PIL import Image
import torch.nn.functional as F

from data.wall_classifier import WallClassifier, get_transforms


class WallTypePredictor:
    """墙类型预测器"""

    def __init__(self, model_path, class_names, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)

        # 加载模型
        self.model = WallClassifier(num_classes=self.num_classes, backbone='efficientnet_b0')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 数据变换
        _, self.transform = get_transforms(224)

        print(f"✅ 加载墙分类器, 类别数: {self.num_classes}")

    def predict(self, image):
        """预测单张图像"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            pred_prob, pred_class = torch.max(probabilities, 1)

        return {
            'class_name': self.class_names[pred_class.item()],
            'class_idx': pred_class.item(),
            'confidence': pred_prob.item(),
            'all_probs': probabilities.cpu().numpy()[0]
        }

    def predict_batch(self, image_patches):
        """批量预测"""
        results = []
        for patch in image_patches:
            results.append(self.predict(patch))
        return results


# 使用示例
if __name__ == "__main__":
    # 假设的类别名称（根据你的实际WALL类型调整）
    wall_classes = [f"WALL{i}" for i in range(1, 24)]

    # 初始化预测器
    predictor = WallTypePredictor('best_wall_classifier.pth', wall_classes)

    # 预测单张图像
    result = predictor.predict("path_to_wall_patch.jpg")
    print(f"预测结果: {result['class_name']}, 置信度: {result['confidence']:.4f}")