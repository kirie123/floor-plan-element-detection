# utils/similarity.py
import os
from pathlib import Path
from typing import List, Dict
import torch
from torch.nn.functional import cosine_similarity
from transformers.image_utils import load_image

from models.dinov3 import dinov3_model


def compute_cross_folder_similarity(
        folder_a: str,
        folder_b: str,
        top_k: int = 5
) -> Dict[str, List[tuple]]:
    """
    计算 folder_a 中每张图与 folder_b 中所有图的相似度

    Returns:
        {
            "a1.jpg": [("b3.jpg", 0.92), ("b1.jpg", 0.88), ...],
            ...
        }
    """
    # 获取所有图片路径
    paths_a = [str(p) for p in Path(folder_a).glob("*") if p.suffix.lower() in {".jpg", ".png"}]
    paths_b = [str(p) for p in Path(folder_b).glob("*") if p.suffix.lower() in {".jpg", ".png"}]

    if not paths_a or not paths_b:
        raise ValueError("One or both folders are empty")

    # 提取 A 和 B 的全局特征
    print(f"Extracting features from {len(paths_a)} images in A...")
    feats_a = []
    for p in paths_a:
        outputs = dinov3_model.forward(p)
        feats_a.append(outputs.pooler_output)  # [1, 1024]
    feats_a = torch.cat(feats_a, dim=0)  # [N, 1024]

    print(f"Extracting features from {len(paths_b)} images in B...")
    feats_b = []
    for p in paths_b:
        outputs = dinov3_model.forward(p)
        feats_b.append(outputs.pooler_output)
    feats_b = torch.cat(feats_b, dim=0)  # [M, 1024]

    # 计算余弦相似度矩阵 [N, M]
    sim_matrix = cosine_similarity(feats_a.unsqueeze(1), feats_b.unsqueeze(0), dim=-1)

    # 为每个 A 图像找 top-k 最相似的 B 图像
    result = {}
    for i, a_path in enumerate(paths_a):
        scores, indices = torch.topk(sim_matrix[i], min(top_k, len(paths_b)))
        matches = [
            (os.path.basename(paths_b[idx]), float(score))
            for score, idx in zip(scores, indices)
        ]
        result[os.path.basename(a_path)] = matches

    return result


# 使用示例
if __name__ == "__main__":
    # 图例文件夹
    path_a = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\图例"
    # 目标文件夹
    path_b = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\目标"
    similarities = compute_cross_folder_similarity(
        folder_a=path_a,  # 图例文件夹
        folder_b=path_b  # 候选实例文件夹
    )
    for a_img, matches in similarities.items():
        print(f"{a_img} 最相似的图:")
        for b_img, score in matches:
            print(f"  - {b_img}: {score:.3f}")