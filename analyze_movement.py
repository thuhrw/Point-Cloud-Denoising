#!/usr/bin/env python3
"""分析预测前后点云的平均移动距离"""
import numpy as np
import os
from pathlib import Path

# 指定路径
NOISY_DIR = Path("./dataset_test_noisy")
PRED_DIR = Path("./results/dataset_test_noisy/shapenet")

def compute_movement_distance(noisy_path, pred_path):
    """计算单个点云的平均移动距离"""
    noisy = np.load(noisy_path)
    pred = np.load(pred_path)

    # 确保形状匹配
    if noisy.shape != pred.shape:
        print(f"  形状不匹配: noisy {noisy.shape} vs pred {pred.shape}")
        return None

    # 计算欧氏距离
    diff = pred - noisy
    distances = np.sqrt((diff ** 2).sum(axis=1))
    mean_dist = distances.mean()
    std_dist = distances.std()
    max_dist = distances.max()

    return mean_dist, std_dist, max_dist

def analyze_directory():
    """分析所有点云的移动距离"""
    results = []

    # 获取所有预测文件
    pred_files = sorted(list(PRED_DIR.glob("**/*.npy")))

    print(f"找到 {len(pred_files)} 个预测文件")
    print("=" * 60)

    for pred_file in pred_files:
        # 只处理 denoised.npy 文件
        if pred_file.name != "denoised.npy":
            continue

        # 构造对应的noisy文件路径
        rel_path = pred_file.relative_to(PRED_DIR)
        # rel_path 格式: 类别ID/样本ID/denoised.npy
        parts = list(rel_path.parts)
        # noisy文件路径: dataset_test_noisy/shapenet/类别ID/样本ID/noisy.npy
        noisy_file = NOISY_DIR / 'shapenet' / parts[0] / parts[1] / 'noisy.npy'

        if not noisy_file.exists():
            print(f"跳过 {rel_path}: 找不到对应的noisy文件")
            continue

        result = compute_movement_distance(noisy_file, pred_file)
        if result is not None:
            mean_dist, std_dist, max_dist = result
            results.append(mean_dist)
            print(f"{rel_path}: mean={mean_dist:.4f}, std={std_dist:.4f}, max={max_dist:.4f}")

    print("=" * 60)
    if results:
        results = np.array(results)
        print(f"\n总体统计:")
        print(f"  平均移动距离: {results.mean():.4f}")
        print(f"  标准差: {results.std():.4f}")
        print(f"  最小值: {results.min():.4f}")
        print(f"  最大值: {results.max():.4f}")

    return results

if __name__ == "__main__":
    analyze_directory()
