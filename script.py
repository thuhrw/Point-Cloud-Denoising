import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

NOISY_ROOT = "/home/ubuntu/starter_code/dataset_test_noisy/shapenet"
DENOISED_ROOT = "/home/ubuntu/starter_code/tmp_predict/dataset_test_noisy/shapenet"
POINT_TARGET = 50000
K_NEIGHBORS = 16  # 更强补全
USE_WEIGHTED = True  # 加权平均（效果提升关键）

all_denoised = []
need_fix_count = 0

for root, dirs, files in os.walk(DENOISED_ROOT):
    for f in files:
        if f == "denoised.npy":
            all_denoised.append(os.path.join(root, f))

print(f"✅ 找到 {len(all_denoised)} 个点云")

for denoised_path in tqdm(all_denoised):
    rel_path = os.path.relpath(denoised_path, DENOISED_ROOT)
    rel_dir = os.path.dirname(rel_path)
    noisy_path = os.path.join(NOISY_ROOT, rel_dir, "noisy.npy")

    denoised = np.load(denoised_path)
    noisy = np.load(noisy_path)

    if denoised.shape[0] < POINT_TARGET:
        need_fix_count += 1
        missing = POINT_TARGET - denoised.shape[0]
        query_points = noisy[-missing:]

        # ===================== 【增强版补全】 =====================
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(denoised)
        distances, indices = nbrs.kneighbors(query_points)

        if USE_WEIGHTED:
            # 加权平均（距离越近权重越大）→ 更平滑、更干净
            weights = np.exp(-distances ** 2 / 0.005)
            weights = weights / weights.sum(axis=1, keepdims=True)
            fill_points = np.einsum("nk,nkd->nd", weights, denoised[indices])
        else:
            fill_points = denoised[indices[:, 0]]

        full = np.concatenate([denoised, fill_points], axis=0)
        full = full[:POINT_TARGET]
        np.save(denoised_path, full.astype(np.float32))

print(f"\n✅ 增强补全完成！修复数量：{need_fix_count}")