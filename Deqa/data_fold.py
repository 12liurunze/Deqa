import json
import numpy as np
from sklearn.model_selection import KFold

# 加载数据集
with open('/home/lrz/Deqa/data/train_diqa_color_fidelity.json', 'r') as f:
    data = json.load(f)  # 假设是列表格式的JSON

# 提取关键信息（图像路径和分数）
image_paths = [item['image'] for item in data]
gt_scores = [item['gt_score'] for item in data]
# 初始化 5-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(image_paths))  # 生成索引分割

# 保存每个 Fold 的数据
for fold, (train_idx, val_idx) in enumerate(folds):
    # 分割数据
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    
    # 保存为 JSON
    with open(f'/home/lrz/Deqa/data/fold/color/fold_{fold+1}_train_co.json', 'w') as f:
        json.dump(train_data, f)
# 检查每个 Fold 的大小
for fold in range(5):
    train_size = len(folds[fold][0])
    val_size = len(folds[fold][1])
    print(f"Fold {fold+1}: Train={train_size}, Val={val_size}")

# 输出示例：
# Fold 1: Train=2800, Val=700
# Fold 2: Train=2800, Val=700
# ...