import json
from collections import defaultdict

# 1. 读取所有文件
file_names = ['/home/lrz/Deqa/results/color_1_new.json', 
              '/home/lrz/Deqa/results/color_2_new.json', 
              '/home/lrz/Deqa/results/color_3_new.json', 
              '/home/lrz/Deqa/results/color_4_new.json', 
              '/home/lrz/Deqa/results/color_5_new.json']
data = []

for file_name in file_names:
    with open(file_name, 'r') as f:
        data.extend(json.load(f))  # 合并所有数据

# 2. 按 image 分组并计算平均值
image_scores = defaultdict(list)
for item in data:
    image_scores[item['image']].append(item['fidelity'])

averages = [
    {"image": image, "fidelity": sum(scores) / len(scores)}
    for image, scores in image_scores.items()
]

# 3. 保存结果
with open('/home/lrz/Deqa/results/averages_3_new.json', 'w') as f:
    json.dump(averages, f, indent=2)

print("平均值计算完成，结果已保存到 averages.json")