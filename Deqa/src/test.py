import json

with open('/home/lrz/Deqa/data/train_diqa_color_fidelity.json', 'r') as f1:
    list1 = json.load(f1)  # list1 是列表

with open('/home/lrz/Deqa/data/val_fidelity.json', 'r') as f2:
    list2 = json.load(f2)  # list2 是列表

merged_list = list1 + list2  # 合并列表

with open('/home/lrz/Deqa/data/merged_fidelity.json', 'w') as out:
    json.dump(merged_list, out, indent=4)