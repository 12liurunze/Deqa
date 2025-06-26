import argparse
import torch
import json
from tqdm import tqdm
from collections import defaultdict
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def wa5(logits):
    """计算加权平均分数（假设 logits 包含 'excellent', 'good', 'fair', 'poor', 'bad'）"""
    import numpy as np
    keys = ["excellent", "good", "fair", "poor", "bad"]
    logprobs = np.array([logits[k] for k in keys])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([5, 4, 3, 2, 1]))

def disable_torch_init():
    """禁用冗余的 PyTorch 初始化以加速模型加载"""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_image(image_file):
    return Image.open(image_file).convert('RGB')

def expand2square(pil_img, background_color):
    """将图像填充为正方形"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def main(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # 数据路径
    image_path = "/home/lrz/Q-Align-main/playground/DIQA-5000_phase1/val/res/"
    json_file = "/home/lrz/Q-Align-main/playground/data/converted_dataset_val.json"
    os.makedirs(f"/home/lrz/Deqa/results", exist_ok=True)

    # 加载数据
    with open(json_file) as f:
        iqadata = json.load(f)

    # 初始化对话模板
    conv_mode = "mplug_owl2"
    conv = conv_templates[conv_mode].copy()
    inp = DEFAULT_IMAGE_TOKEN + "How would you rate the color-fidelity of this image?"
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The color-fidelity of the image is"

    # 定义评分词汇（确保与 wa5 函数匹配）
    toks = ["excellent", "good", "fair", "poor", "bad"]  # 仅保留 5 个词以匹配 wa5
    ids_ = [tokenizer(tok, return_tensors="pt")["input_ids"][0][1] for tok in toks]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    # 评估循环
    image_tensors, batch_data = [], []
    output_file = f"/home/lrz/Deqa/results/color_5_new.json"
    results = []

    for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating")):
        filename = llddata["image"]
        llddata["logits"] = defaultdict(float)

        # 加载并预处理图像
        image = load_image(image_path + filename)
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)
        image_tensors.append(image_tensor)
        batch_data.append(llddata)

        # 批处理推理（每 8 张图像或最后一批）
        if len(image_tensors) == 2 or i == len(iqadata) - 1:
            with torch.inference_mode():
                output_logits = model(
                    input_type=None,
                    input_ids=input_ids.repeat(len(image_tensors), 1),
                    images=torch.cat(image_tensors, 0)
                )["logits"][:, -1]

            # 处理每张图像的 logits
            for j, xllddata in enumerate(batch_data):
                for tok, id_ in zip(toks, ids_):
                    xllddata["logits"][tok] = output_logits[j, id_].item()
                xllddata["score"] = wa5(xllddata["logits"])
                results.append({
                    "image": xllddata["image"],
                    "fidelity": xllddata["score"],
                   
                })

            image_tensors, batch_data = [], []

    # 保存结果
    with open(output_file, "w") as wf:
        json.dump(results, wf, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/lrz/Deqa/output/model/deqa_lora_color_5_new")
    parser.add_argument("--model-base", type=str, default="/home/lrz/Deqa/model_weight")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)