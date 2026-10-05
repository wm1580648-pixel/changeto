

import os
import json
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline
from tqdm import tqdm
import csv

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化情感分析管道
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
    device=device
)

# 初始化CLIP模型（适配ViT-H/14）
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # 兼容ViT-H/14输入
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_model.eval()

def generate_llm_rationales(img_path, text_content, label):
    """
    生成LLM结构化推理结果
    :param img_path: 图片路径
    :param text_content: 文本内容
    :param label: 原始标签（str: negative/neutral/positive）
    :return: 结构化推理字典
    """
    rationales = {
        "TextEvidence": [],
        "ImageEvidence": [],
        "CrossModalRelation": "neutral",
        "FinalLabel": label,
        "Confidence": 0.5,
        "RationaleSummary": "",
        "img_id": ""
    }

    # 1. 文本证据提取
    if text_content:
        try:
            sentiment_results = sentiment_analyzer(text_content[:512])
            text_evidence = [(text_content[:100], sentiment_results[0]['label'])]
            rationales["TextEvidence"] = text_evidence
            rationales["Confidence"] = sentiment_results[0]['score']
        except:
            rationales["TextEvidence"] = [(text_content[:100], "NEUTRAL")]

    # 2. 图像证据提取（基于CLIP的简单语义匹配）
    if img_path and os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_feat = clip_model.get_image_features(**inputs)

            # 情感关键词匹配
            sentiment_texts = ["negative emotion", "neutral emotion", "positive emotion"]
            text_inputs = clip_processor(text=sentiment_texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_feat = clip_model.get_text_features(**text_inputs)

            # 计算相似度
            sim = torch.matmul(F.normalize(image_feat), F.normalize(text_feat).T)
            max_sim_idx = torch.argmax(sim).item()
            img_polarity = ["negative", "neutral", "positive"][max_sim_idx]
            rationales["ImageEvidence"] = [{"Object": "image_content", "Polarity": img_polarity}]
        except:
            rationales["ImageEvidence"] = [{"Object": "unknown", "Polarity": "neutral"}]

    # 3. 跨模态关系判断
    text_polarity = rationales["TextEvidence"][0][1].lower() if rationales["TextEvidence"] else "neutral"
    img_polarity = rationales["ImageEvidence"][0]["Polarity"].lower() if rationales["ImageEvidence"] else "neutral"

    if text_polarity == img_polarity:
        rationales["CrossModalRelation"] = "support"
    elif (text_polarity in ["positive", "negative"]) and (
            img_polarity in ["positive", "negative"]) and text_polarity != img_polarity:
        rationales["CrossModalRelation"] = "conflict"
    elif text_polarity == "neutral" and img_polarity != "neutral":
        rationales["CrossModalRelation"] = "complement"
    elif text_polarity != "neutral" and img_polarity == "neutral":
        rationales["CrossModalRelation"] = "complement"
    else:
        rationales["CrossModalRelation"] = "neutral"

    # 4. 推理总结
    rationales["RationaleSummary"] = f"Text: {text_polarity}, Image: {img_polarity}, Relation: {rationales['CrossModalRelation']}"

    return rationales

def precompute_llm_rationales(dataset_root):
    """为所有数据集生成LLM推理标签"""
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    dataset_names = ["MVSADataSet", "MVSASINGLEDataSet", "Twitter"]
    splits = ["train", "val", "test"]

    for ds_name in dataset_names:
        for split in splits:
            split_dir = os.path.join(dataset_root, ds_name, split)
            csv_path = os.path.join(split_dir, "annotations.csv")
            rationales_save_path = os.path.join(split_dir, "llm_rationales.json")

            if not os.path.exists(csv_path):
                print(f"⚠️ 未找到{csv_path}，跳过该数据集")
                continue

            # 读取CSV
            data = []
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    img_id = row[0].strip('\ufeff').strip()
                    text_label_str = row[1].lower()
                    img_label_str = row[2].lower()
                    data.append((img_id, text_label_str, img_label_str))

            # 生成推理标签
            rationales_dict = {}
            for img_id, text_label_str, img_label_str in tqdm(data, desc=f"处理{ds_name}/{split}"):
                # 加载文本
                text_path = os.path.join(split_dir, "texts", f"{img_id}.txt")
                text_content = ""
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read().strip()

                # 加载图像
                img_path = os.path.join(split_dir, "images", f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(split_dir, "images", f"{img_id}.png")

                # 生成推理标签
                final_label = "neutral"
                if text_label_str != "neutral":
                    final_label = text_label_str
                elif img_label_str != "neutral":
                    final_label = img_label_str

                rationales = generate_llm_rationales(img_path, text_content, final_label)
                rationales["img_id"] = img_id
                rationales_dict[img_id] = rationales

            # 保存推理标签
            with open(rationales_save_path, 'w', encoding='utf-8') as f:
                json.dump(rationales_dict, f, indent=4, ensure_ascii=False)
            print(f"✅ 保存{ds_name}/{split}推理标签到{rationales_save_path}")

# 生成推理标签（按需运行）
if __name__ == "__main__":
    precompute_llm_rationales(r"D:/data")