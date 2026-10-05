import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from common import CLIPTeacher, EMA, AblationR3MSentLoss
from model import R3MSentModel, CrossModalFusion

# -------------------------- 消融实验配置 --------------------------
ABLATION_CONFIGS = {
    # 基准模型（全功能）
    "Baseline (Full Model)": {
        "use_relation_distill": True,
        "use_evidence_distill": True,
        "use_semi_consistency": True,
        "lambda_clip": 1.5,
        "cross_modal_type": "cross_attention"
    },
    # 消融1：移除Cross-Attention（改用简单拼接）
    "Ablation - No Cross-Attention": {
        "use_relation_distill": True,
        "use_evidence_distill": True,
        "use_semi_consistency": True,
        "lambda_clip": 1.5,
        "cross_modal_type": "concat"
    },
    # 消融2：移除CLIP蒸馏
    "Ablation - No CLIP Distill": {
        "use_relation_distill": True,
        "use_evidence_distill": True,
        "use_semi_consistency": True,
        "lambda_clip": 0.0,
        "cross_modal_type": "cross_attention"
    },
    # 消融3：移除关系蒸馏
    "Ablation - No Relation Distill": {
        "use_relation_distill": False,
        "use_evidence_distill": True,
        "use_semi_consistency": True,
        "lambda_clip": 1.5,
        "cross_modal_type": "cross_attention"
    },
    # 消融4：移除证据蒸馏
    "Ablation - No Evidence Distill": {
        "use_relation_distill": True,
        "use_evidence_distill": False,
        "use_semi_consistency": True,
        "lambda_clip": 1.5,
        "cross_modal_type": "cross_attention"
    },
    # 消融5：移除半监督一致性
    "Ablation - No Semi Consistency": {
        "use_relation_distill": True,
        "use_evidence_distill": True,
        "use_semi_consistency": False,
        "lambda_clip": 1.5,
        "cross_modal_type": "cross_attention"
    },
    # 消融6：仅基础模型（无任何蒸馏/融合）
    "Ablation - Only Base Model": {
        "use_relation_distill": False,
        "use_evidence_distill": False,
        "use_semi_consistency": False,
        "lambda_clip": 0.0,
        "cross_modal_type": "concat"
    }
}

# -------------------------- 适配消融的模型重载 --------------------------
class AblationR3MSentModel(R3MSentModel):
    """适配消融实验的模型：支持切换跨模态融合方式（768维）"""

    def __init__(self, cross_modal_type="cross_attention"):
        super().__init__()
        # 重载跨模态融合模块
        if cross_modal_type == "concat":
            # 仅简单拼接（无Cross-Attention）
            self.fusion = nn.Sequential(
                nn.Linear(1536, 768),  # 768*2=1536
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif cross_modal_type == "cross_attention":
            # 原始Cross-Attention融合
            self.fusion = CrossModalFusion(feat_dim=768, num_heads=12)
        self.cross_modal_type = cross_modal_type

    def forward(self, image, text, is_neutral=None):
        return super().forward(image, text, is_neutral)

# -------------------------- 消融实验主函数 --------------------------
def run_ablation_experiments(base_config, ablation_configs):
    from train import train_large_model, LLM_TRAIN_CONFIG

    def json_safe(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_safe(v) for v in obj]
        else:
            return obj

    ablation_results = []
    save_root = "ablation_results"
    os.makedirs(save_root, exist_ok=True)

    for exp_name, exp_config in tqdm(ablation_configs.items(), desc="Running Ablation Experiments"):
        print(f"\n========== 开始消融实验：{exp_name} ==========")
        current_config = base_config.copy()
        current_config.update(exp_config)
        current_config["save_dir"] = os.path.join(save_root, exp_name.replace(" ", "_").lower())
        os.makedirs(current_config["save_dir"], exist_ok=True)

        # 重载模型和损失类
        global R3MSentModel
        R3MSentModel = lambda: AblationR3MSentModel(cross_modal_type=current_config["cross_modal_type"])
        global R3MSentLoss
        R3MSentLoss = lambda config: AblationR3MSentLoss(config)

        try:
            trained_model, metrics = train_large_model(current_config)
            result = {
                "experiment_name": exp_name,
                "best_val_acc": max(metrics["val_acc"]),
                "test_acc": metrics["test_acc"],
                "test_loss": metrics["test_loss"],
                "train_acc_last_epoch": metrics["train_acc"][-1],
                "train_loss_last_epoch": metrics["train_loss"][-1],
                "config": json_safe(current_config)
            }
            ablation_results.append(result)

            with open(os.path.join(current_config["save_dir"], "exp_result.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✅ 实验 {exp_name} 完成，测试集ACC: {metrics['test_acc']:.4f}")
        except Exception as e:
            print(f"❌ 实验 {exp_name} 失败：{str(e)}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(ablation_results)
    results_df = results_df.sort_values(by="test_acc", ascending=False)
    results_df.to_csv(os.path.join(save_root, "ablation_summary.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(save_root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(ablation_results), f, indent=4, ensure_ascii=False)

    print("\n========== 消融实验汇总 ==========")
    print(results_df[["experiment_name", "best_val_acc", "test_acc"]])
    return results_df

# 运行消融实验
if __name__ == "__main__":
    from train import LLM_TRAIN_CONFIG

    # 调整消融实验的训练配置
    base_config = LLM_TRAIN_CONFIG.copy()
    base_config["epochs"] = 5
    base_config["patience"] = 2
    # 运行所有消融实验
    ablation_results_df = run_ablation_experiments(base_config, ABLATION_CONFIGS)