import torch

def main():
    # 延迟导入避免循环
    from ablation import run_ablation_experiments, ABLATION_CONFIGS
    from train import LLM_TRAIN_CONFIG

    # 调整基础配置
    LLM_TRAIN_CONFIG["dataset_name"] = "MVSADataSet"
    LLM_TRAIN_CONFIG["batch_size"] = 10  # 适配ViT-H/14减小batch
    LLM_TRAIN_CONFIG["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 运行消融实验
    results = run_ablation_experiments(LLM_TRAIN_CONFIG, ABLATION_CONFIGS)

    # 输出关键结论
    baseline_acc = results[results["experiment_name"] == "Baseline (Full Model)"]["test_acc"].values[0]
    print(f"\n基础模型测试集ACC: {baseline_acc:.4f}")
    print("\n各模块贡献（相对基础模型的ACC变化）：")
    for _, row in results.iterrows():
        if row["experiment_name"] != "Baseline (Full Model)":
            acc_diff = row["test_acc"] - baseline_acc
            print(f"{row['experiment_name']}: {acc_diff:+.4f}")

if __name__ == "__main__":
    main()