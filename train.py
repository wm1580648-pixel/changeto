import math
import os
from collections import Counter
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from common import CLIPTeacher, EMA, R3MSentLoss
from dataset import build_dataloader
from model import R3MSentModel


LLM_TRAIN_CONFIG = {
    "dataset_root": r"D:/data",
    "dataset_name": r"Twitter",
    "batch_size": 10,
    "gradient_accumulation_steps": 2,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "epochs": 10,
    "warmup_epochs": 5,
    "weight_decay":  1e-4,
    "label_smoothing": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 0 if os.name == "nt" else 4,
    "pin_memory": True,
    "label_mode": "merge",  # merge / image / text / rationale_final
    "use_weighted_sampler":False,
    "use_class_weights": True,
    "use_clip_teacher": True,
    "use_relation_distill": True,
    "use_evidence_distill": True,
    "use_semi_consistency": False,
    "lambda_r": 0.1,
    "lambda_e": 0.1,
    "lambda_u": 0.1,
    "lambda_clip": 0.1,
    "lambda_neutral": 0.1,
    "lambda_sentiment": 0.1,
    "tau": 1.0,
    "clip_model_name": "ViT-H-14",
    "clip_pretrained": "laion2b_s32b_b79k",
    "ema_decay": 0.99,
    "patience": 5,
    "save_dir": "llm_train_checkpoints_twitter_fix",
    "log_interval": 20,
    "save_best_only": True,
    "resume_from_checkpoint": None,
    "seed": 42,
    "monitor_metric": "val_acc",  # val_acc
    "target_val_acc": None,
    "target_acc_tolerance": 0.01,
    "target_acc_patience": 5,
    "auto_disable_broken_rationale": True,
}
def _set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_class_weights(dataset, num_classes=3):
    labels = torch.tensor(dataset.get_final_labels(), dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    inv = counts.sum() / counts
    weights = inv / inv.mean()
    return weights


def _move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def _print_dataset_stats(split_name, dataset):
    idx2label = {0: "negative", 1: "neutral", 2: "positive"}
    final_counts = Counter(dataset.get_final_labels())
    text_counts = Counter(item["text_label"] for item in dataset.data_info)
    img_counts = Counter(item["img_label"] for item in dataset.data_info)

    def _fmt(cnt):
        return ", ".join([f"{idx2label[k]}={int(cnt.get(k, 0))}" for k in range(3)])

    print(
        f"[{split_name}] samples={len(dataset)}, label_mode={dataset.label_mode} | "
        f"final: {_fmt(final_counts)} | text: {_fmt(text_counts)} | image: {_fmt(img_counts)}"
    )


def _count_valid_rationale_labels(dataset):
    count = 0
    for item in dataset.data_info:
        raw = dataset.llm_rationales.get(str(item["img_prefix"]), {})
        label = dataset._label_from_string(raw.get("FinalLabel"))
        if label is not None:
            count += 1
    return count


def train_large_model(config):
    config = dict(config)
    os.makedirs(config["save_dir"], exist_ok=True)
    config.setdefault("label_mode", "merge")
    _set_seed(config.get("seed", 42))

    print("Loading data ...")
    train_loader = build_dataloader(
        dataset_root=config["dataset_root"],
        dataset_name=config["dataset_name"],
        split="train",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        use_weighted_sampler=config.get("use_weighted_sampler", False),
        label_mode=config["label_mode"],
    )
    val_loader = build_dataloader(
        dataset_root=config["dataset_root"],
        dataset_name=config["dataset_name"],
        split="val",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        use_weighted_sampler=False,
        label_mode=config["label_mode"],
    )
    test_loader = build_dataloader(
        dataset_root=config["dataset_root"],
        dataset_name=config["dataset_name"],
        split="test",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        use_weighted_sampler=False,
        label_mode=config["label_mode"],
    )

    _print_dataset_stats("train", train_loader.dataset)
    _print_dataset_stats("val", val_loader.dataset)
    _print_dataset_stats("test", test_loader.dataset)
    if config.get("auto_disable_broken_rationale", True):
        train_valid_rationale = _count_valid_rationale_labels(train_loader.dataset)
        train_size = len(train_loader.dataset)
        train_ratio = float(train_valid_rationale) / max(1, train_size)
        print(
            f"[train] valid rationale FinalLabel ratio={train_valid_rationale}/{train_size} ({train_ratio:.4f})"
        )
        if train_ratio < 0.2:
            if str(config.get("label_mode", "")).lower() == "rationale_final":
                print("Auto fallback: rationale labels too sparse, switching label_mode to merge.")
            if config.get("use_relation_distill", False):
                print("Auto fallback: disable relation distill due to sparse/noisy rationales.")
            if config.get("use_evidence_distill", False):
                print("Auto fallback: disable evidence distill due to sparse/noisy rationales.")
            config["label_mode"] = "merge"
            config["use_relation_distill"] = False
            config["use_evidence_distill"] = False
    if str(config.get("label_mode", "merge")).lower() == "merge":
        train_text_counts = Counter(item["text_label"] for item in train_loader.dataset.data_info)
        if sum(v for k, v in train_text_counts.items() if k != 1) == 0:
            print(
                "Warning: train split text labels are all neutral under merge mode; "
                "this usually causes train/val target mismatch and poor generalization."
            )

    if config.get("use_class_weights", False):
        class_weights = _compute_class_weights(train_loader.dataset).to(config["device"])
        config["class_weights"] = class_weights
        print(f"Class weights: {class_weights.tolist()}")
    else:
        config["class_weights"] = None
        print("Class weights: disabled")

    use_clip_teacher = bool(config.get("use_clip_teacher", True))
    if not use_clip_teacher and float(config.get("lambda_clip", 0.0)) > 0:
        print("Warning: use_clip_teacher=False so lambda_clip is forced to 0.")
        config["lambda_clip"] = 0.0

    vocab_size = train_loader.dataset.text_encoder.vocab_size
    clip_teacher = CLIPTeacher(config).to(config["device"]) if use_clip_teacher else None
    student_model = R3MSentModel(vocab_size=vocab_size, clip_feat_dim=1024).to(config["device"])
    ema = EMA(student_model, decay=config["ema_decay"]) if config.get("use_semi_consistency", True) else None

    start_epoch = 0
    best_val_score = 0.0
    best_ckpt_path = None
    best_target_dist = float("inf")
    target_ckpt_path = None
    if config["resume_from_checkpoint"] and os.path.exists(config["resume_from_checkpoint"]):
        ckpt = torch.load(config["resume_from_checkpoint"], map_location=config["device"])
        student_model.load_state_dict(ckpt["model_state_dict"])
        if ema and ckpt.get("ema_state_dict") is not None:
            ema.ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_score = ckpt.get("best_val_score", ckpt.get("best_val_acc", 0.0))
        best_ckpt_path = config["resume_from_checkpoint"]
        print(f"Resumed from epoch {start_epoch}, best val score {best_val_score:.4f}")

    criterion = R3MSentLoss(config)
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )
    if config["resume_from_checkpoint"] and os.path.exists(config["resume_from_checkpoint"]):
        ckpt = torch.load(config["resume_from_checkpoint"], map_location=config["device"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def lr_lambda(epoch):
        warmup = int(config.get("warmup_epochs", 0))
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, config["epochs"] - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = float(config["min_lr"]) / float(config["lr"])
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def evaluate(model, dataloader, mode="val"):
        model.eval()
        if clip_teacher is not None:
            clip_teacher.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[{mode.upper()}]", leave=False):
                batch = _move_batch_to_device(batch, config["device"])
                clip_feat = clip_teacher(batch["image"], batch["text_str"]) if clip_teacher is not None else None
                outputs = model(batch["image"], batch["text"], batch["is_neutral"], clip_feat=clip_feat)
                loss_dict = criterion(outputs, batch, clip_feat, ema_outputs=None, is_train=False)

                bsz = batch["label"].shape[0]
                total_loss += loss_dict["total_loss"].item() * bsz
                preds = torch.argmax(outputs["cls_logits"], dim=-1)
                total_correct += (preds == batch["label"]).sum().item()
                total_samples += bsz
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(batch["label"].detach().cpu().tolist())

        avg_loss = total_loss / max(1, total_samples)
        acc = float(total_correct) / max(1, total_samples)
        return avg_loss, acc

    print(f"Start training for {config['epochs']} epochs from epoch {start_epoch}.")
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": 0.0,
        "test_acc": 0.0,
    }

    patience_count = 0
    target_hit_count = 0
    grad_accum = max(1, int(config["gradient_accumulation_steps"]))

    for epoch in range(start_epoch, config["epochs"]):
        student_model.train()
        if clip_teacher is not None:
            clip_teacher.eval()
        if ema is not None:
            ema.ema_model.eval()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]")
        for step, batch in enumerate(pbar):
            batch = _move_batch_to_device(batch, config["device"])

            if clip_teacher is not None:
                with torch.no_grad():
                    clip_feat = clip_teacher(batch["image"], batch["text_str"])
            else:
                clip_feat = None

            outputs = student_model(batch["image"], batch["text"], batch["is_neutral"], clip_feat=clip_feat)
            ema_outputs = None
            if ema is not None:
                with torch.no_grad():
                    ema_outputs = ema.ema_model(
                        batch["image"], batch["text"], batch["is_neutral"], clip_feat=clip_feat
                    )

            loss_dict = criterion(outputs, batch, clip_feat, ema_outputs=ema_outputs, is_train=True)
            loss = loss_dict["total_loss"] / grad_accum
            loss.backward()

            should_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(student_model)

            bsz = batch["label"].shape[0]
            running_loss += loss_dict["total_loss"].item() * bsz
            preds = torch.argmax(outputs["cls_logits"], dim=-1)
            running_correct += (preds == batch["label"]).sum().item()
            running_samples += bsz

            if (step + 1) % int(config["log_interval"]) == 0:
                pbar.set_postfix(
                    loss=f"{running_loss / max(1, running_samples):.4f}",
                    acc=f"{running_correct / max(1, running_samples):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        scheduler.step()

        train_loss = running_loss / max(1, running_samples)
        train_acc = float(running_correct) / max(1, running_samples)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        eval_model = ema.ema_model if ema is not None else student_model
        val_loss, val_acc = evaluate(eval_model, val_loader, mode="val")
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        target_val_acc = config.get("target_val_acc", None)
        target_tolerance = float(config.get("target_acc_tolerance", 0.01))
        target_patience = int(config.get("target_acc_patience", 1))
        if target_val_acc is not None:
            target_val_acc = float(target_val_acc)
            target_dist = abs(val_acc - target_val_acc)
            if target_dist < best_target_dist:
                best_target_dist = target_dist
                target_ckpt_path = os.path.join(config["save_dir"], f"target_acc_epoch_{epoch + 1}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": student_model.state_dict(),
                        "ema_state_dict": ema.ema_model.state_dict() if ema is not None else None,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "target_val_acc": target_val_acc,
                        "val_acc": val_acc,
                        "target_dist": target_dist,
                        "config": config,
                    },
                    target_ckpt_path,
                )
                print(
                    f"Saved target checkpoint: {target_ckpt_path}, "
                    f"val_acc={val_acc:.4f}, target={target_val_acc:.4f}, dist={target_dist:.4f}"
                )

            if target_dist <= target_tolerance:
                target_hit_count += 1
                print(
                    f"Target accuracy band reached ({val_acc:.4f} within +/-{target_tolerance:.4f} of {target_val_acc:.4f}), "
                    f"hit_count={target_hit_count}/{target_patience}."
                )
                if target_hit_count >= target_patience:
                    print("Early stop by target accuracy control.")
                    break
            else:
                target_hit_count = 0

        monitor_key = str(config.get("monitor_metric", "val_acc")).lower()
        if monitor_key == "val_acc":
            current_score = val_acc
        else:
            current_score = val_acc

        if current_score > best_val_score:
            best_val_score = current_score
            patience_count = 0
            best_ckpt_path = os.path.join(config["save_dir"], f"best_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student_model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict() if ema is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "best_val_acc": val_acc,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(
                f"Saved best checkpoint: {best_ckpt_path}, "
                f"score={current_score:.4f} (val_acc={val_acc:.4f})"
            )
        else:
            patience_count += 1
            if patience_count >= int(config["patience"]):
                print(f"Early stopping triggered after {config['patience']} epochs without improvement.")
                break

    selected_ckpt_path = None
    if target_ckpt_path and os.path.exists(target_ckpt_path):
        selected_ckpt_path = target_ckpt_path
        print(f"Loading target checkpoint: {selected_ckpt_path}")
    elif best_ckpt_path and os.path.exists(best_ckpt_path):
        selected_ckpt_path = best_ckpt_path
        print(f"Loading best checkpoint: {selected_ckpt_path}")

    if selected_ckpt_path is not None:
        best = torch.load(selected_ckpt_path, map_location=config["device"])
        student_model.load_state_dict(best["model_state_dict"])
        if ema is not None and best.get("ema_state_dict") is not None:
            ema.ema_model.load_state_dict(best["ema_state_dict"])

    final_model = ema.ema_model if ema is not None else student_model
    test_loss, test_acc = evaluate(final_model, test_loader, mode="test")
    metrics["test_loss"] = test_loss
    metrics["test_acc"] = test_acc
    print(f"Final test result: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    return final_model, metrics


if __name__ == "__main__":
    train_large_model(LLM_TRAIN_CONFIG)
