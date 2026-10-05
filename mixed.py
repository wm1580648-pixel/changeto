import argparse
import csv
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEVEN_CLASS_NAMES = [
    "extreme_negative",
    "negative",
    "slightly_negative",
    "neutral",
    "slightly_positive",
    "positive",
    "extreme_positive",
]

SEVEN_CLASS_NAMES_ZH = [
    "极度消极",
    "消极",
    "轻微消极",
    "中性",
    "轻微兴奋",
    "兴奋",
    "极度兴奋",
]

THREE_CLASS_NAMES = ["negative", "neutral", "positive"]
TOKEN_RE = re.compile(r"[A-Za-z0-9_#@']+")

# Ground-truth seven-class mapping generated from (text_label, img_label).
# 0=negative, 1=neutral, 2=positive
PAIR_TO_SEVEN = {
    (0, 0): 0,  # 极度消极
    (0, 1): 1,  # 消极
    (1, 0): 1,  # 消极
    (0, 2): 2,  # 轻微消极
    (1, 1): 3,  # 中性
    (2, 0): 4,  # 轻微兴奋
    (1, 2): 5,  # 兴奋
    (2, 1): 5,  # 兴奋
    (2, 2): 6,  # 极度兴奋
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a 3-class multimodal sentiment model with a rule-based 7-class confusion matrix."
    )
    parser.add_argument("--project-root", default=r"D:\changeto", help="Project root that contains model.py")
    parser.add_argument("--dataset-root", default=r"D:\data", help="Dataset root path")
    parser.add_argument("--dataset-name", default="MVSASINGLEDataSet", help="Dataset name, e.g. Twitter/MVSADataSet/MVSASINGLEDataSet")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluation split")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file or directory. If omitted, the script will auto-pick the latest .pth under the project.",
    )
    parser.add_argument("--label-mode", default="merge", choices=["merge", "image", "text", "rationale_final"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.55,
        help="Low confidence threshold used when splitting negative/positive into weak vs medium strength.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.80,
        help="High confidence threshold used when splitting negative/positive into medium vs extreme strength.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save confusion matrix PNG/CSV and evaluation reports. Default: ./seven_class_eval_<dataset>_<split>",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Only evaluate the first N samples. 0 means use the full split.",
    )
    return parser.parse_args()


def ensure_model_import(project_root):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model import R3MSentModel  # pylint: disable=import-outside-toplevel

    return R3MSentModel


class SimpleTextEncoder:
    def __init__(self, vocab_path=None, train_text_dir=None, max_vocab_size=30000, min_freq=2):
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {"<PAD>": self.pad_idx, "<UNK>": self.unk_idx}
        self.idx2word = {self.pad_idx: "<PAD>", self.unk_idx: "<UNK>"}

        if vocab_path and os.path.exists(vocab_path):
            self._load_vocab(vocab_path)
        elif vocab_path and train_text_dir and os.path.isdir(train_text_dir):
            self._build_and_save_vocab(vocab_path, train_text_dir, max_vocab_size, min_freq)

        self.vocab_size = len(self.word2idx)

    @staticmethod
    def _tokenize(text):
        return TOKEN_RE.findall((text or "").lower())

    def _load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token or token in self.word2idx:
                    continue
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def _build_and_save_vocab(self, vocab_path, train_text_dir, max_vocab_size, min_freq):
        counter = {}
        for name in sorted(os.listdir(train_text_dir)):
            if not name.lower().endswith(".txt"):
                continue
            text_path = os.path.join(train_text_dir, name)
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                for token in self._tokenize(f.read()):
                    counter[token] = counter.get(token, 0) + 1

        kept = []
        for token, freq in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]:
            if freq < min_freq:
                break
            kept.append(token)

        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in kept:
                f.write(token + "\n")

        self._load_vocab(vocab_path)

    def encode(self, text, max_len=48):
        tokens = self._tokenize(text)[:max_len]
        ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        if len(ids) < max_len:
            ids.extend([self.pad_idx] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)


class EvalDataset(Dataset):
    LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(self, dataset_root, dataset_name, split="test", max_text_len=48, label_mode="merge"):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.split = split
        self.max_text_len = max_text_len
        self.label_mode = str(label_mode or "merge").lower()
        self.split_dir = os.path.join(dataset_root, dataset_name, split)
        self.vocab_path = os.path.join(dataset_root, dataset_name, "vocab.txt")
        train_text_dir = os.path.join(dataset_root, dataset_name, "train", "texts")
        self.text_encoder = SimpleTextEncoder(
            vocab_path=self.vocab_path,
            train_text_dir=train_text_dir,
            max_vocab_size=30000,
            min_freq=2,
        )
        self.llm_rationales = self._load_llm_rationales()
        self.data_info = self._load_data_info()
        self.final_labels = [
            self._resolve_final_label(item["img_prefix"], item["text_label"], item["img_label"])
            for item in self.data_info
        ]

    def _load_data_info(self):
        csv_path = os.path.join(self.split_dir, "annotations.csv")
        data_info = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                img_prefix = row[0].strip("\ufeff").strip()
                text_label_str = row[1].strip().lower()
                img_label_str = row[2].strip().lower()
                if text_label_str not in self.LABEL_MAP or img_label_str not in self.LABEL_MAP:
                    continue
                data_info.append(
                    {
                        "img_prefix": img_prefix,
                        "text_label": self.LABEL_MAP[text_label_str],
                        "img_label": self.LABEL_MAP[img_label_str],
                    }
                )
        return data_info

    @staticmethod
    def _merge_label(text_label, img_label):
        if text_label == 1 and img_label == 1:
            return 1
        if text_label != 1:
            return text_label
        return img_label

    @classmethod
    def _label_from_string(cls, label_str):
        if label_str is None:
            return None
        return cls.LABEL_MAP.get(str(label_str).strip().lower())

    def _load_llm_rationales(self):
        path_obj = Path(self.split_dir) / "llm_rationales.json"
        if not path_obj.exists():
            return {}
        try:
            with path_obj.open("r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return {}
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            out = {}
            for item in data:
                img_id = item.get("img_id") or item.get("img_prefix") or item.get("id")
                if img_id is not None:
                    out[str(img_id)] = item
            return out
        return {}

    def _resolve_final_label(self, img_prefix, text_label, img_label):
        if self.label_mode == "image":
            return img_label
        if self.label_mode == "text":
            return text_label
        if self.label_mode == "rationale_final":
            llm_rationale = self.llm_rationales.get(str(img_prefix), {})
            label_from_rationale = self._label_from_string(llm_rationale.get("FinalLabel"))
            if label_from_rationale is not None:
                return label_from_rationale
        return self._merge_label(text_label, img_label)

    @staticmethod
    def _resize_shorter_edge_to_256(image):
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size.")
        if width < height:
            new_width = 256
            new_height = round(height * 256 / width)
        else:
            new_height = 256
            new_width = round(width * 256 / height)
        return image.resize((new_width, new_height), Image.BILINEAR)

    @staticmethod
    def _center_crop_224(image):
        width, height = image.size
        left = max((width - 224) // 2, 0)
        top = max((height - 224) // 2, 0)
        return image.crop((left, top, left + 224, top + 224))

    def _load_image_tensor(self, img_prefix):
        img_jpg = os.path.join(self.split_dir, "images", f"{img_prefix}.jpg")
        img_png = os.path.join(self.split_dir, "images", f"{img_prefix}.png")
        img_path = img_jpg if os.path.exists(img_jpg) else img_png
        image = Image.open(img_path).convert("RGB")
        image = self._resize_shorter_edge_to_256(image)
        image = self._center_crop_224(image)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]
        img_prefix = item["img_prefix"]
        text_label = item["text_label"]
        img_label = item["img_label"]
        final_label = self.final_labels[idx]
        text_path = os.path.join(self.split_dir, "texts", f"{img_prefix}.txt")
        text_content = ""
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read().strip()

        return {
            "image": self._load_image_tensor(img_prefix),
            "text": self.text_encoder.encode(text_content, max_len=self.max_text_len),
            "text_str": text_content,
            "label": torch.tensor(final_label, dtype=torch.long),
            "is_neutral": torch.tensor(final_label == 1, dtype=torch.bool),
            "img_prefix": str(img_prefix),
            "text_label": torch.tensor(text_label, dtype=torch.long),
            "img_label": torch.tensor(img_label, dtype=torch.long),
        }


def auto_find_checkpoint(project_root):
    project_root = Path(project_root)
    candidates = []
    for pattern in ("*.pth", "llm_train_checkpoints*/*.pth", "*checkpoints*/*.pth"):
        candidates.extend(project_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {project_root}")

    def _sort_key(path_obj):
        m = re.search(r"epoch_(\d+)", path_obj.name)
        epoch = int(m.group(1)) if m else -1
        return (epoch, path_obj.stat().st_mtime)

    return sorted(candidates, key=_sort_key)[-1]


def resolve_checkpoint(project_root, checkpoint_arg):
    if checkpoint_arg is None:
        return auto_find_checkpoint(project_root)

    ckpt_path = Path(checkpoint_arg)
    if ckpt_path.is_file():
        return ckpt_path
    if ckpt_path.is_dir():
        candidates = list(ckpt_path.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoint exists in: {ckpt_path}")

        def _sort_key(path_obj):
            m = re.search(r"epoch_(\d+)", path_obj.name)
            epoch = int(m.group(1)) if m else -1
            return (epoch, path_obj.stat().st_mtime)

        return sorted(candidates, key=_sort_key)[-1]

    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_arg}")


def build_output_dir(args):
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path.cwd() / f"seven_class_eval_{args.dataset_name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pair_to_seven_label(text_label, img_label):
    pair = (int(text_label), int(img_label))
    if pair not in PAIR_TO_SEVEN:
        raise ValueError(f"Unsupported label pair: {pair}")
    return PAIR_TO_SEVEN[pair]


def probs_to_seven_label(prob_vec, low_threshold, high_threshold):
    pred3 = int(np.argmax(prob_vec))
    conf = float(prob_vec[pred3])

    if pred3 == 0:
        if conf >= high_threshold:
            return 0
        if conf >= low_threshold:
            return 1
        return 2

    if pred3 == 1:
        return 3

    if conf < low_threshold:
        return 4
    if conf < high_threshold:
        return 5
    return 6


def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def safe_divide(num, den):
    return float(num) / float(den) if den else 0.0


def per_class_accuracy(conf_mat):
    acc = []
    for i in range(conf_mat.shape[0]):
        row_sum = int(conf_mat[i].sum())
        acc.append(safe_divide(conf_mat[i, i], row_sum))
    return acc


def save_matrix_csv(path_obj, conf_mat):
    with path_obj.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + SEVEN_CLASS_NAMES_ZH)
        for idx, row in enumerate(conf_mat):
            writer.writerow([SEVEN_CLASS_NAMES_ZH[idx]] + row.tolist())


def save_per_class_csv(path_obj, conf_mat):
    acc = per_class_accuracy(conf_mat)
    with path_obj.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name_en", "class_name_zh", "support", "correct", "accuracy"])
        for idx in range(conf_mat.shape[0]):
            support = int(conf_mat[idx].sum())
            correct = int(conf_mat[idx, idx])
            writer.writerow(
                [idx, SEVEN_CLASS_NAMES[idx], SEVEN_CLASS_NAMES_ZH[idx], support, correct, f"{acc[idx]:.6f}"]
            )


def save_prediction_details(path_obj, rows):
    with path_obj.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "img_prefix",
                "text_label_3class",
                "img_label_3class",
                "gold_label_7class_id",
                "gold_label_7class_zh",
                "pred_label_3class_id",
                "pred_label_3class_name",
                "pred_confidence",
                "pred_label_7class_id",
                "pred_label_7class_zh",
                "correct_7class",
            ]
        )
        writer.writerows(rows)


def plot_confusion_matrix(conf_mat, path_obj, dataset_name, split_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_mat, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(SEVEN_CLASS_NAMES)))
    ax.set_yticks(range(len(SEVEN_CLASS_NAMES)))
    ax.set_xticklabels(SEVEN_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(SEVEN_CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"7-Class Confusion Matrix - {dataset_name} ({split_name})")

    max_val = conf_mat.max() if conf_mat.size > 0 else 0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            val = int(conf_mat[i, j])
            color = "white" if val > max_val / 2.0 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(path_obj, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate(args):
    R3MSentModel = ensure_model_import(args.project_root)
    checkpoint_path = resolve_checkpoint(args.project_root, args.checkpoint)
    output_dir = build_output_dir(args)

    dataset = EvalDataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        split=args.split,
        label_mode=args.label_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    vocab_size = max(30000, dataset.text_encoder.vocab_size)
    model = R3MSentModel(vocab_size=vocab_size, clip_feat_dim=1024).to(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    y_true_3 = []
    y_pred_3 = []
    y_true_7 = []
    y_pred_7 = []
    detail_rows = []
    seen = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(args.device)
            text = batch["text"].to(args.device)
            is_neutral = batch["is_neutral"].to(args.device)

            outputs = model(image, text, is_neutral=is_neutral, clip_feat=None)
            probs = torch.softmax(outputs["cls_logits"], dim=-1).cpu().numpy()
            pred3 = np.argmax(probs, axis=1)

            batch_size = len(pred3)
            for i in range(batch_size):
                true3 = int(batch["label"][i].item())
                text_label = int(batch["text_label"][i].item())
                img_label = int(batch["img_label"][i].item())
                pred3_i = int(pred3[i])
                pred_conf = float(probs[i][pred3_i])

                true7 = pair_to_seven_label(text_label, img_label)
                pred7 = probs_to_seven_label(probs[i], args.low_threshold, args.high_threshold)

                img_prefix = batch["img_prefix"][i]
                correct_7 = int(true7 == pred7)

                y_true_3.append(true3)
                y_pred_3.append(pred3_i)
                y_true_7.append(true7)
                y_pred_7.append(pred7)
                detail_rows.append(
                    [
                        img_prefix,
                        THREE_CLASS_NAMES[text_label],
                        THREE_CLASS_NAMES[img_label],
                        true7,
                        SEVEN_CLASS_NAMES_ZH[true7],
                        pred3_i,
                        THREE_CLASS_NAMES[pred3_i],
                        f"{pred_conf:.6f}",
                        pred7,
                        SEVEN_CLASS_NAMES_ZH[pred7],
                        correct_7,
                    ]
                )

                seen += 1
                if args.max_samples > 0 and seen >= args.max_samples:
                    break

            if args.max_samples > 0 and seen >= args.max_samples:
                break

    conf_mat_3 = build_confusion_matrix(y_true_3, y_pred_3, num_classes=3)
    conf_mat_7 = build_confusion_matrix(y_true_7, y_pred_7, num_classes=7)

    overall_acc_3 = safe_divide(np.trace(conf_mat_3), conf_mat_3.sum())
    overall_acc_7 = safe_divide(np.trace(conf_mat_7), conf_mat_7.sum())
    per_class_acc_7 = per_class_accuracy(conf_mat_7)

    save_matrix_csv(output_dir / "confusion_matrix_7class.csv", conf_mat_7)
    save_per_class_csv(output_dir / "per_class_accuracy_7class.csv", conf_mat_7)
    save_prediction_details(output_dir / "prediction_details_7class.csv", detail_rows)
    plot_confusion_matrix(conf_mat_7, output_dir / "confusion_matrix_7class.png", args.dataset_name, args.split)

    metrics = {
        "project_root": str(Path(args.project_root).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "checkpoint": str(checkpoint_path.resolve()),
        "label_mode": args.label_mode,
        "device": args.device,
        "samples": len(y_true_7),
        "overall_acc_3class": overall_acc_3,
        "overall_acc_7class": overall_acc_7,
        "per_class_acc_7class": {
            SEVEN_CLASS_NAMES_ZH[i]: per_class_acc_7[i] for i in range(len(SEVEN_CLASS_NAMES_ZH))
        },
        "seven_class_rule": {
            "ground_truth": {
                "negative+negative": "极度消极",
                "negative+neutral / neutral+negative": "消极",
                "negative+positive": "轻微消极",
                "neutral+neutral": "中性",
                "positive+negative": "轻微兴奋",
                "neutral+positive / positive+neutral": "兴奋",
                "positive+positive": "极度兴奋",
            },
            "prediction": {
                "negative": {
                    f"conf >= {args.high_threshold}": "极度消极",
                    f"{args.low_threshold} <= conf < {args.high_threshold}": "消极",
                    f"conf < {args.low_threshold}": "轻微消极",
                },
                "neutral": "中性",
                "positive": {
                    f"conf < {args.low_threshold}": "轻微兴奋",
                    f"{args.low_threshold} <= conf < {args.high_threshold}": "兴奋",
                    f"conf >= {args.high_threshold}": "极度兴奋",
                },
            },
        },
    }

    with (output_dir / "metrics_7class.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    present_classes = [SEVEN_CLASS_NAMES_ZH[i] for i in sorted(set(y_true_7))]
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset_name} | split={args.split} | samples={len(y_true_7)}")
    print(f"3-class accuracy: {overall_acc_3:.4f}")
    print(f"7-class accuracy: {overall_acc_7:.4f}")
    print(f"Ground-truth classes present: {', '.join(present_classes)}")
    print(f"Results saved to: {output_dir}")

    if len(present_classes) < 7:
        print(
            "Warning: this split does not contain all seven ground-truth classes under the current rule. "
            "For example, the Twitter dataset only contains negative-negative / neutral-neutral / positive-positive."
        )


if __name__ == "__main__":
    evaluate(parse_args())
