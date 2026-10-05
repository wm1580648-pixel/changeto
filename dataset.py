import csv
import json
import os
import re
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


TOKEN_RE = re.compile(r"[A-Za-z0-9_#@']+")


class SimpleTextEncoder:
    """Build/load a shared vocabulary and encode text to fixed-length ids."""

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
        counter = Counter()
        for name in sorted(os.listdir(train_text_dir)):
            if not name.lower().endswith(".txt"):
                continue
            text_path = os.path.join(train_text_dir, name)
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                counter.update(self._tokenize(f.read()))

        # Keep the most frequent words so train/val/test share one stable vocab.
        kept = []
        for token, freq in counter.most_common(max_vocab_size):
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
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        if len(ids) < max_len:
            ids.extend([self.pad_idx] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)


class MultimodalSentimentDataset(Dataset):
    LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(self, dataset_root, dataset_name, split="train", max_text_len=48, label_mode="merge"):
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
        self.img_transform = self._get_img_transform()

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

                text_label = self.LABEL_MAP[text_label_str]
                img_label = self.LABEL_MAP[img_label_str]
                data_info.append(
                    {
                        "img_prefix": img_prefix,
                        "text_label": text_label,
                        "img_label": img_label,
                    }
                )

        return data_info

    @staticmethod
    def _merge_label(text_label, img_label):
        # Keep the original project rule to avoid train/eval target mismatch.
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

    def _resolve_final_label(self, img_prefix, text_label, img_label):
        mode = self.label_mode
        if mode == "image":
            return img_label
        if mode == "text":
            return text_label
        if mode == "rationale_final":
            llm_rationale = self.llm_rationales.get(str(img_prefix), {})
            label_from_rationale = self._label_from_string(llm_rationale.get("FinalLabel"))
            if label_from_rationale is not None:
                return label_from_rationale
        return self._merge_label(text_label, img_label)

    @staticmethod
    def _relation_from_labels(text_label, img_label):
        if text_label == img_label:
            return 2  # support
        if text_label == 1 or img_label == 1:
            return 1  # complement
        return 0  # conflict

    def _label2idx(self, label_str):
        label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "0": 0,
            "1": 1,
            "2": 2,
            "support": 2,
            "conflict": 0,
            "complement": 1,
        }
        return label_map.get(str(label_str).strip().lower(), 1)

    def _get_img_transform(self):
        if self.split == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_llm_rationales(self):
        rationales_path = os.path.join(self.split_dir, "llm_rationales.json")
        if not os.path.exists(rationales_path):
            return {}

        try:
            with open(rationales_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception:
            return {}

        try:
            data = json.loads(raw)
            return self._normalize_rationales(data)
        except Exception:
            return self._parse_rationales_relaxed(raw)

    @staticmethod
    def _normalize_rationales(data):
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

    def _parse_rationales_relaxed(self, raw_text):
        """
        Best-effort parser for slightly broken JSON files.
        It extracts key fields by scanning each sample block.
        """
        out = {}
        id_pat = re.compile(r'"(?P<img_id>\d+)"\s*:\s*\{')
        ids = list(id_pat.finditer(raw_text))
        if not ids:
            return {}

        for i, m in enumerate(ids):
            img_id = m.group("img_id")
            start = m.start()
            end = ids[i + 1].start() if i + 1 < len(ids) else len(raw_text)
            seg = raw_text[start:end]

            item = {}
            final_m = re.search(r'"FinalLabel"\s*:\s*"(?P<v>[A-Za-z]+)"', seg)
            if final_m:
                item["FinalLabel"] = final_m.group("v").lower()

            rel_m = re.search(r'"CrossModalRelation"\s*:\s*"(?P<v>[A-Za-z]+)"', seg)
            if rel_m:
                item["CrossModalRelation"] = rel_m.group("v").lower()

            conf_m = re.search(r'"Confidence"\s*:\s*(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', seg)
            if conf_m:
                try:
                    item["Confidence"] = float(conf_m.group("v"))
                except Exception:
                    pass

            pol_m = re.search(r'"Polarity"\s*:\s*"(?P<v>[A-Za-z]+)"', seg)
            if pol_m:
                item["ImageEvidence"] = [{"Polarity": pol_m.group("v").lower()}]

            text_lab_m = re.search(
                r'"TextEvidence"\s*:\s*\[\s*\[\s*"(?:\\.|[^"\\])*"\s*,\s*"(?P<v>[A-Za-z]+)"',
                seg,
                flags=re.S,
            )
            if text_lab_m:
                item["TextEvidence"] = [["", text_lab_m.group("v").upper()]]

            out[str(img_id)] = item

        return out

    def get_final_labels(self):
        return list(self.final_labels)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]
        img_prefix = item["img_prefix"]
        text_label = item["text_label"]
        img_label = item["img_label"]
        final_label = self.final_labels[idx]

        img_jpg = os.path.join(self.split_dir, "images", f"{img_prefix}.jpg")
        img_png = os.path.join(self.split_dir, "images", f"{img_prefix}.png")
        img_path = img_jpg if os.path.exists(img_jpg) else img_png
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        text_path = os.path.join(self.split_dir, "texts", f"{img_prefix}.txt")
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read().strip()
        else:
            text_content = ""

        text_encoded = self.text_encoder.encode(text_content, max_len=self.max_text_len)

        llm_rationale = self.llm_rationales.get(str(img_prefix), {})
        if llm_rationale:
            relation_label = self._label2idx(llm_rationale.get("CrossModalRelation", "complement"))
        else:
            relation_label = self._relation_from_labels(text_label, img_label)
        conf = float(llm_rationale.get("Confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        text_evidence_weight = torch.tensor(conf, dtype=torch.float32)

        img_evidence_weight = 0.5
        img_evidence = llm_rationale.get("ImageEvidence", [])
        if isinstance(img_evidence, list) and len(img_evidence) > 0:
            polarity = str(img_evidence[0].get("Polarity", "neutral")).lower()
            img_evidence_weight = 0.5 if polarity == "neutral" else 0.8
        img_evidence_weight = torch.tensor(img_evidence_weight, dtype=torch.float32)

        is_neutral = (final_label == 1)
        return {
            "image": image,
            "text": text_encoded,
            "text_str": text_content,
            "label": torch.tensor(final_label, dtype=torch.long),
            "is_neutral": torch.tensor(is_neutral, dtype=torch.bool),
            "relation_label": torch.tensor(relation_label, dtype=torch.long),
            "text_evidence_weight": text_evidence_weight,
            "img_evidence_weight": img_evidence_weight,
            "img_prefix": str(img_prefix),
            "text_label": torch.tensor(text_label, dtype=torch.long),
            "img_label": torch.tensor(img_label, dtype=torch.long),
        }


def build_dataloader(
    dataset_root,
    dataset_name,
    split="train",
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    use_weighted_sampler=False,
    label_mode="merge",
):
    dataset = MultimodalSentimentDataset(dataset_root, dataset_name, split, label_mode=label_mode)

    sampler = None
    if split == "train" and use_weighted_sampler:
        labels = dataset.get_final_labels()
        counts = Counter(labels)
        sample_weights = [1.0 / max(1, counts[label]) for label in labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return dataloader
