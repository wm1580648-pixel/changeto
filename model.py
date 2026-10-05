import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=30000, emb_dim=256, hidden_dim=512, padding_idx=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.padding_idx = padding_idx

    def forward(self, text):
        x = self.embedding(text)
        x = self.dropout(x)
        x, _ = self.lstm(x)

        # Masked mean pooling over sequence length.
        mask = (text != self.padding_idx).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        x = (x * mask).sum(dim=1) / denom

        x = self.dropout(x)
        x = self.norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, image):
        x = self.backbone(image)
        x = self.head(x)
        return x


class CrossModalFusion(nn.Module):
    """
    Keep this class name for compatibility with existing code.
    `num_heads` is accepted but unused to avoid breaking old calls.
    """

    def __init__(self, feat_dim=512, num_heads=None):
        super().__init__()
        _ = num_heads
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, text_feat, img_feat, clip_feat=None):
        if clip_feat is None:
            clip_feat = torch.zeros_like(text_feat)
        x = torch.cat([text_feat, img_feat, clip_feat], dim=-1)
        return self.net(x)


class R3MSentModel(nn.Module):
    def __init__(self, feat_dim=512, clip_feat_dim=1024, vocab_size=30000):
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_feat_dim = clip_feat_dim

        self.text_encoder = TextEncoder(vocab_size=vocab_size, hidden_dim=feat_dim)
        self.img_encoder = ImageEncoder(hidden_dim=feat_dim)

        self.clip_adapter = nn.Sequential(
            nn.Linear(clip_feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.fusion = CrossModalFusion(feat_dim=feat_dim)

        self.neutral_cls = nn.Linear(feat_dim, 1)  # logits, no sigmoid
        self.sentiment_cls = nn.Linear(feat_dim, 2)
        self.relation_head = nn.Linear(feat_dim, 3)
        self.text_evidence_head = nn.Linear(feat_dim, 1)
        self.img_evidence_head = nn.Linear(feat_dim, 1)
        self.final_cls = nn.Linear(feat_dim, 3)

        # Alignment target is CLIP space.
        self.clip_align = nn.Linear(feat_dim, clip_feat_dim)

    def forward(self, image, text, is_neutral=None, clip_feat=None):
        _ = is_neutral
        text_feat = self.text_encoder(text)
        img_feat = self.img_encoder(image)

        clip_local = None
        if clip_feat is not None:
            clip_local = self.clip_adapter(clip_feat)

        fusion_feat = self.fusion(text_feat, img_feat, clip_local)
        aligned_feat = self.clip_align(fusion_feat)

        return {
            "fusion_feat": aligned_feat,
            "cls_logits": self.final_cls(fusion_feat),
            "neutral_prob": self.neutral_cls(fusion_feat).squeeze(1),
            "sentiment_logits": self.sentiment_cls(fusion_feat),
            "relation_logits": self.relation_head(fusion_feat),
            "text_evidence": self.text_evidence_head(fusion_feat).squeeze(1),
            "img_evidence": self.img_evidence_head(fusion_feat).squeeze(1),
        }
