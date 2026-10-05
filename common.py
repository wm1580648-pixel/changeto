import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip



class CLIPTeacher(nn.Module):
    """Frozen CLIP encoder that returns fused image-text features."""

    def __init__(self, config):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip is not available. Install open_clip_torch in your training environment."
            )

        self.config = config
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config["clip_model_name"], pretrained=config["clip_pretrained"]
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out_dim = int(self.clip_model.encode_image(dummy).shape[-1])
        self.feature_proj = nn.Identity() if out_dim == 1024 else nn.Linear(out_dim, 1024)

    def forward(self, image, text_str_list):
        tokens = open_clip.tokenize(text_str_list).to(self.config["device"])
        image = image.to(self.config["device"])

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(image)
            txt_feat = self.clip_model.encode_text(tokens)
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            clip_feat = (img_feat + txt_feat) / 2.0

        return self.feature_proj(clip_feat)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        ema_sd = self.ema_model.state_dict()
        model_sd = model.state_dict()
        for k, v in ema_sd.items():
            if not torch.is_floating_point(v):
                ema_sd[k].copy_(model_sd[k])
            else:
                ema_sd[k].mul_(self.decay).add_(model_sd[k], alpha=1.0 - self.decay)


class R3MSentLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        class_weights = config.get("class_weights", None)
        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=config["device"])
        if torch.is_tensor(class_weights):
            class_weights = class_weights.to(config["device"])

        self.cls_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
        self.relation_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.neutral_criterion = nn.BCEWithLogitsLoss()

        self.lambda_neutral = float(config.get("lambda_neutral", 0.3))
        self.lambda_sentiment = float(config.get("lambda_sentiment", 0.6))

    def forward(self, student_outputs, batch, clip_feat, ema_outputs=None, is_train=True):
        _ = is_train
        loss_dict = {}

        label = batch["label"]  # 0:neg, 1:neu, 2:pos
        neutral_target = (label == 1).float()

        loss_dict["cls_loss"] = self.cls_criterion(student_outputs["cls_logits"], label)
        loss_dict["neutral_loss"] = self.neutral_criterion(student_outputs["neutral_prob"], neutral_target)
        loss_dict["neutral_loss"] = loss_dict["neutral_loss"] * self.lambda_neutral

        non_neutral = label != 1
        if non_neutral.any():
            sentiment_label = torch.where(label[non_neutral] == 0, 0, 1)
            sentiment_loss = F.cross_entropy(
                student_outputs["sentiment_logits"][non_neutral], sentiment_label
            )
            loss_dict["sentiment_loss"] = sentiment_loss * self.lambda_sentiment
        else:
            loss_dict["sentiment_loss"] = torch.zeros((), device=label.device)

        loss_dict["relation_loss"] = torch.zeros((), device=label.device)
        if self.config.get("use_relation_distill", True):
            loss_dict["relation_loss"] = self.relation_criterion(
                student_outputs["relation_logits"], batch["relation_label"]
            ) * float(self.config.get("lambda_r", 0.0))

        loss_dict["evidence_loss"] = torch.zeros((), device=label.device)
        if self.config.get("use_evidence_distill", True):
            text_pred = torch.sigmoid(student_outputs["text_evidence"])
            img_pred = torch.sigmoid(student_outputs["img_evidence"])
            text_loss = self.mse_criterion(text_pred, batch["text_evidence_weight"])
            img_loss = self.mse_criterion(img_pred, batch["img_evidence_weight"])
            loss_dict["evidence_loss"] = (text_loss + img_loss) * float(self.config.get("lambda_e", 0.0))

        loss_dict["clip_distill_loss"] = torch.zeros((), device=label.device)
        lambda_clip = float(self.config.get("lambda_clip", 0.0))
        if lambda_clip > 0 and clip_feat is not None:
            student_feat = F.normalize(student_outputs["fusion_feat"], dim=-1)
            clip_feat = F.normalize(clip_feat, dim=-1)
            loss_dict["clip_distill_loss"] = self.mse_criterion(student_feat, clip_feat) * lambda_clip

        loss_dict["semi_consistency_loss"] = torch.zeros((), device=label.device)
        if self.config.get("use_semi_consistency", True) and (ema_outputs is not None):
            tau = float(self.config.get("tau", 1.0))
            student_log_probs = F.log_softmax(student_outputs["cls_logits"] / tau, dim=-1)
            ema_probs = F.softmax(ema_outputs["cls_logits"] / tau, dim=-1)
            loss_dict["semi_consistency_loss"] = (
                self.kl_criterion(student_log_probs, ema_probs) * (tau * tau) * float(self.config.get("lambda_u", 0.0))
            )

        loss_dict["total_loss"] = sum(loss_dict.values())
        return loss_dict


class AblationR3MSentLoss(R3MSentLoss):
    def __init__(self, config):
        super().__init__(config)
