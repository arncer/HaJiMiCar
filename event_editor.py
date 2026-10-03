"""由失败证据预测事件附近及阶段内的连续控制修改。"""
import hashlib
import io
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from repair_physics import TOKEN_DIM, TOKEN_COUNT, CONTEXT_DIM, KNOT_COUNT, FAMILY_NAMES


class EventEditor(nn.Module):
    def __init__(self, hidden=96, use_evidence=True):
        super().__init__()
        self.hidden = hidden
        self.use_evidence = use_evidence
        self.map_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(), nn.Linear(128, hidden))
        self.token_projection = nn.Linear(TOKEN_DIM, hidden)
        self.context_projection = nn.Sequential(nn.Linear(CONTEXT_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.randn(1, TOKEN_COUNT, hidden) * 0.01)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.1, batch_first=True), 2,
            enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(1, KNOT_COUNT, hidden) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden, 4, hidden * 2, dropout=0.1, batch_first=True), 1)
        self.delta_head = nn.Linear(hidden, 2)
        self.family_head = nn.Linear(hidden, len(FAMILY_NAMES))
        self.improvement_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, tokens, context, map_image):
        if tokens.shape[1:] != (TOKEN_COUNT, TOKEN_DIM) or context.shape[-1] != CONTEXT_DIM:
            raise ValueError("事件编辑输入形状不匹配")
        if not self.use_evidence:
            # 有/无证据模型保持相同参数量、场景和计划输入；只去掉物理检查字段。
            tokens = tokens.clone()
            tokens[..., 10:22] = 0
        global_features = self.map_encoder(map_image) + self.context_projection(context)
        memory = self.encoder(self.token_projection(tokens) + self.position + global_features[:, None])
        decoded = self.decoder(self.queries.expand(len(tokens), -1, -1) + global_features[:, None], memory)
        pooled = memory.mean(1)
        return {"delta": torch.tanh(self.delta_head(decoded)),
                "family_logits": self.family_head(pooled),
                "improvement_logit": self.improvement_head(pooled).squeeze(-1)}


def editor_loss(output, batch):
    # 监督来自同一失败初值的真实干预结果；无改善样本学习零编辑及低收益分数。
    weight = 0.25 + batch["improved"].float()
    delta_error = F.smooth_l1_loss(output["delta"], batch["delta"], reduction="none").mean((1, 2))
    regression = (weight * delta_error).mean()
    classification = F.cross_entropy(output["family_logits"], batch["family"].long())
    confidence = F.binary_cross_entropy_with_logits(output["improvement_logit"], batch["improved"].float())
    loss = regression + 0.05 * classification + 0.05 * confidence
    return loss, {"delta": float(regression.detach()), "family": float(classification.detach()),
                  "confidence": float(confidence.detach()), "loss": float(loss.detach())}


def load_editor(path, device):
    content = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "loader_event_editor_v1":
        raise ValueError("不是事件编辑模型")
    model = EventEditor(checkpoint["hidden"], checkpoint["use_evidence"])
    model.load_state_dict(checkpoint["model"])
    checkpoint["sha256"] = hashlib.sha256(content).hexdigest()
    return model.to(device).eval(), checkpoint
