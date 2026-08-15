import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AligningEncoder


class AVMCModule(nn.Module):
    def __init__(
        self,
        n_head=4,
        align_dim=64,
        n_layers=2,
        kernel_size=7,
        dropout=0.1,
        window_size=7,
    ):
        super().__init__()
        self.align_encoder = AligningEncoder(
            proj_dim=align_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            n_head=n_head,
            window_size=window_size,
        )

    def forward(self, attention, mask):
        return self.align_encoder(attention, mask)


class MotionGateCalibration(nn.Module):
    def __init__(self, init_gamma=1.0, init_delta=2.0, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self._delta = nn.Parameter(torch.tensor(init_delta))
        self.eps = eps

    @property
    def delta(self):
        return F.softplus(self._delta)

    def forward(self, seg_logits, disturbance):
        gate = torch.sigmoid(self.gamma - self.delta * disturbance)
        calibrated_logits = seg_logits + torch.log(gate + self.eps)
        return calibrated_logits, gate


class TopKLogSumExpAggregator(nn.Module):
    def __init__(
        self,
        feat_dim,
        topk_ratio=0.1,
        topk_min=5,
        tau=1.0,
        learnable_tau=False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.topk_ratio = topk_ratio
        self.topk_min = topk_min
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer('tau', torch.tensor(tau))

    def forward(self, seg_logits_calib, clip_features, mask=None):
        B, L = seg_logits_calib.shape
        if clip_features.shape[1] != L:
            clip_features = clip_features.permute(0, 2, 1)
        if mask is not None:
            valid_lens = mask.sum(dim=1)
            k_per_sample = torch.clamp(
                (valid_lens * self.topk_ratio).long(),
                min=self.topk_min
            )
            k_per_sample = torch.minimum(k_per_sample, valid_lens.long())
            min_valid_len = valid_lens.min().long().item()
            k = min(k_per_sample.max().item(), min_valid_len)
            k = max(k, 1)
        else:
            k = max(self.topk_min, int(L * self.topk_ratio))
        if mask is not None:
            masked_logits = seg_logits_calib.masked_fill(~mask, float('-inf'))
        else:
            masked_logits = seg_logits_calib
        topk_values, topk_idx = torch.topk(masked_logits, k, dim=1)
        valid_topk_mask = topk_values > float('-inf')
        topk_values_for_lse = topk_values.masked_fill(~valid_topk_mask, float('-inf'))
        vid_logit = self.tau * torch.logsumexp(
            topk_values_for_lse / self.tau, dim=1, keepdim=True
        ) - self.tau * math.log(max(k, 1))
        topk_values_for_softmax = topk_values.masked_fill(~valid_topk_mask, float('-inf'))
        topk_weights = F.softmax(topk_values_for_softmax / self.tau, dim=1)
        topk_weights = torch.nan_to_num(topk_weights, nan=0.0)
        return vid_logit, topk_idx, topk_weights, None


class SegmentHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Conv1d(hidden_dim, 1, 1))
        self.layers = nn.Sequential(*layers)
        bias_value = -math.log((1 - 0.3) / 0.3)
        nn.init.constant_(self.layers[-1].bias, bias_value)

    def forward(self, features, mask=None):
        if features.shape[1] > features.shape[2]:
            features = features.permute(0, 2, 1)
        out = self.layers(features)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            out = out * mask.float()
        return out.squeeze(1)


class FrequencyEvidenceBranch(nn.Module):
    def __init__(
        self,
        videomae_dim=1408,
        hidden_dim=256,
        temporal_dim=16,
        topk_patches=64,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.videomae_dim = videomae_dim
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.topk_patches = topk_patches
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.evidence_mlp = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, temporal_dim),
        )
        self.query_proj = nn.Linear(videomae_dim, hidden_dim)
        self.key_proj = nn.Linear(temporal_dim, hidden_dim)
        self.value_proj = nn.Linear(temporal_dim, hidden_dim)
        self.fusion_proj = nn.Linear(hidden_dim, videomae_dim)
        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, videomae_feats, freq_evidence, mask=None):
        if videomae_feats.shape[1] != videomae_feats.shape[-1]:
            if videomae_feats.shape[1] > videomae_feats.shape[2]:
                videomae_feats = videomae_feats.permute(0, 2, 1)
        B, L, D = videomae_feats.shape
        P = freq_evidence.shape[2]
        K = min(self.temporal_dim, freq_evidence.shape[-1])
        c = freq_evidence[..., :K]
        if K < self.temporal_dim:
            pad = torch.zeros(
                *c.shape[:-1], self.temporal_dim - K,
                device=c.device, dtype=c.dtype
            )
            c = torch.cat([c, pad], dim=-1)
        e = self.evidence_mlp(c)
        patch_scores = e.mean(dim=-1)
        Kp = min(self.topk_patches, P)
        _, topk_indices = torch.topk(patch_scores, Kp, dim=2)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.temporal_dim)
        topk_e = torch.gather(e, 2, topk_indices_expanded)
        q = self.query_proj(videomae_feats)
        k = self.key_proj(topk_e)
        v = self.value_proj(topk_e)
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, Kp, self.num_heads, self.head_dim)
        v = v.view(B, L, Kp, self.num_heads, self.head_dim)
        attn_logits = torch.einsum('blhd,blkhd->blhk', q, k) * self.scale
        attn_weights = F.softmax(attn_logits, dim=2)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.einsum('blhk,blkhd->blhd', attn_weights, v)
        context = context.reshape(B, L, self.hidden_dim)
        fusion_delta = self.fusion_proj(context)
        fused_feats = videomae_feats + fusion_delta
        fused_feats = fused_feats.permute(0, 2, 1)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            fused_feats = fused_feats * mask.float()
        attn_summary = attn_weights.mean(dim=-1)
        return fused_feats, topk_indices, attn_summary
