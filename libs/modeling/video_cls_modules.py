import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm


class AVMCModule(nn.Module):
    def __init__(
        self,
        row_dim=7,
        align_dim=64,
        n_layers=2,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        self.row_dim = row_dim
        self.align_dim = align_dim
        self.row_proj = nn.Linear(row_dim, align_dim)
        nn.init.kaiming_normal_(self.row_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.row_proj.bias, 0)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(align_dim, align_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(align_dim, align_dim, 1),
                nn.Dropout(dropout),
            ))
        self.ln = LayerNorm(align_dim)
        self._lambda_param = nn.Parameter(torch.ones(1) * 2.0)
        self.bias_param = nn.Parameter(torch.zeros(1))

    @property
    def lambda_d(self):
        return F.softplus(self._lambda_param)

    def _head_mean_rows(self, attention):
        if attention.dim() == 4:
            if attention.shape[2] == attention.shape[3]:
                return attention.mean(dim=1)
            B, T, H, W = attention.shape
            return attention.mean(dim=2)
        if attention.dim() == 3:
            B, T, W = attention.shape
            if W != self.row_dim:
                rows = F.adaptive_avg_pool1d(
                    attention.permute(0, 2, 1), self.row_dim
                ).permute(0, 2, 1)
                return rows
            return attention
        raise ValueError(f"Unexpected attention shape: {attention.shape}")

    def forward(self, attention, mask):
        rows = self._head_mean_rows(attention)
        B, T, D = rows.shape
        if D != self.row_dim:
            rows = F.adaptive_avg_pool1d(
                rows.permute(0, 2, 1), self.row_dim
            ).permute(0, 2, 1)
        aligned = self.row_proj(rows)
        aligned = aligned.permute(0, 2, 1)
        mask_float = mask.to(aligned.dtype)
        if mask_float.dim() == 3:
            mask_float = mask_float.squeeze(1)
        aligned = aligned * mask_float.unsqueeze(1)
        for layer in self.layers:
            residual = aligned
            aligned = self.ln(aligned.permute(0, 2, 1)).permute(0, 2, 1)
            aligned = layer(aligned) * mask_float.unsqueeze(1) + residual
        diff = torch.abs(aligned[:, :, 1:] - aligned[:, :, :-1]).mean(dim=1)
        motion = F.pad(diff, (1, 0), value=0)
        disturbance = torch.sigmoid(self.lambda_d * motion + self.bias_param)
        return disturbance * mask_float


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
        topk_patches=64,
        dropout=0.1,
        input_evidence_dim=14,
        temporal_dim=16,
    ):
        super().__init__()
        self.videomae_dim = videomae_dim
        self.hidden_dim = hidden_dim
        self.topk_patches = topk_patches
        self.input_evidence_dim = input_evidence_dim
        self.temporal_dim = temporal_dim
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
        self.scale = hidden_dim ** -0.5

    def _prepare_input(self, freq_evidence):
        c = freq_evidence[..., :self.temporal_dim]
        if c.shape[-1] < self.temporal_dim:
            pad = torch.zeros(
                *c.shape[:-1],
                self.temporal_dim - c.shape[-1],
                device=c.device,
                dtype=c.dtype,
            )
            c = torch.cat([c, pad], dim=-1)
        return c

    def forward(self, videomae_feats, freq_evidence, mask=None):
        if videomae_feats.shape[1] != videomae_feats.shape[-1]:
            if videomae_feats.shape[1] > videomae_feats.shape[2]:
                videomae_feats = videomae_feats.permute(0, 2, 1)
        B, L, D = videomae_feats.shape
        P = freq_evidence.shape[2]
        c = self._prepare_input(freq_evidence)
        e = self.evidence_mlp(c)
        patch_scores = e.mean(dim=-1)
        Kp = min(self.topk_patches, P)
        _, topk_indices = torch.topk(patch_scores, Kp, dim=2)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.temporal_dim)
        topk_e = torch.gather(e, 2, topk_indices_expanded)
        q = self.query_proj(videomae_feats)
        k = self.key_proj(topk_e)
        v = self.value_proj(topk_e)
        attn_logits = torch.einsum('bld,blkd->blk', q, k) * self.scale
        attn_weights = F.softmax(attn_logits, dim=2)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.einsum('blk,blkd->bld', attn_weights, v)
        fusion_delta = self.fusion_proj(context)
        fused_feats = videomae_feats + fusion_delta
        fused_feats = fused_feats.permute(0, 2, 1)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            fused_feats = fused_feats * mask.float()
        return fused_feats, topk_indices, attn_weights
