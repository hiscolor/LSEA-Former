import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CMAMModule(nn.Module):
    """
    Camera Motion Attention Module (CMAM)

    基于 Transformer 的 attention dynamics 计算 motion intensity

    简化实现：直接计算相邻时间步 attention 分布的差异，不使用 alignment encoder
    这避免了 seq_len 固定的问题，更加灵活

    流程:
    1. 获取 attention map A: [B, H, L, L] 或 [B, L, H, W]
    2. 计算相邻时间步 attention 分布的 L1 差异: d_t = |A_t - A_{t-1}|
    3. reduce 到标量: m_t = mean(d_t)
    4. normalize: m_t_norm = sigmoid(lambda * m_t + b)
    """

    def __init__(
        self,
        seq_len=None,
        align_dim=64,
        num_layers=2,
        dropout=0.1,
    ):
        """
        Args:
            seq_len: 已废弃，保留参数兼容性
            align_dim: 已废弃
            num_layers: 已废弃
            dropout: 已废弃
        """
        super().__init__()


        self.lambda_param = nn.Parameter(torch.ones(1) * 2.0)
        self.bias_param = nn.Parameter(torch.zeros(1))

    def forward(self, attention_weights, mask=None):
        """
        Args:
            attention_weights: [B, H, L, L] (full attention) 或 [B, L, H, W] (local attention)
            mask: [B, L] 有效位置 mask (bool)

        Returns:
            motion_intensity: [B, L] 每个时间步的 motion intensity，范围 [0, 1]
        """

        if attention_weights.dim() == 4:
            B = attention_weights.shape[0]




            if attention_weights.shape[2] == attention_weights.shape[3]:

                H, L = attention_weights.shape[1], attention_weights.shape[2]
                att = attention_weights


                att_diff = torch.abs(att[:, :, 1:, :] - att[:, :, :-1, :])

                motion_diff = att_diff.mean(dim=(1, 3))
            else:

                L, H, W = attention_weights.shape[1], attention_weights.shape[2], attention_weights.shape[3]
                att = attention_weights

                att_diff = torch.abs(att[:, 1:, :, :] - att[:, :-1, :, :])

                motion_diff = att_diff.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unexpected attention shape: {attention_weights.shape}")


        motion_intensity = F.pad(motion_diff, (1, 0), value=0)


        motion_intensity = torch.sigmoid(self.lambda_param * motion_intensity + self.bias_param)


        if mask is not None:
            if mask.dim() == 2:
                motion_intensity = motion_intensity * mask.float()
            else:

                mask_2d = mask.view(B, -1)[:, :motion_intensity.shape[1]]
                motion_intensity = motion_intensity * mask_2d.float()

        return motion_intensity


class CMAMFromAMFlow(nn.Module):
    """
    从 AMFlow（attention 差分）计算 motion intensity

    这是一个简化版本，直接使用 backbone 已经计算好的 AMFlow
    """

    def __init__(
        self,
        proj_dim=64,
        n_layers=2,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim


        self.input_proj = nn.Conv1d(1, proj_dim, 1)


        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(proj_dim, proj_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(proj_dim, proj_dim, 1),
                nn.Dropout(dropout),
            ))


        self.ln = nn.LayerNorm(proj_dim)


        self.output_proj = nn.Conv1d(proj_dim, 1, 1)


        self.lambda_param = nn.Parameter(torch.ones(1) * 2.0)
        self.bias_param = nn.Parameter(torch.zeros(1))

    def forward(self, amflow, mask=None):
        """
        Args:
            amflow: [B, L] - AMFlow 值（attention 差分的均值）
            mask: [B, 1, L] 有效位置 mask (bool)

        Returns:
            motion_intensity: [B, L] 范围 [0, 1]
        """
        B, L = amflow.shape

        if mask is not None:
            mask_float = mask.squeeze(1).float() if mask.dim() == 3 else mask.float()
        else:
            mask_float = torch.ones(B, L, device=amflow.device)


        x = amflow.unsqueeze(1)


        x = self.input_proj(x) * mask_float.unsqueeze(1)


        for layer in self.layers:
            residual = x

            x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = layer(x) * mask_float.unsqueeze(1) + residual


        x = self.output_proj(x).squeeze(1)


        motion_intensity = torch.sigmoid(self.lambda_param * x + self.bias_param)
        motion_intensity = motion_intensity * mask_float

        return motion_intensity


class MotionGateCalibration(nn.Module):
    """
    Motion Gate Calibration

    基于 motion intensity 校准 segment logits

    公式:
        c_t = sigmoid(gamma - delta * m_t)
        z'_t = z_t + log(c_t + eps)

    其中:
        - gamma: 可学习偏置
        - delta: 可学习缩放（用 softplus 保证正）
        - m_t: motion intensity [0, 1]
        - z_t: 原始 segment logits
    """

    def __init__(self, init_gamma=1.0, init_delta=2.0, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self._delta = nn.Parameter(torch.tensor(init_delta))
        self.eps = eps

    @property
    def delta(self):
        return F.softplus(self._delta)

    def forward(self, seg_logits, motion_intensity):
        """
        Args:
            seg_logits: [B, L] segment-level logits
            motion_intensity: [B, L] motion intensity [0, 1]

        Returns:
            calibrated_logits: [B, L] 校准后的 logits
            gate_values: [B, L] gate 值 (for visualization)
        """

        gate = torch.sigmoid(self.gamma - self.delta * motion_intensity)


        calibrated_logits = seg_logits + torch.log(gate + self.eps)

        return calibrated_logits, gate


class TopKLogSumExpAggregator(nn.Module):
    """
    Top-k LogSumExp Aggregator

    从 clip-level 聚合到 video-level

    流程:
    1. 按 calibrated logits 选 top-k clips
    2. LogSumExp 聚合: z_lse = tau * logsumexp(z'_topk / tau) - tau * log(k)
    3. 计算权重: w = softmax(z'_topk / tau)
    4. 对 clip features 加权池化: v_vid = sum(w * f_topk)
    5. Video head: vid_logit = Linear(v_vid)
    """

    def __init__(
        self,
        feat_dim,
        topk_ratio=0.1,
        topk_min=5,
        tau=1.0,
        learnable_tau=False,
    ):
        """
        Args:
            feat_dim: clip 特征维度
            topk_ratio: top-k 比例
            topk_min: 最小 k 值
            tau: temperature
            learnable_tau: tau 是否可学习
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.topk_ratio = topk_ratio
        self.topk_min = topk_min

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer('tau', torch.tensor(tau))


        self.video_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, seg_logits_calib, clip_features, mask=None):
        """
        Args:
            seg_logits_calib: [B, L] calibrated segment logits
            clip_features: [B, L, D] 或 [B, D, L] clip 特征
            mask: [B, L] 有效位置 mask

        Returns:
            vid_logit: [B, 1] video-level logit
            topk_idx: [B, K] top-k indices
            topk_weights: [B, K] top-k softmax weights
            pooled_feat: [B, D] 池化后的特征
        """
        B, L = seg_logits_calib.shape


        if clip_features.shape[1] != L:
            clip_features = clip_features.permute(0, 2, 1)

        D = clip_features.shape[-1]


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



        topk_idx_clamped = topk_idx.clamp(0, L - 1)
        topk_idx_expanded = topk_idx_clamped.unsqueeze(-1).expand(-1, -1, D)
        topk_features = torch.gather(clip_features, 1, topk_idx_expanded)



        topk_values_for_lse = topk_values.masked_fill(~valid_topk_mask, float('-inf'))
        z_lse = self.tau * torch.logsumexp(topk_values_for_lse / self.tau, dim=1) - self.tau * math.log(max(k, 1))


        topk_values_for_softmax = topk_values.masked_fill(~valid_topk_mask, float('-inf'))
        topk_weights = F.softmax(topk_values_for_softmax / self.tau, dim=1)

        topk_weights = torch.nan_to_num(topk_weights, nan=0.0)


        pooled_feat = (topk_weights.unsqueeze(-1) * topk_features).sum(dim=1)


        vid_logit = self.video_head(pooled_feat)

        return vid_logit, topk_idx, topk_weights, pooled_feat


class SegmentHead(nn.Module):
    """
    Segment Classification Head

    输出每个 clip 的伪造 logit
    """

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
        """
        Args:
            features: [B, D, L] 或 [B, L, D]
            mask: [B, 1, L] 或 [B, L]

        Returns:
            seg_logits: [B, L]
        """
        B = features.shape[0]





        if features.shape[1] > features.shape[2]:
            features = features.permute(0, 2, 1)

        out = self.layers(features)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            out = out * mask.float()

        return out.squeeze(1)


class FrequencyEvidenceBranch(nn.Module):
    """
    频率取证分支

    方案2: Top-k patch 预筛选 + 条件注意力池化 + 残差融合

    流程:
    1. 对每个 clip 的 P 个 patch，计算异常分数
    2. 选 top-k patch
    3. 用 VideoMAE clip 特征做 query，对 top-k patch 做 attention pooling
    4. 残差融合: f_t = v_t + W * z_t

    支持动态输入维度:
    - 若输入 freq_evidence 的最后一维 != evidence_dim，自动通过 evidence_proj 投影
    - 这允许离线缓存保存原始统计特征 E_raw，而非固定的 128 维
    """

    def __init__(
        self,
        videomae_dim=1408,
        evidence_dim=128,
        hidden_dim=256,
        topk_patches=64,
        num_heads=4,
        dropout=0.1,
        input_evidence_dim=None,
        temporal_pooling="per_clip",
        input_adapter="linear",
        spectral_size=16,
    ):
        """
        Args:
            videomae_dim: VideoMAE 特征维度
            evidence_dim: 内部 evidence 向量维度（投影后）
            hidden_dim: 隐藏层维度
            topk_patches: 每个 clip 保留的 top-k patch 数量
            num_heads: attention heads 数量
            dropout: dropout rate
            input_evidence_dim: 输入 evidence 维度（用于离线缓存），None=与 evidence_dim 相同
            temporal_pooling: patch 排名的时间聚合方式
            input_adapter: linear 或 spectral_conv
            spectral_size: spectral_conv 输入频谱的边长
        """
        super().__init__()
        self.videomae_dim = videomae_dim
        self.evidence_dim = evidence_dim
        self.hidden_dim = hidden_dim
        self.topk_patches = topk_patches
        self.num_heads = num_heads
        self.input_evidence_dim = input_evidence_dim or evidence_dim
        self.input_adapter = input_adapter
        self.spectral_size = spectral_size
        if temporal_pooling not in {"per_clip", "mean", "max", "top25"}:
            raise ValueError(f"Unsupported temporal_pooling: {temporal_pooling}")
        self.temporal_pooling = temporal_pooling

        if input_adapter not in {"linear", "spectral_conv"}:
            raise ValueError(f"Unsupported input_adapter: {input_adapter}")
        if input_adapter == "spectral_conv":
            if self.input_evidence_dim != spectral_size ** 2:
                raise ValueError(
                    "spectral_conv input_evidence_dim must equal spectral_size squared "
                    f"({spectral_size ** 2}), received {self.input_evidence_dim}"
                )
            self.spectral_adapter = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.evidence_proj = nn.Sequential(
                nn.Linear(8, evidence_dim),
                nn.LayerNorm(evidence_dim),
                nn.GELU(),
            )
        else:
            self.spectral_adapter = None

            if self.input_evidence_dim != evidence_dim:
                self.evidence_proj = nn.Sequential(
                    nn.Linear(self.input_evidence_dim, evidence_dim),
                    nn.LayerNorm(evidence_dim),
                    nn.GELU(),
                )
            else:
                self.evidence_proj = None


        self.score_mlp = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )


        self.query_proj = nn.Linear(videomae_dim, hidden_dim)
        self.key_proj = nn.Linear(evidence_dim, hidden_dim)
        self.value_proj = nn.Linear(evidence_dim, hidden_dim)

        self.attn_scale = (hidden_dim // num_heads) ** -0.5
        self.attn_dropout = nn.Dropout(dropout)


        self.fusion_proj = nn.Linear(hidden_dim, videomae_dim)


        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    def forward(self, videomae_feats, freq_evidence, mask=None):
        """
        Args:
            videomae_feats: [B, L, D] 或 [B, D, L] VideoMAE clip 特征
            freq_evidence: [B, L, P, d_e] 频率 patch evidence
                - L: clip 数
                - P: patch 数
                - d_e: evidence 向量维度（可以是 E_raw 或 evidence_dim）
            mask: [B, L] 有效位置 mask

        Returns:
            fused_feats: [B, D, L] 融合后的特征
            topk_indices: [B, L, K] 每个 clip 的 top-k patch indices
            attn_weights: [B, L, K] attention weights
        """

        if videomae_feats.shape[1] != videomae_feats.shape[-1]:
            if videomae_feats.shape[1] > videomae_feats.shape[2]:
                videomae_feats = videomae_feats.permute(0, 2, 1)

        B, L, D = videomae_feats.shape
        P = freq_evidence.shape[2]
        K = min(self.topk_patches, P)

        if self.spectral_adapter is not None:
            if freq_evidence.shape[-1] != self.spectral_size ** 2:
                raise ValueError(
                    "spectral_conv expected flattened spectra with dimension "
                    f"{self.spectral_size ** 2}, received {freq_evidence.shape[-1]}"
                )
            spectral_input = freq_evidence.reshape(
                B * L * P, 1, self.spectral_size, self.spectral_size
            )
            freq_evidence = self.spectral_adapter(spectral_input)
            freq_evidence = freq_evidence.flatten(1).reshape(B, L, P, 8)


        if self.evidence_proj is not None:
            freq_evidence = self.evidence_proj(freq_evidence)



        patch_scores = self.score_mlp(freq_evidence).squeeze(-1)


        if self.temporal_pooling == "per_clip":
            topk_scores, topk_indices = torch.topk(
                patch_scores, K, dim=2
            )
        else:
            valid = (
                mask.bool() if mask is not None
                else torch.ones(B, L, dtype=torch.bool, device=patch_scores.device)
            )
            if valid.dim() == 3:
                valid = valid.squeeze(1)
            if self.temporal_pooling == "mean":
                weights = valid.unsqueeze(-1).to(patch_scores.dtype)
                pooled_scores = (patch_scores * weights).sum(dim=1)
                pooled_scores = pooled_scores / weights.sum(dim=1).clamp(min=1)
            elif self.temporal_pooling == "max":
                pooled_scores = patch_scores.masked_fill(
                    ~valid.unsqueeze(-1), float("-inf")
                ).max(dim=1).values
            else:
                pooled = []
                for sample_scores, sample_valid in zip(patch_scores, valid):
                    sample_scores = sample_scores[sample_valid]
                    count = max(1, math.ceil(sample_scores.shape[0] * 0.25))
                    pooled.append(sample_scores.topk(count, dim=0).values.mean(dim=0))
                pooled_scores = torch.stack(pooled, dim=0)
            _, shared_indices = torch.topk(pooled_scores, K, dim=1)
            topk_indices = shared_indices.unsqueeze(1).expand(-1, L, -1)
            topk_scores = torch.gather(patch_scores, 2, topk_indices)


        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.evidence_dim)
        topk_evidence = torch.gather(freq_evidence, 2, topk_indices_expanded)



        q = self.query_proj(videomae_feats)

        k = self.key_proj(topk_evidence)
        v = self.value_proj(topk_evidence)



        attn_logits = torch.einsum('bld,blkd->blk', q, k) * self.attn_scale
        attn_weights = F.softmax(attn_logits, dim=-1)
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


class VideoClassificationHead(nn.Module):
    """
    完整的视频级二分类 Head

    整合:
    - Segment Head (产生 z_t)
    - CMAM (产生 m_t)
    - Motion Gate Calibration (产生 z'_t)
    - Top-k LogSumExp Aggregator (产生 vid_logit)
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim=256,
        topk_ratio=0.1,
        topk_min=5,
        tau=1.0,
        cmam_align_dim=64,
        enable_cmam=True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.enable_cmam = enable_cmam


        self.segment_head = SegmentHead(
            input_dim=feat_dim,
            hidden_dim=hidden_dim,
        )


        if enable_cmam:
            self.cmam = CMAMFromAMFlow(
                proj_dim=cmam_align_dim,
                n_layers=2,
            )
            self.motion_gate = MotionGateCalibration()


        self.aggregator = TopKLogSumExpAggregator(
            feat_dim=feat_dim,
            topk_ratio=topk_ratio,
            topk_min=topk_min,
            tau=tau,
        )

    def forward(self, features, amflow=None, mask=None):
        """
        Args:
            features: [B, D, L] clip 特征
            amflow: [B, L] attention 差分（用于 CMAM）
            mask: [B, 1, L] 或 [B, L] 有效位置 mask

        Returns:
            dict with:
                - seg_logits: [B, L]
                - seg_logits_calib: [B, L]
                - motion_intensity: [B, L]
                - vid_logit: [B, 1]
                - topk_idx: [B, K]
                - topk_weights: [B, K]
                - pooled_feat: [B, D]
        """
        B, D, L = features.shape


        if mask is not None and mask.dim() == 3:
            mask_2d = mask.squeeze(1)
        elif mask is not None:
            mask_2d = mask
        else:
            mask_2d = None

        seg_logits = self.segment_head(features, mask)


        if self.enable_cmam and amflow is not None:
            motion_intensity = self.cmam(amflow, mask)
            seg_logits_calib, gate_values = self.motion_gate(seg_logits, motion_intensity)
        else:
            motion_intensity = torch.zeros_like(seg_logits)
            seg_logits_calib = seg_logits
            gate_values = torch.ones_like(seg_logits)


        vid_logit, topk_idx, topk_weights, pooled_feat = self.aggregator(
            seg_logits_calib, features, mask_2d
        )

        return {
            'seg_logits': seg_logits,
            'seg_logits_calib': seg_logits_calib,
            'motion_intensity': motion_intensity,
            'gate_values': gate_values,
            'vid_logit': vid_logit,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'pooled_feat': pooled_feat,
        }
