import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionProxy(nn.Module):
    """
    计算相邻时间点的"局部中心相似度变化"作为相机运动强度代理

    输入: B x C x T (特征), B x 1 x T (mask)
    输出: B x T (motion_proxy 值，越大代表运动越剧烈)

    实现方式（轻量）：
    1. 用 1x1 conv 将特征从 C 投影到 proj_dim（如 64）
    2. 对每个时间点 t，计算窗口内各点对中心点的相似度向量 r_t
    3. motion_proxy_t = |r_t - r_{t+1}|_1 （相邻窗口相似度结构的变化）
    4. 视频内归一化

    复杂度: O(B × T × L × proj_dim)，L=window_size 很小，可接受
    """

    def __init__(
        self,
        input_dim,
        proj_dim=64,
        window_size=7,
        norm_type='zscore',
        detach_gradient=True,
        smooth_kernel_size=3,
    ):
        """
        Args:
            input_dim: 输入特征维度 (如 1408)
            proj_dim: 投影后的维度 (如 64)，用于降低计算量
            window_size: 局部窗口大小，必须为奇数 (如 7)
            norm_type: 归一化类型，'zscore' 或 'minmax'
            detach_gradient: 是否阻断梯度，建议默认 True
            smooth_kernel_size: 输出平滑滤波核大小，用于降低插值噪声
        """
        super().__init__()

        assert window_size % 2 == 1, "window_size must be odd"

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.window_size = window_size
        self.half_window = window_size // 2
        self.norm_type = norm_type
        self.detach_gradient = detach_gradient
        self.smooth_kernel_size = smooth_kernel_size


        self.proj = nn.Conv1d(input_dim, proj_dim, kernel_size=1, bias=False)


        if smooth_kernel_size > 1:
            self.smooth = nn.AvgPool1d(
                kernel_size=smooth_kernel_size,
                stride=1,
                padding=smooth_kernel_size // 2
            )
        else:
            self.smooth = None

    def forward(self, feats, mask):
        """
        Args:
            feats: B x C x T 特征张量
            mask: B x 1 x T 有效位置掩码 (bool)

        Returns:
            motion_proxy: B x T 运动强度代理值（已归一化）
        """
        batch_size, channels, seq_len = feats.shape
        device = feats.device


        if self.detach_gradient:
            feats = feats.detach()


        feats_proj = self.proj(feats)


        feats_norm = F.normalize(feats_proj, p=2, dim=1)







        feats_padded = F.pad(
            feats_norm,
            (self.half_window, self.half_window),
            mode='replicate'
        )


        feats_windows = feats_padded.unfold(
            dimension=2,
            size=self.window_size,
            step=1
        )


        center_feats = feats_norm.unsqueeze(-1)



        similarity = (feats_windows * center_feats).sum(dim=1)




        sim_curr = similarity[:, :-1, :]
        sim_next = similarity[:, 1:, :]


        motion_diff = (sim_curr - sim_next).abs().mean(dim=-1)


        motion_proxy = F.pad(motion_diff, (0, 1), mode='replicate')


        mask_float = mask.squeeze(1).float()
        motion_proxy = motion_proxy * mask_float


        if self.smooth is not None:
            motion_proxy = self.smooth(motion_proxy.unsqueeze(1)).squeeze(1)
            motion_proxy = motion_proxy * mask_float


        motion_proxy = self._normalize_per_video(motion_proxy, mask_float)

        return motion_proxy

    def _normalize_per_video(self, motion_proxy, mask):
        """
        对每个视频内部进行归一化

        Args:
            motion_proxy: B x T
            mask: B x T (float)

        Returns:
            normalized motion_proxy: B x T
        """
        batch_size = motion_proxy.shape[0]

        for b in range(batch_size):
            valid_mask = mask[b] > 0.5
            if valid_mask.sum() < 2:
                continue

            valid_values = motion_proxy[b, valid_mask]

            if self.norm_type == 'zscore':
                mean_val = valid_values.mean()
                std_val = valid_values.std() + 1e-6
                motion_proxy[b, valid_mask] = (valid_values - mean_val) / std_val

                motion_proxy[b, valid_mask] = torch.sigmoid(motion_proxy[b, valid_mask])

            elif self.norm_type == 'minmax':
                min_val = valid_values.min()
                max_val = valid_values.max()
                range_val = max_val - min_val + 1e-6
                motion_proxy[b, valid_mask] = (valid_values - min_val) / range_val

            elif self.norm_type == 'robust_zscore':

                median_val = valid_values.median()
                mad = (valid_values - median_val).abs().median() + 1e-6
                motion_proxy[b, valid_mask] = (valid_values - median_val) / (1.4826 * mad)
                motion_proxy[b, valid_mask] = torch.sigmoid(motion_proxy[b, valid_mask])

        return motion_proxy

    def get_segment_motion(self, motion_proxy, segments, mask=None):
        """
        计算每个 segment/proposal 的运动强度

        Args:
            motion_proxy: B x T 运动代理值
            segments: B x N x 2 或 N x 2，segment 的起止时间（特征网格坐标）
            mask: B x T (可选)

        Returns:
            segment_motion: 每个 segment 的平均运动强度
        """
        if segments.dim() == 2:

            segments = segments.unsqueeze(0)
            motion_proxy = motion_proxy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_segments, _ = segments.shape
        seq_len = motion_proxy.shape[1]
        device = motion_proxy.device

        segment_motion = torch.zeros(batch_size, num_segments, device=device)

        for b in range(batch_size):
            for n in range(num_segments):
                start_idx = max(0, int(segments[b, n, 0].item()))
                end_idx = min(seq_len, int(segments[b, n, 1].item()) + 1)

                if end_idx > start_idx:
                    segment_motion[b, n] = motion_proxy[b, start_idx:end_idx].mean()

        if squeeze_output:
            segment_motion = segment_motion.squeeze(0)

        return segment_motion


class MotionAwareScoreCalibrator(nn.Module):
    """
    基于 motion_proxy 对检测分数进行校正

    校正公式（乘性抑制）：
        s'_p = s_p * sigmoid(scale - motion_weight * m_p)

    其中 m_p 是该 proposal 的平均运动强度
    """

    def __init__(
        self,
        motion_weight=2.0,
        scale=1.0,
        calibration_type='multiplicative',
    ):
        """
        Args:
            motion_weight: 运动抑制权重，越大抑制越强
            scale: 偏移量，控制基准抑制程度
            calibration_type: 校正类型
                - 'multiplicative': s' = s * sigmoid(scale - weight * m)
                - 'exponential': s' = s * exp(-weight * m)
        """
        super().__init__()

        self.motion_weight = motion_weight
        self.scale = scale
        self.calibration_type = calibration_type

    def forward(self, scores, segment_motion):
        """
        Args:
            scores: N 个 proposal 的原始分数
            segment_motion: N 个 proposal 的运动强度

        Returns:
            calibrated_scores: 校正后的分数
        """
        if self.calibration_type == 'multiplicative':

            motion_factor = torch.sigmoid(
                self.scale - self.motion_weight * segment_motion
            )
        elif self.calibration_type == 'exponential':

            motion_factor = torch.exp(-self.motion_weight * segment_motion)
        else:
            raise ValueError(f"Unknown calibration_type: {self.calibration_type}")

        calibrated_scores = scores * motion_factor

        return calibrated_scores
