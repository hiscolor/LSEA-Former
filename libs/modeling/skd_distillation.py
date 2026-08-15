import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def bernoulli_entropy(probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 Bernoulli 分布的 entropy
    H(p) = -p*log(p) - (1-p)*log(1-p)

    Args:
        probs: [B, L] 概率值，范围 [0, 1]
        eps: 数值稳定性

    Returns:
        entropy: [B, L]
    """
    probs = probs.clamp(eps, 1 - eps)
    return -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)


def bernoulli_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算 Bernoulli 分布的 KL divergence
    KL(Bern(p) || Bern(q)) = p*log(p/q) + (1-p)*log((1-p)/(1-q))

    Args:
        p: [B, L] teacher 概率
        q: [B, L] student 概率
        eps: 数值稳定性

    Returns:
        kl: [B, L]
    """
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


class SelectiveKnowledgeDistillation(nn.Module):
    """
    选择性知识蒸馏模块

    论文核心逻辑:
    1. Video-level gate G: 只在 teacher 高置信度且预测正确时蒸馏
    2. Clip weights omega_t: 基于 entropy 和 KL 计算，考虑 motion gate
    3. KD losses: 在 motion-calibrated logits z' 上进行蒸馏
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 1.0,
        theta_conf: float = 0.3,
        topk_ratio: float = 0.1,
        topk_min: int = 5,
        kd_clip_weight: float = 1.0,
        kd_vid_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.theta_conf = theta_conf
        self.topk_ratio = topk_ratio
        self.topk_min = topk_min
        self.kd_clip_weight = kd_clip_weight
        self.kd_vid_weight = kd_vid_weight

    def compute_video_gate(
        self,
        teacher_vid_prob: torch.Tensor,
        gt_video_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 video-level gate
        G = 1[|p_vid^T - 0.5| >= theta_conf] * 1[y_hat^T == y]

        Args:
            teacher_vid_prob: [B] teacher 的视频级概率
            gt_video_labels: [B] GT 视频标签

        Returns:
            gate: [B] 0 或 1
        """

        confidence_mask = (torch.abs(teacher_vid_prob - 0.5) >= self.theta_conf).float()



        teacher_pred = (teacher_vid_prob >= 0.5).float()
        correctness_mask = (teacher_pred == gt_video_labels).float()


        gate = confidence_mask * correctness_mask

        return gate

    def compute_clip_weights(
        self,
        teacher_clip_probs: torch.Tensor,
        student_clip_probs: torch.Tensor,
        teacher_motion_gate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 clip-level 蒸馏权重

        omega_t = softmax(beta * s_t) * c_t^T
        s_t = alpha * H_t + (1-alpha) * D_t

        Args:
            teacher_clip_probs: [B, L] teacher 的 clip-level 概率
            student_clip_probs: [B, L] student 的 clip-level 概率
            teacher_motion_gate: [B, L] teacher 的 motion gate 值 c_t
            mask: [B, L] 有效位置 mask

        Returns:
            omega: [B, L] 归一化后的权重，在 mask 内 sum=1
        """

        H_t = bernoulli_entropy(teacher_clip_probs)


        D_t = bernoulli_kl_divergence(teacher_clip_probs, student_clip_probs)


        s_t = self.alpha * H_t + (1 - self.alpha) * D_t


        s_t = s_t.masked_fill(~mask, float('-inf'))


        omega = F.softmax(self.beta * s_t, dim=-1) * teacher_motion_gate


        omega = omega * mask.float()
        omega_sum = omega.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        omega = omega / omega_sum

        return omega

    def compute_topk_omega_bar(
        self,
        seg_logits_calib: torch.Tensor,
        omega: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 video-level 聚合使用的权重 omega_bar
        使用与视频聚合一致的 Top-k 索引集合

        Args:
            seg_logits_calib: [B, L] calibrated segment logits (用于选择 top-k)
            omega: [B, L] clip-level weights
            mask: [B, L] 有效位置 mask

        Returns:
            omega_bar: [B] video-level 权重
            topk_idx: [B, K] top-k indices
        """
        B, L = seg_logits_calib.shape


        valid_lens = mask.sum(dim=1)
        k_per_sample = torch.clamp(
            (valid_lens * self.topk_ratio).long(),
            min=self.topk_min
        )
        k = k_per_sample.max().item()


        masked_logits = seg_logits_calib.masked_fill(~mask, float('-inf'))


        _, topk_idx = torch.topk(masked_logits, k, dim=1)


        topk_omega = torch.gather(omega, 1, topk_idx)


        omega_bar = topk_omega.sum(dim=1)

        return omega_bar, topk_idx

    def compute_kd_losses(
        self,
        teacher_seg_logits_calib: torch.Tensor,
        student_seg_logits_calib: torch.Tensor,
        teacher_vid_logit: torch.Tensor,
        student_vid_logit: torch.Tensor,
        teacher_clip_probs: torch.Tensor,
        student_clip_probs: torch.Tensor,
        teacher_motion_gate: torch.Tensor,
        gt_video_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 SKD 损失

        L_clip_KD = G * tau^2 * sum_t omega_t * KL(...)
        L_vid_KD = G * omega_bar * tau^2 * KL(...)

        Returns:
            dict with:
                - kd_clip_loss: clip-level KD loss
                - kd_vid_loss: video-level KD loss
                - kd_total_loss: weighted sum
                - video_gate: [B] for monitoring
        """
        B, L = teacher_seg_logits_calib.shape
        device = teacher_seg_logits_calib.device
        tau = self.temperature


        teacher_vid_prob = torch.sigmoid(teacher_vid_logit)
        gate = self.compute_video_gate(teacher_vid_prob, gt_video_labels)


        omega = self.compute_clip_weights(
            teacher_clip_probs,
            student_clip_probs,
            teacher_motion_gate,
            mask
        )



        teacher_clip_probs_temp = torch.sigmoid(teacher_seg_logits_calib / tau)
        student_clip_probs_temp = torch.sigmoid(student_seg_logits_calib / tau)

        kl_clip = bernoulli_kl_divergence(
            teacher_clip_probs_temp.detach(),
            student_clip_probs_temp
        )


        kd_clip_per_sample = tau * tau * (omega * kl_clip).sum(dim=-1)
        kd_clip_loss = (gate * kd_clip_per_sample).mean()


        omega_bar, _ = self.compute_topk_omega_bar(
            teacher_seg_logits_calib,
            omega,
            mask
        )

        teacher_vid_prob_temp = torch.sigmoid(teacher_vid_logit / tau)
        student_vid_prob_temp = torch.sigmoid(student_vid_logit / tau)

        kl_vid = bernoulli_kl_divergence(
            teacher_vid_prob_temp.detach(),
            student_vid_prob_temp
        )


        kd_vid_per_sample = omega_bar * tau * tau * kl_vid
        kd_vid_loss = (gate * kd_vid_per_sample).mean()


        kd_total_loss = self.kd_clip_weight * kd_clip_loss + self.kd_vid_weight * kd_vid_loss

        return {
            'kd_clip_loss': kd_clip_loss,
            'kd_vid_loss': kd_vid_loss,
            'kd_total_loss': kd_total_loss,
            'video_gate': gate,
            'clip_weights': omega,
        }


class SKDDistillationWrapper(nn.Module):
    """
    SKD 蒸馏包装器

    将 Student 和 Teacher 模型组合，在训练时计算 SKD 损失
    Teacher 冻结，只训练 Student
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        skd_config: dict = None,
    ):
        """
        Args:
            student: Student 模型 (FCADFormer)
            teacher: Teacher 模型 (预训练并冻结)
            skd_config: SKD 配置字典
        """
        super().__init__()
        self.student = student
        self.teacher = teacher


        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False


        skd_config = skd_config or {}
        self.skd = SelectiveKnowledgeDistillation(
            temperature=skd_config.get('temperature', 4.0),
            alpha=skd_config.get('alpha', 0.5),
            beta=skd_config.get('beta', 1.0),
            theta_conf=skd_config.get('theta_conf', 0.3),
            topk_ratio=skd_config.get('topk_ratio', 0.1),
            topk_min=skd_config.get('topk_min', 5),
            kd_clip_weight=skd_config.get('kd_clip_weight', 1.0),
            kd_vid_weight=skd_config.get('kd_vid_weight', 0.5),
        )


        self.enable_skd = skd_config.get('enabled', True)

    def train(self, mode=True):
        """保持 teacher 在 eval 模式"""
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    def forward(self, video_list):
        """
        Forward pass

        训练时: student forward + teacher forward + SKD losses
        推理时: 只用 student
        """
        if self.training and self.enable_skd:
            return self._training_forward(video_list)
        else:
            return self.student(video_list)

    def _training_forward(self, video_list):
        """训练时的 forward"""

        student_losses = self.student(video_list)


        with torch.no_grad():
            teacher_outputs = self._get_teacher_outputs(video_list)


        student_outputs = self._get_student_outputs_with_grad(video_list)


        gt_video_labels = self._get_gt_video_labels(video_list)


        teacher_L = teacher_outputs['seg_logits'].shape[1]
        student_L = student_outputs['seg_logits'].shape[1]
        min_L = min(teacher_L, student_L)


        teacher_seg_logits_calib = teacher_outputs['seg_logits_calib'][:, :min_L]
        student_seg_logits_calib = student_outputs['seg_logits_calib'][:, :min_L]
        teacher_seg_logits = teacher_outputs['seg_logits'][:, :min_L]
        student_seg_logits = student_outputs['seg_logits'][:, :min_L]
        teacher_motion_gate = teacher_outputs['motion_gate'][:, :min_L]
        mask = student_outputs['mask'][:, :min_L]


        skd_losses = self.skd.compute_kd_losses(
            teacher_seg_logits_calib=teacher_seg_logits_calib,
            student_seg_logits_calib=student_seg_logits_calib,
            teacher_vid_logit=teacher_outputs['vid_logit'],
            student_vid_logit=student_outputs['vid_logit'],
            teacher_clip_probs=torch.sigmoid(teacher_seg_logits),
            student_clip_probs=torch.sigmoid(student_seg_logits),
            teacher_motion_gate=teacher_motion_gate,
            gt_video_labels=gt_video_labels,
            mask=mask,
        )


        combined_losses = student_losses.copy()
        combined_losses['kd_clip_loss'] = skd_losses['kd_clip_loss']
        combined_losses['kd_vid_loss'] = skd_losses['kd_vid_loss']
        combined_losses['final_loss'] = student_losses['final_loss'] + skd_losses['kd_total_loss']


        combined_losses['video_gate_ratio'] = skd_losses['video_gate'].mean()

        return combined_losses

    def _get_student_outputs_with_grad(self, video_list) -> dict:
        """获取 student 的中间输出（带梯度）"""
        batched_inputs, batched_masks = self.student.preprocessing(video_list)


        if self.student.enable_feb and self.student.feb is not None:
            target_len = batched_inputs.shape[-1]
            freq_evidence = self.student._get_freq_evidence(video_list, target_len=target_len)
            if freq_evidence is not None:
                batched_inputs, _, _ = self.student.feb(
                    batched_inputs.permute(0, 2, 1),
                    freq_evidence,
                    batched_masks.squeeze(1)
                )


        feats, masks = self.student.backbone(batched_inputs, batched_masks)
        stem_feat = feats[0]
        stem_mask = masks[0]


        seg_logits = self.student.segment_head(stem_feat, stem_mask)


        motion_intensity = self.student._compute_motion_intensity(stem_feat, stem_mask)
        if self.student.motion_gate is not None:
            seg_logits_calib, gate_values = self.student.motion_gate(seg_logits, motion_intensity)
        else:
            seg_logits_calib = seg_logits
            gate_values = torch.ones_like(seg_logits)


        mask_2d = stem_mask.squeeze(1) if stem_mask.dim() == 3 else stem_mask
        vid_logit, _, _, _ = self.student.aggregator(seg_logits_calib, stem_feat, mask_2d)

        return {
            'seg_logits': seg_logits,
            'seg_logits_calib': seg_logits_calib,
            'vid_logit': vid_logit.squeeze(-1),
            'motion_gate': gate_values,
            'mask': mask_2d.bool(),
        }

    def _get_teacher_outputs(self, video_list) -> dict:
        """获取 teacher 的中间输出 (no grad)"""

        was_training = self.teacher.training
        self.teacher.train()

        batched_inputs, batched_masks = self.teacher.preprocessing(video_list)


        if hasattr(self.teacher, 'enable_feb') and self.teacher.enable_feb and self.teacher.feb is not None:
            target_len = batched_inputs.shape[-1]
            freq_evidence = self.teacher._get_freq_evidence(video_list, target_len=target_len)
            if freq_evidence is not None:
                batched_inputs, _, _ = self.teacher.feb(
                    batched_inputs.permute(0, 2, 1),
                    freq_evidence,
                    batched_masks.squeeze(1)
                )


        feats, masks = self.teacher.backbone(batched_inputs, batched_masks)
        stem_feat = feats[0]
        stem_mask = masks[0]


        seg_logits = self.teacher.segment_head(stem_feat, stem_mask)


        if hasattr(self.teacher, '_compute_motion_intensity'):
            motion_intensity = self.teacher._compute_motion_intensity(stem_feat, stem_mask)
        else:
            motion_intensity = torch.zeros_like(seg_logits)

        if hasattr(self.teacher, 'motion_gate') and self.teacher.motion_gate is not None:
            seg_logits_calib, gate_values = self.teacher.motion_gate(seg_logits, motion_intensity)
        else:
            seg_logits_calib = seg_logits
            gate_values = torch.ones_like(seg_logits)


        mask_2d = stem_mask.squeeze(1) if stem_mask.dim() == 3 else stem_mask
        vid_logit, _, _, _ = self.teacher.aggregator(seg_logits_calib, stem_feat, mask_2d)


        if not was_training:
            self.teacher.eval()

        return {
            'seg_logits': seg_logits,
            'seg_logits_calib': seg_logits_calib,
            'vid_logit': vid_logit.squeeze(-1),
            'motion_gate': gate_values,
            'mask': mask_2d.bool(),
        }

    def _get_gt_video_labels(self, video_list) -> torch.Tensor:
        """获取 GT 视频标签"""
        device = next(self.student.parameters()).device
        gt_labels = []

        for video in video_list:
            if 'video_label' in video:
                label = video['video_label']
                if isinstance(label, torch.Tensor):
                    gt_labels.append(label.to(device).view(1))
                else:
                    gt_labels.append(torch.tensor([label], device=device, dtype=torch.float32))
            else:
                has_segments = video['segments'] is not None and len(video['segments']) > 0
                gt_labels.append(torch.tensor([1.0 if has_segments else 0.0], device=device))

        return torch.cat(gt_labels)


def load_fcad_teacher(teacher_cfg_path: str, teacher_ckpt_path: str, device) -> nn.Module:
    """
    加载 FCAD-Former teacher 模型

    Args:
        teacher_cfg_path: teacher 配置文件路径
        teacher_ckpt_path: teacher checkpoint 路径
        device: 设备

    Returns:
        冻结的 teacher 模型
    """
    from ..core import load_config
    from .models import make_meta_arch


    teacher_cfg = load_config(teacher_cfg_path)


    teacher = make_meta_arch(teacher_cfg['model_name'], **teacher_cfg['model'])


    checkpoint = torch.load(teacher_ckpt_path, map_location=device)

    state_dict = checkpoint.get('state_dict', checkpoint.get('state_dict_ema', checkpoint))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    missing_keys, unexpected_keys = teacher.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Teacher missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Teacher unexpected keys: {unexpected_keys}")

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Loaded FCAD teacher from {teacher_ckpt_path}")

    return teacher
