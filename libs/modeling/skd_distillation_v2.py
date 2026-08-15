import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SKDDistillationWrapperV2(nn.Module):
    """
    Improved SKD wrapper that:
    1. Uses logit MSE matching (not KL with sigmoid)
    2. Properly extracts teacher vid_logit from vid_cls_combined
    3. Supports vanilla KD (no gate) and selective KD (with gate)
    """

    def __init__(self, student, teacher, skd_config=None):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        cfg = skd_config or {}
        self.temperature = cfg.get('temperature', 4.0)
        self.kd_clip_weight = cfg.get('kd_clip_weight', 1.0)
        self.kd_vid_weight = cfg.get('kd_vid_weight', 0.5)
        self.use_gate = cfg.get('theta_conf', 0.3) > 0
        self.theta_conf = cfg.get('theta_conf', 0.3)
        self.alpha = cfg.get('alpha', 0.5)
        self.use_teacher_calibrated_logits = cfg.get('use_teacher_calibrated_logits', True)
        self.use_motion_clip_weight = cfg.get('use_motion_clip_weight', True)

    def train(self, mode=True):
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    def forward(self, video_list):
        if self.training:
            return self._training_forward(video_list)
        return self.student(video_list)

    def _training_forward(self, video_list):

        student_losses = self.student(video_list)


        with torch.no_grad():
            t_out = self._extract_outputs(self.teacher, video_list)

        s_out = self._extract_outputs(self.student, video_list)


        min_L = min(t_out['seg_logits'].shape[1], s_out['seg_logits'].shape[1])
        t_seg = t_out['seg_logits'][:, :min_L].detach()
        s_seg = s_out['seg_logits'][:, :min_L]
        if self.use_teacher_calibrated_logits:
            t_seg_kd = t_out['seg_logits_calib'][:, :min_L].detach()
        else:
            t_seg_kd = t_seg
        s_seg_calib = s_out['seg_logits_calib'][:, :min_L]
        mask = s_out['mask'][:, :min_L].float()
        t_motion = t_out['motion_gate'][:, :min_L].detach()
        t_vid = t_out['vid_logit'].detach()
        s_vid = s_out['vid_logit']

        B = t_seg.shape[0]


        if self.use_gate:
            gt_labels = self._get_gt_labels(video_list)

            t_vid_prob_direct = torch.sigmoid(t_vid)
            t_vid_mil = t_out.get('vid_logit_mil', t_vid)
            t_vid_prob_mil = torch.sigmoid(t_vid_mil.detach())
            t_vid_prob = 0.5 * t_vid_prob_direct + 0.5 * t_vid_prob_mil


            t_pred = (t_vid_prob >= t_vid_prob.mean()).float()
            conf = torch.abs(t_vid_prob - t_vid_prob.mean())
            conf_mask = (conf >= self.theta_conf).float()
            correct = (t_pred == gt_labels).float()
            gate = conf_mask * correct
        else:
            gate = torch.ones(B, device=t_seg.device)


        t_soft = t_seg_kd / self.temperature
        s_soft = s_seg_calib / self.temperature
        clip_mse = ((t_soft - s_soft) ** 2) * mask
        if self.use_motion_clip_weight:
            clip_mse = clip_mse * t_motion
        clip_kd_per_sample = clip_mse.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        kd_clip_loss = (gate * clip_kd_per_sample).sum() / gate.sum().clamp(min=1)


        vid_mse = (t_vid / self.temperature - s_vid / self.temperature) ** 2
        kd_vid_loss = (gate * vid_mse.squeeze(-1) if vid_mse.dim() > 1 else gate * vid_mse).sum() / gate.sum().clamp(min=1)

        kd_total = self.kd_clip_weight * kd_clip_loss + self.kd_vid_weight * kd_vid_loss

        combined = student_losses.copy()
        combined['kd_clip_loss'] = kd_clip_loss
        combined['kd_vid_loss'] = kd_vid_loss
        combined['final_loss'] = student_losses['final_loss'] + kd_total
        combined['video_gate_ratio'] = gate.mean()

        return combined

    def _extract_outputs(self, model, video_list):
        """Extract intermediate outputs from FCADFormer without running full loss computation."""
        was_training = model.training
        model.train()

        batched_inputs, batched_masks = model.preprocessing(video_list)

        if model.enable_feb and model.feb is not None:
            target_len = batched_inputs.shape[-1]
            freq_evidence = model._get_freq_evidence(video_list, target_len=target_len)
            if freq_evidence is not None:
                batched_inputs, _, _ = model.feb(
                    batched_inputs.permute(0, 2, 1),
                    freq_evidence,
                    batched_masks.squeeze(1)
                )


        mask_pool = batched_masks.float()
        raw_avg = (batched_inputs * mask_pool).sum(dim=2) / mask_pool.sum(dim=2).clamp(min=1)
        raw_max = batched_inputs.masked_fill(~batched_masks.bool(), float('-inf')).max(dim=2)[0]

        feats, masks = model.backbone(batched_inputs, batched_masks)
        stem_feat = feats[0]
        stem_mask = masks[0]
        mask_2d = stem_mask.squeeze(1) if stem_mask.dim() == 3 else stem_mask


        stem_det = stem_feat.detach().permute(0, 2, 1)
        attn_scores = model.cls_temporal_attn(stem_det).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask_2d, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        attended_feat = (stem_det * attn_weights.unsqueeze(-1)).sum(dim=1)

        vid_feat = torch.cat([raw_avg, raw_max, attended_feat], dim=1)
        vid_logit = model.vid_cls_combined(vid_feat)

        seg_logits = model.segment_head(stem_feat, stem_mask)

        if hasattr(model, 'enable_cmam') and model.enable_cmam and model.motion_gate is not None:
            motion_intensity = model._compute_motion_intensity(stem_feat, stem_mask)
            seg_logits_calib, gate_values = model.motion_gate(seg_logits, motion_intensity)
        else:
            seg_logits_calib = seg_logits
            gate_values = torch.ones_like(seg_logits)


        vid_logit_mil, _, _, _ = model.aggregator(seg_logits_calib, stem_feat, mask_2d)

        if not was_training:
            model.eval()

        return {
            'seg_logits': seg_logits,
            'seg_logits_calib': seg_logits_calib,
            'vid_logit': vid_logit.squeeze(-1),
            'vid_logit_mil': vid_logit_mil.squeeze(-1),
            'motion_gate': gate_values,
            'mask': mask_2d.bool(),
        }

    def _get_gt_labels(self, video_list):
        device = next(self.student.parameters()).device
        labels = []
        for v in video_list:
            if 'video_label' in v:
                lab = v['video_label']
                lab = lab.item() if isinstance(lab, torch.Tensor) else float(lab)
            else:
                lab = 1.0 if v['segments'] is not None and len(v['segments']) > 0 else 0.0
            labels.append(lab)
        return torch.tensor(labels, device=device, dtype=torch.float32)
