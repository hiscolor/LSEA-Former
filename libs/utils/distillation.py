import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationWrapper(nn.Module):
    """
    Wrapper module that combines Student and Teacher models for distillation training.

    Teacher is frozen (eval mode, no gradients).
    Student is trained with task loss + distillation loss.
    """

    def __init__(
        self,
        student,
        teacher,
        kd_video_weight=0.5,
        kd_temp_weight=1.0,
        temperature=4.0,
    ):
        """
        Args:
            student: Student model (nn.Module)
            teacher: Teacher model (nn.Module), will be frozen
            kd_video_weight: Weight for video-level KL divergence loss
            kd_temp_weight: Weight for temporal scores MSE loss
            temperature: Temperature for KL divergence
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.kd_video_weight = kd_video_weight
        self.kd_temp_weight = kd_temp_weight
        self.temperature = temperature


        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Override train to keep teacher in eval mode."""

        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    def forward(self, video_list):
        """
        Forward pass for distillation training.

        Returns:
            dict: Combined losses (task loss + distillation losses)
        """
        if self.training:

            student_losses = self.student(video_list)


            with torch.no_grad():

                teacher_outputs = self._get_teacher_outputs(video_list)


            student_outputs = self._get_student_outputs(video_list)


            kd_losses = self._compute_kd_losses(student_outputs, teacher_outputs)


            combined_losses = student_losses.copy()
            combined_losses['kd_video_loss'] = kd_losses['kd_video_loss']
            combined_losses['kd_temp_loss'] = kd_losses['kd_temp_loss']
            combined_losses['final_loss'] = (
                student_losses['final_loss'] +
                self.kd_video_weight * kd_losses['kd_video_loss'] +
                self.kd_temp_weight * kd_losses['kd_temp_loss']
            )

            return combined_losses
        else:

            return self.student(video_list)

    def _get_teacher_outputs(self, video_list):
        """Extract teacher's cls_scores and temporal logits."""


        outputs = {}


        self.teacher.eval()


        feats = [vdata['feats'] for vdata in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max()
        batch_size = len(feats)


        device = next(self.teacher.parameters()).device
        batched_inputs = feats[0].new_zeros(batch_size, feats[0].shape[0], max_len)
        batched_masks = feats[0].new_zeros(batch_size, 1, max_len)

        for idx in range(batch_size):
            batched_inputs[idx, :, :feats_lens[idx]] = feats[idx]
            batched_masks[idx, :, :feats_lens[idx]] = 1.0

        batched_inputs = batched_inputs.to(device)
        batched_masks = batched_masks.to(device).bool()


        model = self.teacher.module if hasattr(self.teacher, 'module') else self.teacher
        norm_inputs, reco_result, cls_scores = model.interpolator(batched_inputs, batched_masks)


        backbone_out = model.backbone(
            batched_inputs, norm_inputs, reco_result, batched_masks, return_aux=False
        )
        feats, masks = backbone_out[0], backbone_out[1]
        fpn_feats, fpn_masks = model.neck(feats, masks)
        out_cls_logits = model.cls_head(fpn_feats, fpn_masks)

        outputs['cls_scores'] = cls_scores
        outputs['temporal_logits'] = out_cls_logits[0]

        return outputs

    def _get_student_outputs(self, video_list):
        """Extract student's cls_scores and temporal logits."""
        outputs = {}


        feats = [vdata['feats'] for vdata in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max()
        batch_size = len(feats)


        device = next(self.student.parameters()).device
        batched_inputs = feats[0].new_zeros(batch_size, feats[0].shape[0], max_len)
        batched_masks = feats[0].new_zeros(batch_size, 1, max_len)

        for idx in range(batch_size):
            batched_inputs[idx, :, :feats_lens[idx]] = feats[idx]
            batched_masks[idx, :, :feats_lens[idx]] = 1.0

        batched_inputs = batched_inputs.to(device)
        batched_masks = batched_masks.to(device).bool()


        model = self.student.module if hasattr(self.student, 'module') else self.student
        norm_inputs, reco_result, cls_scores = model.interpolator(batched_inputs, batched_masks)


        backbone_out = model.backbone(
            batched_inputs, norm_inputs, reco_result, batched_masks, return_aux=False
        )
        feats, masks = backbone_out[0], backbone_out[1]
        fpn_feats, fpn_masks = model.neck(feats, masks)
        out_cls_logits = model.cls_head(fpn_feats, fpn_masks)

        outputs['cls_scores'] = cls_scores
        outputs['temporal_logits'] = out_cls_logits[0]

        return outputs

    def _compute_kd_losses(self, student_outputs, teacher_outputs):
        """Compute knowledge distillation losses."""
        losses = {}


        student_video = student_outputs['cls_scores']
        teacher_video = teacher_outputs['cls_scores']


        student_probs = F.log_softmax(
            torch.cat([student_video, -student_video], dim=-1) / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            torch.cat([teacher_video, -teacher_video], dim=-1) / self.temperature, dim=-1
        )
        kd_video_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        kd_video_loss = kd_video_loss * (self.temperature ** 2)

        losses['kd_video_loss'] = kd_video_loss


        student_temp = student_outputs['temporal_logits']
        teacher_temp = teacher_outputs['temporal_logits']


        min_t = min(student_temp.shape[-1], teacher_temp.shape[-1])
        student_temp = student_temp[:, :, :min_t]
        teacher_temp = teacher_temp[:, :, :min_t]


        if student_temp.shape[1] != teacher_temp.shape[1]:

            if teacher_temp.shape[1] > student_temp.shape[1]:
                teacher_temp = F.adaptive_max_pool1d(
                    teacher_temp.transpose(1, 2), student_temp.shape[1]
                ).transpose(1, 2)


        kd_temp_loss = F.mse_loss(
            torch.sigmoid(student_temp),
            torch.sigmoid(teacher_temp.detach())
        )

        losses['kd_temp_loss'] = kd_temp_loss

        return losses


def load_teacher_model(teacher_cfg_path, teacher_ckpt_path, device):
    """
    Load teacher model from checkpoint with its own config.

    Args:
        teacher_cfg_path: Path to teacher's config file
        teacher_ckpt_path: Path to teacher checkpoint
        device: Device to load model to

    Returns:
        Teacher model (frozen, eval mode)
    """
    from ..modeling import make_meta_arch
    from ..core import load_config


    print(f"Loading teacher config from: {teacher_cfg_path}")
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

    print(f"Loaded teacher model from {teacher_ckpt_path}")

    return teacher
