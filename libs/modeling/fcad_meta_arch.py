import math
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from .video_cls_modules import (
    FrequencyEvidenceBranch,
    AVMCModule,
    MotionGateCalibration,
    TopKLogSumExpAggregator,
    SegmentHead,
)

from ..utils import batched_nms


class FCADClsHead(nn.Module):
    """
    1D Conv heads for classification (与原版相同)
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls=[]
    ):
        super().__init__()
        self.act = act_layer()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        return out_logits


class FCADRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression (与原版相同)
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        return out_offsets


@register_meta_arch("FCADFormer")
class FCADFormer(nn.Module):
    """
    FCAD-Former: 基于论文的视频伪造检测模型

    与 UMMAFormer 的区别:
    1. 移除了 TFAA (Temporal Feature Aided Attention)
    2. 移除了 Reconstruction Attention (lh_branch/hh_branch)
    3. 移除了 DeepInterpolator
    4. 使用标准的 ConvTransformerBackbone
    5. 新增 FEB (频率证据分支) - 可选
    6. 新增 CMAM (相机运动感知) - 从 attention 动态计算
    7. 支持 SKD (选择性知识蒸馏) - 通过配置启用
    """

    def __init__(
        self,
        backbone_type,
        fpn_type,
        backbone_arch,
        scale_factor,
        input_dim,
        audio_input_dim,
        max_seq_len,
        max_buffer_len_factor,
        n_head,
        n_mha_win_size,
        embd_kernel_size,
        embd_dim,
        embd_with_ln,
        fpn_dim,
        fpn_with_ln,
        fpn_start_level,
        head_dim,
        regression_range,
        head_num_layers,
        head_kernel_size,
        head_with_ln,
        use_abs_pe,
        use_rel_pe,
        num_classes,
        train_cfg,
        test_cfg,
        mlp_ratio=4.0,
    ):
        super().__init__()


        total_input_dim = input_dim + audio_input_dim
        self.input_dim = input_dim
        self.total_input_dim = total_input_dim

        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len


        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size


        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor


        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']


        fcad_cfg = train_cfg.get('fcad', {})
        self.enable_feb = fcad_cfg.get('enable_feb', False)
        self.enable_cmam = fcad_cfg.get('enable_cmam', True)
        self.lambda_seg = fcad_cfg.get('lambda_seg', 1.0)
        self.lambda_vid = fcad_cfg.get('lambda_vid', 1.0)
        self.topk_ratio = fcad_cfg.get('topk_ratio', 0.1)
        self.topk_min = fcad_cfg.get('topk_min', 5)
        self.tau = fcad_cfg.get('tau', 1.0)


        feb_cfg = fcad_cfg.get('feb', {})
        self.feb_temporal_dim = feb_cfg.get('temporal_dim', 16)
        self.feb_evidence_dim = feb_cfg.get('evidence_dim', 128)
        self.feb_topk_patches = feb_cfg.get('topk_patches', 64)
        self.feb_num_heads = feb_cfg.get('num_heads', 4)
        self.clip_label_thresh = fcad_cfg.get('clip_label_thresh', 0.5)


        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']




        actual_backbone_type = 'convTransformer'

        self.backbone = make_backbone(
            actual_backbone_type,
            **{
                'n_in': total_input_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': 0.0,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'mlp_ratio': mlp_ratio,
            }
        )

        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)
        self.embd_dim = embd_dim


        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln
            }
        )


        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range
            }
        )


        self.cls_head = FCADClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = FCADRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )



        if self.enable_feb:
            self.feb = FrequencyEvidenceBranch(
                videomae_dim=total_input_dim,
                hidden_dim=head_dim,
                temporal_dim=self.feb_temporal_dim,
                evidence_dim=self.feb_evidence_dim,
                topk_patches=self.feb_topk_patches,
                num_heads=self.feb_num_heads,
                dropout=self.train_dropout,
            )
        else:
            self.feb = None


        self.segment_head = SegmentHead(
            input_dim=embd_dim,
            hidden_dim=head_dim,
            num_layers=2,
            kernel_size=3,
            dropout=self.train_dropout,
        )


        if self.enable_cmam:
            stem_window = self.mha_win_size[0] if self.mha_win_size[0] > 0 else 7
            row_dim = 2 * (stem_window // 2) + 1
            self.avmc = AVMCModule(row_dim=row_dim)
            self.motion_gate = MotionGateCalibration()
        else:
            self.avmc = None
            self.motion_gate = None


        self.aggregator = TopKLogSumExpAggregator(
            feat_dim=embd_dim,
            topk_ratio=self.topk_ratio,
            topk_min=self.topk_min,
            tau=self.tau,
        )

        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9


        self._cached_attention = None

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def _format_attention(self, attention, batch_size):
        if attention is None:
            return None
        if attention.dim() == 4:
            if attention.shape[0] == batch_size:
                return attention
            if attention.shape[2] == attention.shape[3]:
                B, H, T, _ = attention.shape
                return attention
            B, T, H, W = attention.shape
            return attention
        if attention.dim() == 3:
            B = batch_size
            total, T, W = attention.shape
            n_head = total // B
            return attention.view(B, n_head, T, W).permute(0, 2, 1, 3)
        return attention

    def _encode_temporal(self, batched_inputs, batched_masks, video_list):
        if self.enable_feb and self.feb is not None:
            target_len = batched_inputs.shape[-1]
            freq_evidence = self._get_freq_evidence(video_list, target_len=target_len)
            if freq_evidence is not None:
                batched_inputs, _, _ = self.feb(
                    batched_inputs.permute(0, 2, 1),
                    freq_evidence,
                    batched_masks.squeeze(1)
                )
        return_aux = self.enable_cmam and self.avmc is not None
        if return_aux:
            feats, masks, aux = self.backbone(
                batched_inputs, batched_masks, return_aux=True
            )
            attention = aux.get('attention')
        else:
            feats, masks = self.backbone(batched_inputs, batched_masks)
            attention = None
        stem_feat = feats[0]
        stem_mask = masks[0]
        mask_2d = stem_mask.squeeze(1) if stem_mask.dim() == 3 else stem_mask
        seg_logits = self.segment_head(stem_feat, stem_mask)
        if self.enable_cmam and self.avmc is not None and attention is not None:
            att = self._format_attention(attention, batched_inputs.shape[0])
            disturbance = self.avmc(att, stem_mask)
            seg_logits_calib, gate_values = self.motion_gate(seg_logits, disturbance)
            motion_intensity = disturbance
        else:
            motion_intensity = torch.zeros_like(seg_logits)
            seg_logits_calib = seg_logits
            gate_values = torch.ones_like(seg_logits)
        vid_logit, topk_idx, topk_weights, _ = self.aggregator(
            seg_logits_calib, stem_feat, mask_2d
        )
        return {
            'batched_inputs': batched_inputs,
            'feats': feats,
            'masks': masks,
            'stem_feat': stem_feat,
            'stem_mask': stem_mask,
            'mask_2d': mask_2d,
            'seg_logits': seg_logits,
            'seg_logits_calib': seg_logits_calib,
            'motion_intensity': motion_intensity,
            'gate_values': gate_values,
            'vid_logit': vid_logit,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
        }

    def forward(self, video_list):
        """
        Forward pass

        Args:
            video_list: list of dict, each contains:
                - feats: [C, T] 视频特征
                - segments: [N, 2] 伪造片段 (训练时)
                - labels: [N] 标签 (训练时)
                - freq_evidence: [L, P, d_e] 频率 patch evidence (可选, FEB 需要)
        """

        batched_inputs, batched_masks = self.preprocessing(video_list)
        encoded = self._encode_temporal(batched_inputs, batched_masks, video_list)
        feats = encoded['feats']
        masks = encoded['masks']
        stem_feat = encoded['stem_feat']
        stem_mask = encoded['stem_mask']
        mask_2d = encoded['mask_2d']
        seg_logits = encoded['seg_logits']
        seg_logits_calib = encoded['seg_logits_calib']
        motion_intensity = encoded['motion_intensity']
        vid_logit = encoded['vid_logit']

        fpn_feats, fpn_masks = self.neck(feats, masks)
        points = self.point_generator(fpn_feats)
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        fpn_masks_squeezed = [x.squeeze(1) for x in fpn_masks]

        if self.training:
            return self._compute_losses(
                video_list, points, fpn_masks_squeezed,
                out_cls_logits, out_offsets,
                seg_logits, seg_logits_calib, vid_logit,
                motion_intensity, mask_2d
            )
        else:
            return self._inference(
                video_list, points, fpn_masks_squeezed,
                out_cls_logits, out_offsets,
                seg_logits, seg_logits_calib, vid_logit,
                motion_intensity
            )

    def _get_freq_evidence(self, video_list, target_len=None):
        """获取频率 patch evidence，并 padding/interpolate 到目标长度"""
        freq_evidence = [x.get('freq_evidence', None) for x in video_list]

        if all(e is None for e in freq_evidence):
            return None


        P, d_e = None, None
        for e in freq_evidence:
            if e is not None:
                P = e.shape[1]
                d_e = e.shape[2]
                break

        if P is None:
            return None


        if target_len is None:
            target_len = max(e.shape[0] if e is not None else 0 for e in freq_evidence)

        if target_len == 0:
            return None


        padded_evidence = []
        for e in freq_evidence:
            if e is None:

                padded = torch.zeros(target_len, P, d_e, device=self.device)
            elif e.shape[0] == target_len:
                padded = e
            elif e.shape[0] < target_len:


                e_flat = e.permute(1, 2, 0).reshape(1, P * d_e, e.shape[0])
                e_interp = torch.nn.functional.interpolate(
                    e_flat.float(), size=target_len, mode='linear', align_corners=False
                )
                padded = e_interp.reshape(P, d_e, target_len).permute(2, 0, 1)
            else:

                e_flat = e.permute(1, 2, 0).reshape(1, P * d_e, e.shape[0])
                e_interp = torch.nn.functional.interpolate(
                    e_flat.float(), size=target_len, mode='linear', align_corners=False
                )
                padded = e_interp.reshape(P, d_e, target_len).permute(2, 0, 1)

            padded_evidence.append(padded.to(self.device))

        return torch.stack(padded_evidence)

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """预处理: 批量化并生成 mask"""
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(feats[0], padding_size, value=padding_val).unsqueeze(0)

        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    def _build_clip_labels(self, video_list, clip_len, device):
        labels = []
        for video in video_list:
            g = torch.zeros(clip_len, device=device)
            segments = video.get('segments')
            if segments is None or len(segments) == 0:
                labels.append(g)
                continue
            feat_stride = float(video['feat_stride'])
            num_frames = float(video['feat_num_frames'])
            segs = segments.to(device)
            for t in range(clip_len):
                clip_start = t * feat_stride
                clip_end = clip_start + num_frames
                clip_dur = max(clip_end - clip_start, 1e-6)
                max_overlap = 0.0
                for seg in segs:
                    ov_start = max(seg[0].item(), clip_start)
                    ov_end = min(seg[1].item(), clip_end)
                    if ov_end > ov_start:
                        max_overlap = max(max_overlap, (ov_end - ov_start) / clip_dur)
                if max_overlap > self.clip_label_thresh:
                    g[t] = 1.0
            labels.append(g)
        return torch.stack(labels, dim=0)

    def _compute_losses(
        self, video_list, points, fpn_masks,
        out_cls_logits, out_offsets,
        seg_logits, seg_logits_calib, vid_logit,
        motion_intensity, mask_2d
    ):
        gt_video_labels = []
        for video in video_list:
            if 'video_label' in video:
                vid_label = video['video_label']
                if isinstance(vid_label, torch.Tensor):
                    gt_video_labels.append(vid_label.to(self.device).view(1))
                else:
                    gt_video_labels.append(torch.tensor([vid_label], device=self.device, dtype=torch.float32))
            else:
                has_segments = video['segments'] is not None and len(video['segments']) > 0
                gt_video_labels.append(torch.tensor([1.0 if has_segments else 0.0], device=self.device))
        gt_video_labels = torch.cat(gt_video_labels)

        B, L = seg_logits_calib.shape
        clip_labels = self._build_clip_labels(video_list, L, self.device)

        vid_smooth = 0.05
        clip_smooth = self.train_label_smoothing
        gt_vid = gt_video_labels.view(-1)
        gt_vid_smooth = gt_vid * (1.0 - vid_smooth) + (1.0 - gt_vid) * vid_smooth
        vid_logit_clamped = torch.clamp(vid_logit.view(-1), -6.0, 6.0)
        vid_loss = F.binary_cross_entropy_with_logits(
            vid_logit_clamped, gt_vid_smooth, reduction='mean'
        )

        gt_clip_smooth = clip_labels * (1.0 - clip_smooth) + (1.0 - clip_labels) * (clip_smooth / 2.0)
        valid_logits = seg_logits_calib.masked_fill(~mask_2d, 0.0)
        valid_targets = gt_clip_smooth.masked_fill(~mask_2d, 0.0)
        denom = mask_2d.float().sum().clamp(min=1.0)
        clip_loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_targets, reduction='sum'
        ) / denom

        final_loss = self.lambda_seg * clip_loss + self.lambda_vid * vid_loss

        return {
            'clip_loss': clip_loss,
            'vid_loss': vid_loss,
            'final_loss': final_loss,
        }

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        """为 localization head 生成标签"""
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        """为单个视频生成标签"""
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            t_mins = center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        max_regress_distance = reg_targets.max(-1)[0]
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        cls_targets.clamp_(min=0.0, max=1.0)

        reg_targets = reg_targets[range(num_pts), min_len_inds]
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    @torch.no_grad()
    def _inference(
        self, video_list, points, fpn_masks,
        out_cls_logits, out_offsets,
        seg_logits, seg_logits_calib, vid_logit,
        motion_intensity
    ):
        """推理"""
        results = []

        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            prob_direct = torch.sigmoid(vid_logit[idx]).item()
            vid_prob = prob_direct


            loc_results = self._inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )

            results_per_vid = {
                'video_id': vidx,
                'fps': fps,
                'duration': vlen,
                'feat_stride': stride,
                'feat_num_frames': nframes,
                'segments': loc_results['segments'],
                'scores': loc_results['scores'],
                'labels': loc_results['labels'],
                'vid_logit': vid_logit[idx].item(),
                'vid_prob': vid_prob,
                'seg_logits': seg_logits[idx].cpu(),
                'seg_logits_calib': seg_logits_calib[idx].cpu(),
                'motion_intensity': motion_intensity[idx].cpu(),
            }

            results.append(results_per_vid)

        results = self._postprocessing(results)

        return results

    @torch.no_grad()
    def _inference_single_video(self, points, fpn_masks, out_cls_logits, out_offsets):
        """单视频 localization 推理"""
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        for cls_i, offsets_i, pts_i, mask_i in zip(
            out_cls_logits, out_offsets, points, fpn_masks
        ):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            pt_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]

        return {
            'segments': segs_all,
            'scores': scores_all,
            'labels': cls_idxs_all
        }

    @torch.no_grad()
    def _postprocessing(self, results):
        """后处理"""
        processed_results = []

        for results_per_vid in results:
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            vid_prob = results_per_vid.get('vid_prob', 1.0)

            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()


            scores = scores * vid_prob


            if self.test_nms_method != 'none' and segs.shape[0] > 0:
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )


            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen

            processed_results.append({
                'video_id': vidx,
                'segments': segs,
                'scores': scores,
                'labels': labels,
                'vid_prob': vid_prob,
                'seg_logits': results_per_vid.get('seg_logits'),
                'seg_logits_calib': results_per_vid.get('seg_logits_calib'),
                'motion_intensity': results_per_vid.get('motion_intensity'),
            })

        return processed_results
