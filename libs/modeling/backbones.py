import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, MutilModelTransformerBlock,ConvBlock, LayerNorm, AligningEncoder)


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,
        n_embd,
        n_head,
        n_embd_ks,
        max_len,
        arch = (2, 2, 5),
        mha_win_size = [-1]*6,
        scale_factor = 2,
        with_ln = False,
        attn_pdrop = 0.0,
        proj_pdrop = 0.0,
        path_pdrop = 0.0,
        use_abs_pe = False,
        use_rel_pe = False,
        use_time_weight = False,
        mlp_ratio = 4.0,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_time_weight = use_time_weight


        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None


        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())


        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)


        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight,
                )
            )


        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight,
                )
            )


        self.apply(self.__init_weights__)

    def __init_weights__(self, module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):


        B, C, T = x.size()


        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )


        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))


        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd

            x = x + pe[:, :, :T] * mask.to(x.dtype)


        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd

            x = x + pe[:, :, :T] * mask.to(x.dtype)


        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)


        out_feats = (x, )
        out_masks = (mask, )


        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,
        n_embd,
        n_embd_ks,
        arch = (2, 2, 5),
        scale_factor = 2,
        with_ln=False,
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor


        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None


        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())


        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))


        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))


        self.apply(self.__init_weights__)

    def __init_weights__(self, module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):


        B, C, T = x.size()


        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )


        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))


        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)


        out_feats = (x, )
        out_masks = (mask, )


        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks

@register_backbone("convHRLRFullResSelfAttTransformerRevised")
class ConvHRLRFullResSelfAttTransformerBackboneRevised(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,
        n_embd,
        n_head,
        n_embd_ks,
        max_len,
        arch = (2, 2, 5),
        mha_win_size = [-1]*6,
        scale_factor = 2,
        with_ln = False,
        attn_pdrop = 0.0,
        proj_pdrop = 0.0,
        path_pdrop = 0.0,
        use_abs_pe = False,
        use_rel_pe = False,
        use_time_weight = False,
        mlp_ratio = 4.0,
        amflow_align = None,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_time_weight = use_time_weight


        if amflow_align is None:
            amflow_align = {}
        self.amflow_enabled = amflow_align.get('enabled', False)
        self.amflow_detach = amflow_align.get('detach_amflow', True)


        if self.amflow_enabled:

            stem_window_size = mha_win_size[0] if mha_win_size[0] > 0 else 7
            self.aligning_encoder = AligningEncoder(
                proj_dim=amflow_align.get('align_proj_dim', 64),
                n_layers=amflow_align.get('align_layers', 2),
                kernel_size=amflow_align.get('kernel_size', 7),
                dropout=amflow_align.get('dropout', 0.1),
                n_head=n_head,
                window_size=stem_window_size,
            )
        else:
            self.aligning_encoder = None


        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None


        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())


        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)


        self.mlp_ratio = mlp_ratio
        self.resselfattention = MutilModelTransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight)


        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight,
                )
            )


        self.branch = nn.ModuleList()
        self.lh_branch = nn.ModuleList()
        self.hh_branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight,
                )
            )
            self.lh_branch.append(
                MutilModelTransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight))
            self.hh_branch.append(
                MutilModelTransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    mlp_ratio=mlp_ratio,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    use_time_weight=self.use_time_weight))

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, norm_x, reco_x, mask, return_aux=False):


        B, C, T = x.size()



        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
            norm_x = torch.cat(
                [proj(s, mask)[0]                    for proj, s in zip(self.proj, norm_x.split(self.n_in, dim=1))
                ], dim=1
            )
            reco_x = torch.cat(
                [proj(s, mask)[0]                    for proj, s in zip(self.proj, reco_x.split(self.n_in, dim=1))
                ], dim=1
            )

        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

            norm_x, _ = self.embd[idx](norm_x, mask)
            norm_x = self.relu(self.embd_norm[idx](norm_x))

            reco_x, _ = self.embd[idx](reco_x, mask)
            reco_x = self.relu(self.embd_norm[idx](reco_x))

        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd

            x = x + pe[:, :, :T] * mask.to(x.dtype)
            norm_x = norm_x + pe[:, :, :T] * mask.to(x.dtype)
            reco_x = reco_x + pe[:, :, :T] * mask.to(x.dtype)


        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd

            x = x + pe[:, :, :T] * mask.to(x.dtype)
            norm_x = norm_x + pe[:, :, :T] * mask.to(x.dtype)
            reco_x = reco_x + pe[:, :, :T] * mask.to(x.dtype)

        x = self.resselfattention(x,mask, x_k=reco_x, mask_k=mask, x_v=x, mask_v=mask)[0]



        attention_matrix = None
        motion_score = None
        aux_outputs = {}

        for idx in range(len(self.stem)):
            is_last_stem = (idx == len(self.stem) - 1)
            need_attn = self.amflow_enabled and is_last_stem and return_aux

            if need_attn:

                x, mask, attention_matrix = self.stem[idx](x, mask, return_attn=True)

                if self.amflow_detach:
                    attention_matrix = attention_matrix.detach()

                motion_score = self.aligning_encoder(attention_matrix, mask)
                aux_outputs['attention_matrix'] = attention_matrix
                aux_outputs['amflow'] = motion_score
                aux_outputs['motion_score'] = motion_score
            else:
                x, mask = self.stem[idx](x, mask)


        lh_feat=x
        lh_mask=mask
        out_feats = [lh_feat]
        out_masks = [lh_mask]

        x,mask = lh_feat,lh_mask
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            lh_feat, lh_mask = self.lh_branch[idx](lh_feat, lh_mask, x_k=F.interpolate(x, scale_factor=self.scale_factor**(idx+1), mode='nearest'), mask_k=lh_mask, x_v=F.interpolate(x, scale_factor=self.scale_factor**(idx+1), mode='nearest'), mask_v=lh_mask)
            out_feats.append(x)
            out_masks.append(mask)
            x, mask = self.hh_branch[idx](x, mask, x_k=F.interpolate(lh_feat, scale_factor=1/(self.scale_factor**(idx+1)), mode='nearest'), mask_k=mask, x_v=F.interpolate(lh_feat, scale_factor=1/(self.scale_factor**(idx+1)), mode='nearest'), mask_v=mask)
        out_feats[0] = lh_feat
        out_masks[0] = lh_mask

        if return_aux and self.amflow_enabled:
            return out_feats, out_masks, aux_outputs
        return out_feats, out_masks
