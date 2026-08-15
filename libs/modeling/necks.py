import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm



@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,
        out_channel,
        scale_factor=2.0,
        start_level=0,
        end_level=-1,
        with_ln=True,
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):

            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln)
            )

            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )

            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):

        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)


        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)


        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.scale_factor, mode='nearest'
            )



        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(used_backbone_levels):
            x, new_mask = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )
            new_fpn_masks += (new_mask, )

        return fpn_feats, new_fpn_masks


@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        scale_factor=2.0,
        start_level=0,
        end_level=-1,
        with_ln=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):

            assert self.in_channels[i] == self.out_channel

            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):

        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)


        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks
