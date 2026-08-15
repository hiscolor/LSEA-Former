import copy
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()

        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)

        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)

        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):


        B, C, T = x.size()

        assert T % self.stride == 0


        out_conv = self.conv(x)

        if self.stride > 1:

            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:

            out_mask = mask.to(x.dtype)


        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask






class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels


        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)


        if self.affine:
            out *= self.weight
            out += self.bias

        return out



def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])


    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)



class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,
        n_head,
        attn_pdrop=0.0,
        proj_pdrop=0.0
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)



        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)


        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):


        B, C, T = x.size()



        k = self.key(x)
        q = self.query(x)
        v = self.value(x)



        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)


        att = (q * self.scale) @ k.transpose(-2, -1)

        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ (v * mask[:, :, :, None].to(v.dtype))

        out = out.transpose(2, 3).contiguous().view(B, C, -1)


        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_qx_stride=1,
        n_kv_stride=1,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)


        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride


        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)


        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)



        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)


        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask, return_attn=False):


        B, C, T = x.size()


        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)

        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)


        q = self.query(q)
        k = self.key(k)
        v = self.value(v)



        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)


        att = (q * self.scale) @ k.transpose(-2, -1)

        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))

        att = F.softmax(att, dim=-1)



        att_for_cmam = None
        if return_attn:

            att_for_cmam = att.detach()

        att = self.attn_drop(att)

        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))

        out = out.transpose(2, 3).contiguous().view(B, C, -1)


        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)

        if return_attn:
            return out, qx_mask, att_for_cmam
        return out, qx_mask



class MaskedMMHCA(nn.Module):
    """
    Multi Head Conv Mutilmodel Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Add mutilmodel attention

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_qx_stride=1,
        n_kv_stride=1,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)


        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride


        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)


        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)



        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)


        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x_q, mask_q ,x_k=None, mask_k=None, x_v=None, mask_v=None):


        B, C, T = x_q.size()


        q, qx_mask = self.query_conv(x_q, mask_q)
        q = self.query_norm(q)

        k, kv_mask = self.key_conv(x_k, mask_k) if x_k is not None else self.key_conv(x_q, mask_q)
        k = self.key_norm(k)
        v, _ = self.value_conv(x_v, mask_v) if x_v is not None else self.value_conv(x_q, mask_q)
        v = self.value_norm(v)


        q = self.query(q)
        k = self.key(k)
        v = self.value(v)



        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)


        att = (q * self.scale) @ k.transpose(-2, -1)

        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))

        att = F.softmax(att, dim=-1)


        att = self.attn_drop(att)

        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))

        out = out.transpose(2, 3).contiguous().view(B, C, -1)


        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask



class LocalMaskedMMHCA(nn.Module):
    """
    Local Multi Head Conv  Mutilmodel Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,
        n_head,
        window_size,
        n_qx_stride=1,
        n_kv_stride=1,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        use_rel_pe=False,
        use_time_weight=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2

        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe
        self.use_time_weight=use_time_weight


        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride


        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)


        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)



        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)


        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)


        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

        if self.use_time_weight:
            self.time_weighting = nn.Parameter(torch.ones(1,self.n_head, self.window_size))

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""


        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )


        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2


        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""

        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())

        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())

        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()

        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )

        x = x.view(total_num_heads, num_chunks, -1)

        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """

        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1


        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)





        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))



        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )





        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )



        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]

        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]


        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1


        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )


        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)


        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x_q, mask_q ,x_k=None, mask_k=None, x_v=None, mask_v=None):


        B, C, T = x_q.size()



        q, qx_mask = self.query_conv(x_q, mask_q)
        q = self.query_norm(q)

        k, kv_mask = self.key_conv(x_k, mask_k) if x_k is not None else  self.key_conv(x_q, mask_q)
        k = self.key_norm(k)
        v, _ = self.value_conv(x_v, mask_v) if x_v is not None else  self.key_conv(x_q, mask_v)
        v = self.value_norm(v)



        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()


        q *= self.scale

        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)


        if self.use_rel_pe:
            att += self.rel_pe

        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))

        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)

        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask


        att = nn.functional.softmax(att, dim=-1)

        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)

        if self.use_time_weight:
            att = att * self.time_weighting[:,:T,:T]
        att = self.attn_drop(att)



        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)

        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask

class MutilModelTransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        use_time_weight=False,
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        self.lnq = LayerNorm(n_embd)
        self.lnk = LayerNorm(n_embd)
        self.lnv = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)


        if mha_win_size > 1:
            self.attn = LocalMaskedMMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
                use_time_weight=use_time_weight
            )
        else:
            self.attn = MaskedMMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )


        if n_ds_strides[0] > 1:
            kernel_size, stride, padding =                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()


        if n_hidden is None:
            n_hidden = int(mlp_ratio * n_embd)
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )


        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()


    def forward(self, x_q, mask_q ,x_k=None, mask_k=None, x_v=None, mask_v=None,pos_embd=None):

        out, out_mask = self.attn(self.lnq(x_q), mask_q,self.lnk(x_k), mask_k,self.lnv(x_v), mask_v)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x_q) * out_mask_float + self.drop_path_attn(out)

        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask






class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,
        n_head,
        window_size,
        n_qx_stride=1,
        n_kv_stride=1,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        use_rel_pe=False,
        use_time_weight = False
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2

        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe
        self.use_time_weight = use_time_weight


        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride


        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)


        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)



        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)


        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)


        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

        if self.use_time_weight:
            self.time_weighting = nn.Parameter(torch.ones(1,self.n_head, self.window_size))

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""


        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )


        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2


        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""

        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())

        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())

        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()

        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )

        x = x.view(total_num_heads, num_chunks, -1)

        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """

        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1


        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)





        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))



        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )





        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )



        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]

        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]


        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1


        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )


        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)


        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask, return_attn=False):


        B, C, T = x.size()



        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)

        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)



        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()


        q *= self.scale

        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)


        if self.use_rel_pe:
            att += self.rel_pe

        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))

        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)

        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask


        att = nn.functional.softmax(att, dim=-1)

        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)




        att_for_cmam = None
        if return_attn:
            att_for_cmam = att.detach()

        if self.use_time_weight:
            att = att * self.time_weighting[:,:T,:T]

        att = self.attn_drop(att)



        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)

        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)

        if return_attn:
            return out, qx_mask, att_for_cmam
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        use_time_weight = False
    ):
        super().__init__()
        assert len(n_ds_strides) == 2

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)


        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
                use_time_weight=use_time_weight,
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )


        if n_ds_strides[0] > 1:
            kernel_size, stride, padding =                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()


        if n_hidden is None:
            n_hidden = int(mlp_ratio * n_embd)
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )


        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None, return_attn=False):

        if return_attn:
            attn_out, out_mask, att_matrix = self.attn(self.ln1(x), mask, return_attn=True)
        else:
            attn_out, out_mask = self.attn(self.ln1(x), mask)
            att_matrix = None

        out_mask_float = out_mask.to(attn_out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(attn_out)

        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        if pos_embd is not None:
            out += pos_embd * out_mask_float

        if return_attn:
            return out, out_mask, att_matrix
        return out, out_mask


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_embd,
        kernel_size=3,
        n_ds_stride=1,
        expansion_factor=2,
        n_out=None,
        act_layer=nn.ReLU,
    ):
        super().__init__()

        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd


        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)


        if n_ds_stride > 1:

            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)


        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)


        out += identity
        out = self.act(out)

        return out, out_mask



class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale




def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)



class MaskedConvTranspose1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros'
    ):
        super().__init__()

        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)

        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                              stride, padding, output_padding=output_padding,dilation=dilation, groups=groups, padding_mode=padding_mode)


    def forward(self, x, mask):


        B, C, T = x.size()

        assert T % self.stride == 0


        out_conv = self.conv(x)

        if self.stride > 1:

            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:

            out_mask = mask.to(x.dtype)


        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask



class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1, dilation=1, norm=True):
        super(DownBlock, self).__init__()
        self.conv_block = MaskedConv1D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self.activation = nn.LeakyReLU(0.2)
        self.batch_normalization = nn.InstanceNorm1d(out_channel)
        self.norm = norm

    def forward(self, x,mask):
        x, mask = self.conv_block(x,mask)
        if self.norm:
            x = self.batch_normalization(x)
        x = self.activation(x)
        return x,mask


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,stride=2, dilation=1, output_padding=0,norm=True, last=False):
        super(UpBlock, self).__init__()
        self.conv_transpose = MaskedConvTranspose1D(
            in_channels=in_channel,
            out_channels=out_channel,
            padding = padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            output_padding=output_padding
        )
        self.activation = nn.Tanh() if last else nn.LeakyReLU(0.2)
        self.batch_normalization = nn.InstanceNorm1d(out_channel)
        self.norm = norm

    def forward(self, x,mask):
        x,mask = self.conv_transpose(x,mask)
        if self.norm:
            x = self.batch_normalization(x)
        x = self.activation(x)

        return x,mask


class Contraction(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_dims=256):
        super(Contraction, self).__init__()
        self.down_1 = DownBlock(in_channel=in_channels, out_channel=hidden_dims)
        self.down_2 = DownBlock(in_channel=hidden_dims, out_channel=hidden_dims*2)
        self.down_3 = DownBlock(in_channel=hidden_dims*2, out_channel=hidden_dims*4)
        self.down_4 = DownBlock(in_channel=hidden_dims*4, out_channel=hidden_dims*8)
        self.down_5 = DownBlock(in_channel=hidden_dims*8, out_channel=out_channels)

    def forward(self, x,mask):
        x,mask = self.down_1(x,mask)
        x,mask = self.down_2(x,mask)
        x,mask = self.down_3(x,mask)
        x,mask = self.down_4(x,mask)
        x,mask = self.down_5(x,mask)
        return x,mask


class Expansion(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_dims=2048, tanh=True):
        super(Expansion, self).__init__()
        self.up_1 = UpBlock(in_channel=in_channels, out_channel=hidden_dims,output_padding=1)
        self.up_2 = UpBlock(in_channel=hidden_dims, out_channel=hidden_dims//2,output_padding=1)
        self.up_3 = UpBlock(in_channel=hidden_dims//2, out_channel=hidden_dims//4,output_padding=1)
        self.up_4 = UpBlock(in_channel=hidden_dims//4, out_channel=hidden_dims//8,output_padding=1)

        self.up_5 = UpBlock(in_channel=hidden_dims//8, out_channel=out_channels,output_padding=1, last=tanh)

    def forward(self, x,mask):
        x,mask = self.up_1(x,mask)
        x,mask = self.up_2(x,mask)
        x,mask = self.up_3(x,mask)
        x,mask = self.up_4(x,mask)
        x,mask = self.up_5(x,mask)
        return x, mask


class DeepInterpolator(nn.Module):
    def __init__(self, in_channels,hidden_channels=512,num_classes=1, norm=True):
        super(DeepInterpolator, self).__init__()
        self.norm = norm
        self.contraction = Contraction(in_channels,hidden_channels)
        self.expansion = Expansion(hidden_channels, in_channels,tanh=False)
        self.conv0 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(hidden_channels),
            nn.LeakyReLU(0.2))
        self.conv1 = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)
        self.conv2 = nn.Linear(hidden_channels, num_classes)
        self.bn1 = LayerNorm(hidden_channels)
        self.dp1 = nn.Dropout(p=0.5)
    def classifier(self, x):
        x = self.conv0(x)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        return x
    def normalize_batch(self, x, return_stats=False):
        mu, sigma = x.mean(), x.std()
        x = (x - mu) / sigma
        if return_stats:
            return x, (mu, sigma)
        return x
    def forward(self, inputs,mask):


        if self.norm:
            inputs = self.normalize_batch(inputs)*mask.detach()
        feat,mask = self.contraction(inputs,mask)
        cls_socres = self.classifier(feat)
        reco,mask = self.expansion(feat,mask)
        return inputs.detach(),reco, cls_socres


class AligningEncoder(nn.Module):
    """
    Aligning Encoder: 按论文公式 (11)-(13) 实现

    论文流程：
    1. 公式 (11): Ã_t = f_align(A_t) - 对每个时间步的 attention 分布投影
    2. 公式 (12): m_t = Mean(|Ã_t - Ã_{t-1}|) - 计算相邻投影后的差分
    3. 公式 (13): m̃_t = σ(λ·m_t + b) - 归一化到 [0, 1]

    输入: attention matrix (B, H, T, T) 或 (B, T, H, W) + mask (B, 1, T)
    输出: motion_intensity (B, T)，范围 [0, 1]
    """

    def __init__(
        self,
        proj_dim=64,
        n_layers=2,
        kernel_size=7,
        dropout=0.1,
        n_head=4,
        window_size=7,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.window_size = window_size




        attn_input_dim = n_head * (2 * (window_size // 2) + 1)
        self.attn_input_dim = attn_input_dim


        self.attn_proj = nn.Linear(attn_input_dim, proj_dim)
        nn.init.kaiming_normal_(self.attn_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.attn_proj.bias, 0)


        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(proj_dim, proj_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(proj_dim, proj_dim, 1),
                nn.Dropout(dropout),
            ))


        self.ln = LayerNorm(proj_dim)


        self._lambda_param = nn.Parameter(torch.ones(1) * 2.0)
        self.bias_param = nn.Parameter(torch.zeros(1))

    @property
    def lambda_d(self):
        return F.softplus(self._lambda_param)

    def forward(self, attention, mask):
        """
        按论文公式 (11)-(13) 处理 attention 矩阵

        Args:
            attention: 完整的 attention 矩阵
                - Full attention: (B, H, T, T) 其中 A[b,h,t,:] 是时间步 t 的 attention 分布
                - Local attention: (B, T, H, W) 其中 A[b,t,h,:] 是时间步 t 的局部 attention
            mask: (B, 1, T) - 有效位置 mask (bool)

        Returns:
            motion_intensity: (B, T) - 运动强度，范围 [0, 1]
        """

        if attention.dim() == 4:
            if attention.shape[2] == attention.shape[3]:

                B, H, T, _ = attention.shape

                attn_rows = attention.permute(0, 2, 1, 3).reshape(B, T, H * T)
            else:

                B, T, H, W = attention.shape

                attn_rows = attention.reshape(B, T, H * W)
        else:
            raise ValueError(f"Unexpected attention shape: {attention.shape}")

        attn_input_dim = attn_rows.shape[-1]


        if attn_input_dim != self.attn_input_dim:


            attn_rows = F.adaptive_avg_pool1d(
                attn_rows.permute(0, 2, 1),
                self.attn_input_dim
            ).permute(0, 2, 1)




        attn_proj = self.attn_proj(attn_rows)


        attn_proj = attn_proj.permute(0, 2, 1)

        mask_float = mask.to(attn_proj.dtype)
        attn_proj = attn_proj * mask_float


        for layer in self.layers:
            residual = attn_proj
            attn_proj = self.ln(attn_proj)
            attn_proj = layer(attn_proj) * mask_float + residual






        attn_diff = torch.abs(attn_proj[:, :, 1:] - attn_proj[:, :, :-1])


        motion_diff = attn_diff.mean(dim=1)


        motion = F.pad(motion_diff, (1, 0), value=0)


        motion_intensity = torch.sigmoid(self.lambda_d * motion + self.bias_param)


        motion_intensity = motion_intensity * mask.squeeze(1).to(motion_intensity.dtype)

        return motion_intensity
