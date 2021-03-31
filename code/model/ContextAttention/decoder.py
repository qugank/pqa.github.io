import torch
from torch import nn as nn

from .layer_norm import LayerNorm
# from .sublayer_connection import SublayerConnection
from .positional_encoding import PositionalEncoding
from .tool import clones


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, enable_future_mask, task, step=None):
        for layer in self.layers:
            x = layer(x + task, memory, src_mask, enable_future_mask, tgt_mask, step=step)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward,
                 self_attention_preprocessor, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attention_preprocessor: PositionalEncoding = self_attention_preprocessor
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.norm_1 = LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout)

        self.norm_2 = LayerNorm(size)
        self.dropout_2 = nn.Dropout(dropout)

        self.norm_3 = LayerNorm(size)
        self.dropout_3 = nn.Dropout(dropout)

    def future_mask(self, size, device, dtype):
        return torch.tril(torch.ones((size, size), device=device, dtype=dtype), diagonal=0)

    def forward(self, input, memory, src_mask, enable_future_mask, target_mask=None, step=None):
        """
        x, memory in shape like [batch, d1, d2, .., dn, depth]
        mask in shape like [batch, d1, d2, .., dn]
        step != None 时, 对x的-2维度按照step截取一定长度, 处理完后将截取后的部分原样拼接回
        """
        input = self.self_attention_preprocessor(input)
        raw_shape = input.shape
        # reshape to [batch, num_block, num_item, depth]
        # 目前还不支持block, block 维度暂设为1
        # TODO: block query
        input = input.view(raw_shape[0], 1, -1, raw_shape[-1])
        memory = memory.view(memory.shape[0], 1, -1, memory.shape[-1])
        target_mask = target_mask.view(target_mask.shape[0], 1, 1, -1) if target_mask is not None else None

        # 根据 stpe 分割序列
        tail = None
        if step is not None:
            x = input[:, :, :step + 1, :]
            tail = input[:, :, step + 1:, :]
            target_mask = target_mask[:, :, :, :step + 1] if target_mask is not None else None
        else:
            x = input

        if src_mask is not None:
            # mask form
            # [batch, d1, d2, ..., dn]
            # reshape into
            # [batch, 1, 1, num_item] when masked it will be apply to [batch, num_block, num_item, num_item]
            src_mask = src_mask.view(src_mask.shape[0], 1, 1, -1)

        # mask掉future item, 和padding mask 协同
        if enable_future_mask:
            future_mask = self.future_mask(x.shape[-2], x.device, x.dtype) \
                .view(1, 1, x.shape[-2], x.shape[-2])  # .repeat(x.shape[0], 1, 1, 1)
        else:
            future_mask = torch.ones((1, 1, x.shape[-2], x.shape[-2]), device=x.device, dtype=torch.float)

        if target_mask is not None:
            target_mask = future_mask.to(torch.bool).__and__(target_mask.to(torch.bool)).to(torch.float)
        else:
            target_mask = future_mask

        raw_x = x
        x = self.norm_1(x)
        x = self.self_attn(x, x, x, target_mask)
        x = raw_x + self.dropout_1(x)

        raw_x = x
        x = self.norm_2(x)
        x = self.src_attn(x, memory, memory, src_mask)
        x = raw_x + self.dropout_2(x)

        raw_x = x
        x = self.norm_3(x)
        x = self.feed_forward(x)
        x = raw_x + self.dropout_3(x)

        # 恢复序列
        if step is not None:
            x = torch.cat([x, tail], dim=-2)

        return x.view(raw_shape)
