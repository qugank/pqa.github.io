from torch import nn as nn

from .layer_norm import LayerNorm
from .tool import clones


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        # self.task_embedding = task_embedding

    def forward(self, x, mask, task):
        "Pass the input (and mask) through each layer in turn."
        # task_emb = self.task_embedding(task)
        for layer in self.layers:
            x = layer(x + task, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, self_attention_preprocessor, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention_preprocessor = self_attention_preprocessor
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm1 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x should be in same shape [batch, d1, d2, ..., dn, depth]
        mask should be in [batch, d1, d2, ..., dn]
        mask should be reshape into [batch, num_block, num_item, num_item]
        x will be flatten before attention
        x 进入后先展平, 运算完成后恢复
        """
        x = self.self_attention_preprocessor(x)
        raw_shape = x.shape
        # num_item = h * w
        # reshape to [batch, num_block, num_item, depth]
        # 目前还不支持block, block 维度暂设为1
        # TODO: block query
        x = x.view(raw_shape[0], 1, -1, raw_shape[-1])
        if mask is not None:
            # mask form
            # [batch, d1, d2, ..., dn]
            # reshape into
            # [batch, 1, 1, num_item] when masked it will be apply to [batch, num_block, num_item, num_item]
            mask = mask.view(mask.shape[0], 1, 1, -1)

        raw_x = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = raw_x + self.dropout1(x)

        raw_x = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = raw_x + self.dropout2(x)

        return x.view(raw_shape)
