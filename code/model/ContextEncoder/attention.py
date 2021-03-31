import math

import torch
from torch import nn


# from . import clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        # 参考t2t原文使用加法方式设置mask
        scores = scores + (1 - mask) * -1e9
    p_attn = torch.softmax(scores, dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.qw = nn.Linear(d_model, d_model)
        self.kw = nn.Linear(d_model, d_model)
        self.vw = nn.Linear(d_model, d_model)
        self.ow = nn.Linear(d_model, d_model)
        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, query, key, value, mask=None):
        """
        q, k, v in shape like [batch, block_num, item_num, depth]
        mask in shape like [batch, block_num, item_num, item_num]
        k, v should have same shape
        """

        batch, block_num, item_num, depth = query.shape
        # mul weight
        query, key, value = self.qw(query), self.kw(key), self.vw(value)
        # split head
        query, key, value = [d.view(*d.shape[:3], self.h, self.d_k).transpose(-2, -3) for d in
                             (query, key, value)]
        # mask should apply multi head dimension also
        # mask [batch, block_num, item_num, item_num] -> [batch, block_num, head, item_num, item_num]
        mask = mask.unsqueeze(-3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-2, -3).contiguous().view(batch, block_num, item_num, -1)
        return self.ow(x)
