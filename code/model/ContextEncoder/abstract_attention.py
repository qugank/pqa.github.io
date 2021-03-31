import torch
from torch import nn as nn

from .generator import Generator
from .embeddings import Embeddings
from .attention import MultiHeadedAttention
from .encoder import EncoderLayer, Encoder
from .feedforward import PositionwiseFeedForward
from .positional_encoding import PositionalEncoding


class ContextEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, src_vocab, task_num, encoder_layer_num=6,
                 d_model=256, d_ff=1024, h=8, dropout=0.1,
                 padding_idx=0, max_shape=(2, 30, 30), attention_dropout=None):
        super().__init__()

        self.encoder = Encoder(
            EncoderLayer(d_model,
                         MultiHeadedAttention(h, d_model, dropout=attention_dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout),
                         PositionalEncoding(d_model, dropout, max_shape),
                         dropout),
            encoder_layer_num)

        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab, padding_idx)
        )

        # self.generator = Generator(d_model, task_num)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask):

        "Take in and process masked src and target sequences."
        # b, h, w
        # b, h, w, d
        x = torch.cat([m.unsqueeze(1) for m in (ctx_x, ctx_y)], dim=1)
        mask = torch.cat([m.unsqueeze(1) for m in (ctx_x_mask, ctx_y_mask)], dim=1)
        x = self.src_embed(x)
        memory = self.encoder(x, mask)
        return memory
