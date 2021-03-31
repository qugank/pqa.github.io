import torch
from torch import nn as nn

from .generator import Generator
from .embeddings import Embeddings
from .attention import MultiHeadedAttention
from .encoder import EncoderLayer, Encoder
from .feedforward import PositionwiseFeedForward
from .positional_encoding import PositionalEncoding
from .decoder import Decoder, DecoderLayer
from .layer_norm import LayerNorm
from model.ContextEncoder import ContextEncoder


class ContextAttention(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, src_vocab, tgt_vocab, task_num=7, encoder_layer_num=6, decoder_layer_num=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1,
                 padding_idx=0, max_shape=(30, 30), attention_dropout=None, enable_decoder=True):
        super().__init__()

        self.context_encoder = ContextEncoder(src_vocab, task_num, encoder_layer_num=6,
                                              d_model=256, d_ff=1024, h=8, dropout=0.1,
                                              padding_idx=0, max_shape=(2, 30, 30), attention_dropout=attention_dropout)

        self.task_fuse = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            LayerNorm(d_model)
        )

        self.task_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.task_readout = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            Generator(d_ff, task_num)
        )

        self.enable_decoder = enable_decoder

        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab, padding_idx)
        )

        self.encoder = Encoder(
            EncoderLayer(d_model,
                         MultiHeadedAttention(h, d_model, dropout=attention_dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout),
                         PositionalEncoding(d_model, dropout, max_shape),
                         dropout),
            encoder_layer_num)
        if self.enable_decoder:
            self.tgt_embed = nn.Sequential(
                Embeddings(d_model, tgt_vocab, padding_idx)
            )
            self.decoder = Decoder(
                DecoderLayer(d_model,
                             MultiHeadedAttention(h, d_model, dropout=attention_dropout),
                             MultiHeadedAttention(h, d_model, dropout=attention_dropout),
                             PositionwiseFeedForward(d_model, d_ff, dropout),
                             PositionalEncoding(d_model, dropout, max_shape),
                             dropout),
                decoder_layer_num)

        self.generator = Generator(d_model, tgt_vocab)
        # self.task_generator = Generator(d_model, task_num)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask, enable_future_mask=True):
        "Take in and process masked src and target sequences."
        # b, h, w
        # b, h, w, d
        task_memory = self.context_encoder(ctx_x, ctx_x_mask, ctx_y, ctx_y_mask)
        task_memory = torch.cat([task_memory[:, 0], task_memory[:, 1]], dim=-1)

        task = task_memory.permute([0, 3, 1, 2]).contiguous()
        task = self.task_pool(task)
        task = task.permute([0, 2, 3, 1]).contiguous()
        task = self.task_readout(task).squeeze(1).squeeze(1) # b, task_num

        task_memory = self.task_fuse(task_memory)
        emb = self.src_embed(src)
        memory = self.encoder(emb, src_mask, task_memory)
        if self.enable_decoder:
            emb = self.tgt_embed(src)
            # x, memory, src_mask, tgt_mask, enable_future_mask, task
            output = self.decoder(emb, memory, src_mask, tgt_mask=src_mask, enable_future_mask=enable_future_mask,
                                  task=task_memory,
                                  step=None)
            return self.generator(output), task
        else:
            return self.generator(memory), task
