import torch
from torch import nn

from model.EncoderAttention.attention import MultiHeadedAttention
from model.EncoderAttention.encoder import Encoder, EncoderLayer
from model.EncoderAttention.feedforward import PositionwiseFeedForward
from model.EncoderAttention.positional_encoding import PositionalEncoding


class TaskClassification(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, task_num, d_model, d_ff, h=8, dropout=0.1, max_shape=(30, 30)):
        super().__init__()
        self.upper = Encoder(
            EncoderLayer(d_model,
                         MultiHeadedAttention(h, d_model, dropout=dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout),
                         PositionalEncoding(d_model, dropout, max_shape),
                         dropout),
            2)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.out = nn.Sequential(
            nn.Linear(d_ff),
            nn.Linear(task_num)
        )

    def forward(self, memory):
        """
        memory: (b, h, w, d_model)
        """
        x = self.upper(memory)
        torch.relu(x)
        # x: (b, h, w, d_model)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x: (b, d_model, h, w)
        x = self.pool(x)
        # x: (b, d_model, 1, 1)
        x = x.squeeze(2)
        x = x.squeeze(3)
        # x: (b, d_model)
        return self.out(x)
