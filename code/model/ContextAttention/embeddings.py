import math

from torch import nn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_index):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_index)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
