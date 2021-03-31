from torch import nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, vocab)
        )

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
