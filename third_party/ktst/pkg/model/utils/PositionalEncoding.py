import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    (very much alike the autobots implementation)"""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # adjust shape to data shape
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, position: Tensor | None = None) -> Tensor:
        if position is None:
            if x.dim() == 3:
                B, T, d = x.shape
                assert T <= self.max_len
                x = x + self.pe[:, :T]
            elif x.dim() == 4:
                B, T, f, d = x.shape
                assert T <= self.max_len
                x = x + self.pe[:, :T].view(1, T, 1, d)
            else:
                raise NotImplementedError()
        else:
            assert x.dim() == 3

            # custom positional encoding per batch
            position = position.unsqueeze(dim=-1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=position.device)
                * (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(position.shape[0], self.max_len, self.d_model)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            assert pe.shape == x.shape

            _, T, d = x.shape
            assert T <= self.max_len
            x = x + self.pe[:, :T]

        return self.dropout(x)
