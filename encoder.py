import torch
from torch import nn
import math
from multiattention import MultiHeadAttention
from transformer import TransformerEmbedding


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        _x = x
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=None,
            key_padding_mask=key_padding_mask
        )
        x = self.layernorm1(_x + self.dropout1(attn_out))

        _x = x
        x = self.layernorm2(_x + self.dropout2(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, d_ff, n_head, n_layers,
                 dropout=0.1, device='cpu', pad_id=0):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pad_id = pad_id

    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is None:
            key_padding_mask = (x == self.pad_id)

        x = self.embedding(x)  # [B,S,D]
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x



