import torch
from torch import nn
import torch.nn.functional as F
import math
from multiattention import MultiHeadAttention
from layernorm import LayerNorm
from transformer import TransformerEmbedding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.d_ff1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x,x,x,mask)
        x = self.dropout1(x)
        x = self.layernorm1(x + _x)
        _x = x
        x = self.d_ff1(x)
        x = self.dropout2(x)
        x = self.layernorm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, d_ff,n_head, n_layers, dropout=0.1,device = 'cpu'):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_head, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


