import torch
from torch import nn
import torch.nn.functional as F
import math

from encoder import PositionwiseFeedForward
from multiattention import MultiHeadAttention
from layernorm import LayerNorm
from transformer import TransformerEmbedding

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention1 = MultiHeadAttention(d_model,n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.droupout1 = nn.Dropout(drop_prob)
        self.cross_attention = MultiHeadAttention(d_model,n_head)
        self.layernorm2 = LayerNorm(d_model)
        self.droupout2 = nn.Dropout(drop_prob)
        self.d_ff1 = PositionwiseFeedForward(d_model,d_ff,drop_prob)
        self.layernorm3 = LayerNorm(d_model)
        self.droupout3 = nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        _x = dec
        x = self.attention1(dec,dec,dec,t_mask)
        x = self.droupout1(x)
        x = self.layernorm1(x + _x)
        _x = x
        x = self.cross_attention(x,enc,enc,s_mask)
        x = self.droupout2(x)
        x = self.layernorm2(x+ _x)
        x = self.d_ff1(x)
        x = self.droupout3(x)
        x = self.layernorm3(x+ _x)
        return x

class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,d_ff,n_head,n_layers,drop_prob,device='cpu'):
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, d_ff, n_head, drop_prob)
                for _ in range(n_layers)
            ]
        )
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self,dec,enc,t_mask,s_mask):
        x =self.embedding(dec)
        for layer in self.layers:
            x = layer(x,enc,t_mask,s_mask)
        x = self.fc(x)
        return x
