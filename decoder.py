import torch
from torch import nn
import math
from encoder import PositionwiseFeedForward
from multiattention import MultiHeadAttention
from layernorm import LayerNorm
from transformer import TransformerEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, drop_prob):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=drop_prob)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout=drop_prob)

        self.ln1 = LayerNorm(d_model)
        self.do1 = nn.Dropout(drop_prob)

        self.ln2 = LayerNorm(d_model)
        self.do2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.ln3 = LayerNorm(d_model)
        self.do3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc,
                tgt_attn_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        _x = dec
        x, _ = self.self_attn(dec, dec, dec,
                              attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)
        x = self.do1(x)
        x = self.ln1(x + _x)


        _x = x
        if torch.rand(1).item() < 0.01:  # 只打印少数 batch 避免刷屏
            print(
                f"[DEBUG] Cross-Attn Input shapes: q={x.shape}, k={enc.shape}, mask={memory_key_padding_mask.shape if memory_key_padding_mask is not None else None}")

        x, _ = self.cross_attn(x, enc, enc,
                               attn_mask=None,
                               key_padding_mask=memory_key_padding_mask)
        x = self.do2(x)
        x = self.ln2(x + _x)


        _x = x
        x = self.ffn(x)
        x = self.do3(x)
        x = self.ln3(x + _x)
        return x


def generate_causal_mask(T, device):
    m = torch.full((T, T), float('-inf'), device=device)
    return torch.triu(m, diagonal=1)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, d_ff, n_head, n_layers, drop_prob, device='cpu'):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_head, drop_prob)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, src_pad_mask, tgt_pad_mask):
        x = self.embedding(dec)
        T = dec.size(1)
        tgt_attn_mask = generate_causal_mask(T, dec.device)
        for layer in self.layers:
            x = layer(x, enc,
                      tgt_attn_mask=tgt_attn_mask,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=src_pad_mask)
        return self.fc(x)

