r'''
import torch
from torch import nn
import math
from encoder import PositionwiseFeedForward
from multiattention import MultiHeadAttention
from layernorm import LayerNorm
from transformer import TransformerEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head,drop_prob):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head,dropout=drop_prob)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout=drop_prob)

        self.ln1 = LayerNorm(d_model,)
        self.do1 = nn.Dropout(drop_prob)

        self.ln2 = LayerNorm(d_model,)
        self.do2 = nn.Dropout(drop_prob)

        #add
        #self.lna = LayerNorm(d_model)
        #self.doa = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.ln3 = LayerNorm(d_model,)
        self.do3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc,
                tgt_attn_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        _x = dec
        x, attn = self.self_attn(dec, dec, dec,
                              attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)

        x = self.do1(x)
        x = self.ln1(x + _x)

        #print(f"  ln1: {x[0, 0, -1, :].tolist()}")




        _x = x

        x, attn_weights = self.cross_attn(x, enc, enc,
                               attn_mask=None,
                               key_padding_mask=memory_key_padding_mask)

        if self.training and not hasattr(self, "debug_printed"):
            with torch.no_grad():
                B, H, Q, K = attn_weights.shape
                w = attn_weights[0, 0, 0]
                print("cross-attn[0,0,0][:10] =", w[:10])
                print("  sum =", w.sum().item(), "std =", w.std().item(), "K =", K)
            self.debug_printed = True

        x = self.do2(x)
        x = self.ln2(x + _x)

        #print(f"  ln2: {x[0, 0, -1, :].tolist()}")

        #add
        #_x = x
        #x, _ = self.self_attn(x, dec, dec,
        #                     attn_mask=tgt_attn_mask,
        #                      key_padding_mask=tgt_key_padding_mask)
        #x = self.doa(x)
        #x = self.lna(x + _x)


        _x = x
        x = self.ffn(x)
        x = self.do3(x)
        x = self.ln3(x + _x)

        return x


def generate_causal_mask(T, device):
    m = torch.full((T, T), float('-inf'), device=device)
    return torch.triu(m, diagonal=1)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, d_ff, n_head, n_layers, pad_id,drop_prob, device='cuda'):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, pad_id, drop_prob,device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_head,drop_prob)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, dec_voc_size)
        #self.history = []

    def forward(self, dec, enc, src_pad_mask, tgt_pad_mask):
        x = self.embedding(dec)
        t = dec.size(1)
        tgt_attn_mask = generate_causal_mask(t, dec.device)
        for layer in self.layers:
            x = layer(x, enc,
                      tgt_attn_mask=tgt_attn_mask,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=src_pad_mask)

            #x_vec = x[0].detach().cpu().numpy()
            #self.history.append(x_vec)
            #import numpy as np
            #import os

            # 创建一个文件夹专门存向量
            #if not os.path.exists("debug_vectors"):
                #os.makedirs("debug_vectors")

            # 使用当前的 history 长度作为 step 序号，防止文件被覆盖
            #step = len(self.history)

            #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{layer}_x_vecd{step}.npy", x_vec)

        return self.fc(x)
'''

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

        self.debug_printed = False

    def forward(
        self, dec, enc,
        tgt_attn_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        residual = dec
        x, attn = self.self_attn(
            dec, dec, dec,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.do1(x)
        x = self.ln1(x + residual)

        if tgt_key_padding_mask is not None:
            x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)

        residual = x
        x, attn_weights = self.cross_attn(
            x, enc, enc,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )

        if self.training and (not self.debug_printed):
            with torch.no_grad():
                B, H, Q, K = attn_weights.shape
                w = attn_weights[0, 0, 0]
                print("cross-attn[0,0,0][:10] =", w[:10])
                print("  sum =", w.sum().item(), "std =", w.std().item(), "K =", K)
            self.debug_printed = True

        x = self.do2(x)
        x = self.ln2(x + residual)

        if tgt_key_padding_mask is not None:
            x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)

        residual = x
        x = self.ffn(x)
        x = self.do3(x)
        x = self.ln3(x + residual)


        if tgt_key_padding_mask is not None:
            x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)

        return x



def generate_causal_mask(T, device):
    m = torch.full((T, T), float('-inf'), device=device)
    return torch.triu(m, diagonal=1)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, d_ff, n_head, n_layers, pad_id,drop_prob, device='cuda'):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, pad_id, drop_prob)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_head,drop_prob)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, dec_voc_size)


    def forward(self, dec, enc, src_pad_mask, tgt_pad_mask):
        x = self.embedding(dec)
        t = dec.size(1)
        tgt_attn_mask = generate_causal_mask(t, dec.device)
        for layer in self.layers:
            x = layer(x, enc,
                      tgt_attn_mask=tgt_attn_mask,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=src_pad_mask)


        return self.fc(x)