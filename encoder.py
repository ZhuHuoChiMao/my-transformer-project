

import torch
from torch import nn
import math
from multiattention import MultiHeadAttention
from transformer import TransformerEmbedding
from layernorm import LayerNorm


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        #self.history_ffn1 = []
        #self.history_ffn2 = []
        #self.history_ffn3 = []
        #self.history_ffn4 = []
        #self.name = name
        #self.layers_i = layers_i

    def forward(self, x):

        #ffn_vec1 = self.linear1(x)[0].detach().cpu().numpy()
        #self.history_ffn1.append(ffn_vec1)
        #ffn_vec2 = self.activation(self.linear1(x)[0]).detach().cpu().numpy()
        #self.history_ffn2.append(ffn_vec2)
        #ffn_vec3 = self.activation(self.linear1(x)[0]).detach().cpu().numpy()
        #self.history_ffn3.append(ffn_vec3)
        #ffn_vec4 = self.linear2(self.activation(self.linear1(x)[0])).detach().cpu().numpy()
        #self.history_ffn4.append(ffn_vec4)

        #import numpy as np
        #import os

        # 创建一个文件夹专门存向量
        #if not os.path.exists("debug_vectors"):
            #os.makedirs("debug_vectors")

        # 使用当前的 history 长度作为 step 序号，防止文件被覆盖
        #step = len(self.history_ffn1)

        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_ffn_vec1{step}.npy", ffn_vec1)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_ffn_vec2{step}.npy", ffn_vec2)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_ffn_vec3{step}.npy", ffn_vec3)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_ffn_vec4{step}.npy", ffn_vec4)

        return self.linear2(self.dropout(self.activation(self.linear1(x))))



class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head,)
        self.layernorm1 = nn.LayerNorm(d_model,)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff,dropout)
        self.layernorm2 = nn.LayerNorm(d_model,)
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
    def __init__(self, enc_voc_size, max_len, d_model, d_ff, n_head, n_layers,pad_id,dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, pad_id, dropout,device, )
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff,dropout)
            for _ in range(n_layers)
        ])
        self.pad_id = pad_id

        #self.history = []

    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is None:
            key_padding_mask = (x == self.pad_id)

        x = self.embedding(x)  # [B,S,D]
        for i, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask=key_padding_mask)
            #x_vec = x[0].detach().cpu().numpy()
            #self.history.append(x_vec)
            #import numpy as np
            #import os

            # 创建一个文件夹专门存向量
            #if not os.path.exists("debug_vectors"):
                #os.makedirs("debug_vectors")
            # 使用当前的 history 长度作为 step 序号，防止文件被覆盖
            #step = len(self.history)

            #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{i}_x_vece{step}.npy", x_vec)

        return x
