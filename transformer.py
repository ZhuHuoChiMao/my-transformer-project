r'''
import torch
from torch import nn
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__(vocab_size, d_model, padding_idx=pad_id)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, pad_id, drop_prob,device,):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model,pad_id)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop = nn.Dropout(p=drop_prob)
        #self.history_t = []
        #self.history_p = []
        #self.history_tp = []
        #self.name = name

    def forward(self, x):

        batch_size, seq_len = x.shape

        tok_emb = self.tok_emb(x)
        #self.history_t.append(tok_emb[0].detach().cpu().numpy())
        pos_emb = self.pos_emb(x)
        #self.history_p.append(pos_emb[0].detach().cpu().numpy())
        x = tok_emb + pos_emb

        #self.history_tp.append(x[0].detach().cpu().numpy())

        # 在 forward 的末尾加上这一段
        import numpy as np
        #import os

        # 创建一个文件夹专门存向量
        #if not os.path.exists("debug_vectors"):
            #os.makedirs("debug_vectors")

        # 保存这三个关键步骤
        # step = len(self.history_t)  # 用长度当序号
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_embedding_t{step}.npy", tok_emb[0].detach().cpu().numpy())
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_embedding_p{step}.npy", pos_emb[0].detach().cpu().numpy())
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_embedding_x{step}.npy", x[0].detach().cpu().numpy())


        return self.drop(x)

'''

import torch
from torch import nn
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__(vocab_size, d_model, padding_idx=pad_id)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 先不绑死 device/dtype
        pos = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term: [ceil(d_model/2)]

        pe[:, 0::2] = torch.sin(pos * div_term)

        # 处理 d_model 为奇数：cos 的列数更少，需要裁剪 div_term
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer("encoding", pe, persistent=True)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.encoding.size(0):
            raise ValueError(f"seq_len={seq_len} > max_len={self.encoding.size(0)}")
        return self.encoding[:seq_len].unsqueeze(0).to(dtype=self.encoding.dtype)



class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, pad_id, drop_prob,device,):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model,pad_id)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop = nn.Dropout(p=drop_prob)


    def forward(self, x):

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb


        return self.drop(x)







