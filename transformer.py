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
    def __init__(self, vocab_size, d_model, max_len, pad_id, drop_prob,device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model,pad_id)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        return self.drop(x)








