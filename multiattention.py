import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Q, _ = q.shape
        Bk, K, _ = k.shape
        assert B == Bk, f"Batch mismatch: q={B}, k={Bk}"
        H, d_k = self.n_head, self.d_k

        # 线性投影 + 拆多头
        q = self.w_q(q).view(B, Q, H, d_k).permute(0, 2, 1, 3)
        k = self.w_k(k).view(B, K, H, d_k).permute(0, 2, 1, 3)
        v = self.w_v(v).view(B, K, H, d_k).permute(0, 2, 1, 3)

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 结构 mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        # padding mask
        if key_padding_mask is not None:
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,K]
            scores = scores.masked_fill(pad, float('-inf'))

        # softmax + dropout
        attn = self.softmax(scores.float()).to(q.dtype)
        attn = self.dropout(attn)

        # 加权求和 + 合并多头
        out = torch.matmul(attn, v)  # [B,H,Q,d_k]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        out = self.w_o(out)
        return out, attn

