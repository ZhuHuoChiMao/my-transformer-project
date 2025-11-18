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

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Q, _ = q.shape
        Bk, K, _ = k.shape
        assert B == Bk, f"Batch mismatch: q={B}, k={Bk}"
        H, d_k = self.n_head, self.d_k


        q = self.w_q(q).view(B, Q, H, d_k).permute(0, 2, 1, 3)
        k = self.w_k(k).view(B, K, H, d_k).permute(0, 2, 1, 3)
        v = self.w_v(v).view(B, K, H, d_k).permute(0, 2, 1, 3)

        '''
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)


        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        # padding mask
        # mask K padding
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                float('-inf')
            )



        attn = self.softmax(scores.float()).to(q.dtype)
        
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        out = self.w_o(out)
        return out, attn
        '''

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: [B, H, Q, K]

        # ------ attn_mask ------
        if attn_mask is not None:
            # attn_mask should broadcast to [B, H, Q, K]
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]
            scores = scores + attn_mask

        # ------ key_padding_mask ------
        if key_padding_mask is not None:
            # key_padding_mask: [B, K]
            # mask to shape [B,1,1,K]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        # ------ softmax ------
        # reshape to combine (B*H)
        scores = scores.view(B * H, Q, K)
        attn = self.softmax(scores)  # [B*H, Q, K]
        attn = attn.view(B, H, Q, K)  # [B, H, Q, K]

        # ------ attention output ------
        out = torch.matmul(attn, v)  # [B, H, Q, d_k]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        out = self.w_o(out)
        return out, attn






