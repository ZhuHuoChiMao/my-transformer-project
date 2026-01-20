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

        #self.history_q = []
        #self.history_k = []
        #self.history_v = []
        #self.history_scores = []
        #self.history_attn = []
        #self.history_attnout = []
        #self.history_out = []
        #self.history_s1 = []
        #self.history_s2 = []

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Q, _ = q.shape
        Bk, K, _ = k.shape
        assert B == Bk, f"Batch mismatch: q={B}, k={Bk}"
        H, d_k = self.n_head, self.d_k


        q = self.w_q(q).view(B, Q, H, d_k).permute(0, 2, 1, 3)
        k = self.w_k(k).view(B, K, H, d_k).permute(0, 2, 1, 3)
        v = self.w_v(v).view(B, K, H, d_k).permute(0, 2, 1, 3)
        #q_vec = q.detach().cpu().numpy()
        #self.history_q.append(q_vec)
        #k_vec = k.detach().cpu().numpy()
        #self.history_k.append(k_vec)
        #v_vec = v.detach().cpu().numpy()
        #self.history_v.append(v_vec)


        scores = torch.matmul(q, k.transpose(-2, -1))
        #scores_s1 = scores.detach().cpu().numpy()
        #self.history_s1.append(scores_s1)

        scores = scores/ math.sqrt(d_k)
        #scores_s2 = scores.detach().cpu().numpy()
        #self.history_s2.append(scores_s2)

        # --- [LATEST] 监控：最新词的原始注意力分数 ---
        # 维度 [B, H, Q_len, K_len]，我们取 Q_len 的最后一行
        #latest_q_scores = scores[:, :, -1, :]
        #print(f"\n[MHA - Latest Word Querying]")
        #print(f"  Latest word scores (before mask/softmax) shape: {latest_q_scores.shape}")

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
        #scores_vec = scores.detach().cpu().numpy()
        #self.history_scores.append(scores_vec)

        attn = self.softmax(scores)  # [B*H, Q, K]
        attn = attn.view(B, H, Q, K)  # [B, H, Q, K]

        # --- [LATEST] 监控：最新词真正看重哪些词 ---
        # 这一行向量的和应该为 1
        #latest_attn_weights = attn[0, 0, -1, :]  # 查看第1个Batch第1个头的最后一行
        #print(f"  Latest word attn weights (Head 0): {latest_attn_weights.tolist()}")
        #attn_vec = attn.detach().cpu().numpy()
        #self.history_attn.append(attn_vec)

        # ------ attention output ------
        out = torch.matmul(attn, v)  # [B, H, Q, d_k]
        #attnout_vec = out.detach().cpu().numpy()
        #self.history_attnout.append(attnout_vec)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        out = self.w_o(out)

        # --- [LATEST] 监控：经过 MHA 后，最新词的向量变成了什么 ---
        #latest_out_vec = out[:, -1, :]
        #print(f"  Final MHA Output for latest word: {latest_out_vec.shape}")
        #out_vec = out.detach().cpu().numpy()
        #self.history_out.append(out_vec)

        return out, attn






