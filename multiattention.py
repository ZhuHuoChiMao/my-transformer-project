
import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head,dropout=0.1,):
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
        #self.name = name
        #self.layers_i = layers_i

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Q, _ = q.shape
        Bk, K, _ = k.shape
        assert B == Bk, f"Batch mismatch: q={B}, k={Bk}"
        H, d_k = self.n_head, self.d_k


        q = self.w_q(q).view(B, Q, H, d_k).permute(0, 2, 1, 3)
        k = self.w_k(k).view(B, K, H, d_k).permute(0, 2, 1, 3)
        v = self.w_v(v).view(B, K, H, d_k).permute(0, 2, 1, 3)

        # 2. 合并回 512 维 [B, Seq, d_model]
        #q_full = q.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        #k_full = k.permute(0, 2, 1, 3).contiguous().view(B, K, self.d_model)
        #v_full = v.permute(0, 2, 1, 3).contiguous().view(B, K, self.d_model)

        # 3. 【核心修改】存储所有向量
        # 存下 Batch 0 的所有序列向量，形状为 [Seq_len, 512]
        #self.history_q.append(q_full[0].detach().cpu().numpy())
        #self.history_k.append(k_full[0].detach().cpu().numpy())
        #self.history_v.append(v_full[0].detach().cpu().numpy())


        scores = torch.matmul(q, k.transpose(-2, -1))
        #scores_s1 = scores[0].detach().cpu().numpy()
        #self.history_s1.append(scores_s1)

        scores = scores/ math.sqrt(d_k)
        #scores_s2 = scores[0].detach().cpu().numpy()
        #self.history_s2.append(scores_s2)


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
        #scores = scores.view(B * H, Q, K)
        scores = scores
        #scores_vec = scores[0].detach().cpu().numpy()
        #self.history_scores.append(scores_vec)

        attn = self.softmax(scores)  # [B*H, Q, K]
        #attn = attn.view(B, H, Q, K)  # [B, H, Q, K]

        #attn_vec = attn[0].detach().cpu().numpy()
        #self.history_attn.append(attn_vec)

        # ------ attention output ------
        out = torch.matmul(attn, v)  # [B, H, Q, d_k]
        #attnout_vec = out.detach().cpu().numpy()
        #self.history_attnout.append(attnout_vec)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, self.d_model)
        out = self.w_o(out)

        #out_vec = out[0].detach().cpu().numpy()
        #self.history_out.append(out_vec)

        #import numpy as np
        #import os

        # 创建一个文件夹专门存向量
        #if not os.path.exists("debug_vectors"):
            #os.makedirs("debug_vectors")

        # 使用当前的 history 长度作为 step 序号，防止文件被覆盖
        #step = len(self.history_q)

        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_q_full{step}.npy", q_full[0].detach().cpu().numpy())
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_k_full{step}.npy", k_full[0].detach().cpu().numpy())
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_v_full{step}.npy", v_full[0].detach().cpu().numpy())
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_scores_s1{step}.npy", scores_s1)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_scores_s2{step}.npy", scores_s2)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_scores_vec{step}.npy", scores_vec)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_attn_vec{step}.npy", attn_vec)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_out_vecm{step}.npy", out_vec)

        return out, attn











