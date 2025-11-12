import torch
from torch import nn
import torch.nn.functional as F
import math
from encoder import Encoder
from decoder import Decoder



class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 n_heads,
                 d_ff,
                 n_layers,
                 drop_prob,
                 device,
                 max_len=100):
        super().__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, d_ff, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, d_ff, n_heads, n_layers, drop_prob, device)
        self.trg_pad_idx = trg_pad_idx
        self.src_pad_idx = src_pad_idx
        self.device = device  # 这里只要能被 torch 使用（'cuda' 或 torch.device）

    # 形状工具：把 [B,Q] → [B,1,Q,1]，把 [B,K] → [B,1,1,K]
    @staticmethod
    def _expand_q_k(q_nonpad: torch.Tensor, k_nonpad: torch.Tensor):
        # q_nonpad / k_nonpad: [B, L] bool, True 表示非 pad（可用）
        q_mask = q_nonpad.unsqueeze(1).unsqueeze(3)  # [B,1,Q,1]
        k_mask = k_nonpad.unsqueeze(1).unsqueeze(2)  # [B,1,1,K]
        return q_mask, k_mask

    def make_pad_mask_qk(self, q_ids: torch.Tensor, k_ids: torch.Tensor,
                         pad_q: int, pad_k: int) -> torch.Tensor:
        # 返回 [B,1,Q,K]，True=允许，False=屏蔽
        q_nonpad = q_ids.ne(pad_q)  # [B,Q]
        k_nonpad = k_ids.ne(pad_k)  # [B,K]
        q_mask, k_mask = self._expand_q_k(q_nonpad, k_nonpad)
        return (q_mask & k_mask)  # [B,1,Q,K]

    def make_causal_mask(self, Lq: int, Lk: int) -> torch.Tensor:
        # 因果遮罩：下三角(含对角) True=允许；上三角 False=屏蔽
        # 返回 [1,1,Q,K] 以便广播
        causal = torch.ones(Lq, Lk, dtype=torch.bool, device=self.device).tril()  # [Q,K]
        return causal.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        B, S = src.size()
        _, T = trg.size()

        # 1) Encoder 自注意力 mask（只需 K 侧 pad，但你当前 MHA 只收一个mask，用 Q&K 也可）
        src_self_mask = self.make_pad_mask_qk(src, src, self.src_pad_idx, self.src_pad_idx)  # [B,1,S,S]

        # 2) Decoder 自注意力：pad ∧ 因果
        tgt_self_pad = self.make_pad_mask_qk(trg, trg, self.trg_pad_idx, self.trg_pad_idx)   # [B,1,T,T]
        tgt_causal   = self.make_causal_mask(T, T)                                           # [1,1,T,T]
        tgt_self_mask = tgt_self_pad & tgt_causal                                            # [B,1,T,T]

        # 3) Cross-Attn：Q 来自 trg，K/V 来自 src。最低要求只遮掉 K 的 pad。
        #    你当前的 MHA 只有一个 mask入口，用 Q&K 也行（等价于把 Q 的 pad 也屏蔽）
        cross_mask = self.make_pad_mask_qk(trg, src, self.trg_pad_idx, self.src_pad_idx)     # [B,1,T,S]

        # 4) Forward
        enc = self.encoder(src, mask=src_self_mask)                # 你的 Encoder 要用这个 mask
        out = self.decoder(trg, enc, t_mask=tgt_self_mask, s_mask=cross_mask)

        return out



