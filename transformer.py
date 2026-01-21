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
        self.history = []

    def forward(self, x):

        batch_size, seq_len = x.shape

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb

        # --- 只查看最新词的监控代码 ---
        # 提取最后一个位置的向量：[batch_size, 1, d_model]
        latest_word_vec = x[:, -1, :]
        # 确保数值为正，防止 log(负数) 报错
        # 我们取绝对值或者只看正值部分
        v = latest_word_vec.flatten()

        # 1. 指数：e 的幂次 (exp)
        # 注意：如果 v 的值大于 88，exp(v) 就会变成 inf (溢出)
        v_exp = torch.exp(v)

        # 2. 对数：以 e 为底 (ln)
        # 我们对绝对值加一个极小值 eps，防止 log(0) 变成 -inf
        eps = 1e-12
        v_log_e = torch.log(torch.abs(v) + eps)

        # 3. 对数：以 10 为底 (log10)
        v_log_10 = torch.log10(torch.abs(v) + eps)
        print(f"\n[Embedding Stage - Latest Word Only]")
        print(f"  Total Batch Length: {batch_size}")
        print(f"  Total Sequence Length: {seq_len}")
        print(f"  Latest Vector | Mean: {latest_word_vec.mean():.4f} | Std: {latest_word_vec.std():.4f} | Norm: {latest_word_vec.norm():.4f}")
        print(f"  Exp (mean): {v_exp.mean().item():.4e}")
        print(f"  Log_e (mean): {v_log_e.mean().item():.4f}")
        print(f"  Log_10 (mean): {v_log_10.mean().item():.4f}")
        print(f"  Vector Snippet (Batch 0, Head 5): {v[0, :5].tolist()}...")

        if seq_len > 1:
            # 存下 Batch 0 的所有词
            self.history.append(x[0].detach().cpu().numpy())
        else:
            # 存下 Batch 0 的最新词
            self.history.append(x[0, -1, :].detach().cpu().numpy())

        return self.drop(x)








