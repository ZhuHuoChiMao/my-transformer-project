import torch
from torch import nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        #self.history = []
    def forward(self,x):
        # --- [LATEST] 仅提取最新词的特征进行观察 ---
        # latest_x 维度: [Batch, d_model]
        latest_x = x[:, -1, :]

        #print(f"\n[LayerNorm - Latest Word Focus]")
        # print(f"  Total Sequence Length: {x.size(1)}")


        mean = x.mean(-1 , keepdim = True)
        var = x.var(-1, unbiased = False , keepdim = True)

        # --- [LATEST] 打印最新词的统计特性 ---
        #latest_mean = mean[:, -1, :]
        #latest_var = var[:, -1, :]
        #print(f"  Latest Word Original Mean: {latest_mean.item():.4f}")
        #print(f"  Latest Word Original Var: {latest_var.item():.4f}")


        output = (x - mean) / (torch.sqrt(var + self.eps))

        output = self.gamma * output + self.beta

        # --- [LATEST] 观察归一化后的最新词向量 ---
        #latest_out = output[:, -1, :]
        #print(f"  Latest Word Normed Mean: {latest_out.mean().item():.4f} (should be ~0)")
        #print(f"  Latest Word Normed Var: {latest_out.var().item():.4f} (should be ~1 if gamma=1)")
        #output_vec = output.detach().cpu().numpy()
        #self.history.append(output_vec)


        return output