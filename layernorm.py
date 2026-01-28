r'''
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
        #self.history_1 = []
        #self.history_2 = []
        #self.name = name
        #self.layers_i = layers_i
    def forward(self,x):

        mean = x.mean(-1 , keepdim = True)
        var = x.var(-1, unbiased = False , keepdim = True)



        output = (x - mean) / (torch.sqrt(var + self.eps))
        #output_vec = output[0].detach().cpu().numpy()
        #self.history_1.append(output_vec)

        output = self.gamma * output + self.beta

        #output_vecc = output[0].detach().cpu().numpy()
        #self.history_2.append(output_vecc)

        #import numpy as np
        #import os

        # 创建一个文件夹专门存向量
        #if not os.path.exists("debug_vectors"):
            #os.makedirs("debug_vectors")

        # 使用当前的 history 长度作为 step 序号，防止文件被覆盖
        #step = len(self.history_1)

        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_output_vecl{step}.npy", output_vec)
        #np.save(fr"C:\Users\acer\PycharmProjects\Transformer\npy\{self.name}_{self.layers_i}_output_vecc{step}.npy", output_vecc)


        return output
'''

import torch
from torch import nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):

        mean = x.mean(-1 , keepdim = True)
        var = x.var(-1, unbiased = False , keepdim = True)
        output = (x - mean) / (torch.sqrt(var + self.eps))
        output = self.gamma * output + self.beta


        return output