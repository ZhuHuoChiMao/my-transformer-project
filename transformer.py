import torch
from torch import nn
import torch.nn.functional as F
import math



from torch import Tensor

#将输入的词汇表索引转换为指定维度的Embedding
#Convertir les indices du vocabulaire d'entrée en vecteurs d'embedding de dimension spécifiée.
#Convert input vocabulary indices into embeddings of a specified dimension.

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding,self).__init__(vocab_size, d_model, padding_idx=1)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len,device):
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size , seq_len = x.size()
        return self.encoding[:seq_len,:]





class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size,d_model)
        self.pos_emb = PositionEmbedding(d_model,max_len,device)
        self.drop = nn.Dropout(p = drop_prob)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x).unsqueeze(0)
        pos_emb = pos_emb.expand_as(tok_emb)
        return self.drop(tok_emb) + self.drop(pos_emb)









