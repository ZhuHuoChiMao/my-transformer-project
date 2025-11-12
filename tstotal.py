import torch
from torch import nn
import torch.nn.functional as F
import math
from encoder import Encoder
from decoder import Decoder



import torch
from torch import nn
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
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        src_pad_mask = (src == self.src_pad_idx)
        tgt_pad_mask = (trg == self.trg_pad_idx)
        enc = self.encoder(src, key_padding_mask=src_pad_mask)


        out = self.decoder(trg, enc, src_pad_mask, tgt_pad_mask)


        return out



