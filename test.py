import torch
from torch import nn
from tstotal import Transformer

batch_size = 2
src_len = 5
trg_len = 6
vocab_size = 1000
pad_idx = 0


src = torch.randint(1, vocab_size, (batch_size, src_len))

trg = torch.randint(1, vocab_size, (batch_size, trg_len))

model = Transformer(
    src_pad_idx=pad_idx,
    trg_pad_idx=pad_idx,
    enc_voc_size=vocab_size,
    dec_voc_size=vocab_size,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    drop_prob=0.1,
    device='cpu',
    max_len=100
)

output = model(src, trg)
print(output.shape)
