import torch,os
from torch import nn
from tstotal import Transformer
import trainclass
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

ds = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk")
train_lines=ds['train']['text']


cd=trainclass.class_datasets(train_lines)
pairs = cd.parse_en_zh_pairs()
dataset_dict=cd.test_datasets(pairs)

data = dataset_dict['train']

tok_en = trainclass.SimpleTokenizer('en')
tok_zh = trainclass.SimpleTokenizer('zh')

tok_en.fit(data['en'])
tok_zh.fit(data['zh'])


dataset = trainclass.SimpleTranslationDataset(data)
collate_fn = trainclass.SimpleTranslationDataset.make_collate_fn(tok_en, tok_zh, max_src_len=32, max_tgt_len=32)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)





pad_idx = 0

model = Transformer(
    src_pad_idx=tok_en.pad_id,
    trg_pad_idx=tok_zh.pad_id,
    enc_voc_size=len(tok_en.itos),
    dec_voc_size=len(tok_zh.itos),
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    drop_prob=0.1,
    device='cuda',
    max_len=100
).to("cuda")

from torch.nn.functional import log_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 载入模型参数（用你保存的路径）
ckpt_path = "/content/drive/MyDrive/transformer_epoch.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
# 单句英->中
@torch.no_grad()
def translate_en2zh(text_en: str, max_len: int = 32):
    # 1) 编码源句：和 collate 一样，src 不加 BOS，末尾加 EOS
    src_ids = tok_en.encode(text_en, add_bos=False, add_eos=True, max_len=max_len)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, src_len]

    # 2) 目标序列以 <bos> 开始
    bos_id, eos_id = tok_zh.bos_id, tok_zh.eos_id
    tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # [1, 1]

    # 3) 逐步解码
    for _ in range(max_len):
        # 你的 Transformer 前向是 model(src, trg)，输出应为 [B, T, V]
        out = model(src, tgt)
        next_logit = out[:, -1, :]          # 取最后一步
        next_token = next_logit.argmax(-1)  # 贪心
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == eos_id:
            break

    # 4) 解码中文：你的 decode 会自动跳过 <pad>/<bos>，遇到 <eos> 截断
    pred_ids = tgt.squeeze(0).tolist()
    print("pred_ids:", pred_ids)
    zh_text = tok_zh.decode(pred_ids)
    return zh_text

# 快速试几句
print(translate_en2zh("good"))
print(translate_en2zh("you are nice"))
print(translate_en2zh("hello"))

def show_tokenization(i):
    en = data[i]["en"]
    zh = data[i]["zh"]

    # --- 英文 ---
    en_tokens = tok_en._tokenize(en)  # 你的tokenizer里公开的是 _tokenize
    en_ids_plain = tok_en.encode(en, add_bos=False, add_eos=False, max_len=32)
    en_ids_eos   = tok_en.encode(en, add_bos=False, add_eos=True,  max_len=32)
    en_decoded   = tok_en.decode(en_ids_eos)

    # 统计 UNK 比例
    unk_ratio_en = sum(1 for x in en_ids_plain if x == tok_en.unk_id) / max(1, len(en_ids_plain))

    print(f"\n=== SAMPLE {i} ===")
    print("EN raw:     ", en)
    print("EN tokens:  ", en_tokens)
    print("EN ids:     ", en_ids_plain)
    print("EN ids+eos: ", en_ids_eos)
    print("EN decoded: ", en_decoded)
    print("EN UNK%:    ", f"{unk_ratio_en:.1%}")

    # --- 中文（你的实现是按字符切分） ---
    zh_tokens = tok_zh._tokenize(zh)
    zh_ids_bos = tok_zh.encode(zh, add_bos=True,  add_eos=False, max_len=32)
    zh_ids_eos = tok_zh.encode(zh, add_bos=False, add_eos=True,  max_len=32)
    zh_decoded = tok_zh.decode(zh_ids_eos)

    unk_ratio_zh = sum(1 for x in zh_ids_eos if x == tok_zh.unk_id) / max(1, len(zh_ids_eos))

    print("ZH raw:     ", zh)
    print("ZH tokens:  ", zh_tokens[:64])  # 过长就截一下展示
    print("ZH ids+bos: ", zh_ids_bos)
    print("ZH ids+eos: ", zh_ids_eos)
    print("ZH decoded: ", zh_decoded)
    print("ZH UNK%:    ", f"{unk_ratio_zh:.1%}")

# 看前 3 条
for i in range(3):
    show_tokenization(i)



'''
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)



num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        src = batch["input_ids"].to("cuda")
        trg = batch["decoder_input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")

        optimizer.zero_grad()
        output = model(src, trg)

        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f" Epoch {epoch+1} finished, avg loss = {avg_loss:.4f}")

    save_path = f"/content/drive/MyDrive/transformer_epoch.pt"
    torch.save(model.state_dict(), save_path)

    import os
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f" 模型已保存到: {save_path}  （大小约 {size_mb:.2f} MB）\n")
'''




