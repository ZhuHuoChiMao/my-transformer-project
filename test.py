import torch
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

        if (i + 1) % 100 == 0:  # 每100个batch打印一次
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch+1} finished, avg loss = {avg_loss:.4f}")

    # ✅ 在每个 epoch 结束后保存一次模型到 Google Drive
    save_path = f"/content/drive/MyDrive/transformer_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), save_path)

    # 打印保存信息确认成功
    import os
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"💾 模型已保存到: {save_path}  （大小约 {size_mb:.2f} MB）\n")

'''
for batch in loader:
    src = batch["input_ids"].to("cuda")
    trg = batch["decoder_input_ids"].to("cuda")

    output = model(src, trg)
    print("output.shape:", output.shape)  # 检查维
    pred_ids = output.argmax(-1)          # 取每步概率最高的 token

    # 解码查看翻译
    for i in range(pred_ids.size(0)):
        print("原句:", tok_en.decode(src[i].tolist()))
        print("翻译:", tok_zh.decode(pred_ids[i].tolist()))
        print("-" * 50)
    break
    
'''


