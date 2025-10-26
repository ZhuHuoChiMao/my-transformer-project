





from datasets import load_dataset
ds = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk")

print(ds)
print(ds["train"][:6])

train_lines=ds['train']['text']
from datasets import Dataset, DatasetDict
# 1) 先把行解析成 (en, zh) 样本对
def parse_en_zh_pairs(text_list):
    en, zh = None, None
    pairs = []
    for line in text_list:
        line = line.strip()
        low = line.lower()
        if low.startswith("english:"):
            en = line.split(":", 1)[1].strip()
        elif low.startswith("mandarin:"):
            zh = line.split(":", 1)[1].strip()
        elif line == "--":
            if en and zh:
                pairs.append((en, zh))
            en, zh = None, None
    if en and zh:
        pairs.append((en, zh))
    return pairs

pairs = parse_en_zh_pairs(train_lines)
print(f"样本对数量: {len(pairs)}")  # 这里应该远小于 111,820（因为多行合成一条样本）

# 2) 构建新的 Dataset（两列：en, zh）
en_list = [en for en, _ in pairs]
zh_list = [zh for _, zh in pairs]
pairs_ds = Dataset.from_dict({"en": en_list, "zh": zh_list})

# 3) 切分为 train/test（例如 90%/10%），设置随机种子保证可复现
splits = pairs_ds.train_test_split(test_size=0.1, seed=42, shuffle=True)

# 如果你还想再从 train 里切出 dev/validation（例如 10%）
tmp = splits["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
dataset_dict = DatasetDict({
    "train": tmp["train"],
    "validation": tmp["test"],
    "test": splits["test"],
})

print(dataset_dict['train'][:4])
print("**********************************************************************************************")

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

# =========
# Tokenizers
# =========

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = SPECIAL_TOKENS

_en_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def basic_en_tokenize(text: str) -> List[str]:
    """
    英文：按词和标点切分（Hello, world! -> ["Hello", ",", "world", "!"]）
    数字独立成 token，撇号内部视为词的一部分（e.g. don't）
    """
    return [m.group(0) for m in _en_word_re.finditer(text)]

def basic_zh_tokenize(text: str) -> List[str]:
    """
    中文：按字切分（去掉空白），标点自然成为单个字符 token
    """
    return [ch for ch in text if not ch.isspace()]

@dataclass
class SimpleTokenizer:
    """一个非常简单的 tokenizer：支持 fit、encode、decode。"""
    lang: str                       # 'en' or 'zh'
    stoi: Dict[str, int] = None
    itos: List[str] = None

    def __post_init__(self):
        self.reset_vocab()

    def reset_vocab(self):
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = list(SPECIAL_TOKENS)

    def _tokenize(self, text: str) -> List[str]:
        if self.lang == 'en':
            return basic_en_tokenize(text)
        elif self.lang == 'zh':
            return basic_zh_tokenize(text)
        else:
            raise ValueError(f"Unsupported lang: {self.lang}")

    def fit(self, texts: List[str], min_freq: int = 1):
        """
        从文本列表构建词表。min_freq 可用于过滤低频词（默认不过滤）
        """
        from collections import Counter
        self.reset_vocab()
        cnt = Counter()
        for t in texts:
            cnt.update(self._tokenize(t))
        # 追加到词表
        for tok, freq in cnt.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    @property
    def pad_id(self): return self.stoi[PAD]
    @property
    def bos_id(self): return self.stoi[BOS]
    @property
    def eos_id(self): return self.stoi[EOS]
    @property
    def unk_id(self): return self.stoi[UNK]

    def encode(
        self, text: str, add_bos: bool=False, add_eos: bool=False, max_len: int=None
    ) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in toks]
        if add_bos: ids = [self.bos_id] + ids
        if add_eos: ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        if self.lang == 'en':
            # 英文：简单规则把标点贴回去
            out = []
            for t in toks:
                if re.match(r"^[^\w\s]$", t):  # 单个标点
                    out.append(t)
                elif len(out) == 0:
                    out.append(t)
                elif re.match(r"^[^\w\s]$", out[-1]):  # 前一个是标点
                    out.append(t)
                else:
                    out.append(" " + t)
            return "".join(out)
        else:
            # 中文：直接拼接
            return "".join(toks)

# =========
# Dataset & Collate
# =========

class SimpleTranslationDataset(Dataset):
    def __init__(self, data: Dict[str, List[str]]):
        """
        data: {'en': [...], 'zh': [...]}
        """
        self.src = data['en']
        self.tgt = data['zh']
        assert len(self.src) == len(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {"src": self.src[idx], "tgt": self.tgt[idx]}

def pad_sequences(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (padded, attention_mask)
    padded: [B, T]，mask: [B, T]，pad 位置为 0
    """
    max_seq_len = max((len(s) for s in seqs), default=0)
    max_len = max(32, max_seq_len)
    batch = []
    mask = []
    for s in seqs:
        pad_len = max_len - len(s)
        batch.append(s + [pad_id] * pad_len)
        mask.append([1] * len(s) + [0] * pad_len)
    return torch.tensor(batch, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

def make_collate_fn(tok_src: SimpleTokenizer, tok_tgt: SimpleTokenizer,
                    max_src_len: int=None, max_tgt_len: int=None):
    """
    组装 encoder/decoder 所需张量：
    - input_ids, attention_mask
    - decoder_input_ids（以 <bos> 开头）
    - labels（以 <eos> 结尾，pad 处设为 -100 以忽略 loss）
    """
    def collate(batch: List[Dict[str, str]]):
        src_ids = [
            tok_src.encode(item["src"], add_bos=False, add_eos=True, max_len=max_src_len)
            for item in batch
        ]
        # decoder 输入以 <bos> 开头（不含 eos）
        dec_in_ids = [
            tok_tgt.encode(item["tgt"], add_bos=True, add_eos=False, max_len=max_tgt_len)
            for item in batch
        ]
        # labels 以 <eos> 结尾（不含 bos）
        labels = [
            tok_tgt.encode(item["tgt"], add_bos=False, add_eos=True, max_len=max_tgt_len)
            for item in batch
        ]

        input_ids, attention_mask = pad_sequences(src_ids, tok_src.pad_id)
        decoder_input_ids, decoder_attention_mask = pad_sequences(dec_in_ids, tok_tgt.pad_id)
        labels_pad, _ = pad_sequences(labels, tok_tgt.pad_id)

        # 让 CrossEntropyLoss 忽略 pad
        labels_pad[labels_pad == tok_tgt.pad_id] = -100

        return {
            "input_ids": input_ids,                               # [B, S]
            "attention_mask": attention_mask,                     # [B, S]
            "decoder_input_ids": decoder_input_ids,               # [B, T]
            "decoder_attention_mask": decoder_attention_mask,     # [B, T]
            "labels": labels_pad,                                 # [B, T]
        }
    return collate

# =========
# 使用示例（你的这条数据）
# =========

data = dataset_dict['train']

# 1) 构建 tokenizer 并拟合词表
tok_en = SimpleTokenizer('en')
tok_zh = SimpleTokenizer('zh')

tok_en.fit(data['en'])
tok_zh.fit(data['zh'])

# 2) 构建数据集 & DataLoader
dataset = SimpleTranslationDataset(data)
collate_fn = make_collate_fn(tok_en, tok_zh, max_src_len=32, max_tgt_len=32)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

for i, batch in enumerate(loader):
    if i >= 5:
        break
    print(f"\n===== Batch {i} =====")
    for k, v in batch.items():
        print(k, v.shape, "\n", v)


print("**********************************************************************************************")


