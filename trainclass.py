from datasets import load_dataset
from datasets import Dataset, DatasetDict
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, DatasetDict

class class_datasets:
    def __init__(self,train_lines):
        self.train_lines=train_lines

    def parse_en_zh_pairs(self,):
        en, zh = None, None
        pairs = []
        for line in self.train_lines:
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

    def test_datasets(self,pairs):
        en_list = [en for en, _ in pairs]
        zh_list = [zh for _, zh in pairs]
        pairs_ds = Dataset.from_dict({"en": en_list, "zh": zh_list})


        splits = pairs_ds.train_test_split(test_size=0.1, seed=42, shuffle=True)


        tmp = splits["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
        dataset_dict = DatasetDict({
            "train": tmp["train"],
            "validation": tmp["test"],
            "test": splits["test"],
        })

        return dataset_dict



class SimpleTokenizer:
    def __init__(self, lang: str):
        self.lang = lang
        self.SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.PAD, self.BOS, self.EOS, self.UNK = self.SPECIAL_TOKENS
        self._en_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")
        self.reset_vocab()

    def basic_en_tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in self._en_word_re.finditer(text)]

    @staticmethod
    def basic_zh_tokenize(text: str) -> List[str]:
        return [ch for ch in text if not ch.isspace()]

    def reset_vocab(self):
        self.stoi = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.itos = list(self.SPECIAL_TOKENS)

    def _tokenize(self, text: str) -> List[str]:
        if self.lang == 'en':
            return self.basic_en_tokenize(text)
        elif self.lang == 'zh':
            return self.basic_zh_tokenize(text)
        else:
            raise ValueError(f"Unsupported lang: {self.lang}")

    def fit(self, texts: List[str], min_freq: int = 1):
        from collections import Counter
        self.reset_vocab()
        cnt = Counter()
        for t in texts:
            cnt.update(self._tokenize(t))
        for tok, freq in cnt.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    @property
    def pad_id(self): return self.stoi[self.PAD]
    @property
    def bos_id(self): return self.stoi[self.BOS]
    @property
    def eos_id(self): return self.stoi[self.EOS]
    @property
    def unk_id(self): return self.stoi[self.UNK]

    def encode(self, text: str, add_bos=False, add_eos=False, max_len=None) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in toks]
        if add_bos: ids = [self.bos_id] + ids
        if add_eos: ids = ids + [self.eos_id]
        if max_len is not None: ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else self.UNK)
        if self.lang == 'en':
            out = []
            for t in toks:
                if re.match(r"^[^\w\s]$", t):
                    out.append(t)
                elif len(out) == 0:
                    out.append(t)
                elif re.match(r"^[^\w\s]$", out[-1]):
                    out.append(t)
                else:
                    out.append(" " + t)
            return "".join(out)
        else:
            return "".join(toks)




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

    @staticmethod
    def pad_sequences(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_seq_len = max((len(s) for s in seqs), default=0)
        max_len = max(32, max_seq_len)
        batch = []
        mask = []
        for s in seqs:
            pad_len = max_len - len(s)
            batch.append(s + [pad_id] * pad_len)
            mask.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(batch, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    @staticmethod
    def make_collate_fn(tok_src: SimpleTokenizer, tok_tgt: SimpleTokenizer,
                        max_src_len: int = None, max_tgt_len: int = None):

        def collate(batch: List[Dict[str, str]]):
            src_ids = [
                tok_src.encode(item["src"], add_bos=False, add_eos=True, max_len=max_src_len)
                for item in batch
            ]
            dec_in_ids = [
                tok_tgt.encode(item["tgt"], add_bos=True, add_eos=False, max_len=max_tgt_len)
                for item in batch
            ]
            labels = [
                tok_tgt.encode(item["tgt"], add_bos=False, add_eos=True, max_len=max_tgt_len)
                for item in batch
            ]

            input_ids, attention_mask = SimpleTranslationDataset.pad_sequences(src_ids, tok_src.pad_id)
            decoder_input_ids, decoder_attention_mask = SimpleTranslationDataset.pad_sequences(dec_in_ids, tok_tgt.pad_id)
            labels_pad, _ = SimpleTranslationDataset.pad_sequences(labels, tok_tgt.pad_id)

            labels_pad[labels_pad == tok_tgt.pad_id] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels_pad,
            }

        return collate

