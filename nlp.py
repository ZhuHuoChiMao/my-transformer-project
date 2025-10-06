import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
import re, math, numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# Preprocessor
class TextPreprocessor:
    def __init__(self, language='french'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str):
        text_clean = re.sub(r'[^\w\s]', '', text)

        text_normalized = re.sub(r'\s+', ' ', text_clean).strip()

        words = word_tokenize(text_normalized, language='french')

        filtered_tokens = [word for word in words if word.lower() not in self.stop_words]

        stems = [self.stemmer.stem(word) for word in filtered_tokens]

        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in stems]

        return lemmatized_words


# TF-IDF
class TF_IDF:
    def __init__(self):
        self.vocab = None
        self.index = None
        self.idf = None

    def fit(self, corpus_tokens):
        N = len(corpus_tokens)
        df_counts = Counter()
        for doc in corpus_tokens:
            for t in set(doc):
                df_counts[t] += 1

        self.vocab = sorted(df_counts.keys())
        self.index = {t: i for i, t in enumerate(self.vocab)}
        V = len(self.vocab)
        self.idf = np.zeros(V, dtype=float)
        for t, j in self.index.items():
            self.idf[j] = math.log((1 + N) / (1 + df_counts[t])) + 1.0
        return self.idf

    def to_vector(self, doc_tokens):
        V = len(self.vocab)
        vec = np.zeros(V, dtype=float)
        cnt = Counter(doc_tokens)
        L = sum(cnt.values()) or 1
        for t, c in cnt.items():
            j = self.index.get(t)
            if j is not None:
                tf = c / L
                vec[j] = tf * self.idf[j]
        return vec

    def transform(self, docs_tokens):
        return np.vstack([self.to_vector(doc) for doc in docs_tokens])

    def fit_transform(self, corpus_tokens):
        self.fit(corpus_tokens)
        return self.transform(corpus_tokens)


class feedforward:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def TrainModel(self):
        ds = load_dataset(self.dataset_name)
        pdf = ds["train"].to_pandas()[["text_fr", "labels"]].dropna()
        X_raw = pdf["text_fr"].astype(str).tolist()
        y = pdf["labels"].map({"ham": 0, "spam": 1}).to_numpy()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, stratify=y, random_state=42
        )

        tp = TextPreprocessor()
        train_tokens = [tp.preprocess(t) for t in X_train_raw]
        test_tokens = [tp.preprocess(t) for t in X_test_raw]

        tfidf = TF_IDF()
        Xtr = tfidf.fit_transform(train_tokens)
        Xte = tfidf.transform(test_tokens)

        clf = ComplementNB(alpha=1.0)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print("Accuracy:", acc,
              "\nPrecision:", prec,
              "\nRecall:", rec,
              "\nF1:", f1)
        print("Confusion_matrix:\n", cm)

    def TrainModel_DL(self, num_epochs=5, batch_size=32, lr=1e-4, max_len=50):
        ds = load_dataset(self.dataset_name)
        pdf = ds["train"].to_pandas()[["text_fr", "labels"]].dropna()
        X_raw = pdf["text_fr"].astype(str).tolist()
        y = pdf["labels"].map({"ham": 0, "spam": 1}).to_numpy()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, stratify=y, random_state=42
        )

        tp = TextPreprocessor()
        train_tokens = [tp.preprocess(t) for t in X_train_raw]
        test_tokens = [tp.preprocess(t) for t in X_test_raw]
        vocab = {w: i + 2 for i, w in enumerate(set(sum(train_tokens, [])))}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1

    def encode(tokens):
        return [vocab.get(w, 1) for w in tokens][:max_len]

    def pad(seq):
        return seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

        Xtr_ids = torch.tensor([pad(encode(t)) for t in train_tokens])
        Xte_ids = torch.tensor([pad(encode(t)) for t in test_tokens])
        ytr = torch.tensor(y_train)
        yte = torch.tensor(y_test)

        train_ds = TensorDataset(Xtr_ids, ytr)
        test_ds = TensorDataset(Xte_ids, yte)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = TransformerClassifier(
            src_pad_idx=0,
            enc_voc_size=len(vocab),
            d_model=128,
            n_heads=4,
            d_ff=256,
            n_layers=2,
            drop_prob=0.1,
            device="cpu",
            max_len=max_len,
            num_classes=2
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print("DL Model Evaluation:")
        print("Accuracy:", acc,
              "\nPrecision:", prec,
              "\nRecall:", rec,
              "\nF1:", f1)
        print("Confusion_matrix:\n", cm)


fd = feedforward("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset")
fd.TrainModel()
fd.TrainModel_DL()