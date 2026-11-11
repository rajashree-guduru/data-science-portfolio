"""PyTorch intent classifier (fastText-style hashed n-gram bag of embeddings)."""
from __future__ import annotations

import re
import zlib

import numpy as np
import torch
from torch import nn

TOKEN_RE = re.compile(r"[a-z0-9#$']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HashedNgramEncoder:
    """Maps text to hashed word unigram + bigram ids (stable across runs via crc32)."""

    def __init__(self, n_buckets: int = 2**18):
        self.n_buckets = n_buckets

    def _h(self, s: str) -> int:
        return zlib.crc32(s.encode()) % self.n_buckets

    def encode(self, text: str) -> list[int]:
        toks = tokenize(text)
        ids = [self._h(t) for t in toks]
        ids += [self._h(a + " " + b) for a, b in zip(toks, toks[1:])]
        return ids or [0]


class FastTextClassifier(nn.Module):
    def __init__(self, n_buckets: int, dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_buckets, dim, mode="mean", sparse=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, flat_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.emb(flat_ids, offsets)))


class IntentModel:
    """Wraps encoder + network with fit / predict_proba, and tracks the training vocabulary."""

    def __init__(self, dim: int = 64, n_buckets: int = 2**18, seed: int = 0, version: str = "v1"):
        self.encoder = HashedNgramEncoder(n_buckets)
        self.dim, self.seed, self.version = dim, seed, version
        self.labels: list[str] = []
        self.vocab: set[str] = set()
        self.net: FastTextClassifier | None = None

    def _batch(self, texts):
        seqs = [self.encoder.encode(t) for t in texts]
        offsets = torch.tensor([0] + [len(s) for s in seqs[:-1]]).cumsum(0)
        flat = torch.tensor([i for s in seqs for i in s])
        return flat, offsets

    def fit(self, texts, labels, val_texts=None, val_labels=None, epochs=10, lr=0.01, batch_size=64):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.labels = sorted(set(labels))
        lab2id = {l: i for i, l in enumerate(self.labels)}
        y = np.array([lab2id[l] for l in labels])
        self.vocab = {t for x in texts for t in tokenize(x)}
        self.net = FastTextClassifier(self.encoder.n_buckets, self.dim, len(self.labels))
        # sparse gradients for the large hashed embedding table keep training fast on CPU
        opt_sparse = torch.optim.SparseAdam(list(self.net.emb.parameters()), lr=lr)
        opt_dense = torch.optim.Adam(list(self.net.out.parameters()), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        history = []
        texts = list(texts)
        for ep in range(epochs):
            self.net.train()
            order = rng.permutation(len(texts))
            total = 0.0
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                flat, off = self._batch([texts[i] for i in idx])
                opt_sparse.zero_grad()
                opt_dense.zero_grad()
                loss = loss_fn(self.net(flat, off), torch.tensor(y[idx]))
                loss.backward()
                opt_sparse.step()
                opt_dense.step()
                total += loss.item() * len(idx)
            rec = {"epoch": ep + 1, "train_loss": total / len(texts)}
            if val_texts is not None:
                rec["val_acc"] = float(np.mean(self.predict(val_texts) == np.array(val_labels)))
            history.append(rec)
        return history

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 1024) -> np.ndarray:
        self.net.eval()
        out = []
        texts = list(texts)
        for s in range(0, len(texts), batch_size):
            flat, off = self._batch(texts[s:s + batch_size])
            out.append(torch.softmax(self.net(flat, off), dim=-1).numpy())
        return np.vstack(out)

    def predict(self, texts) -> np.ndarray:
        return np.array(self.labels)[self.predict_proba(texts).argmax(1)]

    def predict_one(self, text: str) -> tuple[str, float]:
        p = self.predict_proba([text])[0]
        return self.labels[int(p.argmax())], float(p.max())

    def oov_rate(self, texts) -> float:
        toks = [t for x in texts for t in tokenize(x) if not t.startswith("#") and not t.isdigit()]
        return float(np.mean([t not in self.vocab for t in toks])) if toks else 0.0
