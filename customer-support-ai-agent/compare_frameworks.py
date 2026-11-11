"""Train the same intent model in PyTorch and in TensorFlow/Keras, then compare them.

Usage:  python compare_frameworks.py        (about 1-2 minutes on a laptop CPU)
Writes: results/framework_comparison.csv

Both models use the same data splits, the same text-to-ids step, the same layer sizes,
the same optimizer (Adam, learning rate 0.01), batch size 64 and 10 epochs.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import fill_placeholders, load_dataset, make_splits
from src.model import IntentModel
from src.model_tf import KerasIntentModel

SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "results")


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    sp = make_splits(load_dataset(), SEED)
    # Same filled-in text for both frameworks, so only the framework differs.
    X_tr, y_tr = [fill_placeholders(t, rng) for t in sp["train"].instruction], list(sp["train"].intent)
    X_va, y_va = [fill_placeholders(t, rng) for t in sp["val"].instruction], list(sp["val"].intent)
    X_ty, y_ty = [fill_placeholders(t, rng) for t in sp["typo_holdout"].instruction], list(sp["typo_holdout"].intent)

    rows = []
    for name, model in [("PyTorch (custom training loop, SparseAdam)", IntentModel(seed=SEED)),
                        ("TensorFlow / Keras (compile + fit)", KerasIntentModel(seed=SEED))]:
        start = time.time()
        model.fit(X_tr, y_tr, epochs=10)
        train_s = time.time() - start
        start = time.time()
        for t in X_va[:500]:
            model.predict_one(t)
        ms_per_msg = (time.time() - start) / 500 * 1000
        pv, pt = model.predict(X_va), model.predict(X_ty)
        rows.append({
            "framework": name,
            "clean_val_accuracy": round(accuracy_score(y_va, pv), 4),
            "clean_val_macro_f1": round(f1_score(y_va, pv, average="macro"), 4),
            "typo_holdout_accuracy": round(accuracy_score(y_ty, pt), 4),
            "typo_holdout_macro_f1": round(f1_score(y_ty, pt, average="macro"), 4),
            "train_seconds": round(train_s, 1),
            "ms_per_message": round(ms_per_msg, 2),
        })
        print(rows[-1])

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/framework_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
