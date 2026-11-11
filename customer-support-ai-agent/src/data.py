"""Data loading, placeholder filling and train / production splits.

Dataset: Bitext Customer Support LLM Chatbot Training Dataset (26,872 customer
messages, 27 intents, 11 categories), licensed CDLA-Sharing-1.0.
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Each row carries linguistic "flags". Flag Z marks messages with spelling errors
and typos. We keep Z messages OUT of training and use them later to simulate a
real-world shift in how customers write (covariate drift).
"""
from __future__ import annotations

import os
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bitext_customer_support.csv")

_NAMES = ["Alex", "Priya", "Jordan", "Mei", "Carlos", "Fatima", "Sam", "Olga"]
_ACCOUNT = ["premium", "standard", "business", "free", "gold"]
_CITIES = ["Austin", "Denver", "Seattle", "Boston", "Chicago"]
_COUNTRIES = ["Canada", "Mexico", "Germany", "India", "Japan"]


def load_dataset(path: str = DEFAULT_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    df = df.dropna(subset=["instruction", "intent"]).reset_index(drop=True)
    df["has_typo"] = df["flags"].astype(str).str.contains("Z")
    return df


def random_order_id(rng: np.random.Generator) -> str:
    return f"#{rng.integers(10000, 99999)}"


def fill_placeholders(text: str, rng: np.random.Generator, order_id: str | None = None) -> str:
    """Replace Bitext {{placeholders}} with realistic values."""
    order_id = order_id or random_order_id(rng)
    repl = {
        "{{Order Number}}": order_id,
        "{{Invoice Number}}": f"INV-{rng.integers(1000, 9999)}",
        "{{Person Name}}": str(rng.choice(_NAMES)),
        "{{Account Type}}": str(rng.choice(_ACCOUNT)),
        "{{Account Category}}": str(rng.choice(_ACCOUNT)),
        "{{Refund Amount}}": f"${rng.integers(10, 400)}",
        "{{Currency Symbol}}": "$",
        "{{Delivery City}}": str(rng.choice(_CITIES)),
        "{{Delivery Country}}": str(rng.choice(_COUNTRIES)),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return re.sub(r"\{\{[^}]+\}\}", "", text).strip()


def make_splits(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Clean messages -> train 60% / val 10% / production pool 30%.
    Typo messages (never trained on by v1) -> production pool 60% / stress-test val 15% / final holdout 25%."""
    clean = df[~df["has_typo"]]
    typo = df[df["has_typo"]]
    train, rest = train_test_split(clean, train_size=0.6, stratify=clean["intent"], random_state=seed)
    val, prod_clean = train_test_split(rest, train_size=0.25, stratify=rest["intent"], random_state=seed)
    typo_prod, rest_t = train_test_split(typo, train_size=0.6, stratify=typo["intent"], random_state=seed)
    typo_val, typo_holdout = train_test_split(rest_t, train_size=0.375, stratify=rest_t["intent"], random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "prod_clean": prod_clean.reset_index(drop=True),
        "typo_prod": typo_prod.reset_index(drop=True),
        "typo_val": typo_val.reset_index(drop=True),
        "typo_holdout": typo_holdout.reset_index(drop=True),
    }
