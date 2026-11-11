"""TensorFlow / Keras version of the intent classifier.

This rebuilds the PyTorch model in src/model.py with the SAME design, so the two
frameworks can be compared fairly:

    message -> hashed word + word-pair ids -> embedding per id -> average -> dropout -> 27 scores

Key difference to remember for interviews:
  * PyTorch: you write the training loop yourself (see IntentModel.fit in model.py).
  * Keras:   you describe the model, call compile(), then fit() runs the loop for you.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import HashedNgramEncoder  # reuse the exact same text-to-ids step as PyTorch


class KerasIntentModel:
    """Same interface as the PyTorch IntentModel: fit, predict_proba, predict, predict_one."""

    def __init__(self, dim: int = 64, n_buckets: int = 2**18, seed: int = 0, max_len: int = 48):
        self.encoder = HashedNgramEncoder(n_buckets)  # text -> list of slot numbers (0 .. n_buckets-1)
        self.n_buckets = n_buckets
        self.dim = dim                                # 64 numbers learned per slot, same as PyTorch
        self.seed = seed
        self.max_len = max_len                        # messages are short (max 16 words), so 48 ids is plenty
        self.labels: list[str] = []
        self.model: tf.keras.Model | None = None

    # ---- turning text into a fixed-size grid of numbers ------------------------------
    def _to_ids(self, texts) -> np.ndarray:
        """Keras wants every row the same length, so we pad short messages with 0.

        Slot numbers are shifted by +1 so that 0 can mean "padding, ignore me".
        """
        out = np.zeros((len(texts), self.max_len), dtype="int32")
        for i, t in enumerate(texts):
            ids = [x + 1 for x in self.encoder.encode(t)][: self.max_len]
            out[i, : len(ids)] = ids
        return out

    # ---- describing the model ---------------------------------------------------------
    def _build(self, n_classes: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.max_len,), dtype="int32")      # one row of ids per message
        # Embedding: a learned table with one 64-number vector per slot.
        # mask_zero=True tells the next layer to ignore the padding zeros.
        x = tf.keras.layers.Embedding(self.n_buckets + 1, self.dim, mask_zero=True)(inputs)
        # Average the vectors of all words in the message -> one 64-number summary
        # (this is what nn.EmbeddingBag(mode="mean") does in PyTorch).
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)                                 # randomly drop 20% during training
        # Dense = linear layer; softmax turns 27 scores into probabilities that add to 1.
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs)

    # ---- training ---------------------------------------------------------------------
    def fit(self, texts, labels, val_texts=None, val_labels=None, epochs: int = 10, lr: float = 0.01,
            batch_size: int = 64):
        tf.keras.utils.set_random_seed(self.seed)        # same results on every run
        self.labels = sorted(set(labels))
        lab2id = {l: i for i, l in enumerate(self.labels)}
        x = self._to_ids(texts)
        y = np.array([lab2id[l] for l in labels])

        self.model = self._build(len(self.labels))
        # compile() = choose the optimizer and the loss. Same choices as the PyTorch version:
        # Adam optimizer, cross-entropy loss (the "sparse" version takes label numbers, not one-hot).
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        val = None
        if val_texts is not None:
            val = (self._to_ids(val_texts), np.array([lab2id[l] for l in val_labels]))
        # fit() runs the whole training loop for us: batches, loss, gradients, optimizer steps.
        hist = self.model.fit(x, y, validation_data=val, epochs=epochs, batch_size=batch_size,
                              shuffle=True, verbose=0)
        return [{"epoch": i + 1, "train_loss": hist.history["loss"][i],
                 **({"val_acc": hist.history["val_accuracy"][i]} if val else {})}
                for i in range(epochs)]

    # ---- predicting -------------------------------------------------------------------
    def predict_proba(self, texts) -> np.ndarray:
        return self.model.predict(self._to_ids(list(texts)), batch_size=1024, verbose=0)

    def predict(self, texts) -> np.ndarray:
        return np.array(self.labels)[self.predict_proba(texts).argmax(1)]

    def predict_one(self, text: str) -> tuple[str, float]:
        p = self.model(self._to_ids([text]), training=False).numpy()[0]  # direct call is faster for one message
        return self.labels[int(p.argmax())], float(p.max())
