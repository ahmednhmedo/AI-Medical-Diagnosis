# inference_lstm.py
"""
Hybrid LSTM + TF-IDF ensemble inference.

Loads the LSTM model and combines its predictions with the TF-IDF baseline
using an optimal ensemble weight learned during training.
"""
import os
import json
import pickle
from typing import List, Tuple, Union

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_lstm")
BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_baseline")


class LSTMClassifier:
    def __init__(self, art_dir: str = ART_DIR, baseline_dir: str = BASELINE_DIR):
        self.art_dir = art_dir
        self.baseline_dir = baseline_dir

        # Load config
        config_path = os.path.join(art_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.maxlen = cfg.get("maxlen", 50)
            self.ensemble_weight = cfg.get("ensemble_weight", 0.5)
        else:
            self.maxlen = 50
            self.ensemble_weight = 0.5

        # Load LSTM
        self.model = tf.keras.models.load_model(os.path.join(art_dir, "model.keras"))
        with open(os.path.join(art_dir, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(os.path.join(art_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        # Try to load TF-IDF for ensemble
        self.tfidf_vec = None
        self.tfidf_clf = None
        self.tfidf_le = None
        try:
            import joblib
            vec_p = os.path.join(baseline_dir, "vectorizer.pkl")
            clf_p = os.path.join(baseline_dir, "clf.pkl")
            le_p = os.path.join(baseline_dir, "label_encoder.pkl")
            if os.path.exists(vec_p) and os.path.exists(clf_p):
                self.tfidf_vec = joblib.load(vec_p)
                self.tfidf_clf = joblib.load(clf_p)
                with open(le_p, "rb") as f:
                    self.tfidf_le = pickle.load(f)
        except Exception:
            pass

        # Build label alignment map once
        self._build_label_map()

    def _build_label_map(self):
        """Align LSTM and TF-IDF label spaces for ensemble."""
        self.all_labels = sorted(set(self.label_encoder.classes_))
        if self.tfidf_le is not None:
            self.all_labels = sorted(
                set(self.label_encoder.classes_) | set(self.tfidf_le.classes_)
            )
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.all_labels)}
        self.n_aligned = len(self.all_labels)

    def _ensemble_probs(self, texts, lstm_probs):
        """Combine LSTM + TF-IDF probabilities."""
        if self.tfidf_vec is None or self.tfidf_clf is None:
            return lstm_probs, self.label_encoder

        w = self.ensemble_weight
        tfidf_probs = self.tfidf_clf.predict_proba(self.tfidf_vec.transform(texts))

        # Map both to aligned label space
        aligned = np.zeros((len(texts), self.n_aligned))
        for j, lbl in enumerate(self.label_encoder.classes_):
            aligned[:, self.label_to_idx[lbl]] += w * lstm_probs[:, j]
        for j, lbl in enumerate(self.tfidf_le.classes_):
            aligned[:, self.label_to_idx[lbl]] += (1 - w) * tfidf_probs[:, j]

        return aligned, None  # labels come from self.all_labels

    def predict_topk(
        self, texts: Union[str, List[str]], k: int = 5, rescale_topk: bool = True
    ) -> List[List[Tuple[str, float]]]:
        if isinstance(texts, str):
            inputs = [texts]
        else:
            inputs = list(texts)

        cleaned = [t if isinstance(t, str) else "" for t in inputs]

        # LSTM prediction
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        X = pad_sequences(seqs, maxlen=self.maxlen, padding="post", truncating="post")
        lstm_probs = self.model.predict(X, verbose=0)

        # Ensemble
        ens_probs, _ = self._ensemble_probs(cleaned, lstm_probs)

        outputs = []
        for row in ens_probs:
            top_idx = np.argsort(-row)[:k]
            pairs = []
            for i in top_idx:
                if i < self.n_aligned:
                    lbl = self.all_labels[i]
                else:
                    lbl = f"label_{i}"
                pairs.append((lbl, float(row[i])))

            if rescale_topk:
                total = sum(s for _, s in pairs)
                if total > 0:
                    pairs = [(l, s / total) for l, s in pairs]

            outputs.append(pairs)

        return outputs
