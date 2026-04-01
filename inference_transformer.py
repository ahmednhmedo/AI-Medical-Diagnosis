"""
inference_transformer.py — Inference class for BioBERT transformer classifier.

Usage
    from inference_transformer import TransformerClassifier
    clf = TransformerClassifier()
    results = clf.predict_topk("fever, cough, chest pain", k=5)
"""

import os
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_transformer")
MAX_LEN = 256


class TransformerClassifier:
    def __init__(self, artifact_dir: str = ARTIFACT_DIR):
        model_dir = os.path.join(artifact_dir, "model")
        tok_dir = os.path.join(artifact_dir, "tokenizer")
        le_path = os.path.join(artifact_dir, "label_encoder.pkl")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Transformer model not found at {model_dir}. "
                "Run train_transformer.py first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        with open(le_path, "rb") as f:
            self.le = pickle.load(f)

    @torch.no_grad()
    def predict_topk(self, texts, k: int = 5, rescale_topk: bool = True):
        """Return top-k predictions for each input text.

        Parameters
            texts : str or list[str]
            k     : number of top predictions
            rescale_topk : normalise returned scores to sum to 1

        Returns
            list[list[tuple(label, score)]]
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        enc = self.tokenizer(
            texts,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for row in probs:
            top_idx = row.argsort()[-k:][::-1]
            pairs = [(self.le.classes_[i], float(row[i])) for i in top_idx]
            if rescale_topk:
                total = sum(s for _, s in pairs)
                if total > 0:
                    pairs = [(lbl, s / total) for lbl, s in pairs]
            results.append(pairs)

        return results if not single else results
