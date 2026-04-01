"""
inference_baseline.py
Lightweight loader + predictor for the TF-IDF baseline.
Returns: List[List[ (label, score_float) ]]
"""
import os
from typing import List, Tuple, Union
import joblib
import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_baseline")

class BaselineClassifier:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.vectorizer = None
        self.clf = None
        self.label_encoder = None
        self._load()

    def _load(self):
        vec_p = os.path.join(self.artifacts_dir, "vectorizer.pkl")
        clf_p = os.path.join(self.artifacts_dir, "clf.pkl")
        le_p  = os.path.join(self.artifacts_dir, "label_encoder.pkl")

        if not (os.path.exists(vec_p) and os.path.exists(clf_p) and os.path.exists(le_p)):
            raise FileNotFoundError(
                f"Artifacts not found in {self.artifacts_dir}. "
                "Expected vectorizer.pkl, clf.pkl, label_encoder.pkl"
            )

        self.vectorizer = joblib.load(vec_p)
        self.clf        = joblib.load(clf_p)
        self.label_encoder = joblib.load(le_p)

    def predict_topk(
        self,
        texts: Union[str, List[str]],
        k: int = 5,
        rescale_topk: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Accepts either a single string or a list of strings.
        Returns for each input a list of (label, score) tuples sorted by score desc.
        If rescale_topk=True, the returned scores for the top-k are normalized to sum=1
        (makes them easier to read when there are many classes).
        """
        # Normalize input to list
        if isinstance(texts, str):
            inputs = [texts]
        elif isinstance(texts, list):
            inputs = texts
        else:
            raise ValueError("texts must be a string or list of strings.")

        outputs: List[List[Tuple[str, float]]] = []

        X = self.vectorizer.transform([t if isinstance(t, str) else "" for t in inputs])
        probs_all = self.clf.predict_proba(X)  # shape (n_inputs, n_classes)

        for probs in probs_all:
            # top-k indices
            idx = np.argsort(-probs)[:k]
            labels = self.label_encoder.inverse_transform(idx)
            scores = probs[idx].astype(float)

            if rescale_topk:
                ssum = float(scores.sum())
                if ssum > 0:
                    scores = scores / ssum

            # build list of tuples (label, score)
            out = [(str(lbl), float(score)) for lbl, score in zip(labels, scores)]
            outputs.append(out)

        return outputs
