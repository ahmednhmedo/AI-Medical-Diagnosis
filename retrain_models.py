"""
retrain_models.py  -  Retrain TF-IDF baseline and hybrid LSTM model.

Strategy:
  - TF-IDF:  word+char ngrams + LogisticRegression (strong baseline)
  - LSTM:    trains on same data distribution as inference (short queries)
             + uses label-smoothing + deeper augmentation from symptoms+causes
  - Ensemble at inference: LSTM predictions weighted-averaged with TF-IDF
"""

import os
import re
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
import joblib

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "Medical_Diagnosis")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# -- helpers ------------------------------------------------------------------

def clean(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s,.\-]", " ", text)
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


STOPWORDS = {
    "also", "may", "include", "usually", "often", "sometimes", "however",
    "common", "called", "known", "people", "condition", "treatment",
    "doctor", "gp", "nhs", "visit", "see", "get", "help", "go", "can",
    "make", "well", "might", "could", "would", "one", "two", "first",
    "use", "used", "using", "need", "able", "much", "many", "way",
    "take", "keep", "things", "feel", "find", "time", "day", "week",
}


def medical_tokens(text):
    return [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]


# -- data loading -------------------------------------------------------------

def load_all_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    symptom_docs = list(db["Synonym_Expanded_Illnesses"].find({}, {"_id": 0}))
    aug_docs = list(db["Augmented_Illnesses_Train"].find({}, {"_id": 0}))
    client.close()
    print(f"Symptom records: {len(symptom_docs)}, Augmented records: {len(aug_docs)}")
    return symptom_docs, aug_docs


# -- augmentation -------------------------------------------------------------

def _safe_sample(tokens, k):
    k = min(k, len(tokens))
    return random.sample(tokens, k) if k > 0 else tokens[:1]


def make_variants(symptom_text, cause_text="", n=15):
    """Generate diverse short query variants combining symptoms and causes."""
    sym_tokens = medical_tokens(symptom_text)
    cause_tokens = medical_tokens(cause_text)
    all_tokens = sym_tokens + cause_tokens
    variants = []

    if len(sym_tokens) < 2 and len(all_tokens) < 2:
        return [symptom_text] if symptom_text.strip() else []

    # Full symptom text
    if sym_tokens:
        variants.append(" ".join(sym_tokens[:20]))

    # Contiguous spans from symptoms (preserves "chest pain", "sore throat")
    for _ in range(3):
        src = sym_tokens if sym_tokens else all_tokens
        span = min(random.randint(3, 10), len(src))
        start = random.randint(0, max(0, len(src) - span))
        variants.append(" ".join(src[start:start + span]))

    # Comma-separated keyword subsets (how users actually type)
    for _ in range(4):
        src = sym_tokens if sym_tokens else all_tokens
        lo, hi = min(2, len(src)), min(7, len(src))
        k = random.randint(lo, hi) if lo < hi else lo
        variants.append(", ".join(_safe_sample(src, k)))

    # Short 2-3 keyword queries (hardest case)
    for _ in range(3):
        src = sym_tokens if sym_tokens else all_tokens
        lo, hi = min(2, len(src)), min(3, len(src))
        k = random.randint(lo, hi) if lo < hi else lo
        variants.append(", ".join(_safe_sample(src, k)))

    # Mixed symptom+cause queries (real users describe both)
    if sym_tokens and cause_tokens:
        for _ in range(2):
            n_sym = random.randint(1, min(3, len(sym_tokens)))
            n_cause = random.randint(1, min(2, len(cause_tokens)))
            mixed = _safe_sample(sym_tokens, n_sym) + _safe_sample(cause_tokens, n_cause)
            random.shuffle(mixed)
            variants.append(", ".join(mixed))

    # Shuffled tokens (different ordering of same symptoms)
    if len(sym_tokens) >= 3:
        shuffled = sym_tokens.copy()
        random.shuffle(shuffled)
        variants.append(" ".join(shuffled[:8]))

    return variants[:n]


def build_training_data(symptom_docs, aug_docs):
    """Build combined training dataset for TF-IDF."""
    texts, labels = [], []

    for doc in symptom_docs:
        name = doc.get("illness_name", "").strip()
        if not name:
            continue
        symptoms = clean(doc.get("symptoms", ""))
        causes = clean(doc.get("causes", ""))
        if not symptoms and not causes:
            continue

        variants = make_variants(symptoms, causes, n=15)
        texts.extend(variants)
        labels.extend([name] * len(variants))

    # Add augmented paraphrased data (trimmed)
    for doc in aug_docs:
        name = doc.get("illness_name", "").strip()
        text = doc.get("text", "").strip()
        if name and text:
            tokens = text.split()[:100]
            texts.append(" ".join(tokens))
            labels.append(name)

    return texts, labels


def build_eval_queries(symptom_docs):
    """Deterministic eval queries per illness."""
    queries, true_labels = [], []
    for doc in symptom_docs:
        name = doc.get("illness_name", "").strip()
        symptoms = clean(doc.get("symptoms", ""))
        if not name or not symptoms:
            continue
        tokens = medical_tokens(symptoms)
        if len(tokens) < 3:
            continue
        random.seed(hash(name) + 999)
        k = random.randint(2, min(4, len(tokens)))
        queries.append(", ".join(random.sample(tokens, k)))
        true_labels.append(name)
    random.seed(SEED)
    return queries, true_labels


# -- evaluation ---------------------------------------------------------------

def eval_topk(probs, y_true, k_values=[1, 3, 5]):
    results = {}
    for k in k_values:
        correct = sum(
            1 for i in range(len(y_true))
            if y_true[i] in probs[i].argsort()[-k:]
        )
        results[f"acc@{k}"] = correct / len(y_true)
    return results


# -- TF-IDF training ----------------------------------------------------------

def train_tfidf(texts, labels, eval_queries, eval_labels):
    print("\n" + "=" * 60)
    print("  TRAINING TF-IDF BASELINE")
    print("=" * 60)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Samples: {len(texts)}, Classes: {num_classes}")

    word_vec = TfidfVectorizer(
        max_features=60000, min_df=2, sublinear_tf=True,
        ngram_range=(1, 2), analyzer="word",
    )
    char_vec = TfidfVectorizer(
        max_features=40000, min_df=3, sublinear_tf=True,
        ngram_range=(3, 5), analyzer="char_wb",
    )
    vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])
    X = vectorizer.fit_transform(texts)
    print(f"Features: {X.shape[1]}")

    clf = LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced",
        solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X, y)

    # Evaluate
    if eval_queries:
        X_eval = vectorizer.transform(eval_queries)
        probs_eval = clf.predict_proba(X_eval)
        eval_y = le.transform(eval_labels)
        m = eval_topk(probs_eval, eval_y)
        f1 = f1_score(eval_y, probs_eval.argmax(axis=1), average="macro", zero_division=0)
        print(f"Eval: Acc@1={m['acc@1']:.3f}  Acc@3={m['acc@3']:.3f}  "
              f"Acc@5={m['acc@5']:.3f}  F1={f1:.3f}")

    # Save
    out_dir = os.path.join(BASE_DIR, "artifacts_baseline")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "clf.pkl"))
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"num_classes": num_classes, "train_samples": len(texts)}, f)
    print(f"Saved to {out_dir}/")

    _show_predictions(lambda q: clf.predict_proba(vectorizer.transform([q]))[0], le)
    return vectorizer, clf, le


# -- LSTM training (with TF-IDF ensemble) -------------------------------------

def train_lstm(texts, labels, eval_queries, eval_labels, symptom_docs=None,
               tfidf_vec=None, tfidf_clf=None, tfidf_le=None):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, Bidirectional, LSTM, GlobalMaxPooling1D,
        Dense, Dropout, SpatialDropout1D, BatchNormalization, Concatenate,
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("  TRAINING LSTM (symptom-focused + deeper architecture)")
    print("=" * 60)

    # Build LSTM training data from symptom docs only (short queries)
    lstm_texts, lstm_labels = [], []
    for doc in symptom_docs:
        name = doc.get("illness_name", "").strip()
        if not name:
            continue
        symptoms = clean(doc.get("symptoms", ""))
        causes = clean(doc.get("causes", ""))
        if not symptoms and not causes:
            continue
        variants = make_variants(symptoms, causes, n=20)
        lstm_texts.extend(variants)
        lstm_labels.extend([name] * len(variants))

    le = LabelEncoder()
    y = le.fit_transform(lstm_labels)
    num_classes = len(le.classes_)
    print(f"Samples: {len(lstm_texts)}, Classes: {num_classes}")

    VOCAB_SIZE = 25000
    tokenizer = KerasTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(lstm_texts)
    actual_vocab = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    seqs = tokenizer.texts_to_sequences(lstm_texts)
    MAXLEN = min(int(np.percentile([len(s) for s in seqs], 97)), 60)
    print(f"Vocab: {actual_vocab}, MaxLen: {MAXLEN}")

    X = pad_sequences(seqs, maxlen=MAXLEN, padding="post", truncating="post")

    X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
        X, y, lstm_texts, test_size=0.15, random_state=SEED, stratify=y,
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), cw))

    # Architecture: deeper LSTM with residual-style dense layers
    inp = Input(shape=(MAXLEN,))
    x = Embedding(actual_vocab, 128)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )

    # ---- Evaluate standalone LSTM ----
    val_probs = model.predict(X_val, verbose=0)
    vm = eval_topk(val_probs, y_val)
    vf1 = f1_score(y_val, val_probs.argmax(axis=1), average="macro", zero_division=0)
    print(f"\nLSTM Val: Acc@1={vm['acc@1']:.3f}  Acc@3={vm['acc@3']:.3f}  "
          f"Acc@5={vm['acc@5']:.3f}  F1={vf1:.3f}")

    # ---- Find best ensemble weight on validation set ----
    best_w, best_acc = 0.5, 0
    if tfidf_vec is not None and tfidf_clf is not None:
        tfidf_val_probs = tfidf_clf.predict_proba(tfidf_vec.transform(texts_val))
        # Align label spaces (both should have same le)
        tfidf_label_map = {lbl: i for i, lbl in enumerate(tfidf_le.classes_)}
        lstm_label_map = {lbl: i for i, lbl in enumerate(le.classes_)}

        # Build aligned probability matrices
        all_labels = sorted(set(le.classes_) | set(tfidf_le.classes_))
        n_aligned = len(all_labels)
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        # Map LSTM probs to aligned space
        lstm_aligned = np.zeros((len(y_val), n_aligned))
        for j, lbl in enumerate(le.classes_):
            if lbl in label_to_idx:
                lstm_aligned[:, label_to_idx[lbl]] = val_probs[:, j]

        # Map TF-IDF probs to aligned space
        tfidf_aligned = np.zeros((len(y_val), n_aligned))
        for j, lbl in enumerate(tfidf_le.classes_):
            if lbl in label_to_idx:
                tfidf_aligned[:, label_to_idx[lbl]] = tfidf_val_probs[:, j]

        # Map true labels to aligned space
        y_val_aligned = np.array([label_to_idx[le.classes_[yi]] for yi in y_val])

        for w in np.arange(0.1, 0.9, 0.05):
            ens = w * lstm_aligned + (1 - w) * tfidf_aligned
            acc3 = sum(
                1 for i in range(len(y_val_aligned))
                if y_val_aligned[i] in ens[i].argsort()[-3:]
            ) / len(y_val_aligned)
            if acc3 > best_acc:
                best_acc = acc3
                best_w = round(w, 2)

        ens_probs = best_w * lstm_aligned + (1 - best_w) * tfidf_aligned
        em = eval_topk(ens_probs, y_val_aligned)
        print(f"Ensemble (w_lstm={best_w}): Acc@1={em['acc@1']:.3f}  "
              f"Acc@3={em['acc@3']:.3f}  Acc@5={em['acc@5']:.3f}")

    # ---- Evaluate on eval queries ----
    if eval_queries:
        eval_seqs = tokenizer.texts_to_sequences(eval_queries)
        X_eval = pad_sequences(eval_seqs, maxlen=MAXLEN, padding="post", truncating="post")
        eval_y = le.transform(eval_labels)
        eval_probs = model.predict(X_eval, verbose=0)
        em = eval_topk(eval_probs, eval_y)
        ef1 = f1_score(eval_y, eval_probs.argmax(axis=1), average="macro", zero_division=0)
        print(f"LSTM Eval: Acc@1={em['acc@1']:.3f}  Acc@3={em['acc@3']:.3f}  "
              f"Acc@5={em['acc@5']:.3f}  F1={ef1:.3f}")

        # Ensemble eval
        if tfidf_vec is not None:
            tfidf_eval_probs = tfidf_clf.predict_proba(tfidf_vec.transform(eval_queries))
            lstm_eval_al = np.zeros((len(eval_y), n_aligned))
            for j, lbl in enumerate(le.classes_):
                if lbl in label_to_idx:
                    lstm_eval_al[:, label_to_idx[lbl]] = eval_probs[:, j]
            tfidf_eval_al = np.zeros((len(eval_y), n_aligned))
            for j, lbl in enumerate(tfidf_le.classes_):
                if lbl in label_to_idx:
                    tfidf_eval_al[:, label_to_idx[lbl]] = tfidf_eval_probs[:, j]
            eval_y_al = np.array([label_to_idx[le.classes_[yi]] for yi in eval_y])
            ens_eval = best_w * lstm_eval_al + (1 - best_w) * tfidf_eval_al
            eem = eval_topk(ens_eval, eval_y_al)
            print(f"Ensemble Eval: Acc@1={eem['acc@1']:.3f}  Acc@3={eem['acc@3']:.3f}  "
                  f"Acc@5={eem['acc@5']:.3f}")

    # ---- Save ----
    out_dir = os.path.join(BASE_DIR, "artifacts_lstm")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    config = {
        "maxlen": MAXLEN, "vocab_size": actual_vocab,
        "num_classes": num_classes, "ensemble_weight": best_w,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved to {out_dir}/ (ensemble_weight={best_w})")

    _show_predictions(
        lambda q: model.predict(
            pad_sequences(tokenizer.texts_to_sequences([q]), maxlen=MAXLEN,
                          padding="post", truncating="post"), verbose=0
        )[0], le
    )


# -- display ------------------------------------------------------------------

SAMPLE_QUERIES = [
    "fever, headache, sore throat",
    "chest pain, shortness of breath",
    "stomach pain, nausea, vomiting",
    "skin rash, itching",
    "cough, runny nose, sneezing",
    "joint pain, swelling, stiffness",
    "dizziness, fainting, blurred vision",
    "difficulty breathing, wheezing",
]

def _show_predictions(predict_fn, le):
    print("\nSample predictions:")
    for q in SAMPLE_QUERIES:
        probs = predict_fn(q)
        top3 = probs.argsort()[-3:][::-1]
        print(f"  '{q}'")
        for i in top3:
            print(f"     {float(probs[i]):.3f}  {le.classes_[i]}")


# -- main ---------------------------------------------------------------------

if __name__ == "__main__":
    symptom_docs, aug_docs = load_all_data()
    texts, labels = build_training_data(symptom_docs, aug_docs)
    eval_queries, eval_labels = build_eval_queries(symptom_docs)

    print(f"TF-IDF data: {len(texts)} samples, {len(set(labels))} classes")
    print(f"Eval: {len(eval_queries)} queries")

    vec, clf, le = train_tfidf(texts, labels, eval_queries, eval_labels)
    train_lstm(texts, labels, eval_queries, eval_labels,
               symptom_docs=symptom_docs,
               tfidf_vec=vec, tfidf_clf=clf, tfidf_le=le)

    print("\n" + "=" * 60)
    print("  DONE -- Both models retrained")
    print("  Restart Flask app: python app.py")
    print("=" * 60)
