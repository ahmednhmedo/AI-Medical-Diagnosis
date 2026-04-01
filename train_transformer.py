"""
train_transformer.py - Fine-tune BioBERT for multi-class symptom -> disease classification.

Optimised for CPU training:
  - Shorter MAX_LEN (128) since symptom queries are short
  - Freezes bottom BERT layers (only fine-tunes top 4 + classifier)
  - Progress bars with ETA so you know it's still running
  - Label smoothing for better generalisation

Usage
    python train_transformer.py            # train + save locally
    python train_transformer.py --mongo    # also upload to MongoDB GridFS
"""

import os
import sys
import json
import time
import pickle
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# -- Config ------------------------------------------------------------------

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MAX_LEN = 128                  # 128 is plenty for short symptom queries (was 256)
BATCH_SIZE = 16
EPOCHS = 4
LR = 3e-5                     # slightly higher LR since we freeze most layers
FREEZE_LAYERS = 6              # freeze bottom 6 of 12 BERT layers (speed + quality balance)
WARMUP_RATIO = 0.1
LABEL_SMOOTHING = 0.1          # helps with 291-class classification
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts_transformer")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "Medical_Diagnosis")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Data loading ------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    for coll_name in ["Augmented_Illnesses_Train", "Augmented_Illnesses"]:
        docs = list(db[coll_name].find({}, {"_id": 0}))
        if docs:
            print(f"[data] Loaded {len(docs)} records from {coll_name}")
            client.close()
            return pd.DataFrame(docs)

    docs = list(db["Synonym_Expanded_Illnesses"].find({}, {"_id": 0}))
    client.close()
    if not docs:
        raise RuntimeError("No training data found in MongoDB")
    print(f"[data] Fallback: loaded {len(docs)} records from Synonym_Expanded_Illnesses")
    df = pd.DataFrame(docs)
    df["text"] = df.apply(
        lambda r: " ".join(
            str(r.get(f, "")) for f in ["symptoms", "causes", "warnings", "recommendations"]
        ),
        axis=1,
    )
    return df


# -- Dataset -----------------------------------------------------------------

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -- Layer freezing ----------------------------------------------------------

def freeze_bert_layers(model, num_freeze=8):
    """Freeze the embedding layer and bottom N transformer layers.

    This dramatically speeds up CPU training because backprop only
    runs through the top (12 - num_freeze) layers + classifier head.
    """
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze bottom layers
    for i in range(num_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Frozen {num_freeze}/12 layers. Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.1f}%)")


# -- Training loop with progress ---------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    desc = f"Epoch {epoch}/{total_epochs} [train]"
    pbar = tqdm(loader, desc=desc, ncols=100, leave=True)

    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Label smoothing cross-entropy
        loss = F.cross_entropy(outputs.logits, labels, label_smoothing=LABEL_SMOOTHING)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, desc="[val]"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total = 0

    pbar = tqdm(loader, desc=desc, ncols=100, leave=True)
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item() * len(labels)
        total += len(labels)

        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, f1, np.array(all_preds), np.array(all_labels)


def topk_accuracy(model, loader, k=3):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            topk_preds = logits.topk(k, dim=-1).indices
            correct += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += len(labels)
    return correct / total


# -- Main --------------------------------------------------------------------

def main():
    upload_mongo = "--mongo" in sys.argv
    t0 = time.time()

    # 1. Load data
    df = load_training_data()

    if "text" not in df.columns:
        text_cols = ["symptoms", "causes", "warnings", "recommendations"]
        df["text"] = df.apply(
            lambda r: " ".join(str(r.get(c, "")) for c in text_cols), axis=1
        )

    label_col = "illness_name"
    df = df.dropna(subset=["text", label_col])
    df = df[df["text"].str.strip().astype(bool)]

    counts = Counter(df[label_col])
    valid = {k for k, v in counts.items() if v >= 2}
    df = df[df[label_col].isin(valid)].reset_index(drop=True)
    print(f"[data] {len(df)} samples, {len(valid)} classes")

    # 2. Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)
    texts = df["text"].tolist()

    # 3. Split - 1 sample per class for validation
    class_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    val_idx_set = set()
    train_idx_set = set()
    rng = np.random.RandomState(42)
    for lbl, idxs in class_indices.items():
        rng.shuffle(idxs)
        val_idx_set.add(idxs[0])
        train_idx_set.update(idxs[1:])

    train_idx = sorted(train_idx_set)
    val_idx = sorted(val_idx_set)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = labels[np.array(train_idx)]
    val_labels = labels[np.array(val_idx)]
    print(f"[split] train={len(train_texts)}, val={len(val_texts)}")

    # 4. Tokenizer & datasets
    print(f"\n[config] Device: {DEVICE}")
    print(f"[config] MAX_LEN={MAX_LEN}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}")
    print(f"[config] Freezing bottom {FREEZE_LAYERS}/12 BERT layers")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MedicalDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = MedicalDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Time estimate
    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * EPOCHS
    # Rough CPU estimate: ~0.8s/batch with frozen layers, ~2.5s without
    est_sec_per_batch = 1.2 if FREEZE_LAYERS > 0 else 2.5
    est_total_min = (total_batches * est_sec_per_batch) / 60
    print(f"\n[estimate] {batches_per_epoch} batches/epoch x {EPOCHS} epochs = {total_batches} total batches")
    print(f"[estimate] Estimated training time: ~{est_total_min:.0f} minutes on CPU")
    print(f"[estimate] Started at: {time.strftime('%H:%M:%S')}")
    est_end = time.localtime(time.time() + total_batches * est_sec_per_batch)
    print(f"[estimate] Expected finish: ~{time.strftime('%H:%M:%S', est_end)}")
    print()

    # 5. Model + freeze layers
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    ).to(DEVICE)
    freeze_bert_layers(model, FREEZE_LAYERS)

    # 6. Optimizer + scheduler (higher LR for classifier head)
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters()
                     if p.requires_grad and "classifier" not in n],
         "lr": LR, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                     if p.requires_grad and "classifier" in n],
         "lr": LR * 3, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 7. Train
    best_f1 = 0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch, EPOCHS)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  ")
        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (EPOCHS - epoch)

        print(
            f"  -> train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} | "
            f"time={epoch_time:.0f}s, ~{remaining:.0f}s remaining"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best model (f1={val_f1:.3f})")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # 8. Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation on validation set:")
    print(f"{'='*60}")
    val_loss, val_acc, val_f1, preds, golds = evaluate(model, val_loader, desc="Final eval")
    topk3 = topk_accuracy(model, val_loader, k=3)
    topk5 = topk_accuracy(model, val_loader, k=5)

    print(f"\n  Accuracy:       {val_acc:.3f}")
    print(f"  F1 (macro):     {val_f1:.3f}")
    print(f"  Top-3 accuracy: {topk3:.3f}")
    print(f"  Top-5 accuracy: {topk5:.3f}")

    total_time = time.time() - t0
    print(f"\n  Total training time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    # 9. Save artefacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model.save_pretrained(os.path.join(ARTIFACT_DIR, "model"))
    tokenizer.save_pretrained(os.path.join(ARTIFACT_DIR, "tokenizer"))

    with open(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    metrics = {
        "accuracy": round(val_acc, 4),
        "f1_macro": round(val_f1, 4),
        "top3_accuracy": round(topk3, 4),
        "top5_accuracy": round(topk5, 4),
        "epochs": EPOCHS,
        "num_classes": num_classes,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "training_time_min": round(total_time / 60, 1),
    }
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[save] Artefacts saved to {ARTIFACT_DIR}/")

    # 10. Optionally save to MongoDB GridFS
    if upload_mongo:
        from model_store import save_model_to_mongo
        save_model_to_mongo(
            artifact_dir=ARTIFACT_DIR,
            model_name="biobert_classifier_v1",
            model_type="Transformer (BioBERT)",
            labels=list(le.classes_),
            metrics=metrics,
        )
        print("[mongo] Model uploaded to MongoDB GridFS")


if __name__ == "__main__":
    main()
