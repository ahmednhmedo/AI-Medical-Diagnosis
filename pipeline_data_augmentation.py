"""
05b_data_augmentation_train_only.py
Augment ONLY the training originals.

Input  : MongoDB -> Train_Illnesses_Base (originals only)
Output : MongoDB -> Augmented_Illnesses_Train (original + paraphrases as 'text')
"""

import re
import random
from typing import List
from pymongo import MongoClient

# HF paraphraser (CPU-friendly; falls back to lightweight heuristics if HF load fails)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MONGO_URI   = "mongodb://localhost:27017/"
DB_NAME     = "Medical_Diagnosis"
SRC_COLL    = "Train_Illnesses_Base"
DEST_COLL   = "Augmented_Illnesses_Train"

PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"
NUM_PARAPHRASES  = 3
MAX_LEN          = 256
random.seed(42)

def to_plain_text(d):
    parts = [str(d.get(k, "")).strip() for k in ["illness_name", "symptoms", "causes", "warnings", "recommendations"]]
    txt = " ".join([p for p in parts if p])
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def light_heuristic_augs(text: str) -> List[str]:
    """Cheap, deterministic noise when HF model cannot load."""
    outs = set()
    # clause shuffle (split on commas/semicolons)
    clauses = re.split(r"[;,.]", text)
    clauses = [c.strip() for c in clauses if c.strip()]
    if len(clauses) >= 3:
        shuffled = clauses[:]
        random.shuffle(shuffled)
        outs.add(re.sub(r"\s+", " ", ". ".join(shuffled)).strip())

    # simple synonym-ish swaps (very light)
    swaps = [
        ("fever", "high temperature"),
        ("cough", "coughing"),
        ("pain", "discomfort"),
        ("shortness of breath", "breathlessness"),
        ("nausea", "feeling sick"),
    ]
    t = text
    for a, b in swaps:
        t = re.sub(fr"\b{re.escape(a)}\b", b, t, flags=re.I)
    if t != text:
        outs.add(t)

    # small adjective insertion
    outs.add(re.sub(r"\bsevere\b", "very severe", text, flags=re.I))

    return [o for o in outs if o and o != text]

def load_paraphraser():
    try:
        tok = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL)
        print("✅ Paraphraser model loaded.")
        return tok, mdl
    except Exception as e:
        print(f"⚠️ Could not load paraphraser: {e}\n   → Will use lightweight heuristic augmentation.")
        return None, None

def paraphrase_batch(tokenizer, model, text: str, n: int) -> List[str]:
    if tokenizer is None or model is None:
        return light_heuristic_augs(text)[:n]

    prompt = f"paraphrase: {text} </s>"
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    # generate n variations with diverse beam search sampling
    outputs = model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=4,
        num_return_sequences=max(1, n),
        do_sample=True,
        top_p=0.92,
        top_k=50
    )
    outs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    outs = [re.sub(r"\s+", " ", o).strip() for o in outs]
    outs = [o for o in outs if o and o.lower() != text.lower()]
    # dedup
    seen, unique = set(), []
    for o in outs:
        if o not in seen:
            seen.add(o)
            unique.append(o)
    return unique[:n]

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    src  = db[SRC_COLL]
    dest = db[DEST_COLL]

    docs = list(src.find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError(f"No data in '{SRC_COLL}'. Run 05a first.")

    # wipe destination
    dest.delete_many({})

    tokenizer, model = load_paraphraser()

    total_written = 0
    for d in docs:
        base_text = to_plain_text(d)
        if not base_text:
            continue

        # include the original
        records = [{
            "illness_name": d["illness_name"],
            "text": base_text,
            "source": "base",
            "variant_id": 0
        }]

        # add paraphrases
        paras = paraphrase_batch(tokenizer, model, base_text, NUM_PARAPHRASES)
        for i, p in enumerate(paras, start=1):
            records.append({
                "illness_name": d["illness_name"],
                "text": p,
                "source": "paraphrase",
                "variant_id": i
            })

        dest.insert_many(records)
        total_written += len(records)

    print(f"✅ Wrote {total_written} train records to '{DEST_COLL}' "
          f"(originals + up to {NUM_PARAPHRASES} paraphrases each).")

if __name__ == "__main__":
    main()
