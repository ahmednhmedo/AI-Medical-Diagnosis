# ==============================================================
# 🩺 Semantic Enrichment + UMLS Linking + BioBERT Embeddings
# One-pass, optimized, resumable pipeline
#
# Collections (MongoDB: Medical_Diagnosis):
#   - Preprocessed_Illnesses (INPUT)
#   - Synonym_Expanded_Illnesses (OUTPUT)
#   - UMLS_Enriched_Illnesses (OUTPUT)
#   - Illness_Vectors_Refined (OUTPUT)
#
# Requirements (install once):
#   pip install -U spacy scispacy==0.5.1 tqdm pymongo sklearn sentence-transformers
#   # models:
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
#   # (or md/sm if needed)
# ==============================================================

import os, gc, pickle, json
from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient

# ---- NLP / Linking
import spacy
from scispacy.linking import EntityLinker
from spacy.language import Language

# ---- Embeddings
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Try sentence-transformers first (faster sentence pooling), then HF transformers
_USE_ST = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _USE_ST = False
    from transformers import AutoTokenizer, AutoModel


# ==============================================================
# 0) Config
# ==============================================================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "Medical_Diagnosis"

SRC_PREPRO   = "Preprocessed_Illnesses"
DST_SYNONYM  = "Synonym_Expanded_Illnesses"
DST_UMLS     = "UMLS_Enriched_Illnesses"
DST_VECTORS  = "Illness_Vectors_Refined"

CHECKPOINT_DIR = "./embedding_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Biomedical model preference order (largest → smallest)
SPACY_MODELS = ["en_core_sci_lg", "en_core_sci_md", "en_core_sci_sm"]

# Sentence-Transformer BioBERT (fast)
ST_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# HF Transformers fallback (mean pooling)
HF_MODEL = "dmis-lab/biobert-base-cased-v1.1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch sizes
UMLS_BATCH_SIZE = 16          # doc.text batches into nlp.pipe()
MONGO_WRITE_BATCH = 64        # insert_many batch size
EMBED_BATCH_MAX = 64          # only used for SentenceTransformer encode()


# ==============================================================
# 1) Mongo helpers
# ==============================================================

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


# ==============================================================
# 2) Synonym Expansion
# ==============================================================

medical_synonyms = {
    "heart attack": ["myocardial infarction", "cardiac arrest"],
    "flu": ["influenza", "viral infection"],
    "fever": ["high temperature", "pyrexia"],
    "stroke": ["cerebrovascular accident", "brain ischemia"],
    "diabetes": ["high blood sugar", "hyperglycemia"],
}

def expand_synonyms(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text
    low = t.lower()
    for key, syns in medical_synonyms.items():
        if key in low:
            # append all synonyms once
            t += " " + " ".join(syns)
    return t

def run_synonym_expansion() -> pd.DataFrame:
    src = db[SRC_PREPRO]
    df = pd.DataFrame(list(src.find({}, {"_id": 0})))
    print(f"✅ Loaded {len(df)} preprocessed records.")

    for col in ["symptoms", "causes", "warnings", "recommendations"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(expand_synonyms)
        else:
            df[col] = ""

    # Save
    db[DST_SYNONYM].delete_many({})
    if len(df) > 0:
        db[DST_SYNONYM].insert_many(df.to_dict(orient="records"))
    print(f"✅ Saved {len(df)} records to '{DST_SYNONYM}'.")
    return df


# ==============================================================
# 3) UMLS Linking with SciSpaCy (fast via nlp.pipe)
# ==============================================================

def load_spacy_umls_pipeline():
    nlp = None
    last_err = None
    for model in SPACY_MODELS:
        try:
            nlp = spacy.load(model)
            print(f"✅ Loaded spaCy model: {model}")
            break
        except Exception as e:
            last_err = e
            continue
    if nlp is None:
        raise RuntimeError(f"Unable to load any SciSpaCy model. Last error: {last_err}")

    # add UMLS linker once
    if "scispacy_linker" not in nlp.pipe_names:
        linker = EntityLinker(resolve_abbreviations=True, name="umls")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "name": "umls"})
        print("✅ UMLS linker added.")
    else:
        print("ℹ️ UMLS linker already present.")

    linker = nlp.get_pipe("scispacy_linker")
    print("🔧 Pipeline:", nlp.pipe_names)
    return nlp, linker

def extract_umls_from_docs(
    nlp, linker, docs: Iterable[spacy.tokens.Doc]
) -> List[List[Dict[str, Any]]]:
    """
    For each Doc, return list of linked entities with CUI/score/canonical_name.
    """
    all_entities: List[List[Dict[str, Any]]] = []
    for doc in docs:
        ents = []
        for ent in doc.ents:
            if hasattr(ent._, "umls_ents") and ent._.umls_ents:
                cui, score = ent._.umls_ents[0]
                concept = linker.umls.cui_to_entity.get(cui)
                ents.append({
                    "text": ent.text,
                    "cui": cui,
                    "concept_name": concept.canonical_name if concept else "",
                    "score": float(score),
                })
        all_entities.append(ents)
    return all_entities

def run_umls_linking():
    nlp, linker = load_spacy_umls_pipeline()
    src = db[DST_SYNONYM]
    dest = db[DST_UMLS]
    dest.delete_many({})

    cursor = list(src.find({}, {"_id": 0}))
    total = len(cursor)
    print(f"🔎 UMLS linking for {total} records...")

    # Prepare texts to pass through spaCy (symptoms-heavy for clinical relevance)
    def make_text(doc):
        return " ".join([
            str(doc.get("symptoms", "")),
            str(doc.get("causes", "")),
            str(doc.get("warnings", "")),
            str(doc.get("recommendations", "")),
        ]).strip()

    texts = [make_text(d) for d in cursor]

    batch = []
    count = 0

    # Process with nlp.pipe for speed
    for i in tqdm(range(0, total, UMLS_BATCH_SIZE)):
        docs = list(nlp.pipe(texts[i:i+UMLS_BATCH_SIZE]))
        ent_lists = extract_umls_from_docs(nlp, linker, docs)

        for j, ents in enumerate(ent_lists):
            doc = cursor[i + j]
            out = {
                "illness_name": doc.get("illness_name"),
                "umls_mappings": ents,            # list of {text, cui, concept_name, score}
                "scraped_at": doc.get("scraped_at")
            }
            batch.append(out)
            count += 1

        if len(batch) >= MONGO_WRITE_BATCH:
            dest.insert_many(batch)
            batch.clear()
            gc.collect()

    if batch:
        dest.insert_many(batch)

    print(f"✅ UMLS linking complete. Total processed: {count}")


# ==============================================================
# 4) BioBERT Embeddings (refined) + checkpoint + resume
# ==============================================================

class Embedder:
    """
    Wrapper that prefers SentenceTransformer BioBERT; falls back to HF Transformers.
    Normalizes output for cosine similarity.
    """
    def __init__(self):
        self.use_st = _USE_ST
        if self.use_st:
            try:
                self.model = SentenceTransformer(ST_MODEL, device=str(DEVICE))
                self.dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Loaded SentenceTransformer: {ST_MODEL} on {DEVICE}")
            except Exception as e:
                print(f"⚠️ ST model failed ({e}). Falling back to HF Transformers...")
                self.use_st = False

        if not self.use_st:
            from transformers import AutoTokenizer, AutoModel
            self.tok = AutoTokenizer.from_pretrained(HF_MODEL)
            self.model = AutoModel.from_pretrained(HF_MODEL).to(DEVICE).eval()
            self.dim = self.model.config.hidden_size
            print(f"✅ Loaded HF Transformers: {HF_MODEL} on {DEVICE}")

    def encode_many(self, texts: List[str]) -> np.ndarray:
        # Returns (N, D) normalized embeddings
        if self.use_st:
            vecs = self.model.encode(
                texts, show_progress_bar=False, batch_size=min(EMBED_BATCH_MAX, max(8, len(texts))),
                normalize_embeddings=True, convert_to_numpy=True
            )
            return vecs

        # HF fallback: mean-pooling last_hidden_state
        vecs = []
        from transformers import AutoTokenizer  # (already imported above if used)
        with torch.no_grad():
            for t in texts:
                if not isinstance(t, str) or not t.strip():
                    vecs.append(np.zeros(self.dim, dtype=np.float32))
                    continue
                inputs = self.tok(
                    t, return_tensors="pt", truncation=True, max_length=256, padding=True
                ).to(DEVICE)
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)[0]
                # L2 normalize
                norm = np.linalg.norm(emb) + 1e-9
                vecs.append(emb / norm)
        return np.vstack(vecs)

def load_checkpoint() -> set:
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("progress_")]
    if not files:
        return set()
    latest = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    with open(os.path.join(CHECKPOINT_DIR, latest), "rb") as f:
        done = pickle.load(f)
    return set(done)

def save_checkpoint(done: set):
    fname = os.path.join(CHECKPOINT_DIR, f"progress_{len(done)}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(sorted(list(done)), f)

def build_embeddings():
    src = db[DST_SYNONYM]      # use symptom text (most predictive)
    dest = db[DST_VECTORS]
    dest.delete_many({})

    rows = list(src.find({}, {"_id": 0}))
    print(f"🧬 Preparing embeddings for {len(rows)} illnesses...")
    if not rows:
        print("⚠️ No data found in Synonym_Expanded_Illnesses. Aborting.")
        return

    # Text builder (weight symptoms highest by repetition to bias the encoder)
    def text_for_embedding(doc):
        s = str(doc.get("symptoms", ""))
        c = str(doc.get("causes", ""))
        w = str(doc.get("warnings", ""))
        r = str(doc.get("recommendations", ""))
        return " ".join([(s + " ") * 3, (c + " ") * 2, w, r]).strip()

    names = [d["illness_name"] for d in rows]
    texts = [text_for_embedding(d) for d in rows]

    embedder = Embedder()

    done = load_checkpoint()
    print(f"▶️ Resuming embeddings. Already completed: {len(done)}")

    batch_docs, batch_texts = [], []
    written = 0

    for name, txt, doc in tqdm(zip(names, texts, rows), total=len(rows)):
        if name in done:
            continue
        batch_docs.append({
            "illness_name": name,
            "scraped_at": doc.get("scraped_at")
        })
        batch_texts.append(txt)

        if len(batch_texts) >= EMBED_BATCH_MAX:
            vecs = embedder.encode_many(batch_texts)
            payload = [{
                "illness_name": d["illness_name"],
                "embedding_vector": vecs[i].astype(np.float32).tolist(),
                "scraped_at": d["scraped_at"]
            } for i, d in enumerate(batch_docs)]

            dest.insert_many(payload)
            for d in batch_docs:
                done.add(d["illness_name"])
            save_checkpoint(done)

            written += len(payload)
            batch_docs, batch_texts = [], []
            gc.collect()

    # flush last
    if batch_texts:
        vecs = embedder.encode_many(batch_texts)
        payload = [{
            "illness_name": d["illness_name"],
            "embedding_vector": vecs[i].astype(np.float32).tolist(),
            "scraped_at": d["scraped_at"]
        } for i, d in enumerate(batch_docs)]
        dest.insert_many(payload)
        for d in batch_docs:
            done.add(d["illness_name"])
        save_checkpoint(done)
        written += len(payload)

    print(f"✅ Stored {written} refined embeddings into '{DST_VECTORS}'.")


# ==============================================================
# 5) Quick sanity test
# ==============================================================

def quick_similarity_test(query: str, top_k=5):
    vecs = list(db[DST_VECTORS].find({}, {"_id":0, "embedding_vector":1, "illness_name":1}))
    if not vecs:
        print("⚠️ No vectors found. Run embedding phase first.")
        return
    X = np.array([v["embedding_vector"] for v in vecs], dtype=np.float32)
    names = [v["illness_name"] for v in vecs]

    # Reuse Embedder for query encoding
    emb = Embedder().encode_many([query])[0].reshape(1, -1)
    sims = cosine_similarity(emb, X)[0]
    idx = sims.argsort()[-top_k:][::-1]

    print(f"\n🧩 Top {top_k} matches for: {query!r}\n")
    for i in idx:
        print(f"• {names[i]}  →  similarity {sims[i]:.3f}")


# ==============================================================
# 🚀 Main
# ==============================================================

if __name__ == "__main__":
    # 1) Synonym expansion (idempotent)
    run_synonym_expansion()

    # 2) UMLS linking (uses SciSpaCy large model if available)
    #    Skip if you want to avoid the ~1GB linker memory footprint.
    try:
        run_umls_linking()
    except Exception as e:
        print(f"⚠️ UMLS linking failed or skipped: {e}")

    # 3) Embeddings with BioBERT (resumable, normalized)
    build_embeddings()

    # 4) Sanity test
    quick_similarity_test("fever, cough, chest pain", top_k=5)
