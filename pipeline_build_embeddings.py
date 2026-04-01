# ===========================================
# 📘 Phase 4.2 - Refine and Rebuild Illness Embeddings for API
# ===========================================
from pymongo import MongoClient
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# ===========================================
# 🔌 1. Connect to MongoDB
# ===========================================
client = MongoClient("mongodb://localhost:27017/")
db = client["Medical_Diagnosis"]

# Load from your synonym-expanded collection
df = pd.DataFrame(list(db["Synonym_Expanded_Illnesses"].find({}, {"_id": 0})))
print(f"✅ Loaded {len(df)} synonym-expanded records")

# ===========================================
# 🧠 2. Load medical embedding model (BioBERT/ClinicalBERT)
# ===========================================
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

def get_embedding(text: str):
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text, show_progress_bar=False)

# ===========================================
# ⚗️ 3. Weighted text concatenation
# ===========================================
def build_weighted_text(row):
    # Simple concatenation (no weights)
    return " ".join([
        str(row.get("symptoms", "")),
        str(row.get("causes", "")),
        str(row.get("warnings", "")),
        str(row.get("recommendations", ""))
    ])

# ===========================================
# 🧩 4. Create embeddings with normalization + deduplication
# ===========================================
records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    illness_name = row.get("illness_name", "").strip()
    if not illness_name:
        continue

    text = build_weighted_text(row)
    emb = get_embedding(text)
    emb = normalize(emb.reshape(1, -1))[0]  # normalize for cosine sim

    records.append({
        "illness_name": illness_name,
        "embedding_vector": emb.tolist(),
        "scraped_at": row.get("scraped_at")
    })

# Deduplicate illnesses by name (keep last occurrence)
unique_records = {r["illness_name"]: r for r in records}
records = list(unique_records.values())
print(f"✅ {len(records)} unique embeddings generated")

# ===========================================
# 💾 5. Save to MongoDB
# ===========================================
db["Illness_Vectors_Refined"].delete_many({})
db["Illness_Vectors_Refined"].insert_many(records)
print(f"✅ Saved refined embeddings to 'Illness_Vectors_Refined' collection")

# ===========================================
# 🔍 6. Quick Sanity Test
# ===========================================
vectors = np.array([r["embedding_vector"] for r in records])
names = [r["illness_name"] for r in records]

from sklearn.metrics.pairwise import cosine_similarity

def find_similar(symptom_text, top_k=5):
    q_vec = get_embedding(symptom_text).reshape(1, -1)
    sims = cosine_similarity(q_vec, vectors)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    print(f"\n🧩 Top {top_k} matches for: '{symptom_text}'\n")
    for i in top_idx:
        print(f"• {names[i]}  →  similarity {sims[i]:.3f}")

find_similar("fever, cough, chest pain")
