from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# =====================================================
# 🧠 CONFIGURATION
# =====================================================
SNOMED_CONCEPT_PATH = r"E:\Python\Medical Diagnosis AI\SNOMED CT\SNOMED CT\SnomedCT_InternationalRF2_PRODUCTION_20250601T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20250601.txt"
SNOMED_DESC_PATH = r"E:\Python\Medical Diagnosis AI\SNOMED CT\SNOMED CT\SnomedCT_InternationalRF2_PRODUCTION_20250601T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20250601.txt"
DB_NAME = "Medical_Diagnosis"

# =====================================================
# ⚙️ LOAD SNOMED DATA
# =====================================================
print("📥 Loading SNOMED concept and description data...")

desc_cols = [
    "id", "effectiveTime", "active", "moduleId", "conceptId",
    "languageCode", "typeId", "term", "caseSignificanceId"
]

# Load the descriptions (names/synonyms)
desc = pd.read_csv(SNOMED_DESC_PATH, sep="\t", usecols=desc_cols, dtype=str)

# Load only the concept IDs and active flag
concepts = pd.read_csv(SNOMED_CONCEPT_PATH, sep="\t", usecols=["id", "active"], dtype=str)

# Filter to active concepts/descriptions
desc = desc[desc["active"] == "1"]
concepts = concepts[concepts["active"] == "1"]

# Join descriptions with concepts
desc = desc.merge(concepts, left_on="conceptId", right_on="id", suffixes=("", "_concept"))
desc = desc[["conceptId", "term"]].drop_duplicates()

print(f"✅ Loaded {len(desc):,} active SNOMED English descriptions.")

# =====================================================
# ⚙️ LOAD UMLS concepts from MongoDB
# =====================================================
client = MongoClient("mongodb://localhost:27017/")
db = client[DB_NAME]

print("📥 Loading UMLS-enriched illnesses...")
umls_data = list(db["UMLS_Enriched_Illnesses"].find({}, {"_id": 0, "illness_name": 1, "umls_mappings": 1}))

records = []
for illness in umls_data:
    illness_name = illness.get("illness_name", "")
    for m in illness.get("umls_mappings", []):
        if m.get("concept_name") and m.get("cui"):
            records.append({
                "illness_name": illness_name,
                "cui": m["cui"],
                "umls_concept": m["concept_name"]
            })

df_umls = pd.DataFrame(records)
print(f"✅ Loaded {len(df_umls):,} UMLS-linked concepts from MongoDB.")

# =====================================================
# 🧠 BUILD TF-IDF MODEL
# =====================================================
print("🔍 Building TF-IDF vectorizer for SNOMED terms...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
snomed_vectors = vectorizer.fit_transform(desc["term"].astype(str))

# Function to find best match in SNOMED
def find_best_snomed_match(umls_term):
    if not isinstance(umls_term, str) or not umls_term.strip():
        return None
    query_vec = vectorizer.transform([umls_term])
    sims = cosine_similarity(query_vec, snomed_vectors).flatten()
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]
    best_row = desc.iloc[best_idx]
    return {
        "snomed_id": best_row["conceptId"],
        "snomed_term": best_row["term"],
        "similarity": float(best_score)
    }

# =====================================================
# 🔗 MAP UMLS → SNOMED
# =====================================================
mapped = []
for _, row in tqdm(df_umls.iterrows(), total=len(df_umls), desc="🔗 Mapping"):
    try:
        match = find_best_snomed_match(row["umls_concept"])
        if match:
            mapped.append({
                "illness_name": row["illness_name"],
                "cui": row["cui"],
                "umls_concept": row["umls_concept"],
                **match
            })
    except Exception as e:
        continue

df_mapped = pd.DataFrame(mapped)
print(f"✅ Mapped {len(df_mapped):,} UMLS concepts to SNOMED terms.")

# =====================================================
# 💾 SAVE TO MONGODB
# =====================================================
dest = db["UMLS_SNOMED_Linked"]
dest.delete_many({})
dest.insert_many(df_mapped.to_dict(orient="records"))

print(f"✅ Saved {len(df_mapped):,} SNOMED-linked records to 'UMLS_SNOMED_Linked'.")
print("\n🎯 Example link:")
print(df_mapped.head(5).to_string(index=False))
