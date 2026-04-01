"""
build_snomed_map.py  -  Map each illness to its correct SNOMED CT code.

Strategy:
  1. Try exact match (illness name == SNOMED synonym term)
  2. Try normalized match (lowercase, strip suffixes)
  3. Try fuzzy substring match with scoring
  4. Store verified mapping in MongoDB 'Illness_SNOMED_Codes' collection
"""

import os
import re
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "Medical_Diagnosis")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SNOMED CT description files
SNOMED_PATHS = [
    os.path.join(BASE_DIR, "SNOMED CT", "SNOMED CT",
                 "SnomedCT_InternationalRF2_PRODUCTION_20250601T120000Z",
                 "Snapshot", "Terminology",
                 "sct2_Description_Snapshot-en_INT_20250601.txt"),
    os.path.join(BASE_DIR, "SNOMED CT", "SNOMED CT",
                 "SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20250301T120000Z",
                 "Snapshot", "Terminology",
                 "sct2_Description_Snapshot-en_US1000124_20250301.txt"),
]

# Manual overrides for illnesses where fuzzy matching fails
MANUAL_OVERRIDES = {
    "Alcohol-related liver disease": ("41309000", "Alcoholic liver disease"),
    "Middle ear infection (otitis media)": ("65363002", "Otitis media"),
    "Breathing problems in children": ("267036007", "Dyspnea"),
    "Allergies": ("421961002", "Allergy"),
    "Overactive thyroid": ("34486009", "Hyperthyroidism"),
    "Dementia with Lewy bodies": ("312991009", "Dementia with Lewy bodies"),
    "Gallbladder cancer": ("363358000", "Cancer of gallbladder"),
    "Nasopharyngeal cancer": ("91742005", "Nasopharyngeal carcinoma"),
    "Subacromial pain syndrome": ("239962005", "Subacromial bursitis"),
    "Nasal and sinus cancer": ("254590004", "Cancer of nasal sinus"),
    "Type 1 diabetes": ("46635009", "Type 1 diabetes mellitus"),
    "Type 2 diabetes": ("44054006", "Type 2 diabetes mellitus"),
    "If your child has cold or flu symptoms": ("6142004", "Influenza"),
    "Bowel polyps": ("68496003", "Polyp of intestine"),
    "Golfers elbow": ("53286005", "Medial epicondylitis"),
    "Mechanical neck pain": ("209560008", "Cervicalgia"),
    "Phobias": ("386810004", "Phobic disorder"),
    "Brain tumours: Teenagers and young adults": ("126952004", "Brain tumour"),
    "PIMS": ("1119302008", "Pediatric inflammatory multisystem syndrome"),
    "Testicular cancer": ("363449006", "Cancer of testis"),
    "Liver tumours": ("126851005", "Tumour of liver"),
    "Penile cancer": ("363450006", "Cancer of penis"),
}


def normalize(text):
    """Normalize illness name for matching."""
    text = text.lower().strip()
    # Remove parenthetical abbreviations: "Whooping cough (pertussis)" -> "whooping cough"
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    # Remove common suffixes
    for suffix in [": children", ": teenagers and young adults", ": adults"]:
        text = text.replace(suffix, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_snomed_terms():
    """Load all active SNOMED CT synonym terms."""
    frames = []
    for path in SNOMED_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", dtype=str)
            df = df[df["active"] == "1"]
            frames.append(df[["conceptId", "term", "typeId"]])
            print(f"Loaded {len(df)} descriptions from {os.path.basename(path)}")

    if not frames:
        raise FileNotFoundError("No SNOMED CT description files found")

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate
    combined = combined.drop_duplicates(subset=["conceptId", "term"])
    combined = combined.dropna(subset=["term"])
    combined = combined[combined["term"].str.strip() != ""]
    print(f"Total unique SNOMED terms: {len(combined)}")
    return combined


def build_mapping():
    """Map each illness to its SNOMED code."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Load illness names
    illnesses = list(db["Conditions"].find({}, {"_id": 0, "condition": 1}))
    if not illnesses:
        illnesses = list(db["Preprocessed_Illnesses"].find({}, {"_id": 0, "illness_name": 1}))
    illness_names = [d.get("condition") or d.get("illness_name", "") for d in illnesses]
    illness_names = [n.strip() for n in illness_names if n.strip()]
    print(f"\nIllnesses to map: {len(illness_names)}")

    # Load SNOMED terms
    snomed_df = load_snomed_terms()

    # Build lookup: normalized term -> (conceptId, original term)
    # Prefer synonyms (typeId 900000000000013009) over FSN
    synonyms = snomed_df[snomed_df["typeId"] == "900000000000013009"]
    fsn = snomed_df[snomed_df["typeId"] == "900000000000003001"]

    # Build term -> concept lookup
    term_lookup = {}
    for _, row in synonyms.iterrows():
        key = row["term"].lower().strip()
        term_lookup[key] = (row["conceptId"], row["term"])
    for _, row in fsn.iterrows():
        key = row["term"].lower().strip()
        # Remove semantic tag from FSN: "Asthma (disorder)" -> "asthma"
        clean_key = re.sub(r"\s*\([^)]*\)\s*$", "", key).strip()
        if clean_key not in term_lookup:
            term_lookup[clean_key] = (row["conceptId"], row["term"])

    # TF-IDF fallback for fuzzy matching
    all_snomed_terms = list(set(
        [row["term"] for _, row in synonyms.iterrows()]
    ))
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=100000, sublinear_tf=True,
    )
    snomed_matrix = tfidf.fit_transform(all_snomed_terms)
    snomed_term_to_concept = {}
    for _, row in synonyms.iterrows():
        snomed_term_to_concept[row["term"]] = row["conceptId"]
    for _, row in fsn.iterrows():
        clean = re.sub(r"\s*\([^)]*\)\s*$", "", row["term"]).strip()
        snomed_term_to_concept[clean] = row["conceptId"]

    results = []
    matched = 0
    fuzzy_matched = 0

    for illness in illness_names:
        norm = normalize(illness)
        record = {
            "illness_name": illness,
            "snomed_code": None,
            "snomed_term": None,
            "match_type": "none",
            "confidence": 0.0,
        }

        # 0. Manual override (highest priority)
        if illness in MANUAL_OVERRIDES:
            cid, term = MANUAL_OVERRIDES[illness]
            record.update({
                "snomed_code": cid, "snomed_term": term,
                "match_type": "manual", "confidence": 1.0,
            })
            matched += 1
            results.append(record)
            continue

        # 1. Exact match
        if norm in term_lookup:
            cid, term = term_lookup[norm]
            record.update({
                "snomed_code": cid, "snomed_term": term,
                "match_type": "exact", "confidence": 1.0,
            })
            matched += 1
            results.append(record)
            continue

        # 2. Try common name variants
        found = False
        variants = [
            illness.lower(),
            norm,
            norm.replace("-", " "),
            norm.replace("'s", ""),
            re.sub(r"\s*(disease|syndrome|disorder)\s*$", "", norm).strip(),
        ]
        for v in variants:
            if v in term_lookup:
                cid, term = term_lookup[v]
                record.update({
                    "snomed_code": cid, "snomed_term": term,
                    "match_type": "variant", "confidence": 0.95,
                })
                matched += 1
                found = True
                break
        if found:
            results.append(record)
            continue

        # 3. Fuzzy TF-IDF match
        query_vec = tfidf.transform([illness])
        sims = cosine_similarity(query_vec, snomed_matrix)[0]
        best_idx = sims.argmax()
        best_score = float(sims[best_idx])
        best_term = all_snomed_terms[best_idx]

        if best_score >= 0.5:
            cid = snomed_term_to_concept.get(best_term, "")
            record.update({
                "snomed_code": cid, "snomed_term": best_term,
                "match_type": "fuzzy", "confidence": round(best_score, 3),
            })
            fuzzy_matched += 1

        results.append(record)

    # Store in MongoDB
    db["Illness_SNOMED_Codes"].drop()
    if results:
        db["Illness_SNOMED_Codes"].insert_many(results)

    print(f"\nResults:")
    print(f"  Exact/variant matches: {matched}/{len(illness_names)}")
    print(f"  Fuzzy matches: {fuzzy_matched}/{len(illness_names)}")
    print(f"  No match: {len(illness_names) - matched - fuzzy_matched}/{len(illness_names)}")

    # Show some examples
    print("\nSample mappings:")
    for r in results[:15]:
        status = "OK" if r["snomed_code"] else "MISS"
        print(f"  [{status}] {r['illness_name']}")
        if r["snomed_code"]:
            print(f"       SNOMED: {r['snomed_code']} | {r['snomed_term']} ({r['match_type']}, {r['confidence']})")

    # Show misses
    misses = [r for r in results if not r["snomed_code"]]
    if misses:
        print(f"\nUnmatched ({len(misses)}):")
        for r in misses[:20]:
            print(f"  - {r['illness_name']}")

    client.close()
    return results


if __name__ == "__main__":
    build_mapping()
