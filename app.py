"""
app.py — Unified Flask API for Medical Diagnosis AI

Three API groups (per project spec):
    1. Data Handling API   — scrape NHS Inform, manage conditions in MongoDB
    2. Preprocessing API   — clean / normalise raw scraped data
    3. Model API           — train, load, serve predictions, manage saved models

Run:  python app.py
"""

import os
import re
import json
import pickle
import subprocess
import sys
import threading
from datetime import datetime, timezone
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import nltk

# Ensure NLTK data is available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── App setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
CORS(app)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "Medical_Diagnosis")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── MongoDB helper ──────────────────────────────────────────────────────────

_mongo_client = None

def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA HANDLING API — scrape NHS, list / query conditions
# ═══════════════════════════════════════════════════════════════════════════

NHS_AZ_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"


def _scrape_condition_page(url: str) -> dict:
    """Scrape a single NHS Inform condition page → dict of sections."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    sections = []
    for h2 in soup.find_all("h2"):
        title = h2.get_text(strip=True)
        parts = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            text = sib.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)
        sections.append({"section_title": title, "description": " ".join(parts)})
    return {"sections": sections}


@app.route("/api/scrape", methods=["POST"])
def scrape_nhs():
    """Scrape NHS Inform A-Z and store raw documents in MongoDB.

    Optional body: {"limit": N}  to scrape only first N conditions.
    """
    db = get_db()
    limit = (request.json or {}).get("limit", None)

    try:
        resp = requests.get(NHS_AZ_URL, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch A-Z index: {e}"}), 502

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a_tag in soup.select("a[href*='/illnesses-and-conditions/']"):
        href = a_tag.get("href", "")
        name = a_tag.get_text(strip=True)
        if name and href and href != NHS_AZ_URL:
            full = href if href.startswith("http") else f"https://www.nhsinform.scot{href}"
            links.append((name, full))

    # Deduplicate
    seen = set()
    unique_links = []
    for name, url in links:
        if url not in seen:
            seen.add(url)
            unique_links.append((name, url))
    if limit:
        unique_links = unique_links[:int(limit)]

    inserted = 0
    errors = []
    for name, url in unique_links:
        page = _scrape_condition_page(url)
        if "error" in page:
            errors.append({"condition": name, "error": page["error"]})
            continue

        doc = {
            "illness_name": name,
            "url": url,
            "sections": page["sections"],
            "scraped_at": datetime.now(timezone.utc),
        }
        db["Illnesses"].update_one(
            {"illness_name": name}, {"$set": doc}, upsert=True
        )
        inserted += 1

    return jsonify({
        "message": f"Scraped {inserted} conditions",
        "errors": errors[:20],
        "total_links": len(unique_links),
    })


@app.route("/api/conditions", methods=["GET"])
def list_conditions():
    """List all conditions in the Conditions collection."""
    db = get_db()
    coll = db["Conditions"]
    docs = list(coll.find({}, {"_id": 0}))

    # Fallback: try Preprocessed_Illnesses if Conditions is empty
    if not docs:
        docs = list(db["Preprocessed_Illnesses"].find({}, {"_id": 0}))

    return jsonify({"count": len(docs), "conditions": docs})


@app.route("/api/conditions/<name>", methods=["GET"])
def get_condition(name):
    """Get a single condition by name (case-insensitive partial match)."""
    db = get_db()
    regex = re.compile(re.escape(name), re.IGNORECASE)

    for coll_name in ["Conditions", "Preprocessed_Illnesses"]:
        doc = db[coll_name].find_one(
            {"$or": [{"condition": regex}, {"illness_name": regex}]},
            {"_id": 0},
        )
        if doc:
            return jsonify(doc)

    return jsonify({"error": f"Condition '{name}' not found"}), 404


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING API — clean, normalise, structure raw scraped data
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

SECTION_KEYWORDS = {
    "symptoms": ["symptom"],
    "causes": ["cause", "risk factor"],
    "warnings": ["emergency", "seek", "warning", "urgent", "when to get"],
    "recommendations": ["treat", "self-care", "manage", "prevent", "living with"],
}


def _classify_section(title: str) -> str:
    """Map a section title to one of: symptoms, causes, warnings, recommendations, or ''."""
    title_lower = title.lower()
    for field, keywords in SECTION_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return field
    return ""


def _clean_text(text: str) -> str:
    """Lowercase, strip HTML artefacts, normalise whitespace, lemmatise."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s,.\-]", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def _preprocess_single(raw_doc: dict) -> dict:
    """Convert a raw Illnesses document into a cleaned Conditions record."""
    name = raw_doc.get("illness_name", "").strip()
    sections = raw_doc.get("sections", [])

    result = {
        "condition": name,
        "illness_name": name,
        "symptoms": "",
        "causes": "",
        "warnings": "",
        "recommendations": "",
    }

    for sec in sections:
        field = _classify_section(sec.get("section_title", ""))
        if field:
            existing = result[field]
            new_text = _clean_text(sec.get("description", ""))
            result[field] = f"{existing} {new_text}".strip() if existing else new_text

    return result


@app.route("/api/preprocess", methods=["POST"])
def preprocess_data():
    """Clean all raw Illnesses documents → Conditions + Preprocessed_Illnesses collections.

    Optional body: {"source": "Illnesses"}  to specify source collection.
    """
    db = get_db()
    source = (request.json or {}).get("source", "Illnesses")

    raw_docs = list(db[source].find({}, {"_id": 0}))
    if not raw_docs:
        return jsonify({"error": f"No documents in '{source}'"}), 404

    cleaned = []
    for doc in raw_docs:
        record = _preprocess_single(doc)
        if record["condition"]:
            cleaned.append(record)

    # Deduplicate by condition name
    unique = {}
    for rec in cleaned:
        unique[rec["condition"]] = rec
    cleaned = list(unique.values())

    # Write to both Conditions (spec format) and Preprocessed_Illnesses (legacy)
    if cleaned:
        for coll_name in ["Conditions", "Preprocessed_Illnesses"]:
            coll = db[coll_name]
            for rec in cleaned:
                coll.update_one(
                    {"illness_name": rec["illness_name"]},
                    {"$set": rec},
                    upsert=True,
                )

    return jsonify({
        "message": f"Preprocessed {len(cleaned)} conditions",
        "sample": cleaned[:3] if cleaned else [],
    })


@app.route("/api/preprocess/single", methods=["POST"])
def preprocess_single():
    """Clean a single raw text blob.

    Body: {"text": "raw symptom text to clean"}
    """
    data = request.json or {}
    text = data.get("text", "")
    cleaned = _clean_text(text)
    return jsonify({"original": text, "cleaned": cleaned})


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL API — train, predict, manage models
# ═══════════════════════════════════════════════════════════════════════════

# ── Lazy model loaders ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_tfidf():
    from inference_baseline import BaselineClassifier
    return BaselineClassifier(os.path.join(BASE_DIR, "artifacts_baseline"))


@lru_cache(maxsize=1)
def _load_lstm():
    from inference_lstm import LSTMClassifier
    return LSTMClassifier(os.path.join(BASE_DIR, "artifacts_lstm"))


@lru_cache(maxsize=1)
def _load_transformer():
    model_dir = os.path.join(BASE_DIR, "artifacts_transformer", "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            "Transformer model not trained yet. "
            "Run 'python train_transformer.py' first, or use tfidf/lstm."
        )
    from inference_transformer import TransformerClassifier
    return TransformerClassifier(os.path.join(BASE_DIR, "artifacts_transformer"))


MODEL_LOADERS = {
    "tfidf": _load_tfidf,
    "tf-idf": _load_tfidf,
    "baseline": _load_tfidf,
    "lstm": _load_lstm,
    "dl": _load_lstm,
    "rnn": _load_lstm,
    "transformer": _load_transformer,
    "bert": _load_transformer,
    "biobert": _load_transformer,
}


def _resolve_model_name(name: str) -> str:
    """Normalise user-supplied model name."""
    name = name.lower().strip()
    if name in ("tf-idf", "tfidf", "baseline"):
        return "tfidf"
    if name in ("lstm", "dl", "rnn"):
        return "lstm"
    if name in ("transformer", "bert", "biobert"):
        return "transformer"
    return name


def _get_snomed_code(condition_name: str) -> dict:
    """Look up SNOMED CT code for a condition from Illness_SNOMED_Codes collection."""
    db = get_db()
    regex = re.compile(re.escape(condition_name), re.IGNORECASE)
    doc = db["Illness_SNOMED_Codes"].find_one(
        {"illness_name": regex},
        {"_id": 0, "snomed_code": 1, "snomed_term": 1, "match_type": 1, "confidence": 1},
    )
    if doc and doc.get("snomed_code"):
        return {
            "snomed_code": doc["snomed_code"],
            "snomed_term": doc.get("snomed_term", ""),
        }
    return {"snomed_code": None, "snomed_term": ""}


def _get_recommendations(condition_name: str) -> dict:
    """Fetch warnings & recommendations from the Conditions collection."""
    db = get_db()
    regex = re.compile(re.escape(condition_name), re.IGNORECASE)
    for coll in ["Conditions", "Preprocessed_Illnesses", "Synonym_Expanded_Illnesses"]:
        doc = db[coll].find_one(
            {"$or": [{"condition": regex}, {"illness_name": regex}]},
            {"_id": 0, "warnings": 1, "recommendations": 1},
        )
        if doc:
            return {
                "warnings": doc.get("warnings", ""),
                "recommendations": doc.get("recommendations", ""),
            }
    return {"warnings": "", "recommendations": ""}


def _classify_age_group(age: str) -> str:
    """Classify age into group for post-prediction reranking."""
    age = (age or "").strip()
    if not age:
        return ""
    try:
        n = int(age)
        if n < 13:
            return "child"
        elif n < 20:
            return "teen"
        elif n < 60:
            return "adult"
        else:
            return "elderly"
    except ValueError:
        return age.lower()


# Keywords that hint a condition is more relevant for a demographic
AGE_RELEVANCE = {
    "child": ["children", "child", "pediatric", "infant", "baby", "young"],
    "teen": ["teenager", "adolescent", "young adult", "teen"],
    "elderly": ["elderly", "older", "geriatric", "age-related", "degenerative"],
}
GENDER_RELEVANCE = {
    "male": ["prostate", "testicular", "penile"],
    "female": ["ovarian", "cervical", "menstrual", "pregnancy", "uterine", "breast"],
}


def _rerank_with_metadata(predictions: list, age: str = "", gender: str = "") -> list:
    """Boost scores for conditions that match patient demographics.

    Does NOT modify the query text (avoids train-inference mismatch).
    Instead, applies a small score boost to predictions whose label or
    associated text matches the patient's age/gender profile.
    """
    age_group = _classify_age_group(age)
    gender = (gender or "").strip().lower()

    if not age_group and not gender:
        return predictions

    boosted = []
    for pred in predictions:
        label_lower = pred["label"].lower()
        boost = 0.0

        # Age-based boosting
        if age_group in AGE_RELEVANCE:
            for kw in AGE_RELEVANCE[age_group]:
                if kw in label_lower:
                    boost += 0.05
                    break

        # Gender-based boosting
        if gender in GENDER_RELEVANCE:
            for kw in GENDER_RELEVANCE[gender]:
                if kw in label_lower or kw in pred.get("warnings", "").lower() or kw in pred.get("recommendations", "").lower():
                    boost += 0.03
                    break

        pred_copy = dict(pred)
        pred_copy["score"] = round(pred_copy["score"] + boost, 4)
        boosted.append(pred_copy)

    # Re-sort by score and renormalise
    boosted.sort(key=lambda x: x["score"], reverse=True)
    total = sum(p["score"] for p in boosted)
    if total > 0:
        for p in boosted:
            p["score"] = round(p["score"] / total, 4)

    return boosted


# ── Prediction endpoint ────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint.

    Body: {
        "text": "fever, cough, chest pain"  OR  "symptoms": "...",
        "model": "tfidf" | "lstm" | "transformer",
        "k": 5,
        "age": "35",       (optional)
        "gender": "female"  (optional)
    }

    Returns: {
        "model": "tfidf",
        "predictions": [
            {"label": "Pneumonia", "score": 0.42, "warnings": "...", "recommendations": "..."},
            ...
        ]
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"detail": "Invalid JSON"}), 400

    # Extract text
    text = data.get("text") or data.get("symptoms") or ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).strip()

    if not text:
        return jsonify({"detail": "Missing 'text' or 'symptoms' field"}), 400

    # Model selection
    model_name = _resolve_model_name(str(data.get("model", "tfidf")))
    k = int(data.get("k", data.get("top_k", 5)))

    # Metadata (used for post-prediction reranking, NOT query modification)
    age = str(data.get("age", ""))
    gender = str(data.get("gender", ""))

    # Load model and predict on raw symptom text (no augmentation)
    loader = MODEL_LOADERS.get(model_name)
    if loader is None:
        return jsonify({"detail": f"Unknown model '{model_name}'. Use tfidf/lstm/transformer."}), 400

    try:
        clf = loader()
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 500
    except Exception as e:
        return jsonify({"detail": f"Model '{model_name}' failed to load: {e}"}), 500

    try:
        preds = clf.predict_topk(text, k=k)
    except Exception as e:
        return jsonify({"detail": f"Prediction error: {e}"}), 500

    # Normalise shape: preds is List[List[Tuple]] -- take first list for single input
    if preds and isinstance(preds[0], list):
        preds = preds[0]

    # Attach recommendations from DB
    results = []
    for item in preds:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            label, score = item
        elif isinstance(item, dict):
            label = item.get("label", "")
            score = item.get("score", 0.0)
        else:
            continue

        rec = _get_recommendations(label)
        snomed = _get_snomed_code(label)
        results.append({
            "label": str(label),
            "score": round(float(score), 4),
            "snomed_code": snomed["snomed_code"],
            "snomed_term": snomed["snomed_term"],
            "warnings": rec["warnings"],
            "recommendations": rec["recommendations"],
        })

    # Apply demographic reranking if age/gender provided
    if age or gender:
        results = _rerank_with_metadata(results, age, gender)

    return jsonify({
        "model": model_name,
        "predictions": results,
        "metadata": {"age": age, "gender": gender} if (age or gender) else {},
    })


# ── Training endpoints ─────────────────────────────────────────────────────

@app.route("/api/train/<model_type>", methods=["POST"])
def train_model(model_type):
    """Trigger model training in a background thread.

    model_type: "baseline" | "lstm" | "transformer"
    Optional body: {"upload_mongo": true}
    """
    upload = (request.json or {}).get("upload_mongo", False)

    script_map = {
        "baseline": "09_ml_baseline_tfidf_classifier.py",
        "tfidf": "09_ml_baseline_tfidf_classifier.py",
        "lstm": "10_dl_lstm_classifier.py",
        "transformer": "train_transformer.py",
        "bert": "train_transformer.py",
    }

    script = script_map.get(model_type.lower())
    if not script:
        return jsonify({"error": f"Unknown model type: {model_type}"}), 400

    script_path = os.path.join(BASE_DIR, script)
    if not os.path.exists(script_path):
        return jsonify({"error": f"Script not found: {script}"}), 404

    cmd = [sys.executable, script_path]
    if upload and model_type.lower() in ("transformer", "bert"):
        cmd.append("--mongo")

    def _run():
        subprocess.run(cmd, cwd=BASE_DIR)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({
        "message": f"Training {model_type} started in background",
        "script": script,
    })


# ── Model management ───────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all models saved in MongoDB."""
    from model_store import list_models as _list
    models = _list()
    for m in models:
        m["created"] = str(m.get("created", ""))
    return jsonify({"models": models})


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    """Delete a model from MongoDB GridFS."""
    from model_store import delete_model as _del
    ok = _del(name)
    if ok:
        return jsonify({"message": f"Deleted model '{name}'"})
    return jsonify({"error": f"Model '{name}' not found"}), 404


@app.route("/api/models/save", methods=["POST"])
def save_model():
    """Save local model artefacts to MongoDB GridFS.

    Body: {
        "artifact_dir": "artifacts_baseline",
        "model_name": "tfidf_v1",
        "model_type": "TF-IDF + LogisticRegression"
    }
    """
    data = request.json or {}
    adir = data.get("artifact_dir", "")
    mname = data.get("model_name", "")
    mtype = data.get("model_type", "unknown")

    if not adir or not mname:
        return jsonify({"error": "artifact_dir and model_name required"}), 400

    full_dir = os.path.join(BASE_DIR, adir) if not os.path.isabs(adir) else adir
    if not os.path.isdir(full_dir):
        return jsonify({"error": f"Directory not found: {full_dir}"}), 404

    # Try to extract labels from label_encoder
    labels = []
    le_path = os.path.join(full_dir, "label_encoder.pkl")
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        labels = list(le.classes_) if hasattr(le, "classes_") else []

    # Try to load metrics
    metrics = {}
    metrics_path = os.path.join(full_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    from model_store import save_model_to_mongo
    doc_id = save_model_to_mongo(full_dir, mname, mtype, labels, metrics)

    return jsonify({"message": f"Saved '{mname}' to MongoDB", "id": doc_id})


# ── Model comparison ────────────────────────────────────────────────────────

@app.route("/api/compare", methods=["POST"])
def compare_models():
    """Compare all available models on the same input.

    Body: {"text": "fever, cough, chest pain", "k": 5}

    Returns predictions from each model side-by-side with metrics.
    """
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text") or data.get("symptoms") or "").strip()
    if not text:
        return jsonify({"detail": "Missing 'text' field"}), 400
    k = int(data.get("k", 5))

    results = {}
    model_names = ["tfidf", "lstm", "transformer"]

    for mname in model_names:
        loader = MODEL_LOADERS.get(mname)
        if not loader:
            continue
        try:
            clf = loader()
            preds = clf.predict_topk(text, k=k)
            if preds and isinstance(preds[0], list):
                preds = preds[0]

            ranked = []
            for item in preds:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    label, score = item
                else:
                    continue
                rec = _get_recommendations(label)
                ranked.append({
                    "label": str(label),
                    "score": round(float(score), 4),
                    "warnings": rec["warnings"],
                    "recommendations": rec["recommendations"],
                })
            results[mname] = {"predictions": ranked, "status": "ok"}
        except Exception as e:
            results[mname] = {"predictions": [], "status": f"error: {e}"}

    # Load stored metrics from MongoDB
    stored_metrics = {}
    try:
        from model_store import list_models as _list
        for m in _list():
            stored_metrics[m["name"]] = m.get("metrics", {})
    except Exception:
        pass

    return jsonify({
        "query": text,
        "top_k": k,
        "models": results,
        "stored_metrics": stored_metrics,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Web UI
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Medical Diagnosis AI - Flask API")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    print()
    print("  Endpoints:")
    print("  -- Data Handling --")
    print("  POST /api/scrape          Scrape NHS Inform -> MongoDB")
    print("  GET  /api/conditions      List all conditions")
    print("  GET  /api/conditions/<n>  Get condition by name")
    print()
    print("  -- Preprocessing --")
    print("  POST /api/preprocess        Clean raw data -> Conditions collection")
    print("  POST /api/preprocess/single  Clean a single text blob")
    print()
    print("  -- Model --")
    print("  POST /predict               Symptom -> diagnosis prediction")
    print("  POST /api/train/<type>       Train model (baseline/lstm/transformer)")
    print("  GET  /api/models             List saved models in MongoDB")
    print("  POST /api/models/save        Upload model artefacts to GridFS")
    print("  DELETE /api/models/<name>    Delete a model from MongoDB")
    print()
    print("  -- UI --")
    print("  GET  /                       Web diagnosis interface")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
