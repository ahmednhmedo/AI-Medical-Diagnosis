"""
model_store.py — Save / load trained models and metadata via MongoDB GridFS.

Collections used
    models   – metadata document (name, type, labels, metrics, gridfs_id, created)
    fs.files / fs.chunks – GridFS binary storage for serialised model artefacts

Public API
    save_model_to_mongo(artifact_dir, model_name, model_type, labels, metrics)
    load_model_from_mongo(model_name, out_dir)
    list_models()
    delete_model(model_name)
"""

import io
import os
import pickle
import zipfile
from datetime import datetime, timezone

import gridfs
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "Medical_Diagnosis")

_client = None
_db = None
_fs = None


def _connect():
    global _client, _db, _fs
    if _db is None:
        _client = MongoClient(MONGO_URI)
        _db = _client[DB_NAME]
        _fs = gridfs.GridFS(_db)
    return _db, _fs


# ── save ────────────────────────────────────────────────────────────────────

def _zip_directory(directory: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, directory)
                zf.write(full, arcname)
    return buf.getvalue()


def save_model_to_mongo(
    artifact_dir: str,
    model_name: str,
    model_type: str,
    labels: list,
    metrics: dict,
) -> str:
    """Zip *artifact_dir*, store in GridFS, write metadata to ``models`` collection.

    Returns the inserted metadata document's ``_id`` as a string.
    """
    db, fs = _connect()

    # Remove previous version with same name (keeps history clean)
    prev = db["models"].find_one({"name": model_name})
    if prev and "gridfs_id" in prev:
        try:
            fs.delete(prev["gridfs_id"])
        except Exception:
            pass
        db["models"].delete_one({"_id": prev["_id"]})

    # Zip and store
    blob = _zip_directory(artifact_dir)
    gridfs_id = fs.put(blob, filename=f"{model_name}.zip")

    doc = {
        "name": model_name,
        "type": model_type,
        "gridfs_id": gridfs_id,
        "labels": labels,
        "metrics": metrics,
        "created": datetime.now(timezone.utc),
    }
    result = db["models"].insert_one(doc)
    print(f"[model_store] Saved '{model_name}' -> GridFS id {gridfs_id}")
    return str(result.inserted_id)


# ── load ────────────────────────────────────────────────────────────────────

def load_model_from_mongo(model_name: str, out_dir: str) -> dict:
    """Download model artefacts from GridFS, unzip into *out_dir*.

    Returns the metadata document (dict).
    """
    db, fs = _connect()
    doc = db["models"].find_one({"name": model_name})
    if doc is None:
        raise FileNotFoundError(f"No model named '{model_name}' in MongoDB")

    blob = fs.get(doc["gridfs_id"]).read()
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        zf.extractall(out_dir)

    print(f"[model_store] Loaded '{model_name}' -> {out_dir}")
    return doc


# ── list / delete ───────────────────────────────────────────────────────────

def list_models() -> list:
    db, _ = _connect()
    return list(
        db["models"].find(
            {},
            {"_id": 0, "name": 1, "type": 1, "labels": 1, "metrics": 1, "created": 1},
        )
    )


def delete_model(model_name: str) -> bool:
    db, fs = _connect()
    doc = db["models"].find_one({"name": model_name})
    if doc is None:
        return False
    if "gridfs_id" in doc:
        try:
            fs.delete(doc["gridfs_id"])
        except Exception:
            pass
    db["models"].delete_one({"_id": doc["_id"]})
    print(f"[model_store] Deleted '{model_name}'")
    return True


# ── CLI helper ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    action = sys.argv[1] if len(sys.argv) > 1 else "list"

    if action == "list":
        for m in list_models():
            m["created"] = str(m.get("created", ""))
            print(json.dumps(m, indent=2))

    elif action == "save" and len(sys.argv) >= 4:
        # Usage: python model_store.py save <artifact_dir> <model_name> [model_type]
        adir = sys.argv[2]
        mname = sys.argv[3]
        mtype = sys.argv[4] if len(sys.argv) > 4 else "unknown"

        # Attempt to load label_encoder to get labels
        le_path = os.path.join(adir, "label_encoder.pkl")
        labels = []
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                le = pickle.load(f)
            labels = list(le.classes_) if hasattr(le, "classes_") else []

        save_model_to_mongo(adir, mname, mtype, labels, {})

    elif action == "delete" and len(sys.argv) >= 3:
        delete_model(sys.argv[2])

    else:
        print("Usage:")
        print("  python model_store.py list")
        print("  python model_store.py save <artifact_dir> <model_name> [model_type]")
        print("  python model_store.py delete <model_name>")
