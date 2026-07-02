# AI Medical Diagnosis

An AI-powered medical diagnosis **support** system that predicts probable medical
conditions from free-text symptom descriptions. It scrapes and processes medical
condition data from NHS Inform, applies NLP preprocessing, trains multiple
ML / deep-learning models, maps predictions to SNOMED CT codes, stores models in
MongoDB GridFS, and serves everything through a Flask REST API with a simple web
interface.

> ⚠️ **Medical disclaimer:** This project is for educational and research purposes
> only. It is **not** a medical device or a certified diagnostic tool. See the
> [Important Medical Disclaimer](#important-medical-disclaimer) section before use.

---

## Project Overview

The system is designed to:

- Accept **free-text symptom descriptions** from the user.
- Predict **probable medical conditions**.
- Rank predictions using **probability / Top-K scoring**.
- Attach useful metadata such as **warnings**, **recommendations**, and
  **SNOMED CT codes** when available.
- Optionally **rerank** results using patient **age / gender** (post-prediction
  demographic weighting — the query text is not modified).
- Provide access through both a **Flask REST API** and a **web UI**.

---

## Key Features

- **NHS Inform scraping** — pulls condition pages from the NHS Inform A-Z index.
- **MongoDB storage** — raw scraped data and cleaned/preprocessed condition data.
- **NLP preprocessing** — HTML/character cleaning, tokenization, English stopword
  removal, and WordNet lemmatization.
- **Multiple model approaches:**
  - **TF-IDF + Logistic Regression** baseline.
  - **BiLSTM + TF-IDF ensemble** (weighted-average of LSTM and TF-IDF probabilities).
  - **BioBERT transformer** classifier (`dmis-lab/biobert-base-cased-v1.1`).
- **SNOMED CT mapping** — maps predicted conditions to SNOMED CT codes/terms.
- **Flask REST API** — grouped into Data Handling, Preprocessing, and Model APIs.
- **Web UI** — `templates/index.html`, served by Flask at `/`.
- **Model storage / versioning** — trained artifact directories are zipped and
  stored in MongoDB GridFS via `model_store.py`.
- **Top-K predictions** and **optional demographic reranking** (age/gender).

---

## Architecture

```
NHS Inform A-Z (web)
        │  Web scraping (requests + BeautifulSoup)
        ▼
     MongoDB
   (raw + preprocessed conditions, SNOMED codes, model metadata)
        │  NLP preprocessing (clean, tokenize, stopwords, lemmatize)
        ▼
  Model training
  ├─ TF-IDF + Logistic Regression   → artifacts_baseline/
  ├─ BiLSTM + TF-IDF ensemble       → artifacts_lstm/
  └─ BioBERT transformer            → artifacts_transformer/
        │  (artifacts also zipped into MongoDB GridFS)
        ▼
     Flask API (app.py)
        │  /predict, /api/compare, model + data endpoints
        ▼
  Web interface (templates/index.html)  +  JSON API responses
```

Only components that exist in this repository are shown above.

---

## Tech Stack

| Layer            | Technology                                   | Purpose                                            |
| ---------------- | -------------------------------------------- | -------------------------------------------------- |
| Language         | Python                                       | Core implementation                                |
| Web framework    | Flask, Flask-CORS                            | REST API + serving the web UI                      |
| Database         | MongoDB, PyMongo, GridFS                     | Data storage and model artifact persistence        |
| Web scraping     | Requests, BeautifulSoup4                     | Fetching and parsing NHS Inform pages              |
| NLP              | NLTK                                         | Tokenization, stopwords, lemmatization             |
| Classical ML     | scikit-learn                                 | TF-IDF vectorization + Logistic Regression         |
| Deep learning    | TensorFlow / Keras                           | BiLSTM model                                       |
| Transformers     | PyTorch, HuggingFace Transformers, BioBERT   | BioBERT fine-tuning and inference                  |
| Embeddings       | sentence-transformers                        | Listed in `requirements.txt`                       |
| Data handling    | Pandas, NumPy, joblib, tqdm                  | Data processing and utilities                      |
| Frontend         | HTML + Pico CSS (dark theme)                 | Web interface (`templates/index.html`)             |

> SciSpacy (for UMLS linking) is referenced as an **optional** dependency in
> `requirements.txt` and is not installed by default.

---

## Repository Structure

```
Medical Diagnosis AI/
├── app.py                          # Flask API (Data / Preprocessing / Model groups) + web UI
├── requirements.txt                # Python dependencies
├── PROJECT_DOCUMENTATION.md        # Detailed project documentation (source of truth)
│
├── retrain_models.py               # Trains TF-IDF baseline + BiLSTM (ensemble) models
├── train_transformer.py            # Fine-tunes BioBERT transformer (supports --mongo)
├── build_snomed_map.py             # Maps conditions to SNOMED CT codes
├── model_store.py                  # Save/load/list/delete models in MongoDB GridFS
│
├── inference_baseline.py           # TF-IDF baseline inference class
├── inference_lstm.py               # LSTM + TF-IDF ensemble inference class
├── inference_transformer.py        # BioBERT inference class
│
├── pipeline_semantic_enrichment.py # Semantic/UMLS enrichment pipeline
├── pipeline_train_test_split.py    # Train/test splitting pipeline
├── pipeline_data_augmentation.py   # Data augmentation pipeline
├── pipeline_build_embeddings.py    # BioBERT embedding pipeline
├── pipeline_umls_snomed_map.py     # UMLS-to-SNOMED linking pipeline
│
├── artifacts_baseline/             # TF-IDF artifacts: vectorizer.pkl, clf.pkl, label_encoder.pkl, metrics.json
├── artifacts_lstm/                 # LSTM artifacts: model.keras, tokenizer.pkl, label_encoder.pkl, config.json
├── artifacts_transformer/          # BioBERT artifacts: model/, tokenizer/, label_encoder.pkl, metrics.json
│
├── templates/
│   └── index.html                  # Web UI (dark theme, Pico CSS, SNOMED tags)
│
├── web_scrapping.ipynb             # Notebook: scraping
├── data_preprocessing.ipynb        # Notebook: preprocessing
└── semantic_embedding_biobert.ipynb# Notebook: BioBERT embeddings
```

> **Note:** A `SNOMED CT/` directory is referenced by the SNOMED mapping code, but
> it is excluded from version control via `.gitignore` and is therefore **not**
> part of this repository. Large model files (`artifacts_baseline/clf.pkl` and
> `artifacts_transformer/model/model.safetensors`) are tracked with **Git LFS**.

---

## Models

### 1. TF-IDF + Logistic Regression (baseline)

- **Training:** `retrain_models.py`
- **Inference:** `inference_baseline.py`
- **Artifacts:** `artifacts_baseline/` (`vectorizer.pkl`, `clf.pkl`,
  `label_encoder.pkl`, `metrics.json`)
- **Purpose:** A strong keyword-matching baseline using word + character n-gram
  TF-IDF features fed into a Logistic Regression classifier over the condition labels.

### 2. BiLSTM + TF-IDF Ensemble (deep learning)

- **Training:** `retrain_models.py`
- **Inference:** `inference_lstm.py`
- **Artifacts:** `artifacts_lstm/` (`model.keras`, `tokenizer.pkl`,
  `label_encoder.pkl`, `config.json`)
- **Purpose:** A Bidirectional LSTM whose predictions are combined with the TF-IDF
  model via a weighted average. The ensemble weight and sequence length are stored
  in `config.json` (e.g. `ensemble_weight`, `maxlen`, `num_classes`).

### 3. BioBERT Transformer

- **Training:** `train_transformer.py` (use `--mongo` to also upload to GridFS)
- **Inference:** `inference_transformer.py`
- **Artifacts:** `artifacts_transformer/` (`model/`, `tokenizer/`,
  `label_encoder.pkl`, `metrics.json`)
- **Purpose:** Fine-tunes `dmis-lab/biobert-base-cased-v1.1` for multi-class
  symptom → condition classification. Optimized for CPU training (shorter max
  sequence length, frozen lower BERT layers, label smoothing).

Metrics are stored in each model's artifact folder when available (see
[Results / Evaluation](#results--evaluation)).

---

## Dataset and Data Pipeline

1. **Scraping** — `POST /api/scrape` fetches the NHS Inform A-Z index, extracts
   condition page links, and scrapes each page's `<h2>` sections.
2. **Raw storage** — each condition is upserted into the MongoDB `Illnesses`
   collection with its URL, sections, and a scrape timestamp.
3. **Section extraction** — during preprocessing, section titles are classified
   into **symptoms**, **causes**, **warnings**, and **recommendations** using
   keyword heuristics.
4. **Cleaning** — text is lowercased, stripped of HTML/special characters,
   whitespace-normalized, stopword-filtered, and lemmatized.
5. **Cleaned storage** — results are written to the `Conditions` (and legacy
   `Preprocessed_Illnesses`) collections.
6. **Training / evaluation** — models are trained on the processed data via the
   training scripts and evaluated with Top-K accuracy / F1.
7. **Prediction** — at inference, predictions are enriched with warnings,
   recommendations, and SNOMED CT codes looked up from MongoDB.

No external dataset download is bundled in this repository; data is acquired by
running the scraper against NHS Inform.

---

## SNOMED CT Mapping

**Script:** `build_snomed_map.py`

The SNOMED mapping component maps each predicted condition to a SNOMED CT code and
term, which are stored in the `Illness_SNOMED_Codes` MongoDB collection and attached
to prediction responses (`snomed_code`, `snomed_term`). According to the project
documentation, mapping uses a tiered strategy (exact, variant, fuzzy, and manual
overrides) with an associated `match_type` and `confidence` per condition.

> The underlying SNOMED CT description files are **not** included in this
> repository (the `SNOMED CT/` directory is gitignored). SNOMED CT is subject to
> its own licensing/access terms depending on your country and usage, which you
> are responsible for obtaining.

---

## API Endpoints

All routes are defined in `app.py`.

| Method   | Endpoint                  | Purpose                                                      |
| -------- | ------------------------- | ----------------------------------------------------------- |
| `GET`    | `/`                       | Web diagnosis interface                                      |
| `GET`    | `/healthz`                | Health check (`{"status": "ok"}`)                           |
| `POST`   | `/api/scrape`             | Scrape NHS Inform A-Z into MongoDB (optional `{"limit": N}`)|
| `GET`    | `/api/conditions`         | List all conditions                                         |
| `GET`    | `/api/conditions/<name>`  | Get a condition by name (case-insensitive partial match)    |
| `POST`   | `/api/preprocess`         | Clean all raw data → `Conditions` collection                |
| `POST`   | `/api/preprocess/single`  | Clean a single text blob (`{"text": "..."}`)                |
| `POST`   | `/predict`                | Predict conditions from symptom text                        |
| `POST`   | `/api/train/<model_type>` | Trigger training in a background thread (see note below)    |
| `GET`    | `/api/models`             | List models saved in MongoDB                                |
| `POST`   | `/api/models/save`        | Save local artifact directory to GridFS                     |
| `DELETE` | `/api/models/<name>`      | Delete a model from MongoDB GridFS                           |
| `POST`   | `/api/compare`            | Compare all models on the same input                        |

> **Note on `/api/train/<model_type>`:** The endpoint launches a training script by
> name. In the current code, `transformer`/`bert` map to `train_transformer.py`
> (present in the repo), while `baseline`/`tfidf` and `lstm` map to script
> filenames that are **not** present in this repository. To train the baseline and
> LSTM models, run `retrain_models.py` directly (see
> [Model Training](#model-training)).

---

## Example Prediction Request

`/predict` accepts either a `text` or `symptoms` field, a `model` name
(`tfidf` / `lstm` / `transformer`, plus aliases such as `baseline`, `rnn`, `bert`,
`biobert`), a `k` (or `top_k`) count, and optional `age` / `gender`.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "fever, dry cough, chest pain",
        "model": "transformer",
        "k": 5,
        "age": "35",
        "gender": "male"
      }'
```

Example response shape (values are placeholders):

```json
{
  "model": "transformer",
  "predictions": [
    {
      "label": "<condition name>",
      "score": 0.0,
      "snomed_code": "<snomed id or null>",
      "snomed_term": "<snomed term>",
      "warnings": "<warning text>",
      "recommendations": "<recommendation text>"
    }
  ],
  "metadata": { "age": "35", "gender": "male" }
}
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ahmednhmedo/AI-Medical-Diagnosis.git
   cd AI-Medical-Diagnosis
   ```

2. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Linux / macOS:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start MongoDB** locally (default `mongodb://localhost:27017/`).

5. **Configure optional environment variables** (defaults shown):

   | Variable    | Default                       | Purpose             |
   | ----------- | ----------------------------- | ------------------- |
   | `MONGO_URI` | `mongodb://localhost:27017/`  | MongoDB connection  |
   | `MONGO_DB`  | `Medical_Diagnosis`           | Database name       |

   Windows (PowerShell):
   ```powershell
   $env:MONGO_URI = "mongodb://localhost:27017/"
   $env:MONGO_DB  = "Medical_Diagnosis"
   ```

   Linux / macOS:
   ```bash
   export MONGO_URI="mongodb://localhost:27017/"
   export MONGO_DB="Medical_Diagnosis"
   ```

> On first run, `app.py` downloads the required NLTK resources
> (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) automatically.

---

## How to Run

1. **Start the Flask app**

   ```bash
   python app.py
   ```
   The server listens on `http://localhost:5000`.

2. **Open the web interface** at `http://localhost:5000`.

3. **Scrape data** into MongoDB:

   ```bash
   curl -X POST http://localhost:5000/api/scrape
   # Optional: scrape only the first N conditions
   curl -X POST http://localhost:5000/api/scrape \
     -H "Content-Type: application/json" -d '{"limit": 10}'
   ```

4. **Preprocess data**:

   ```bash
   curl -X POST http://localhost:5000/api/preprocess
   ```

5. **Train models** (see [Model Training](#model-training)).

6. **Run predictions** via `/predict` (see
   [Example Prediction Request](#example-prediction-request)) or the web UI.

---

## Model Training

Train the classical + LSTM models and the transformer with the provided scripts:

```bash
# TF-IDF baseline + BiLSTM ensemble → artifacts_baseline/ and artifacts_lstm/
python retrain_models.py

# BioBERT transformer → artifacts_transformer/  (add --mongo to also upload to GridFS)
python train_transformer.py
python train_transformer.py --mongo

# Build SNOMED CT code mappings
python build_snomed_map.py
```

> Transformer fine-tuning is computationally expensive and can take a significant
> amount of time and memory, especially on CPU. The bundled transformer metrics
> report a training time of roughly 38 minutes for the recorded run.

Trained artifact directories can also be pushed to MongoDB GridFS via the API:

```bash
curl -X POST http://localhost:5000/api/models/save \
  -H "Content-Type: application/json" \
  -d '{"artifact_dir": "artifacts_baseline", "model_name": "tfidf_v1", "model_type": "TF-IDF + LogisticRegression"}'
```

---

## Web Interface

The web UI (`templates/index.html`, served by Flask at `/`) provides:

- A **symptoms** text input.
- Optional **age** and **gender** fields for demographic reranking.
- A **model** selector.
- A **Top-K** selector.
- A results area that displays ranked predictions, including SNOMED CT tags,
  warnings, and recommendations.

It is styled with **Pico CSS** using a dark theme.

---

## Results / Evaluation

Evaluation metrics are stored in the model artifact folders when available:

- `artifacts_transformer/metrics.json` contains verified metrics for the BioBERT run.
- `artifacts_baseline/metrics.json` contains run metadata
  (`num_classes`, `train_samples`) but no accuracy fields in the committed file.
- `artifacts_lstm/` contains `config.json` (ensemble weight, sequence length, etc.)
  rather than a `metrics.json`.

**BioBERT transformer — verified from `artifacts_transformer/metrics.json`:**

| Metric        | Value  |
| ------------- | ------ |
| Accuracy@1    | 0.8179 |
| F1 (macro)    | 0.7781 |
| Accuracy@3    | 0.9278 |
| Accuracy@5    | 0.9691 |
| Epochs        | 4      |
| Classes       | 291    |

The project documentation (`PROJECT_DOCUMENTATION.md`) additionally reports a
comparison across all three models (e.g. TF-IDF baseline ≈ 63% Acc@1, BiLSTM
ensemble ≈ 53% Acc@1). Only the BioBERT figures above are independently
verifiable from a committed `metrics.json` file in this repository; the baseline
and LSTM figures come from the documentation and are not reproduced in their
artifact metrics files.

---

## Important Medical Disclaimer

This project is for **educational and research purposes only**. It is **not** a
medical device, **not** a certified diagnostic tool, and should **not** be used as
a substitute for professional medical advice, diagnosis, or treatment. Predictions
are model estimates and may be incorrect. Always consult a qualified healthcare
professional for any medical concerns or before making any health-related decisions.

---

## Future Improvements

The following are **future work** ideas, not currently implemented:

- Clinical validation of predictions.
- Authentication / access control for the API.
- Docker support (no Dockerfile is currently present).
- Automated tests.
- CI/CD pipelines.
- Model monitoring and drift detection.
- Prediction explainability.
- Deployment configuration.

---

## Author

**Ahmed Nhmedo**

---

## License

No license file is currently included in this repository.
