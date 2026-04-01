# Medical Diagnosis AI - Project Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Solution Architecture](#2-solution-architecture)
3. [Database Design](#3-database-design)
4. [Data Acquisition (Web Scraping)](#4-data-acquisition)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Design and Training](#6-model-design-and-training)
7. [SNOMED CT Medical Code Mapping](#7-snomed-ct-medical-code-mapping)
8. [Flask API Documentation](#8-flask-api-documentation)
9. [Web User Interface](#9-web-user-interface)
10. [Model Evaluation Report](#10-model-evaluation-report)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [File Structure](#12-file-structure)

---

## 1. Project Overview

This project implements an AI-powered medical diagnosis system that:

1. **Scrapes** 291 medical conditions from NHS Inform A-Z index into MongoDB
2. **Preprocesses** symptoms, causes, warnings, and recommendations using NLP techniques
3. **Trains three NLP models** for symptom-to-disease classification:
   - TF-IDF + Logistic Regression (baseline)
   - Bidirectional LSTM with TF-IDF ensemble (deep learning)
   - BioBERT Transformer (state-of-the-art)
4. **Predicts** probable conditions from free-text symptom descriptions
5. **Ranks** conditions by probability with Top-K scoring
6. **Attaches** warnings, recommendations, and SNOMED CT medical codes to predictions
7. **Serves** everything through a Flask REST API with a web interface
8. **Stores** trained models in MongoDB GridFS for persistence and versioning

### Technologies Used
- **Language**: Python 3.11
- **Database**: MongoDB with GridFS
- **Web Framework**: Flask with CORS
- **ML/NLP**: scikit-learn, TensorFlow/Keras, PyTorch, HuggingFace Transformers
- **Pre-trained Model**: dmis-lab/biobert-base-cased-v1.1
- **Medical Terminology**: SNOMED CT (International + US editions)
- **NLP Tools**: NLTK, SciSpacy, TF-IDF vectorization
- **Frontend**: HTML5 with Pico CSS (dark theme)

---

## 2. Solution Architecture

```
+------------------+     +-------------------+     +------------------+
|   NHS Inform     |     |    MongoDB        |     |   Flask API      |
|   A-Z Website    |---->|                   |<--->|   (app.py)       |
|                  |     | - Conditions      |     |                  |
+------------------+     | - Models (GridFS) |     | Data Handling API|
   Web Scraping          | - SNOMED Codes    |     | Preprocessing API|
                         | - Augmented Data  |     | Model API        |
                         +-------------------+     +--------+---------+
                                                            |
                              +-----------------------------+
                              |
              +---------------+---------------+----------------+
              |               |               |                |
        +-----------+   +-----------+   +------------+   +-----------+
        | TF-IDF +  |   | BiLSTM +  |   | BioBERT    |   | Web UI    |
        | Logistic  |   | TF-IDF    |   | Transformer|   | (HTML)    |
        | Regression|   | Ensemble  |   |            |   |           |
        +-----------+   +-----------+   +------------+   +-----------+
        artifacts_      artifacts_      artifacts_        templates/
        baseline/       lstm/           transformer/      index.html
```

### Data Flow
1. **Scraping**: NHS Inform pages -> raw HTML -> parsed sections -> MongoDB `Illnesses` collection
2. **Preprocessing**: Raw text -> tokenization -> stopword removal -> lemmatization -> `Conditions` collection
3. **Augmentation**: Training data expanded with symptom query variants for better inference
4. **Training**: Augmented data -> 3 model architectures -> artifacts saved locally + MongoDB GridFS
5. **Inference**: User symptoms -> model prediction -> probability ranking -> SNOMED codes + warnings/advice -> JSON response

---

## 3. Database Design

### MongoDB Database: `Medical_Diagnosis`

#### Conditions Collection (291 documents)
| Field | Type | Description |
|-------|------|-------------|
| condition | String | Disease/condition name (e.g., "Asthma") |
| symptoms | String | Preprocessed symptoms text |
| causes | String | Preprocessed causes text |
| warnings | String | Emergency/warning advice |
| recommendations | String | Treatment and self-care advice |

#### Models Collection (3 documents)
| Field | Type | Description |
|-------|------|-------------|
| name | String | Model identifier (e.g., "biobert_classifier_v1") |
| type | String | Architecture type (e.g., "Transformer (BioBERT)") |
| gridfs_id | ObjectId | Reference to model binary in GridFS |
| labels | Array | List of 291 condition labels |
| metrics | JSON | Evaluation metrics (accuracy, F1, Top-K) |
| created | DateTime | Timestamp when model was saved |

#### Illness_SNOMED_Codes Collection (291 documents)
| Field | Type | Description |
|-------|------|-------------|
| illness_name | String | Condition name |
| snomed_code | String | SNOMED CT concept ID |
| snomed_term | String | Official SNOMED CT term |
| match_type | String | "exact", "variant", "manual", or "fuzzy" |
| confidence | Float | Match confidence score (0.0 - 1.0) |

#### Other Collections
| Collection | Documents | Purpose |
|---|---|---|
| Illnesses | 386 | Raw scraped data from NHS Inform |
| Preprocessed_Illnesses | 291 | Cleaned and normalized illness data |
| Synonym_Expanded_Illnesses | 291 | Enriched with UMLS synonyms |
| Augmented_Illnesses_Train | 1,746 | Training data with symptom query variants |
| UMLS_Enriched_Illnesses | 291 | UMLS entity-linked records |
| Illness_Vectors_Refined | 291 | BioBERT semantic embeddings |
| fs.files / fs.chunks | 3 / 1,811 | GridFS binary storage for model artifacts |

---

## 4. Data Acquisition

**Script**: `app.py` -> `POST /api/scrape`
**Notebook**: `web_scrapping.ipynb`

### Process
1. Fetches the NHS Inform A-Z index page: `https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/`
2. Extracts all condition page links (291 conditions)
3. For each condition page, scrapes:
   - **Symptoms**: Signs and symptoms of the condition
   - **Causes**: What causes the condition
   - **Warnings**: When to seek emergency care
   - **Recommendations**: Treatment and self-care advice
4. Stores each condition as a document in the MongoDB `Illnesses` collection
5. Uses BeautifulSoup for HTML parsing and requests for HTTP fetching

### Data Coverage
- 291 unique medical conditions scraped
- Conditions range from common (Cold, Flu, Asthma) to serious (Cancer types, Stroke)
- Includes age-specific variants (e.g., "Acute lymphoblastic leukaemia: Children")

---

## 5. Data Preprocessing

**Script**: `app.py` -> `POST /api/preprocess`
**Notebook**: `data_preprocessing.ipynb`

### Pipeline Steps

1. **Text Cleaning**
   - Remove HTML tags and special characters
   - Remove URLs and email addresses
   - Normalize whitespace

2. **Tokenization**
   - Split text into individual words using NLTK word_tokenize

3. **Stopword Removal**
   - Remove common English stopwords (the, is, at, etc.)
   - Preserves medically significant terms

4. **Lemmatization**
   - Reduce words to base form using WordNet Lemmatizer
   - Example: "breathing" -> "breathe", "symptoms" -> "symptom"

5. **Output**
   - Cleaned text stored in `Conditions` collection with fields: condition, symptoms, causes, warnings, recommendations

### Data Augmentation (for training)

**Script**: `retrain_models.py` -> `make_variants()`

To bridge the gap between long medical descriptions (training data) and short user queries (inference), we generate training variants:

- **Contiguous spans**: Random consecutive chunks from symptom text
- **Keyword subsets**: Random selection of medical keywords
- **Short queries**: 3-6 word snippets mimicking real user input
- **Mixed queries**: Combinations of symptoms + causes

This produces 1,746 training samples from 291 conditions (6x augmentation).

---

## 6. Model Design and Training

### Model 1: TF-IDF + Logistic Regression (Baseline)

**Script**: `retrain_models.py`
**Inference**: `inference_baseline.py`
**Artifacts**: `artifacts_baseline/`

#### Architecture
```
Input Text -> TF-IDF Vectorizer -> Logistic Regression -> Softmax -> 291 classes
```

#### TF-IDF Configuration
- **Word n-grams**: (1, 2) - unigrams and bigrams
- **Character n-grams**: (3, 5) - captures partial word matches
- **Max features**: 80,000
- **Sublinear TF**: True (applies log scaling)

#### Classifier
- Logistic Regression with C=1.0
- Balanced class weights (handles class imbalance)
- Multi-class: one-vs-rest strategy

#### Why This Works
TF-IDF captures keyword frequency patterns. When a user types "chest pain, shortness of breath", TF-IDF finds conditions whose training text has high overlap with these keywords. Simple but effective for exact keyword matching.

---

### Model 2: Bidirectional LSTM + TF-IDF Ensemble (Deep Learning)

**Script**: `retrain_models.py`
**Inference**: `inference_lstm.py`
**Artifacts**: `artifacts_lstm/`

#### Architecture
```
Input Text -> Tokenizer -> Embedding(128) -> SpatialDropout1D(0.3)
           -> Bidirectional LSTM(128) -> GlobalMaxPooling1D
           -> Dense(256) -> BatchNorm -> Dropout(0.4)
           -> Dense(128) -> Dropout(0.3)
           -> Dense(291, softmax)
```

#### Key Design Decisions
- **Bidirectional LSTM**: Reads text both forward and backward, capturing context in both directions
- **SpatialDropout1D**: Drops entire embedding channels, prevents co-adaptation
- **GlobalMaxPooling1D**: Takes the maximum activation across all timesteps, focusing on the most important features
- **BatchNormalization**: Stabilizes training, allows higher learning rates

#### Ensemble Strategy
The LSTM predictions are combined with TF-IDF predictions using a weighted average:

```
final_probs = w * lstm_probs + (1 - w) * tfidf_probs
```

The optimal weight `w` is found by evaluating all values from 0.1 to 0.9 on the validation set and selecting the one with highest Top-3 accuracy. This is stored in `config.json`.

#### Label Space Alignment
Since LSTM and TF-IDF may have slightly different label encoders, the ensemble aligns both to a shared label space before combining probabilities.

---

### Model 3: BioBERT Transformer (State-of-the-Art)

**Script**: `train_transformer.py`
**Inference**: `inference_transformer.py`
**Artifacts**: `artifacts_transformer/`

#### Architecture
```
Input Text -> BioBERT Tokenizer -> Token IDs + Attention Mask
           -> BioBERT Encoder (12 layers, 768 hidden)
           -> [CLS] token representation
           -> Classification Head -> Softmax -> 291 classes
```

#### Pre-trained Model
- **dmis-lab/biobert-base-cased-v1.1**: BERT model pre-trained on PubMed biomedical literature
- Already understands medical terminology, drug names, anatomical terms
- Fine-tuned on our 291-class symptom-to-disease classification task

#### Training Optimizations (for CPU)
- **Layer freezing**: Bottom 6 of 12 BERT layers frozen (only fine-tune top 6 + classifier)
  - Reduces trainable parameters from 108M to 43M (39.9%)
  - Prevents overfitting on small dataset (1,746 samples)
  - ~2x faster training on CPU
- **MAX_LEN = 128**: Sufficient for symptom queries (attention is O(n^2))
- **Label smoothing = 0.1**: Softens overconfident predictions across 291 classes
- **Differential learning rate**: Classifier head uses 3x higher LR than BERT layers
- **Linear warmup + decay**: 10% warmup steps, then linear decay

#### Why BioBERT Outperforms
- Pre-trained on 28+ million PubMed abstracts - already "knows" medicine
- Contextual embeddings: understands that "chest pain" in cardiac vs muscular contexts are different
- Attention mechanism: can focus on the most diagnostically relevant symptoms
- Transfer learning: leverages massive biomedical knowledge despite small fine-tuning dataset

---

## 7. SNOMED CT Medical Code Mapping

**Script**: `build_snomed_map.py`

### What is SNOMED CT?
SNOMED CT (Systematized Nomenclature of Medicine - Clinical Terms) is the international standard for clinical terminology. Every medical condition has a unique numeric code used in electronic health records worldwide.

### Mapping Strategy (3-tier)
1. **Exact Match** (confidence 1.0): Condition name matches a SNOMED synonym exactly
   - Example: "Asthma" -> SNOMED 21341004
2. **Variant Match** (confidence 0.95): Normalized name matches after removing suffixes, hyphens, possessives
   - Example: "Parkinson's disease" -> "parkinson disease" -> SNOMED 49049000
3. **Fuzzy Match** (confidence 0.5-1.0): TF-IDF character n-gram similarity when exact matching fails
   - Example: "Tourette's syndrome" -> "Tourette syndrome" (0.852 similarity)
4. **Manual Override** (confidence 1.0): 22 conditions with known incorrect fuzzy matches corrected by hand
   - Example: "Type 1 diabetes" -> SNOMED 46635009 (not generic "Diabetes type")

### Results
- **241 exact/manual matches** (82.8%)
- **50 fuzzy matches** (17.2%)
- **0 unmatched** (100% coverage)
- SNOMED data sources: International (June 2025) + US Edition (March 2025)

### Integration
Each prediction response includes `snomed_code` and `snomed_term` alongside the disease label, probability score, warnings, and recommendations.

---

## 8. Flask API Documentation

**File**: `app.py`

The API is organized into three groups as required by the project specification:

### Group 1: Data Handling API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scrape` | POST | Scrapes NHS Inform A-Z pages and stores in MongoDB |
| `/api/conditions` | GET | Returns all conditions from the database |
| `/api/conditions/<name>` | GET | Returns a specific condition by name |

**Example - Scrape:**
```bash
curl -X POST http://localhost:5000/api/scrape
```
Response: `{"message": "Scraped 291 conditions", "count": 291}`

### Group 2: Preprocessing API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/preprocess` | POST | Cleans all raw data -> Conditions collection |
| `/api/preprocess/single` | POST | Cleans a single text blob |

**Example - Preprocess single text:**
```bash
curl -X POST http://localhost:5000/api/preprocess/single \
  -H "Content-Type: application/json" \
  -d '{"text": "The patient has severe headaches and blurred vision."}'
```

### Group 3: Model API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Main prediction endpoint |
| `/api/train/<type>` | POST | Train a model (baseline/lstm/transformer) |
| `/api/models` | GET | List all saved models in MongoDB |
| `/api/models/save` | POST | Upload model artifacts to GridFS |
| `/api/models/<name>` | DELETE | Delete a model from MongoDB |
| `/api/compare` | POST | Compare all models side-by-side |

**Example - Predict:**
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

**Response:**
```json
{
  "model": "transformer",
  "predictions": [
    {
      "label": "Costochondritis",
      "score": 0.2924,
      "snomed_code": "393606005",
      "snomed_term": "Costochondritis",
      "warnings": "",
      "recommendations": "costochondritis often get better..."
    },
    ...
  ],
  "metadata": {"age": "35", "gender": "male"}
}
```

### Demographic Reranking
When age and/or gender are provided, predictions are reranked using post-prediction demographic weighting. This boosts conditions more relevant to the patient's demographic (e.g., childhood diseases for young patients, prostate conditions for males). The original model predictions are not modified -- only the ranking order changes.

---

## 9. Web User Interface

**File**: `templates/index.html`

The web UI provides a user-friendly interface at `http://localhost:5000/`:

- **Symptoms input**: Free-text textarea for describing symptoms
- **Patient metadata**: Optional age and gender fields for demographic reranking
- **Model selection**: Dropdown to choose TF-IDF, LSTM, or BioBERT
- **Top-K selector**: Choose how many predictions to display (3, 5, or 10)
- **Results display**: Shows ranked predictions with:
  - Disease name and confidence score
  - SNOMED CT code (as a styled tag)
  - Warning messages (yellow text)
  - Recommendations/treatment advice (green text)
- **Dark theme**: Professional medical interface using Pico CSS

---

## 10. Model Evaluation Report

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy@1 | Correct disease in top-1 prediction |
| Accuracy@3 | Correct disease in top-3 predictions |
| Accuracy@5 | Correct disease in top-5 predictions |
| F1 Macro | Harmonic mean of precision and recall, averaged across all classes |

### Results Comparison

| Model | Acc@1 | Acc@3 | Acc@5 | F1 Macro |
|-------|-------|-------|-------|----------|
| TF-IDF + Logistic Regression | 63.1% | 80.0% | 86.2% | 56.2% |
| BiLSTM + TF-IDF Ensemble | 53.4% | 76.2%* | 84.5% | 46.7% |
| **BioBERT Transformer** | **81.8%** | **92.8%** | **96.9%** | **77.8%** |

*Ensemble accuracy (standalone LSTM: 67.9% Acc@3)

### Analysis

1. **BioBERT is the clear winner** with 81.8% top-1 accuracy -- nearly 19 percentage points above the TF-IDF baseline. Its pre-training on biomedical text gives it a fundamental advantage in understanding medical terminology.

2. **TF-IDF performs surprisingly well** as a baseline (63.1% Acc@1). Its strength is keyword matching -- when symptoms contain distinctive words that strongly associate with a specific condition.

3. **LSTM benefits from ensemble** with TF-IDF. The standalone LSTM only achieves 53.4% Acc@1, but the ensemble combines the LSTM's sequential understanding with TF-IDF's keyword strength.

4. **Top-5 accuracy** is high across all models (84-97%), meaning the correct diagnosis is almost always in the top 5 suggestions. This is clinically useful as a decision-support tool.

5. **Score distribution**: Scores appear "low" (e.g., 25%) because probability is distributed across 291 classes. A score of 0.25 for one condition actually means the model is quite confident, since random chance would give 0.34%.

### Model Storage
All three models are saved in MongoDB GridFS with full metadata:
- Binary model files (zipped artifact directories)
- Label encoders for all 291 classes
- Training metrics and hyperparameters
- Timestamps for version tracking

---

## 11. How to Run the Project

### Prerequisites
```
Python 3.11+
MongoDB (running on localhost:27017)
```

### Install Dependencies
```bash
pip install flask flask-cors pymongo requests beautifulsoup4 nltk
pip install scikit-learn numpy pandas
pip install tensorflow
pip install torch transformers
pip install tqdm joblib
```

### Step-by-Step Execution

#### Step 1: Scrape Data
```bash
# Option A: Via API
python app.py  # Start server
curl -X POST http://localhost:5000/api/scrape

# Option B: Via notebook
# Open and run web_scrapping.ipynb
```

#### Step 2: Preprocess Data
```bash
# Via API
curl -X POST http://localhost:5000/api/preprocess

# Or via notebook
# Open and run data_preprocessing.ipynb
```

#### Step 3: Train Models

```bash
# Train TF-IDF baseline + LSTM ensemble (saves to MongoDB)
python retrain_models.py

# Train BioBERT transformer (saves to MongoDB)
python train_transformer.py --mongo

# Build SNOMED CT code mappings
python build_snomed_map.py
```

#### Step 4: Run the Application
```bash
python app.py
# Open http://localhost:5000 in your browser
```

### Environment Variables (optional)
```
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=Medical_Diagnosis
```

---

## 12. File Structure

```
Medical Diagnosis AI/
|
|-- app.py                          # Main Flask API (all 3 API groups + web UI)
|-- retrain_models.py               # Trains TF-IDF baseline + LSTM ensemble
|-- train_transformer.py            # Fine-tunes BioBERT transformer
|-- build_snomed_map.py             # Maps conditions to SNOMED CT codes
|
|-- inference_baseline.py           # TF-IDF inference class
|-- inference_lstm.py               # LSTM + TF-IDF ensemble inference class
|-- inference_transformer.py        # BioBERT inference class
|-- model_store.py                  # MongoDB GridFS model save/load/delete
|
|-- artifacts_baseline/             # TF-IDF model artifacts
|   |-- vectorizer.pkl              #   TF-IDF vectorizer
|   |-- clf.pkl                     #   Logistic Regression classifier
|   |-- label_encoder.pkl           #   Label encoder (291 classes)
|   |-- metrics.json                #   Evaluation metrics
|
|-- artifacts_lstm/                 # LSTM model artifacts
|   |-- model.keras                 #   Trained Keras BiLSTM model
|   |-- tokenizer.pkl               #   Text tokenizer
|   |-- label_encoder.pkl           #   Label encoder
|   |-- config.json                 #   MAXLEN + ensemble weight
|
|-- artifacts_transformer/          # BioBERT model artifacts
|   |-- model/                      #   HuggingFace model weights
|   |-- tokenizer/                  #   BioBERT tokenizer
|   |-- label_encoder.pkl           #   Label encoder
|   |-- metrics.json                #   Evaluation metrics
|
|-- templates/
|   |-- index.html                  # Web UI (dark theme, SNOMED codes)
|
|-- pipeline_semantic_enrichment.py # UMLS entity linking with SciSpacy
|-- pipeline_train_test_split.py    # Stratified train/test splitting
|-- pipeline_data_augmentation.py   # T5 paraphrasing augmentation
|-- pipeline_build_embeddings.py    # BioBERT semantic embeddings
|-- pipeline_umls_snomed_map.py     # UMLS-to-SNOMED linking
|
|-- web_scrapping.ipynb             # Jupyter notebook for scraping
|-- data_preprocessing.ipynb        # Jupyter notebook for preprocessing
|-- semantic_embedding_biobert.ipynb # Jupyter notebook for embeddings
|
|-- SNOMED CT/                      # SNOMED CT description files (International + US)
|
|-- MedicalDiagnosisProject_StudentGuide (1).pdf  # Project requirements
|-- PROJECT_DOCUMENTATION.md        # This document
```

---

## Checklist: PDF Requirements Coverage

| # | Requirement | Status | Implementation |
|---|-------------|--------|----------------|
| 1 | Scrape NHS Inform A-Z into MongoDB | DONE | app.py POST /api/scrape, web_scrapping.ipynb |
| 2 | Preprocess symptoms, causes, warnings, recommendations | DONE | app.py POST /api/preprocess, data_preprocessing.ipynb |
| 3 | Build NLP hybrid models (RNN, LSTM, Transformers) | DONE | retrain_models.py (TF-IDF + LSTM), train_transformer.py (BioBERT) |
| 4 | Predict conditions from free-text + user metadata | DONE | app.py POST /predict with age/gender reranking |
| 5 | Rank conditions with probabilities | DONE | Top-K softmax probabilities, rescaled to sum to 1.0 |
| 6 | Flask API with scraping and prediction endpoints | DONE | 3 API groups: Data Handling, Preprocessing, Model |
| 7 | Save trained models to MongoDB | DONE | model_store.py with GridFS, 3 models stored |
| DB | Conditions Collection (condition, symptoms, causes, warnings, recommendations) | DONE | 291 documents |
| DB | Models Collection (name, type, gridfs_id, labels, metrics, created) | DONE | 3 model documents |
| T1 | Data Preprocessing | DONE | Tokenization, stopwords, lemmatization |
| T2 | Baseline Models (TF-IDF + Logistic Regression) | DONE | 63.1% Acc@1, 86.2% Acc@5 |
| T3 | Deep Learning Models (RNN/LSTM) | DONE | BiLSTM ensemble, 76.2% Acc@3 |
| T4 | Transformers (BERT/BioBERT) | DONE | BioBERT, 81.8% Acc@1, 96.9% Acc@5 |
| T5 | Model Comparison | DONE | POST /api/compare, evaluation metrics for all 3 |
| T6 | Probability Ranking | DONE | Sorted by score, Top-K configurable |
| T7 | Recommendations | DONE | Warnings + recommendations from DB attached to predictions |
| + | SNOMED CT codes | DONE | 291/291 conditions mapped, displayed in predictions |
| + | Web UI | DONE | Dark theme, model selector, SNOMED tags |
| + | Documentation | DONE | This document |
