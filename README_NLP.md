# 🏦 Banking Intent Classifier — NLP Assignment

Fine-tuned **DistilBERT** model for intent classification on the [Banking77](https://huggingface.co/datasets/banking77) dataset, achieving **>90% accuracy**. Includes a TF-IDF + Logistic Regression baseline and a live inference pipeline that returns human-readable category names.

---

## 📌 Overview

Online banking assistants need to understand the **intent** behind a customer's message (e.g. "Why was my card declined?" → `card_not_working`). This project:

1. **Explores** the Banking77 dataset (77 intent classes, ~10k queries)
2. Trains a **baseline** classifier (TF-IDF + Logistic Regression)
3. Fine-tunes **DistilBERT** for sequence classification
4. **Evaluates** both models on a held-out test set
5. Provides a **live inference** pipeline for custom queries

---

## 🗂️ Project Structure

```
├── NLP_Assignment.ipynb    # Main notebook
├── train.csv               # Training data (downloaded at runtime)
├── test.csv                # Test data (downloaded at runtime)
├── results/                # Model checkpoints (created at runtime)
└── logs/                   # Training logs (created at runtime)
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Baseline model | scikit-learn `LogisticRegression` + `TfidfVectorizer` |
| Transformer model | `distilbert-base-uncased` (HuggingFace) |
| Training framework | HuggingFace `Trainer` API |
| Metrics | Accuracy, Macro-F1 |
| Dataset | [Banking77](https://huggingface.co/datasets/banking77) |
| Runtime | Google Colab / Kaggle |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install transformers evaluate accelerate datasets scikit-learn pandas
```

### 2. Download the data

The first notebook cell handles this automatically:

```bash
wget -q -nc "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
wget -q -nc "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
```

### 3. Run the notebook

Open `NLP_Assignment.ipynb` and run all cells in order.

---

## 🔄 Pipeline Walkthrough

### Part 1 — Data Exploration

```python
train_df.shape   # (10003, 2)
test_df.shape    # (3080, 2)

train_df['category'].value_counts()  # 77 classes, ~130 samples each
```

- **10 003** training samples, **3 080** test samples
- **77** intent classes — near-balanced (~100–150 samples per class)
- Columns: `text` (query string), `category` (intent label)

---

### Part 2 — Baseline: TF-IDF + Logistic Regression

```
Raw text
   │
   ▼
TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)
   │
   ▼
LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=2000)
   │
   ▼
Predicted category (string label)
```

The baseline provides a strong starting point for comparison against the transformer.

---

### Part 3 — DistilBERT Fine-Tuning

```
Raw text
   │
   ▼
LabelEncoder  →  integer label IDs (0–76)
   │
   ▼
Train (85%) / Validation (15%) split
   │
   ▼
DistilBertTokenizer  (max_length=512, padding, truncation)
   │
   ▼
DistilBertForSequenceClassification (num_labels=77)
   │
   ▼
Trainer (6 epochs, lr=2e-5, batch=16, best model by macro-F1)
   │
   ▼
Evaluation on test set
```

**Training arguments:**

| Parameter | Value |
|---|---|
| Epochs | 6 |
| Batch size (train/eval) | 16 |
| Learning rate | 2e-5 |
| Evaluation strategy | every epoch |
| Best model selection | highest macro-F1 on validation set |

---

### Part 4 — Metrics

**Why Accuracy?**
Measures the overall percentage of correctly classified queries — intuitive and easy to interpret.

**Why Macro-F1?**
The harmonic mean of precision and recall, averaged equally across all 77 classes. More robust to class imbalance than micro-F1, and a better signal for rare intent classes.

**Target:** ≥ 90% on both metrics.

---

### Part 5 — Live Inference

The `pipeline("text-classification", ...)` wrapper is used for inference. Because HuggingFace pipelines return raw `"LABEL_<id>"` strings, the predicted integer ID is decoded back to the original category name using the fitted `LabelEncoder`:

```python
result = classifier(text)[0]                           # {'label': 'LABEL_4', 'score': 0.97}
label_id = int(result['label'].split('_')[1])          # 4
category = label_encoder.inverse_transform([label_id])[0]  # 'card_not_working'
```

**Example queries and predictions:**

| Query | Predicted Intent |
|---|---|
| How can I troubleshoot issues with card acceptance at ATMs? | `card_acceptance` |
| What should I do if my virtual card is not working for online transactions? | `virtual_card_not_working` |
| How can I resolve problems with contactless payments not working? | `contactless_not_working` |

---

## 📊 Results

| Model | Validation Accuracy | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.85 | ~0.85 | — |
| DistilBERT (fine-tuned) | **>0.92** | **>0.92** | **>0.91** |

---

## 📝 Notes

- Do **not** use `datasets.load('banking77')` — the assignment requires working with the raw CSV files.
- The test set is used only for final evaluation, never for validation during training.
- Training takes ~20–30 minutes on a T4 GPU (Google Colab free tier).

---

## 📄 License

This project is for educational purposes.
