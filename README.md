# Maternal Risk Classification using Machine Learning (SKI 2023)

> **Dataset:** 2023 Indonesian Health Survey (Survei Kesehatan Indonesia - SKI) — Ministry of Health of Indonesia  
> **Author / Penulis:** Attala Alif Ramadhani Tri Hida  
> **Models:** Random Forest vs XGBoost · **Framework:** CRISP-DM  
> **Task:** 3-Class Multiclass Risk Classification — Low Risk / High Risk / Very High Risk

---

## Overview

This repository contains the complete, reproducible machine learning pipeline for classifying maternal health risk levels based on the **Poedji Rochjati Score Card (KSPR)** parameters derived from the SKI 2023 national survey dataset ($N = 211,351$).

### Key Methodological Highlights:

| Methodological Aspect | Technical Implementation |
|---|---|
| **Leakage-Free Pipeline** | `imblearn.Pipeline` — SMOTE resampling ($k=5$) applied strictly inside Cross-Validation folds |
| **Data Cleaning** | Excluded administrative IDs and post-hoc cesarean delivery columns to eliminate data leakage |
| **Hyperparameter Tuning** | `RandomizedSearchCV` · 5-Fold Stratified CV (Random Forest & XGBoost) |
| **Statistical Validation** | 10-Fold CV + **Wilcoxon Signed-Rank Test** for statistical significance testing |
| **Explainable AI (XAI)** | **SHAP** — Summary Bee Swarm Plots, Local Patient Waterfall Plots, Dependence Plots |
| **Automated Reporting** | Generates publication-grade figures (200 DPI) and full Word reports (`laporan_hasil_eksperimen.docx`) |

---

## Repository Contents

```
.
├── main.py       ← Single master pipeline & figure/report generator
├── README.md     ← Documentation & setup guide
└── .gitignore    ← Excludes dataset CSVs, model caches, output figures, and Word reports
```

> **Privacy & Reproducibility:** Raw survey CSV datasets, trained model binary caches (`model_cache.pkl`), generated output figures, and Word document reports are excluded from this repository per `.gitignore` guidelines to protect respondent privacy and maintain a lightweight code repository.

---

## Getting Started & Execution Guide

### 1. Install Required Dependencies

Ensure Python $\ge 3.9$ is installed, then install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            imbalanced-learn shap scipy joblib pillow python-docx
```

### 2. Prepare the Dataset

Place your SKI 2023 dataset CSV file in the same directory as `main.py` and name it `dataset_ski_2023.csv`:

```
.
├── dataset_ski_2023.csv
└── main.py
```

> **Privacy Note:** The SKI 2023 national survey dataset is managed by the [Ministry of Health of Indonesia (Kemenkes RI)](https://layanandata.kemkes.go.id/) and is kept private.

### 3. Run the Pipeline

Execute the master pipeline script:

```bash
python main.py
```

**Automatic Pipeline Behavior:**
1. **Dataset Preprocessing:** Loads `dataset_ski_2023.csv`, drops administrative IDs and leakage columns, performs an 80/20 stratified split into training ($n = 169,080$) and hold-out test ($n = 42,271$) sets.
2. **Model Training & Tuning (First Run):** If `model_cache.pkl` is absent, trains baseline models, performs 5-fold CV hyperparameter tuning using `RandomizedSearchCV`, evaluates hold-out predictions, and automatically saves `model_cache.pkl`.
3. **Cache Loading (Subsequent Runs):** Loads `model_cache.pkl` directly for instant execution (< 10 seconds).
4. **Statistical Validation:** Computes 10-Fold CV summaries, per-class specificity scores, and the Wilcoxon Signed-Rank Test.
5. **Figure Generation:** Renders Figures 2–6 in standard publication font sizes to `figures_final/` and `figures_en/`.
6. **Word Report Export:** Automatically creates `laporan_hasil_eksperimen.docx` containing complete execution tables, statistical test results, and embedded high-resolution figures.

---

## Experimental Results Summary

Evaluated on the independent hold-out test set ($n = 42,271$):

| Model Variant | Accuracy | F1-Macro | ROC-AUC (Macro OvR) | PR-AUC (Macro OvR) |
|---|---|---|---|---|
| Random Forest (Baseline) | 0.9477 | 0.9169 | 0.9931 | 0.9712 |
| XGBoost (Baseline) | 0.9510 | 0.9200 | 0.9944 | 0.9745 |
| Random Forest (Tuned) | 0.9480 | 0.9184 | 0.9934 | 0.9720 |
| **XGBoost (Tuned - Best Model)** | **0.9516** | **0.9207** | **0.9946** | **0.9754** |

### Statistical Significance:
* **Wilcoxon Signed-Rank Test:** $Z = 0.0000$, $p = 1.50 \times 10^{-4}$ ($p < 0.05$) — the performance superiority of Tuned XGBoost over Tuned Random Forest is **statistically significant**.

---

## Author & Citation

* **Author:** Attala Alif Ramadhani Tri Hida  
* **Repository:** [https://github.com/attaramadhani/laporan-crisp-dm](https://github.com/attaramadhani/laporan-crisp-dm)  
* **Purpose:** Research and development of early-warning decision support systems for maternal health risk assessment.
