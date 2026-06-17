# Maternal Risk Classification using Machine Learning

> **Dataset:** 2023 Indonesian Health Survey (SKI) — Ministry of Health of Indonesia  
> **Models:** Random Forest vs XGBoost · **Framework:** CRISP-DM  
> **Task:** 3-class classification — Low Risk / High Risk / Very High Risk

---

## Overview

This study applies a comparative machine learning approach to classify maternal risk levels based on the **Poedji Rochjati Score Card (KSPR)** parameters derived from the SKI 2023 national survey dataset.

Key methodological features:

| Aspect | Implementation |
|---|---|
| Pipeline (no leakage) | `imblearn.Pipeline` — SMOTE applied only inside CV folds |
| Hyperparameter tuning | `RandomizedSearchCV` · 5-Fold CV · 50 iter (RF), 60 iter (XGBoost) |
| Statistical validation | 10-Fold CV + **Wilcoxon Signed-Rank Test** |
| Interpretability (XAI) | **SHAP** — Summary Plot, Waterfall, Dependence Plots |

---

## Repository Contents

```
.
├── main.py       ← single unified pipeline script
├── README.md
└── .gitignore
```

> The raw dataset, trained model cache, output figures, and manuscript drafts
> are excluded from this repository (see `.gitignore`).

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            imbalanced-learn shap scipy joblib pillow
```

### 2. Prepare dataset

Place the SKI 2023 dataset file in the same directory as `main.py`,
then set the `DATA_FILE` variable at the top of `main.py` accordingly.

> **Privacy:** The dataset is sourced from the
> [Ministry of Health of Indonesia](https://layanandata.kemkes.go.id/)
> and is excluded from this repository to protect respondent privacy.

### 3. Run

```bash
python main.py
```

**First run** (~30–60 min, hardware-dependent):
1. Loads and preprocesses the dataset
2. Trains baseline models (RF & XGBoost, without tuning)
3. Performs hyperparameter tuning via `RandomizedSearchCV` (5-Fold CV)
4. Saves trained models to a local cache file
5. Runs 10-Fold CV + Wilcoxon Signed-Rank Test
6. Generates all figures to `figures_en/` folder

**Subsequent runs** (cache found → training skipped, ~5–10 min):  
Loads the cached models and proceeds directly to statistical validation and figure generation.

---

## Results

| Model | Accuracy | F1-Macro | ROC-AUC (OvR) |
|---|---|---|---|
| RF Baseline | 0.9539 | 0.8474 | 0.9913 |
| XGB Baseline | 0.9565 | 0.8566 | 0.9940 |
| RF Tuned | 0.9568 | 0.8635 | 0.9914 |
| **XGB Tuned** | **0.9614** | **0.9207** | **0.9946** |

Wilcoxon Signed-Rank Test: **p < 0.05** — the performance difference between
Tuned XGBoost and Tuned Random Forest is statistically significant.

---

*This project is developed for scientific research purposes in the context of
maternal health risk early-warning systems.*
