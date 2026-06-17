# Pregnancy Risk Classification Using Machine Learning (KSPR)

> **Dataset:** 2023 Indonesian Health Survey (SKI) · N = 211,351 records  
> **Models:** Random Forest vs XGBoost · **Framework:** CRISP-DM  
> **Task:** 3-class maternal risk classification — Low / High / Very High Risk

---

## 🚀 Overview

This project performs a comparative analysis between **Random Forest** (parallel bagging) and **XGBoost** (sequential boosting) for classifying maternal risk levels.  
Key methodological highlights:

| Feature | Detail |
|---|---|
| Anti-leakage pipeline | `imblearn.Pipeline` — SMOTE only inside CV folds |
| Tuning | `RandomizedSearchCV` · 5-Fold CV · 50 iter RF, 60 iter XGB |
| Statistical test | 10-Fold CV + **Wilcoxon Signed-Rank Test** |
| Interpretability (XAI) | **SHAP** — Summary, Waterfall, Dependence Plots |
| Output figures | Saved to `figures_en/` (all labels in English) |

---

## 📁 Repository Structure

```
.
├── main.py                         ← ✅ SINGLE unified script (run this)
├── final_dataset_kspr_attala.csv   ← ⛔ excluded via .gitignore (privacy)
├── v5_cache.pkl                    ← ⛔ excluded (auto-generated on first run)
├── figures_en/                     ← output figures (English labels)
│   ├── fig_methodology.png
│   ├── fig_smote_distribution.png
│   ├── fig_confusion_matrix.png
│   ├── fig_confusion_matrix_norm.png
│   ├── fig_feature_importance.png
│   ├── fig_roc_curves.png
│   ├── fig_pr_curves.png
│   ├── fig_shap_summary_bar.png
│   ├── fig_shap_summary.png
│   ├── fig_shap_waterfall.png
│   ├── fig_shap_dep_miscarriage.png
│   ├── fig_shap_dep_parity.png
│   └── fig_shap_dep.png
├── humanized_article_en.md         ← manuscript draft (English)
├── humanized_article.md            ← manuscript draft (Bahasa Indonesia)
└── .gitignore
```

> **Legacy scripts** (`laporan_crisp_dm_v5.py`, `analisis_statistik_v5.py`,  
> `generate_english_plots_v3.py`, `fix_shap_bar.py`) are kept for archival  
> reference but are **superseded by `main.py`**.

---

## 🛠️ How to Run

### 1 · Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            imbalanced-learn shap scipy joblib pillow
```

### 2 · Place dataset
Put `final_dataset_kspr_attala.csv` in the same directory as `main.py`.

### 3 · Execute
```bash
python main.py
```

**What happens on first run (~30–60 min, hardware-dependent):**
1. Loads and preprocesses the SKI 2023 dataset (211,351 rows)
2. Trains RF & XGBoost baseline models
3. Performs hyperparameter tuning (RandomizedSearchCV, 5-Fold CV)
4. Saves trained models to `v5_cache.pkl`
5. Runs 10-Fold CV + Wilcoxon Signed-Rank Test
6. Generates all figures to `figures_en/`

**On subsequent runs** (cache found → stages 3 & 4 skipped, ~5–10 min):
- Loads `v5_cache.pkl` and jumps directly to STAGE 5 & 6.

---

## 📈 Key Results

| Model | Accuracy | F1-Macro | ROC-AUC (OvR) |
|---|---|---|---|
| RF Baseline | 0.9539 | 0.8474 | 0.9913 |
| XGB Baseline | 0.9565 | 0.8566 | 0.9940 |
| RF Tuned | 0.9568 | 0.8635 | 0.9914 |
| **XGB Tuned** | **0.9614** | **0.9207** | **0.9946** |

**Statistical test:** Wilcoxon p < 0.05 → XGBoost superiority is statistically significant.

---

## 🔒 Privacy Note

The raw dataset (`final_dataset_kspr_attala.csv`) sourced from the  
[Ministry of Health of Indonesia](https://layanandata.kemkes.go.id/) is  
excluded from this repository to protect health respondent privacy.

---

## 🇮🇩 Ringkasan (Bahasa Indonesia)

Proyek ini mengklasifikasikan risiko kehamilan (Rendah / Tinggi / Sangat Tinggi) menggunakan dataset SKI 2023 (211.351 data ibu hamil).  
Cukup jalankan **satu script**:

```bash
python main.py
```

Script ini mencakup seluruh pipeline: preprocessing, training, tuning, uji statistik Wilcoxon (10-Fold CV), dan pembuatan semua gambar berbahasa Inggris ke folder `figures_en/`.

---

*Developed for scientific article publication and maternal early-warning system integration.*
