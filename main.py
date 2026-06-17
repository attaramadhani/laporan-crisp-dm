"""
main.py  ──  Maternal Risk Classification using SKI 2023 Dataset
================================================================
Unified pipeline script. Run this single file end-to-end.

Stages
------
  STAGE 1 – Data Loading & Preprocessing
  STAGE 2 – Train/Test Split (80/20 Stratified)
  STAGE 3 – Baseline Models (RF & XGBoost without tuning)
  STAGE 4 – Hyperparameter Tuning (ImbPipeline + RandomizedSearchCV, 5-Fold CV)
  STAGE 5 – Statistical Validation (10-Fold CV, Wilcoxon Signed-Rank Test)
  STAGE 6 – Figure Generation (English labels, saved to figures_en/)

Cache
-----
  All trained models and predictions are serialised to v5_cache.pkl.
  On subsequent runs, STAGE 3 & 4 are skipped automatically (loaded from cache).

Hardware target : MSI Cyborg 15 A13V (i5-13420H, 12 threads)
Dataset         : final_dataset_kspr_attala.csv
Python ≥ 3.9    : pip install scikit-learn xgboost imbalanced-learn shap
                  matplotlib seaborn scipy joblib pillow
"""

import warnings
warnings.filterwarnings("ignore")

import io, os, sys, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import shap
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, balanced_accuracy_score,
                                      classification_report, confusion_matrix,
                                      f1_score, matthews_corrcoef,
                                      roc_auc_score, roc_curve, auc,
                                      precision_recall_curve,
                                      average_precision_score)
from sklearn.model_selection  import (train_test_split, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.preprocessing    import label_binarize
from xgboost                  import XGBClassifier
from imblearn.over_sampling   import SMOTE
from imblearn.pipeline        import Pipeline as ImbPipeline
from scipy.stats              import wilcoxon

# ═══════════════════════════════════════════════════════════════════════════ #
#  CONFIGURATION                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #
DATA_FILE    = "final_dataset_kspr_attala.csv"
CACHE_FILE   = "v5_cache.pkl"
OUTPUT_DIR   = "figures_en"
RANDOM_STATE = 42
N_JOBS       = 10     # adjust to your CPU thread count
N_ITER_RF    = 50     # RandomizedSearchCV iterations for Random Forest
N_ITER_XGB   = 60     # RandomizedSearchCV iterations for XGBoost

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── English feature name translation dictionary ──────────────────────────── #
FEATURE_TRANSLATION = {
    "id_anggota_rt"                        : "RT Member ID",
    "id_rumah_tangga"                      : "Household ID",
    "id_provinsi"                          : "Province ID",
    "id_kabupaten"                         : "Regency ID",
    "tipe_desa_kota"                       : "Urban/Rural Residence",
    "umur_ibu_tahun"                       : "Maternal Age (years)",
    "pendidikan"                           : "Maternal Education Level",
    "pekerjaan"                            : "Maternal Occupation",
    "umur_hamil_pertama"                   : "Age at First Pregnancy",
    "total_hamil_gravida"                  : "Gravida (Total Pregnancies)",
    "total_lahir_paritas"                  : "Parity (Prior Births)",
    "total_keguguran_abortus"              : "Total Miscarriages",
    "pernah_melahirkan_periode_ini"        : "Recent Birth (in Period)",
    "status_hamil_kembar"                  : "Multiple Pregnancy (Twins)",
    "periksa_kehamilan_medis"              : "ANC Access (Medical)",
    "usia_kandungan_periksa_pertama_bulan" : "Gestational Age at First ANC (months)",
    "faskes_anc_tersering"                 : "Most Frequent ANC Facility",
    "anc_ukur_tinggi_badan"               : "ANC Height Check",
    "anc_timbang_berat"                   : "ANC Weight Monitored",
    "anc_tensi_darah"                     : "ANC Blood Pressure Check",
    "anc_tes_hb"                          : "ANC Hemoglobin Test",
    "penolong_persalinan"                 : "Birth Attendant",
    "tempat_persalinan"                   : "Place of Delivery",
    "metode_persalinan_sesar"             : "Cesarean Delivery Method",
    "terima_gizi_karena_anemia"           : "Received Anemia Nutrition",
    "terima_gizi_karena_anemia_2"         : "Received Anemia Nutrition (Phase 2)",
    "konsumsi_tablet_tambah_darah"        : "Iron Pill Consumption",
    "aman_tidak_ada_faktor_risiko"        : "Safe (No Risk Factors)",
    "muntah_diare_saat_hamil"             : "Severe Vomiting/Diarrhea",
    "demam_saat_hamil"                    : "Gestational Fever",
    "hipertensi_saat_hamil"              : "Gestational Hypertension",
    "janin_kurang_gerak"                 : "Reduced Fetal Movement",
    "pendarahan_saat_hamil"              : "Gestational Bleeding",
    "ketuban_pecah_dini_saat_hamil"      : "Premature Rupture of Membranes",
    "sakit_kencing_saat_hamil"           : "Urinary Pain during Pregnancy",
    "batuk_lama_saat_hamil"              : "Chronic Cough during Pregnancy",
    "sesak_napas_saat_hamil"             : "Dyspnea during Pregnancy",
    "nyeri_dada_saat_hamil"              : "Chest Pain during Pregnancy",
    "bengkak_kaki_saat_hamil"            : "Swollen Feet during Pregnancy",
    "kejang_saat_hamil"                  : "Gestational Convulsions",
    "komplikasi_lain_saat_hamil"         : "Other Pregnancy Complications",
    "tidak_ada_komplikasi_hamil"         : "No Gestational Complications",
    "posisi_janin_sungsang"              : "Breech Fetal Position",
    "pendarahan_saat_bersalin"           : "Delivery Bleeding",
    "kejang_saat_bersalin"              : "Delivery Convulsions",
    "ketuban_pecah_dini_saat_bersalin"  : "PROM during Delivery",
    "partus_lama"                        : "Prolonged Labor",
    "lilitan_tali_pusar"                : "Nuchal Cord",
    "plasenta_previa"                   : "Placenta Previa",
    "plasenta_tertinggal"               : "Retained Placenta",
    "hipertensi_saat_bersalin"          : "Delivery Hypertension",
    "komplikasi_lain_saat_bersalin"     : "Other Delivery Complications",
    "tidak_ada_komplikasi_bersalin"     : "No Prior Delivery Complications",
    "pendarahan_nifas"                  : "Postpartum Hemorrhage",
    "cairan_berbau_nifas"               : "Foul Postpartum Discharge",
    "bengkak_pusing_nifas"              : "Postpartum Edema & Dizziness",
    "kejang_nifas"                      : "Postpartum Convulsions",
    "demam_nifas"                       : "Postpartum Fever",
    "payudara_bengkak_nifas"            : "Mastitis/Breast Engorgement",
    "depresi_nifas"                     : "Postpartum Depression",
    "tidak_ada_komplikasi_nifas"        : "No Postpartum Complications",
    "label_risiko"                      : "Risk Label",
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  STAGE 1 – LOAD DATA                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #
log("STAGE 1 ─ Loading dataset...")
if not os.path.exists(DATA_FILE):
    log(f"ERROR: '{DATA_FILE}' not found. Aborting.")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns.")

# ═══════════════════════════════════════════════════════════════════════════ #
#  STAGE 2 – PREPROCESSING & TRAIN/TEST SPLIT                                #
# ═══════════════════════════════════════════════════════════════════════════ #
log("STAGE 2 ─ Preprocessing & stratified 80/20 split...")

# Drop leakage-prone and ID columns
drop_cols = ["label_risiko",
             "id_anggota_rt", "id_rumah_tangga", "id_provinsi", "id_kabupaten"]
if "metode_persalinan_sesar" in df.columns: drop_cols.append("metode_persalinan_sesar")
if "operasi_caesar"          in df.columns: drop_cols.append("operasi_caesar")
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(drop_cols, axis=1)
y = df["label_risiko"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Documentation: SMOTE counts (simulated once, outside CV)
_sm = SMOTE(random_state=RANDOM_STATE)
_, _y_sm = _sm.fit_resample(X_train, y_train)
dist_after_smote = dict(pd.Series(_y_sm).value_counts().sort_index())
del _sm, _y_sm

log(f"  Train n={len(X_train):,}  |  Test n={len(X_test):,}  |  Features={X.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════ #
#  STAGE 3 & 4 – TRAINING / TUNING  (skipped if cache exists)               #
# ═══════════════════════════════════════════════════════════════════════════ #
if os.path.exists(CACHE_FILE):
    log(f"STAGE 3/4 ─ Cache found → loading '{CACHE_FILE}' (skipping training)...")
    cache = joblib.load(CACHE_FILE)

    acc_rf_b     = cache["acc_rf_b"];    f1_rf_b    = cache["f1_rf_b"]
    roc_rf_b     = cache["roc_rf_b"];    cm_rf_b    = cache["cm_rf_b"]
    report_rf_b  = cache["report_rf_b"]; y_pred_rf_b  = cache["y_pred_rf_b"]
    y_proba_rf_b = cache["y_proba_rf_b"]; time_rf_base = cache["time_rf_base"]

    acc_xgb_b     = cache["acc_xgb_b"];   f1_xgb_b  = cache["f1_xgb_b"]
    roc_xgb_b     = cache["roc_xgb_b"];   cm_xgb_b  = cache["cm_xgb_b"]
    report_xgb_b  = cache["report_xgb_b"]; y_pred_xgb_b  = cache["y_pred_xgb_b"]
    y_proba_xgb_b = cache["y_proba_xgb_b"]; time_xgb_base = cache["time_xgb_base"]

    acc_rf_t     = cache["acc_rf_t"];   f1_rf_t    = cache["f1_rf_t"]
    roc_rf_t     = cache["roc_rf_t"];   cm_rf_t    = cache["cm_rf_t"]
    report_rf_t  = cache["report_rf_t"]; y_pred_rf_t  = cache["y_pred_rf_t"]
    y_proba_rf_t = cache["y_proba_rf_t"]; rf_tuned  = cache["rf_tuned"]
    rf_best      = cache["rf_best"];    time_rf_tune  = cache["time_rf_tune"]

    acc_xgb_t     = cache["acc_xgb_t"];  f1_xgb_t   = cache["f1_xgb_t"]
    roc_xgb_t     = cache["roc_xgb_t"];  cm_xgb_t   = cache["cm_xgb_t"]
    report_xgb_t  = cache["report_xgb_t"]; y_pred_xgb_t  = cache["y_pred_xgb_t"]
    y_proba_xgb_t = cache["y_proba_xgb_t"]; xgb_tuned = cache["xgb_tuned"]
    xgb_best      = cache["xgb_best"];  time_xgb_tune = cache["time_xgb_tune"]

    log(f"  RF  Baseline → Acc={acc_rf_b:.4f}  F1={f1_rf_b:.4f}  AUC={roc_rf_b:.4f}")
    log(f"  XGB Baseline → Acc={acc_xgb_b:.4f}  F1={f1_xgb_b:.4f}  AUC={roc_xgb_b:.4f}")
    log(f"  RF  Tuned    → Acc={acc_rf_t:.4f}  F1={f1_rf_t:.4f}  AUC={roc_rf_t:.4f}")
    log(f"  XGB Tuned    → Acc={acc_xgb_t:.4f}  F1={f1_xgb_t:.4f}  AUC={roc_xgb_t:.4f}")

else:
    # ── STAGE 3: Baseline Models ──────────────────────────────────────────── #
    log("STAGE 3 ─ Training baseline models (SMOTE applied outside pipeline)...")
    _smote = SMOTE(random_state=RANDOM_STATE)
    X_train_s, y_train_s = _smote.fit_resample(X_train, y_train)

    t0 = time.time()
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf_base.fit(X_train_s, y_train_s)
    time_rf_base = time.time() - t0
    y_pred_rf_b  = rf_base.predict(X_test)
    y_proba_rf_b = rf_base.predict_proba(X_test)
    acc_rf_b  = accuracy_score(y_test, y_pred_rf_b)
    f1_rf_b   = f1_score(y_test, y_pred_rf_b, average="macro")
    roc_rf_b  = roc_auc_score(y_test, y_proba_rf_b, multi_class="ovr")
    report_rf_b = classification_report(y_test, y_pred_rf_b, output_dict=True)
    cm_rf_b   = confusion_matrix(y_test, y_pred_rf_b)

    t0 = time.time()
    xgb_base = XGBClassifier(
        random_state=RANDOM_STATE, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist", verbosity=0, n_jobs=N_JOBS)
    xgb_base.fit(X_train_s, y_train_s)
    time_xgb_base = time.time() - t0
    y_pred_xgb_b  = xgb_base.predict(X_test)
    y_proba_xgb_b = xgb_base.predict_proba(X_test)
    acc_xgb_b  = accuracy_score(y_test, y_pred_xgb_b)
    f1_xgb_b   = f1_score(y_test, y_pred_xgb_b, average="macro")
    roc_xgb_b  = roc_auc_score(y_test, y_proba_xgb_b, multi_class="ovr")
    report_xgb_b = classification_report(y_test, y_pred_xgb_b, output_dict=True)
    cm_xgb_b  = confusion_matrix(y_test, y_pred_xgb_b)

    log(f"  RF  Baseline → Acc={acc_rf_b:.4f}  F1={f1_rf_b:.4f}  ({time_rf_base:.1f}s)")
    log(f"  XGB Baseline → Acc={acc_xgb_b:.4f}  F1={f1_xgb_b:.4f}  ({time_xgb_base:.1f}s)")

    # ── STAGE 4: Hyperparameter Tuning with ImbPipeline ───────────────────── #
    log(f"STAGE 4 ─ Tuning RF ({N_ITER_RF} iter × 5-Fold) and XGB ({N_ITER_XGB} iter × 5-Fold)...")
    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Random Forest ---
    rf_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    rf_param_dist = {
        "clf__n_estimators":      [100, 150, 200, 250, 300],
        "clf__max_depth":         [10, 15, 20, 25],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__bootstrap":         [True],
    }
    t0 = time.time()
    rf_rs = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=N_ITER_RF, cv=cv_tune,
        scoring="f1_macro", n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=0)
    rf_rs.fit(X_train, y_train)
    time_rf_tune = time.time() - t0
    rf_tuned      = rf_rs.best_estimator_
    y_pred_rf_t   = rf_tuned.predict(X_test)
    y_proba_rf_t  = rf_tuned.predict_proba(X_test)
    acc_rf_t   = accuracy_score(y_test, y_pred_rf_t)
    f1_rf_t    = f1_score(y_test, y_pred_rf_t, average="macro")
    roc_rf_t   = roc_auc_score(y_test, y_proba_rf_t, multi_class="ovr")
    report_rf_t = classification_report(y_test, y_pred_rf_t, output_dict=True)
    cm_rf_t    = confusion_matrix(y_test, y_pred_rf_t)
    rf_best    = {k.replace("clf__", ""): v for k, v in rf_rs.best_params_.items()}
    log(f"  RF  Tuned → CV-F1={rf_rs.best_score_:.4f}  AUC={roc_rf_t:.4f}  ({time_rf_tune/60:.1f} min)")

    # --- XGBoost ---
    xgb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   XGBClassifier(
            random_state=RANDOM_STATE, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist", verbosity=0, n_jobs=1)),
    ])
    xgb_param_dist = {
        "clf__n_estimators":     [100, 150, 200, 250, 300],
        "clf__max_depth":        [3, 4, 5, 6, 7],
        "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__gamma":            [0, 0.1, 0.2],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_alpha":        [0, 0.1, 0.5],
        "clf__reg_lambda":       [0.5, 1.0, 1.5],
    }
    t0 = time.time()
    xgb_rs = RandomizedSearchCV(
        xgb_pipe, xgb_param_dist, n_iter=N_ITER_XGB, cv=cv_tune,
        scoring="f1_macro", n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=0)
    xgb_rs.fit(X_train, y_train)
    time_xgb_tune = time.time() - t0
    xgb_tuned      = xgb_rs.best_estimator_
    y_pred_xgb_t   = xgb_tuned.predict(X_test)
    y_proba_xgb_t  = xgb_tuned.predict_proba(X_test)
    acc_xgb_t   = accuracy_score(y_test, y_pred_xgb_t)
    f1_xgb_t    = f1_score(y_test, y_pred_xgb_t, average="macro")
    roc_xgb_t   = roc_auc_score(y_test, y_proba_xgb_t, multi_class="ovr")
    report_xgb_t = classification_report(y_test, y_pred_xgb_t, output_dict=True)
    cm_xgb_t    = confusion_matrix(y_test, y_pred_xgb_t)
    xgb_best    = {k.replace("clf__", ""): v for k, v in xgb_rs.best_params_.items()}
    log(f"  XGB Tuned → CV-F1={xgb_rs.best_score_:.4f}  AUC={roc_xgb_t:.4f}  ({time_xgb_tune/60:.1f} min)")

    # ── Save cache ────────────────────────────────────────────────────────── #
    log(f"  Saving cache to '{CACHE_FILE}'...")
    joblib.dump({
        "acc_rf_b": acc_rf_b,   "f1_rf_b": f1_rf_b,    "roc_rf_b": roc_rf_b,
        "report_rf_b": report_rf_b, "cm_rf_b": cm_rf_b,
        "y_pred_rf_b": y_pred_rf_b, "y_proba_rf_b": y_proba_rf_b,
        "time_rf_base": time_rf_base,

        "acc_xgb_b": acc_xgb_b, "f1_xgb_b": f1_xgb_b,  "roc_xgb_b": roc_xgb_b,
        "report_xgb_b": report_xgb_b, "cm_xgb_b": cm_xgb_b,
        "y_pred_xgb_b": y_pred_xgb_b, "y_proba_xgb_b": y_proba_xgb_b,
        "time_xgb_base": time_xgb_base,

        "acc_rf_t": acc_rf_t,   "f1_rf_t": f1_rf_t,    "roc_rf_t": roc_rf_t,
        "report_rf_t": report_rf_t, "cm_rf_t": cm_rf_t,
        "y_pred_rf_t": y_pred_rf_t, "y_proba_rf_t": y_proba_rf_t,
        "rf_tuned": rf_tuned, "rf_best": rf_best, "time_rf_tune": time_rf_tune,

        "acc_xgb_t": acc_xgb_t, "f1_xgb_t": f1_xgb_t,  "roc_xgb_t": roc_xgb_t,
        "report_xgb_t": report_xgb_t, "cm_xgb_t": cm_xgb_t,
        "y_pred_xgb_t": y_pred_xgb_t, "y_proba_xgb_t": y_proba_xgb_t,
        "xgb_tuned": xgb_tuned, "xgb_best": xgb_best, "time_xgb_tune": time_xgb_tune,
    }, CACHE_FILE)
    log("  Cache saved.")


# ═══════════════════════════════════════════════════════════════════════════ #
#  STAGE 5 – STATISTICAL VALIDATION                                          #
#  10-Fold CV  +  Wilcoxon Signed-Rank Test  +  Specificity per class        #
# ═══════════════════════════════════════════════════════════════════════════ #
log("STAGE 5 ─ 10-Fold CV + Wilcoxon Signed-Rank Test (est. 5–10 min)...")
cv_wilcoxon = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

rf_cv_f1   = cross_val_score(rf_tuned,  X, y, cv=cv_wilcoxon, scoring="f1_macro",  n_jobs=N_JOBS)
xgb_cv_f1  = cross_val_score(xgb_tuned, X, y, cv=cv_wilcoxon, scoring="f1_macro",  n_jobs=N_JOBS)
rf_cv_acc  = cross_val_score(rf_tuned,  X, y, cv=cv_wilcoxon, scoring="accuracy",  n_jobs=N_JOBS)
xgb_cv_acc = cross_val_score(xgb_tuned, X, y, cv=cv_wilcoxon, scoring="accuracy",  n_jobs=N_JOBS)

stat_w, p_val_w = wilcoxon(rf_cv_f1, xgb_cv_f1)

def _specificity(cm):
    """Return per-class specificity from a confusion matrix."""
    specs = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return specs

spec_rf  = _specificity(cm_rf_t)
spec_xgb = _specificity(cm_xgb_t)

print("\n" + "═"*60)
print("  STATISTICAL VALIDATION RESULTS")
print("═"*60)
print(f"  RF  10-Fold Accuracy : {rf_cv_acc.mean():.4f} ± {rf_cv_acc.std():.4f}")
print(f"  RF  10-Fold F1-Macro : {rf_cv_f1.mean():.4f} ± {rf_cv_f1.std():.4f}")
print(f"  XGB 10-Fold Accuracy : {xgb_cv_acc.mean():.4f} ± {xgb_cv_acc.std():.4f}")
print(f"  XGB 10-Fold F1-Macro : {xgb_cv_f1.mean():.4f} ± {xgb_cv_f1.std():.4f}")
print(f"  Wilcoxon Z-stat      : {stat_w:.4f}")
print(f"  Wilcoxon p-value     : {p_val_w:.5e}  ({'SIGNIFICANT' if p_val_w < 0.05 else 'NOT SIGNIFICANT'})")
print(f"  Specificity RF  [Low / High / VHigh] : {spec_rf[0]:.4f} / {spec_rf[1]:.4f} / {spec_rf[2]:.4f}")
print(f"  Specificity XGB [Low / High / VHigh] : {spec_xgb[0]:.4f} / {spec_xgb[1]:.4f} / {spec_xgb[2]:.4f}")
print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  STAGE 6 – FIGURE GENERATION  (English labels → figures_en/)              #
# ═══════════════════════════════════════════════════════════════════════════ #
log(f"STAGE 6 ─ Generating figures → '{OUTPUT_DIR}/'...")

def _save(path: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, path), dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"  Saved: {path}")

class_colors     = ["#e84393", "#f59e0b", "#10b981"]
class_labels_roc = ["Low Risk (0)", "High Risk (1)", "Very High Risk (2)"]

X_test_en = X_test.rename(columns=FEATURE_TRANSLATION)


# ── Fig 1: Methodology Flowchart ─────────────────────────────────────────── #
def _draw_flowchart(save_name: str):
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.axis("off")
    boxes = [
        ("Raw SKI 2023 Survey Data\n(N = 211,351)",                       0.5, 0.95, 0.60, 0.05),
        ("Feature Audit\n(Verify missing values; 0 missing in primary attributes)", 0.5, 0.86, 0.75, 0.05),
        ("Feature Exclusion\n(Remove ID & cesarean delivery columns to prevent leakage)", 0.5, 0.77, 0.80, 0.05),
        ("Stratified Split\n(80% Train Set: n = 169,080; 20% Test Set: n = 42,271)", 0.5, 0.68, 0.75, 0.05),
        ("Training Partition\n(n = 169,080)",                             0.25, 0.57, 0.35, 0.05),
        ("Isolated Test Set\n(n = 42,271; Zero Leakage)",                 0.75, 0.57, 0.40, 0.05),
        ("SMOTE Resampling\n(k = 5; applied only inside CV folds)",       0.25, 0.45, 0.40, 0.05),
        ("Model Training & Tuning\n(RandomizedSearchCV on F1-Macro)",     0.25, 0.33, 0.42, 0.05),
        ("Model Performance Evaluation\n(Accuracy, Balanced Accuracy, F1-Macro, Specificity, MCC)", 0.5, 0.21, 0.85, 0.05),
        ("SHAP Explanations\n(Global feature importance & local patient predictions)", 0.5, 0.10, 0.80, 0.05),
    ]
    for text, x, y, w, h in boxes:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor="#f8f9fa", edgecolor="#1d3557", lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9, color="#1d3557", fontweight="bold")
    arrows = [
        ((0.5, 0.885), (0.5, 0.925)), ((0.5, 0.795), (0.5, 0.835)),
        ((0.5, 0.705), (0.5, 0.745)), ((0.25, 0.595), (0.5, 0.655)),
        ((0.75, 0.595), (0.5, 0.655)), ((0.25, 0.475), (0.25, 0.545)),
        ((0.25, 0.355), (0.25, 0.425)), ((0.45, 0.235), (0.25, 0.305)),
        ((0.55, 0.235), (0.75, 0.545)), ((0.5, 0.125), (0.5, 0.185)),
    ]
    for xy, xytext in arrows:
        ax.annotate("", xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle="->", color="#1d3557", lw=1.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"  Saved: {save_name}")

_draw_flowchart("fig_methodology.png")


# ── Fig 2: SMOTE Class Distribution ──────────────────────────────────────── #
categories = ["Low Risk", "High Risk", "Very High Risk"]
before = [35725, 117990, 15365]
after  = [dist_after_smote.get(k, 0) for k in [0, 1, 2]]
x = np.arange(len(categories)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, before, w, label="Before SMOTE", color="#ff7f0e", alpha=0.85)
b2 = ax.bar(x + w/2, after,  w, label="After SMOTE",  color="#1f77b4", alpha=0.85)
for rects in (b1, b2):
    for rect in rects:
        ax.annotate(f"{rect.get_height():,}",
                    xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_title("Training Set Class Distribution Before and After SMOTE Rebalancing",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=10)
ax.legend(frameon=True, facecolor="white", edgecolor="none")
ax.grid(True, alpha=0.3, axis="y", linestyle="--")
_save("fig_smote_distribution.png")


# ── Fig 3: Confusion Matrix Comparison (4 models) ────────────────────────── #
fig_cm, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (cm, ttl) in zip(axes.flatten(), [
        (cm_rf_b,  "Random Forest – Baseline (No Resampling)"),
        (cm_xgb_b, "XGBoost – Baseline (No Resampling)"),
        (cm_rf_t,  "Random Forest – Tuned (Pipeline + SMOTE)"),
        (cm_xgb_t, "XGBoost – Tuned (Pipeline + SMOTE)"),
]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low Risk", "High Risk", "Very High Risk"],
                yticklabels=["Low Risk", "High Risk", "Very High Risk"],
                annot_kws={"size": 12, "weight": "bold"}, cbar=True)
    ax.set_title(ttl, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
plt.suptitle("Confusion Matrix Comparison – Baseline vs Tuned Models",
             fontsize=14, fontweight="bold", y=0.98)
_save("fig_confusion_matrix.png")


# ── Fig 3b: Normalized Confusion Matrix (Tuned XGBoost) ──────────────────── #
cm_norm = cm_xgb_t.astype("float") / cm_xgb_t.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["Low Risk", "High Risk", "Very High Risk"],
            yticklabels=["Low Risk", "High Risk", "Very High Risk"],
            annot_kws={"size": 12, "weight": "bold"})
plt.title("Normalized Confusion Matrix – Tuned XGBoost Model",
          fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label", fontsize=11)
_save("fig_confusion_matrix_norm.png")


# ── Fig 4: Feature Importance Comparison ─────────────────────────────────── #
fig_fi, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, pipe_model, color, ttl in zip(
    axes,
    [rf_tuned, xgb_tuned],
    ["#4682B4", "#2E8B57"],
    ["Random Forest (Tuned)", "XGBoost (Tuned)"],
):
    clf = pipe_model.named_steps["clf"]
    imp = clf.feature_importances_
    idx = np.argsort(imp)[-15:]
    y_labels = [FEATURE_TRANSLATION.get(X.columns[i], X.columns[i].replace("_", " ").title())
                for i in idx]
    ax.barh(range(len(idx)), imp[idx], color=color, align="center", alpha=0.85)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top 15 Feature Importance\n{ttl}", fontsize=12,
                 fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
_save("fig_feature_importance.png")


# ── Fig 5: ROC Curves (4 models, OvR per class) ──────────────────────────── #
fig_roc, axes_roc = plt.subplots(2, 2, figsize=(14, 11))
_roc_models = [
    ("RF Baseline",         cache["y_proba_rf_b"]  if os.path.exists(CACHE_FILE) else y_proba_rf_b),
    ("XGBoost Baseline",    cache["y_proba_xgb_b"] if os.path.exists(CACHE_FILE) else y_proba_xgb_b),
    ("RF Tuned (Pipeline)", cache["y_proba_rf_t"]  if os.path.exists(CACHE_FILE) else y_proba_rf_t),
    ("XGB Tuned (Pipeline)",cache["y_proba_xgb_t"] if os.path.exists(CACHE_FILE) else y_proba_xgb_t),
]
for ax, (model_name, y_proba) in zip(axes_roc.flatten(), _roc_models):
    for i, (color, lbl) in enumerate(zip(class_colors, class_labels_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{lbl} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random Classifier")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve – {model_name}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle("ROC Curve One-vs-Rest (OvR) – Baseline vs Tuned Models",
             fontsize=14, fontweight="bold", y=0.98)
_save("fig_roc_curves.png")


# ── Fig 6: Precision-Recall Curve (Tuned XGBoost) ───────────────────────── #
plt.figure(figsize=(8, 7))
for i, (color, lbl) in enumerate(zip(class_colors, class_labels_roc)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba_xgb_t[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_proba_xgb_t[:, i])
    plt.plot(recall, precision, color=color, lw=2, label=f"{lbl} (PR-AUC={ap:.3f})")
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel("Recall (Sensitivity)", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.title("Precision-Recall Curve (OvR) – Tuned XGBoost Model",
          fontsize=12, fontweight="bold", pad=15)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3, linestyle="--")
_save("fig_pr_curves.png")


# ── SHAP computations ─────────────────────────────────────────────────────── #
log("  Computing SHAP values (sample n=300)...")
X_test_samp    = X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE)
X_test_samp_en = X_test_samp.rename(columns=FEATURE_TRANSLATION)

rf_clf  = rf_tuned.named_steps["clf"]
xgb_clf = xgb_tuned.named_steps["clf"]
explainer_rf  = shap.TreeExplainer(rf_clf)
explainer_xgb = shap.TreeExplainer(xgb_clf)
shap_rf  = explainer_rf.shap_values(X_test_samp)
shap_xgb = explainer_xgb.shap_values(X_test_samp)

# Extract class-2 SHAP values for XGBoost
if isinstance(shap_xgb, list) and len(shap_xgb) > 2:
    shap_xgb_c2 = shap_xgb[2]
elif hasattr(shap_xgb, "shape") and len(shap_xgb.shape) == 3:
    shap_xgb_c2 = shap_xgb[:, :, 2]
else:
    shap_xgb_c2 = shap_xgb


# ── Fig 7: SHAP Global Bar Plot — RF vs XGBoost (side-by-side, fixed layout) #
def _shap_bar_to_pil(shap_vals, X_en, title, figsize=(9, 8)):
    """Render one SHAP bar plot into a PIL Image (no overlap artifacts)."""
    plt.close("all")
    shap.summary_plot(shap_vals, X_en, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close("all")
    return Image.open(buf).copy()

log("  Rendering SHAP bar plots...")
img_rf  = _shap_bar_to_pil(shap_rf,  X_test_samp_en,
                             "SHAP Global Feature Importance\n(Tuned Random Forest)")
img_xgb = _shap_bar_to_pil(shap_xgb, X_test_samp_en,
                             "SHAP Global Feature Importance\n(Tuned XGBoost)")

# Equalise heights then stitch
h_max   = max(img_rf.size[1], img_xgb.size[1])
gap     = 20

def _pad_h(img, h):
    if img.size[1] == h: return img
    out = Image.new("RGB", (img.size[0], h), (255, 255, 255))
    out.paste(img, (0, (h - img.size[1]) // 2))
    return out

img_rf  = _pad_h(img_rf,  h_max)
img_xgb = _pad_h(img_xgb, h_max)
combined = Image.new("RGB", (img_rf.size[0] + gap + img_xgb.size[0], h_max), (255, 255, 255))
combined.paste(img_rf,  (0, 0))
combined.paste(img_xgb, (img_rf.size[0] + gap, 0))
combined.save(os.path.join(OUTPUT_DIR, "fig_shap_summary_bar.png"), dpi=(150, 150))
log("  Saved: fig_shap_summary_bar.png")


# ── Fig 8: SHAP Summary Dot Plot (XGBoost, class 2) ─────────────────────── #
plt.figure(figsize=(10, 6.5))
shap.summary_plot(shap_xgb_c2, X_test_samp_en, show=False)
plt.title("XGBoost SHAP Summary Plot – Very High Risk Class",
          fontsize=13, fontweight="bold", pad=15)
_save("fig_shap_summary.png")


# ── Fig 9: SHAP Waterfall Plot (one True Positive, class 2) ──────────────── #
log("  Computing full-test-set SHAP for waterfall...")
shap_xgb_full = explainer_xgb.shap_values(X_test)
if isinstance(shap_xgb_full, list) and len(shap_xgb_full) > 2:
    shap_full_c2 = shap_xgb_full[2]
elif hasattr(shap_xgb_full, "shape") and len(shap_xgb_full.shape) == 3:
    shap_full_c2 = shap_xgb_full[:, :, 2]
else:
    shap_full_c2 = shap_xgb_full

if isinstance(explainer_xgb.expected_value, (list, np.ndarray)):
    ev = np.array(explainer_xgb.expected_value).ravel()[2]
else:
    ev = explainer_xgb.expected_value

vhr_idx = np.where(y_test == 2)[0]
tp_idx  = next(i for i in vhr_idx if y_pred_xgb_t[i] == 2)

plt.figure(figsize=(12, 8))
shap.waterfall_plot(shap.Explanation(
    values=shap_full_c2[tp_idx],
    base_values=ev,
    data=X_test_en.iloc[tp_idx],
    feature_names=X_test_en.columns.tolist(),
), show=False)
plt.title("SHAP Waterfall Plot for a Very High Risk Patient",
          fontsize=14, fontweight="bold", pad=15)
_save("fig_shap_waterfall.png")


# ── Fig 10 & 11: SHAP Dependence Plots ───────────────────────────────────── #
for feat, fname, title in [
    ("Total Miscarriages",
     "fig_shap_dep_miscarriage.png",
     "SHAP Dependence Plot – Miscarriage History (Very High Risk Class)"),
    ("Parity (Prior Births)",
     "fig_shap_dep_parity.png",
     "SHAP Dependence Plot – Parity (Very High Risk Class)"),
    ("Maternal Age (years)",
     "fig_shap_dep.png",
     "SHAP Dependence Plot – Maternal Age (Very High Risk Class)"),
]:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feat, shap_xgb_c2, X_test_samp_en, show=False)
    plt.title(title, fontsize=11, fontweight="bold", pad=15)
    _save(fname)


log("═" * 60)
log("All stages completed successfully.")
log(f"  Figures saved to : ./{OUTPUT_DIR}/")
log(f"  Model cache      : ./{CACHE_FILE}")
log("═" * 60)
