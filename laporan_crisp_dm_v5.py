"""
laporan_crisp_dm_v5.py
======================
Mode: SILENT (verbose=0, warnings suppressed)
Tujuan: Pipeline ML lengkap dengan perbaikan metodologi dari v2:
        - SMOTE dimasukkan ke dalam imblearn.Pipeline agar tidak bocor ke CV
        - N_ITER ditingkatkan (50 RF, 60 XGB) untuk pencarian yang lebih luas
        - max_depth=None dihapus dari search space untuk mencegah overfitting
        - Baseline (tanpa tuning) + Tuned (dengan pipeline CV yang benar)
        - Laporan Word format CRISP-DM lengkap

Hardware target: MSI Cyborg 15 A13V (i5-13420H, 12 threads)
Dataset        : final_dataset_kspr_attala.csv
"""

import warnings
warnings.filterwarnings('ignore')

import os, sys, time, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, accuracy_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

try:
    from docx import Document
    from docx.shared import Inches, Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    print("ERROR: python-docx tidak terinstall. Jalankan: pip install python-docx")
    sys.exit(1)

# =========================================================== #
#  KONSTANTA KONFIGURASI                                      #
# =========================================================== #
N_JOBS       = 10     # 10 dari 12 thread
RANDOM_STATE = 42
N_ITER_RF    = 50     # ditingkatkan dari 20 → 50 (lebih luas)
N_ITER_XGB   = 60     # ditingkatkan dari 25 → 60 (lebih luas)
OUTPUT_FILE  = 'output_eksperimen/Laporan_CRISP_DM_Komputasi_v5.docx'
CACHE_FILE   = 'output_eksperimen/v5_cache.pkl'
DATA_FILE    = 'final_dataset_kspr_attala.csv'

class_labels = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================================================== #
#  1. LOAD DATA                                               #
# =========================================================== #
log("1/6 Load data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    log(f"ERROR: File '{DATA_FILE}' tidak ditemukan.")
    sys.exit(1)
except Exception as e:
    log(f"ERROR saat membaca file: {e}")
    sys.exit(1)

n_rows_raw      = len(df)
n_cols_raw      = len(df.columns)
target_dist_raw = df['label_risiko'].value_counts().sort_index().to_dict()
missing_total   = int(df.isnull().sum().sum())

# =========================================================== #
#  2. PREPROCESSING                                           #
# =========================================================== #
log("2/6 Preprocessing...")

id_cols   = ['id_anggota_rt', 'id_rumah_tangga', 'id_provinsi', 'id_kabupaten']
drop_cols = ['label_risiko']

for c in id_cols:
    if c in df.columns:
        drop_cols.append(c)

if 'metode_persalinan_sesar' in df.columns:
    drop_cols.append('metode_persalinan_sesar')
if 'operasi_caesar' in df.columns:
    drop_cols.append('operasi_caesar')

X = df.drop(drop_cols, axis=1)
y = df['label_risiko']
n_features = X.shape[1]

# =========================================================== #
#  3. TRAIN-TEST SPLIT                                        #
#     SMOTE tidak dilakukan di sini — hanya split saja        #
#     SMOTE akan masuk ke dalam pipeline CV                   #
# =========================================================== #
log("3/6 Train-Test Split (SMOTE masuk ke pipeline CV)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
dist_train_original = y_train.value_counts().sort_index().to_dict()

# Hitung distribusi SMOTE untuk dokumentasi (simulasi di luar CV)
_smote_doc = SMOTE(random_state=RANDOM_STATE)
_, y_train_s_doc = _smote_doc.fit_resample(X_train, y_train)
dist_after_smote = pd.Series(y_train_s_doc).value_counts().sort_index().to_dict()
del _smote_doc, y_train_s_doc

os.makedirs('output_eksperimen', exist_ok=True)

# =========================================================== #
#  CACHING: Load hasil jika sudah pernah dihitung             #
# =========================================================== #
if os.path.exists(CACHE_FILE):
    log(f"[CACHE] Ditemukan cache '{CACHE_FILE}' — skip training & tuning, load langsung.")
    cache = joblib.load(CACHE_FILE)
    # Baseline
    acc_rf_b      = cache['acc_rf_b']
    f1_rf_b       = cache['f1_rf_b']
    roc_rf_b      = cache['roc_rf_b']
    report_rf_b   = cache['report_rf_b']
    cm_rf_b       = cache['cm_rf_b']
    y_pred_rf_b   = cache['y_pred_rf_b']
    y_proba_rf_b  = cache['y_proba_rf_b']
    time_rf_base  = cache['time_rf_base']

    acc_xgb_b     = cache['acc_xgb_b']
    f1_xgb_b      = cache['f1_xgb_b']
    roc_xgb_b     = cache['roc_xgb_b']
    report_xgb_b  = cache['report_xgb_b']
    cm_xgb_b      = cache['cm_xgb_b']
    y_pred_xgb_b  = cache['y_pred_xgb_b']
    y_proba_xgb_b = cache['y_proba_xgb_b']
    time_xgb_base = cache['time_xgb_base']

    # Tuned
    acc_rf_t      = cache['acc_rf_t']
    f1_rf_t       = cache['f1_rf_t']
    roc_rf_t      = cache['roc_rf_t']
    report_rf_t   = cache['report_rf_t']
    cm_rf_t       = cache['cm_rf_t']
    y_pred_rf_t   = cache['y_pred_rf_t']
    y_proba_rf_t  = cache['y_proba_rf_t']
    rf_tuned      = cache['rf_tuned']
    rf_best       = cache['rf_best']
    time_rf_tune  = cache['time_rf_tune']

    acc_xgb_t     = cache['acc_xgb_t']
    f1_xgb_t      = cache['f1_xgb_t']
    roc_xgb_t     = cache['roc_xgb_t']
    report_xgb_t  = cache['report_xgb_t']
    cm_xgb_t      = cache['cm_xgb_t']
    y_pred_xgb_t  = cache['y_pred_xgb_t']
    y_proba_xgb_t = cache['y_proba_xgb_t']
    xgb_tuned     = cache['xgb_tuned']
    xgb_best      = cache['xgb_best']
    time_xgb_tune = cache['time_xgb_tune']

    log(f"  RF  Baseline -> Acc={acc_rf_b:.4f}  F1={f1_rf_b:.4f}  ROC={roc_rf_b:.4f}  (from cache)")
    log(f"  XGB Baseline -> Acc={acc_xgb_b:.4f}  F1={f1_xgb_b:.4f}  ROC={roc_xgb_b:.4f}  (from cache)")
    log(f"  RF  Tuned    -> CV-best=cached  Test ROC={roc_rf_t:.4f}  (from cache)")
    log(f"  XGB Tuned    -> CV-best=cached  Test ROC={roc_xgb_t:.4f}  (from cache)")

else:
    # =========================================================== #
    #  4. BASELINE MODELS (SMOTE di luar, tanpa tuning)           #
    # =========================================================== #
    log("4/6 Baseline models (SMOTE di luar, tanpa tuning)...")

    smote_base = SMOTE(random_state=RANDOM_STATE)
    X_train_s, y_train_s = smote_base.fit_resample(X_train, y_train)

    # --- Random Forest Baseline ---
    t0 = time.time()
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf_base.fit(X_train_s, y_train_s)
    time_rf_base = time.time() - t0

    y_pred_rf_b  = rf_base.predict(X_test)
    y_proba_rf_b = rf_base.predict_proba(X_test)
    acc_rf_b     = accuracy_score(y_test, y_pred_rf_b)
    f1_rf_b      = f1_score(y_test, y_pred_rf_b, average='macro')
    roc_rf_b     = roc_auc_score(y_test, y_proba_rf_b, multi_class='ovr')
    report_rf_b  = classification_report(y_test, y_pred_rf_b, output_dict=True)
    cm_rf_b      = confusion_matrix(y_test, y_pred_rf_b)

    # --- XGBoost Baseline ---
    t0 = time.time()
    xgb_base = XGBClassifier(
        random_state=RANDOM_STATE, objective='multi:softprob',
        eval_metric='mlogloss', n_jobs=N_JOBS, tree_method='hist', verbosity=0
    )
    xgb_base.fit(X_train_s, y_train_s)
    time_xgb_base = time.time() - t0

    y_pred_xgb_b  = xgb_base.predict(X_test)
    y_proba_xgb_b = xgb_base.predict_proba(X_test)
    acc_xgb_b     = accuracy_score(y_test, y_pred_xgb_b)
    f1_xgb_b      = f1_score(y_test, y_pred_xgb_b, average='macro')
    roc_xgb_b     = roc_auc_score(y_test, y_proba_xgb_b, multi_class='ovr')
    report_xgb_b  = classification_report(y_test, y_pred_xgb_b, output_dict=True)
    cm_xgb_b      = confusion_matrix(y_test, y_pred_xgb_b)

    log(f"  RF  Baseline -> Acc={acc_rf_b:.4f}  F1={f1_rf_b:.4f}  ROC={roc_rf_b:.4f}  ({time_rf_base:.2f}s)")
    log(f"  XGB Baseline -> Acc={acc_xgb_b:.4f}  F1={f1_xgb_b:.4f}  ROC={roc_xgb_b:.4f}  ({time_xgb_base:.2f}s)")

    # =========================================================== #
    #  5. HYPERPARAMETER TUNING DENGAN IMBPIPELINE                #
    # =========================================================== #
    log("5/6 Hyperparameter Tuning (ImbPipeline + RandomizedSearchCV, mode silent)...")

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # -- RF Pipeline + Param Space --
    rf_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf',   RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
    ])
    rf_param_dist = {
        'clf__n_estimators':      [100, 150, 200, 250, 300],
        'clf__max_depth':         [10, 15, 20, 25],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf':  [1, 2, 4],
        'clf__max_features':      ['sqrt', 'log2'],
        'clf__bootstrap':         [True],
    }
    log(f"  Tuning RF (ImbPipeline): {N_ITER_RF} iter x 5 fold = {N_ITER_RF*5} fits ...")
    t0 = time.time()
    rf_rs = RandomizedSearchCV(
        rf_pipe, param_distributions=rf_param_dist,
        n_iter=N_ITER_RF, cv=cv_strat, scoring='f1_macro',
        n_jobs=N_JOBS, pre_dispatch='2*n_jobs',
        random_state=RANDOM_STATE, verbose=0, refit=True
    )
    rf_rs.fit(X_train, y_train)
    time_rf_tune = time.time() - t0

    rf_tuned     = rf_rs.best_estimator_
    y_pred_rf_t  = rf_tuned.predict(X_test)
    y_proba_rf_t = rf_tuned.predict_proba(X_test)
    acc_rf_t     = accuracy_score(y_test, y_pred_rf_t)
    f1_rf_t      = f1_score(y_test, y_pred_rf_t, average='macro')
    roc_rf_t     = roc_auc_score(y_test, y_proba_rf_t, multi_class='ovr')
    report_rf_t  = classification_report(y_test, y_pred_rf_t, output_dict=True)
    cm_rf_t      = confusion_matrix(y_test, y_pred_rf_t)
    rf_best      = {k.replace('clf__', ''): v for k, v in rf_rs.best_params_.items()}
    log(f"  RF  Tuned  -> CV-best={rf_rs.best_score_:.4f}  Test ROC={roc_rf_t:.4f}  ({time_rf_tune/60:.1f} mnt)")

    # -- XGB Pipeline + Param Space --
    xgb_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf',   XGBClassifier(
            random_state=RANDOM_STATE, objective='multi:softprob',
            eval_metric='mlogloss', tree_method='hist', verbosity=0, n_jobs=1
        ))
    ])
    xgb_param_dist = {
        'clf__n_estimators':     [100, 150, 200, 250, 300],
        'clf__max_depth':        [3, 4, 5, 6, 7],
        'clf__learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'clf__subsample':        [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'clf__gamma':            [0, 0.1, 0.2],
        'clf__min_child_weight': [1, 3, 5],
        'clf__reg_alpha':        [0, 0.1, 0.5],
        'clf__reg_lambda':       [0.5, 1.0, 1.5],
    }
    log(f"  Tuning XGB (ImbPipeline): {N_ITER_XGB} iter x 5 fold = {N_ITER_XGB*5} fits ...")
    t0 = time.time()
    xgb_rs = RandomizedSearchCV(
        xgb_pipe, param_distributions=xgb_param_dist,
        n_iter=N_ITER_XGB, cv=cv_strat, scoring='f1_macro',
        n_jobs=N_JOBS, pre_dispatch='2*n_jobs',
        random_state=RANDOM_STATE, verbose=0, refit=True
    )
    xgb_rs.fit(X_train, y_train)
    time_xgb_tune = time.time() - t0

    xgb_tuned     = xgb_rs.best_estimator_
    y_pred_xgb_t  = xgb_tuned.predict(X_test)
    y_proba_xgb_t = xgb_tuned.predict_proba(X_test)
    acc_xgb_t     = accuracy_score(y_test, y_pred_xgb_t)
    f1_xgb_t      = f1_score(y_test, y_pred_xgb_t, average='macro')
    roc_xgb_t     = roc_auc_score(y_test, y_proba_xgb_t, multi_class='ovr')
    report_xgb_t  = classification_report(y_test, y_pred_xgb_t, output_dict=True)
    cm_xgb_t      = confusion_matrix(y_test, y_pred_xgb_t)
    xgb_best      = {k.replace('clf__', ''): v for k, v in xgb_rs.best_params_.items()}
    log(f"  XGB Tuned  -> CV-best={xgb_rs.best_score_:.4f}  Test ROC={roc_xgb_t:.4f}  ({time_xgb_tune/60:.1f} mnt)")

    # =========================================================== #
    #  SIMPAN CACHE                                               #
    # =========================================================== #
    log(f"[CACHE] Menyimpan hasil ke '{CACHE_FILE}'...")
    joblib.dump({
        # Baseline RF
        'acc_rf_b': acc_rf_b, 'f1_rf_b': f1_rf_b, 'roc_rf_b': roc_rf_b,
        'report_rf_b': report_rf_b, 'cm_rf_b': cm_rf_b,
        'y_pred_rf_b': y_pred_rf_b, 'y_proba_rf_b': y_proba_rf_b,
        'time_rf_base': time_rf_base,
        # Baseline XGB
        'acc_xgb_b': acc_xgb_b, 'f1_xgb_b': f1_xgb_b, 'roc_xgb_b': roc_xgb_b,
        'report_xgb_b': report_xgb_b, 'cm_xgb_b': cm_xgb_b,
        'y_pred_xgb_b': y_pred_xgb_b, 'y_proba_xgb_b': y_proba_xgb_b,
        'time_xgb_base': time_xgb_base,
        # Tuned RF
        'acc_rf_t': acc_rf_t, 'f1_rf_t': f1_rf_t, 'roc_rf_t': roc_rf_t,
        'report_rf_t': report_rf_t, 'cm_rf_t': cm_rf_t,
        'y_pred_rf_t': y_pred_rf_t, 'y_proba_rf_t': y_proba_rf_t,
        'rf_tuned': rf_tuned, 'rf_best': rf_best, 'time_rf_tune': time_rf_tune,
        # Tuned XGB
        'acc_xgb_t': acc_xgb_t, 'f1_xgb_t': f1_xgb_t, 'roc_xgb_t': roc_xgb_t,
        'report_xgb_t': report_xgb_t, 'cm_xgb_t': cm_xgb_t,
        'y_pred_xgb_t': y_pred_xgb_t, 'y_proba_xgb_t': y_proba_xgb_t,
        'xgb_tuned': xgb_tuned, 'xgb_best': xgb_best, 'time_xgb_tune': time_xgb_tune,
    }, CACHE_FILE)
    log(f"[CACHE] Tersimpan. Run ulang berikutnya akan langsung pakai cache ini.")

# =========================================================== #
#  PLOTS                                                      #
# =========================================================== #
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

# ── Confusion Matrices (4 model) ──
fig_cm, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_data = [
    (cm_rf_b,  'Random Forest – Baseline'),
    (cm_xgb_b, 'XGBoost – Baseline'),
    (cm_rf_t,  'Random Forest – Tuned (Pipeline)'),
    (cm_xgb_t, 'XGBoost – Tuned (Pipeline)'),
]
for ax, (cm, ttl) in zip(axes.flatten(), plot_data):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rendah','Sedang','Tinggi'],
                yticklabels=['Rendah','Sedang','Tinggi'])
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.suptitle('Confusion Matrix – Baseline vs Tuned (ImbPipeline)', fontsize=13, fontweight='bold')
plt.tight_layout()
cm_buf = fig_to_bytes(fig_cm)
plt.close()

# ── Feature Importance (dari classifier dalam pipeline) ──
fig_fi, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, pipe_model, color, ttl in zip(
    axes,
    [rf_tuned, xgb_tuned],
    ['steelblue', 'forestgreen'],
    ['Random Forest (Tuned)', 'XGBoost (Tuned)'],
):
    clf_step = pipe_model.named_steps['clf']
    imp = clf_step.feature_importances_
    idx = np.argsort(imp)[-15:]
    ax.barh(range(len(idx)), imp[idx], color=color, align='center')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([X.columns[i] for i in idx], fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 15 Feature Importance\n{ttl}', fontweight='bold')
plt.tight_layout()
fi_buf = fig_to_bytes(fig_fi)
plt.close()

# ── ROC-AUC Comparison Bar Chart ──
fig_roc, ax = plt.subplots(figsize=(11, 5))
models_lbl = ['RF\nBaseline', 'XGB\nBaseline', 'RF\nTuned', 'XGB\nTuned']
roc_scores = [roc_rf_b, roc_xgb_b, roc_rf_t, roc_xgb_t]
colors_roc = ['#5885AF', '#3E7A5E', '#1C4E80', '#0C5928']
bars = ax.bar(models_lbl, roc_scores, color=colors_roc, width=0.5)
ax.set_ylim(max(0.80, min(roc_scores) - 0.03), 1.0)
ax.set_ylabel('ROC-AUC Score (OvR)')
ax.set_title('Perbandingan ROC-AUC Score – Baseline vs Tuned (ImbPipeline)', fontweight='bold')
for bar, score in zip(bars, roc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
roc_buf = fig_to_bytes(fig_roc)
plt.close()

# ── ROC Curve (OvR per kelas) – semua 4 model ──
log("Plot kurva ROC (OvR per kelas, 4 model)...")

classes_list  = [0, 1, 2]
class_colors  = ['#e84393', '#f59e0b', '#10b981']
class_labels_roc = ['Rendah (0)', 'Sedang (1)', 'Tinggi (2)']

y_test_bin = label_binarize(y_test, classes=classes_list)

_roc_models = [
    ('RF Baseline',          y_proba_rf_b,  '#5885AF'),
    ('XGBoost Baseline',     y_proba_xgb_b, '#3E7A5E'),
    ('RF Tuned (Pipeline)',  y_proba_rf_t,  '#1C4E80'),
    ('XGB Tuned (Pipeline)', y_proba_xgb_t, '#0C5928'),
]

# 4 sub-plot: satu per model, setiap sub-plot ada 3 kurva kelas
fig_roc_curve, axes_roc = plt.subplots(2, 2, figsize=(14, 11))
for ax, (model_name, y_proba, base_color) in zip(axes_roc.flatten(), _roc_models):
    for i, (cls, color, lbl) in enumerate(zip(classes_list, class_colors, class_labels_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{lbl} (AUC = {roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'ROC Curve – {model_name}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('ROC Curve One-vs-Rest (OvR) – Baseline vs Tuned (ImbPipeline)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
roc_curve_buf = fig_to_bytes(fig_roc_curve)
plt.close()

# ── SHAP (dari classifier tuned) ──
log("Kalkulasi SHAP values (subset 300 untuk efisiensi)...")
X_test_sample = (X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE)
                 if len(X_test) > 300 else X_test)

rf_clf_step  = rf_tuned.named_steps['clf']
xgb_clf_step = xgb_tuned.named_steps['clf']

# SHAP RF
explainer_rf   = shap.TreeExplainer(rf_clf_step)
shap_values_rf = explainer_rf.shap_values(X_test_sample)

plt.figure()
shap.summary_plot(shap_values_rf, X_test_sample, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance (Random Forest Tuned)", pad=20)
plt.tight_layout()
shap_rf_buf = fig_to_bytes(plt.gcf())
plt.close()

if isinstance(shap_values_rf, list) and len(shap_values_rf) > 2:
    plt.figure()
    shap.summary_plot(shap_values_rf[2], X_test_sample, show=False)
    plt.title("Random Forest SHAP – Risiko TINGGI", pad=20)
    plt.tight_layout()
    shap_rf_tinggi_buf = fig_to_bytes(plt.gcf())
    plt.close()
elif hasattr(shap_values_rf, 'shape') and len(shap_values_rf.shape) == 3:
    plt.figure()
    shap.summary_plot(shap_values_rf[:, :, 2], X_test_sample, show=False)
    plt.title("Random Forest SHAP – Risiko TINGGI", pad=20)
    plt.tight_layout()
    shap_rf_tinggi_buf = fig_to_bytes(plt.gcf())
    plt.close()
else:
    shap_rf_tinggi_buf = None

# SHAP XGB
explainer_xgb   = shap.TreeExplainer(xgb_clf_step)
shap_values_xgb = explainer_xgb.shap_values(X_test_sample)

plt.figure()
shap.summary_plot(shap_values_xgb, X_test_sample, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance (XGBoost Tuned)", pad=20)
plt.tight_layout()
shap_xgb_buf = fig_to_bytes(plt.gcf())
plt.close()

if isinstance(shap_values_xgb, list) and len(shap_values_xgb) > 2:
    plt.figure()
    shap.summary_plot(shap_values_xgb[2], X_test_sample, show=False)
    plt.title("XGBoost SHAP – Risiko TINGGI", pad=20)
    plt.tight_layout()
    shap_xgb_tinggi_buf = fig_to_bytes(plt.gcf())
    plt.close()
elif hasattr(shap_values_xgb, 'shape') and len(shap_values_xgb.shape) == 3:
    plt.figure()
    shap.summary_plot(shap_values_xgb[:, :, 2], X_test_sample, show=False)
    plt.title("XGBoost SHAP – Risiko TINGGI", pad=20)
    plt.tight_layout()
    shap_xgb_tinggi_buf = fig_to_bytes(plt.gcf())
    plt.close()
else:
    shap_xgb_tinggi_buf = None

# =========================================================== #
#  6. BUAT DOKUMEN WORD                                       #
# =========================================================== #
log("6/6 Membuat dokumen Word...")
os.makedirs('output_eksperimen', exist_ok=True)

doc = Document()
sec = doc.sections[0]
sec.page_width    = Cm(21);  sec.page_height   = Cm(29.7)
sec.left_margin   = Cm(3);   sec.right_margin  = Cm(2.5)
sec.top_margin    = Cm(2.5); sec.bottom_margin = Cm(2.5)

def h(lvl, txt):
    p = doc.add_heading(txt, level=lvl)
    p.paragraph_format.space_before = Pt(10 if lvl == 1 else 6)
    p.paragraph_format.space_after  = Pt(4)
    return p

def para(txt, bold=False, italic=False, align=None):
    p = doc.add_paragraph(txt)
    if bold or italic:
        for run in p.runs:
            run.bold   = bold
            run.italic = italic
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(4)
    return p

def bullet(txt):
    doc.add_paragraph(txt, style='List Bullet')

def add_table(headers, rows_data):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = 'Table Grid'
    hd_row = tbl.rows[0].cells
    for i, h_txt in enumerate(headers):
        hd_row[i].text = h_txt
        for run in hd_row[i].paragraphs[0].runs:
            run.bold = True
    for row_vals in rows_data:
        row = tbl.add_row().cells
        for i, val in enumerate(row_vals):
            row[i].text = str(val)
    doc.add_paragraph()
    return tbl

def add_monospaced_block(text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.name = 'Courier New'
        run.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(10)
    return p

# ═══════════════════════════════════════════════════════════ #
#  HALAMAN JUDUL                                              #
# ═══════════════════════════════════════════════════════════ #
h(0, '')
p_title = doc.add_heading('Laporan Penelitian (Versi 5)', 0)
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p_sub = doc.add_heading(
    'Klasifikasi Risiko Kehamilan menggunakan\n'
    'Random Forest & XGBoost\n'
    'SMOTE dalam ImbPipeline + Hyperparameter Tuning (Metodologi Benar)\n'
    '(Dataset Final: KSPR Attala)', 1)
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

for txt in [
    'Kerangka Kerja: CRISP-DM (Cross-Industry Standard Process for Data Mining)',
    'Hardware: MSI Cyborg 15 A13V (Intel Core i5-13420H)',
    f'Tanggal Eksperimen: {time.strftime("%d %B %Y")}',
]:
    para(txt, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  RINGKASAN EKSEKUTIF                                        #
# ═══════════════════════════════════════════════════════════ #
h(1, 'Ringkasan Eksekutif')

best_scores = {
    'RF Baseline':  roc_rf_b,
    'XGB Baseline': roc_xgb_b,
    'RF Tuned':     roc_rf_t,
    'XGB Tuned':    roc_xgb_t,
}
best_name = max(best_scores, key=best_scores.get)
best_roc  = best_scores[best_name]
best_f1   = {'RF Baseline': f1_rf_b, 'XGB Baseline': f1_xgb_b,
             'RF Tuned': f1_rf_t, 'XGB Tuned': f1_xgb_t}[best_name]

para(
    f'Penelitian versi 5 ini menerapkan perbaikan metodologi kritis dibanding versi 2: '
    f'SMOTE diintegrasikan ke dalam imblearn.Pipeline sehingga resampling hanya terjadi '
    f'pada training fold — menghindari kebocoran data (data leakage) ke validation fold '
    f'dalam proses cross-validation. N_ITER ditingkatkan menjadi {N_ITER_RF} (RF) dan '
    f'{N_ITER_XGB} (XGB) untuk pencarian hyperparameter yang lebih luas. '
    f'Dataset final_dataset_kspr_attala.csv dengan {n_rows_raw:,} observasi digunakan. '
    f'Model terbaik adalah {best_name} dengan ROC-AUC = {best_roc:.4f}.'
)

add_table(
    ['Metrik', 'RF Baseline', 'XGB Baseline', 'RF Tuned', 'XGB Tuned'],
    [
        ('Accuracy',  f'{acc_rf_b:.4f}',  f'{acc_xgb_b:.4f}',  f'{acc_rf_t:.4f}',  f'{acc_xgb_t:.4f}'),
        ('F1-Macro',  f'{f1_rf_b:.4f}',   f'{f1_xgb_b:.4f}',   f'{f1_rf_t:.4f}',   f'{f1_xgb_t:.4f}'),
        ('ROC-AUC',   f'{roc_rf_b:.4f}',  f'{roc_xgb_b:.4f}',  f'{roc_rf_t:.4f}',  f'{roc_xgb_t:.4f}'),
        ('Waktu Fit', f'{time_rf_base:.2f}s', f'{time_xgb_base:.2f}s',
                      f'{time_rf_tune/60:.1f}m', f'{time_xgb_tune/60:.1f}m'),
    ]
)
doc.add_picture(roc_buf, width=Inches(5.5))
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 1: BUSINESS UNDERSTANDING                              #
# ═══════════════════════════════════════════════════════════ #
h(1, '1. Business Understanding (Pemahaman Bisnis)')
para(
    'Kematian ibu merupakan salah satu indikator derajat kesehatan yang paling sensitif. '
    'Deteksi dini risiko kehamilan sangat penting agar intervensi medis dapat diberikan tepat waktu. '
    'Skor Poedji Rochjati (KSPR) adalah sistem pakar berbasis skor yang digunakan bidan/dokter '
    'untuk menentukan tingkat risiko kehamilan. Otomasi klasifikasi ini melalui machine learning '
    'diharapkan dapat mempercepat dan menstandardisasi proses identifikasi risiko.'
)
h(2, '1.1 Tujuan Bisnis')
for t in [
    'Membangun model ML yang akurat untuk klasifikasi risiko kehamilan (multikelas).',
    'Membandingkan performa Random Forest vs XGBoost secara kuantitatif.',
    'Menangani imbalanced class dengan SMOTE yang diintegrasikan ke dalam pipeline CV.',
    'Meningkatkan performa model melalui Hyperparameter Tuning dengan metodologi yang benar.',
    'Mengidentifikasi faktor klinis paling berpengaruh menggunakan SHAP.',
]:
    bullet(t)

h(2, '1.2 Kriteria Keberhasilan')
add_table(
    ['Kriteria', 'Target'],
    [
        ('ROC-AUC (OvR)',          '≥ 0.90'),
        ('F1-score Macro',         '≥ 0.80'),
        ('Tidak ada data leakage', 'Terbukti dari analisis fitur & pipeline CV'),
        ('Waktu inferensi',        '< 1 detik per observasi'),
    ]
)
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 2: DATA UNDERSTANDING                                  #
# ═══════════════════════════════════════════════════════════ #
h(1, '2. Data Understanding (Pemahaman Data)')
h(2, '2.1 Sumber dan Deskripsi Data')
para(
    'Dataset bersumber dari Survei Kesehatan Indonesia (SKI) 2023. '
    'Label risiko kehamilan ditentukan berdasarkan Skor Poedji Rochjati (KSPR) '
    'yang merupakan sistem pakar yang diakui secara klinis di Indonesia. '
    'Distribusi kelas bersifat imbalanced: kelas Sedang mendominasi dataset.'
)
add_table(
    ['Karakteristik', 'Nilai'],
    [
        ('Jumlah Observasi',           f'{n_rows_raw:,}'),
        ('Jumlah Kolom (raw)',          str(n_cols_raw)),
        ('Jumlah Fitur (setelah prep)', str(n_features)),
        ('Missing Value',               str(missing_total)),
        ('Format',                      'CSV'),
        ('Target Variable',             'label_risiko'),
        ('Jumlah Kelas Target',         '3'),
    ]
)
h(2, '2.2 Distribusi Kelas Data Asli')
add_table(
    ['Kelas', 'Label', 'Jumlah', 'Proporsi (%)'],
    [
        (str(k), class_labels[k], str(v), f'{v/n_rows_raw*100:.1f}%')
        for k, v in sorted(target_dist_raw.items())
    ]
)
h(2, '2.3 Identifikasi Potensi Masalah')
for item in [
    'Imbalanced Class: Kelas Rendah (0) sangat sedikit dibanding Sedang (1) → diatasi dengan SMOTE di dalam pipeline.',
    'Data Leakage (Fitur): "operasi_caesar" / "metode_persalinan_sesar" merupakan informasi post-partum → dihapus.',
    'Data Leakage (CV): SMOTE di luar CV menyebabkan data sintetis bocor ke validation fold → diatasi dengan ImbPipeline.',
    'Identifikasi ID unik (id_anggota, rumah_tangga, dll.) → didrop untuk mencegah overfitting.',
]:
    bullet(item)
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 3: DATA PREPARATION                                    #
# ═══════════════════════════════════════════════════════════ #
h(1, '3. Data Preparation (Persiapan Data)')
h(2, '3.1 Penghapusan Fitur Leakage & ID')
para(
    'Kolom unik (ID) serta variabel metode persalinan/caesar dihapus karena merepresentasikan '
    'informasi yang tidak tersedia saat prediksi dilakukan (informasi paska persalinan). '
    'Penggunaan variabel ini akan membuat evaluasi model tidak realistis.'
)
h(2, '3.2 Train-Test Split')
add_table(
    ['Subset', 'Proporsi', 'Jumlah Baris', 'Keterangan'],
    [
        ('Training Set', '80%', f'{len(X_train):,}', 'Digunakan melatih + tuning model'),
        ('Test Set',     '20%', f'{len(X_test):,}',  'Hanya digunakan evaluasi akhir'),
    ]
)
para('Parameter stratify=y memastikan proporsi kelas sama di training dan test set.')

h(2, '3.3 SMOTE – Penanganan Imbalanced Class')
add_table(
    ['Kelas', 'Label', 'Sebelum SMOTE', 'Estimasi Sesudah SMOTE'],
    [
        (str(k), class_labels[k], str(dist_train_original.get(k, 0)), str(dist_after_smote.get(k, 0)))
        for k in sorted(dist_after_smote.keys())
    ]
)
para(
    'SMOTE diterapkan di dalam imblearn.Pipeline sehingga resampling hanya terjadi pada '
    'training fold di setiap iterasi cross-validation. Validation fold selalu menggunakan '
    'data asli (tidak sintetis), menghasilkan estimasi performa yang lebih realistis. '
    'Ini adalah perbaikan metodologi utama dibanding versi 2.'
)
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 4: MODELING                                            #
# ═══════════════════════════════════════════════════════════ #
h(1, '4. Modeling (Pemodelan)')
h(2, '4.1 Deskripsi Algoritma')
h(3, 'Random Forest')
para(
    'Random Forest adalah metode ensemble berbasis bagging yang membangun sejumlah '
    'pohon keputusan (decision tree) secara paralel. Setiap pohon dilatih pada subset '
    'acak dari data dan fitur. Prediksi akhir ditentukan melalui voting mayoritas.'
)
h(3, 'XGBoost (Extreme Gradient Boosting)')
para(
    'XGBoost adalah metode ensemble berbasis boosting yang membangun pohon secara '
    'sekuensial. XGBoost digunakan karena tingkat akurasinya yang tinggi, efisiensi '
    'komputasi, dan robustness terhadap outlier.'
)
h(2, '4.2 Perbaikan Metodologi v5: ImbPipeline')
para(
    'Perbedaan utama v5 dari v2 adalah penggunaan imblearn.pipeline.Pipeline '
    '(ImbPipeline) yang menggabungkan SMOTE dan model ke dalam satu pipeline. '
    'Saat RandomizedSearchCV melakukan cross-validation, pipeline memastikan:'
)
for item in [
    'Training fold: SMOTE diterapkan → model dilatih pada data yang di-resample.',
    'Validation fold: Data asli (tidak sintetis) → estimasi performa yang valid.',
    'Tidak ada data sintetis dari validation fold yang "membocor" ke proses seleksi parameter.',
]:
    bullet(item)

h(2, '4.3 Strategi Hyperparameter Tuning')
para('RandomizedSearchCV dipilih dibanding GridSearchCV karena lebih hemat komputasi.')
add_table(
    ['Parameter Konfigurasi', 'Nilai', 'Keterangan'],
    [
        ('n_iter RF',  str(N_ITER_RF),  f'{N_ITER_RF}×5 = {N_ITER_RF*5} fits (ditingkatkan dari 20)'),
        ('n_iter XGB', str(N_ITER_XGB), f'{N_ITER_XGB}×5 = {N_ITER_XGB*5} fits (ditingkatkan dari 25)'),
        ('cv',         '5 (Stratified)', 'StratifiedKFold menjaga proporsi kelas tiap fold'),
        ('scoring',    'f1_macro',       'Evaluasi merata untuk semua kelas'),
        ('n_jobs',     str(N_JOBS),      'Alokasikan ke thread CPU'),
        ('max_depth',  '10–25 (tanpa None)', 'None dihapus untuk cegah overfitting'),
    ]
)
h(2, '4.4 Hyperparameter Terbaik')
para(f'RF Best Params  : {rf_best}')
para(f'XGB Best Params : {xgb_best}')
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 5: EVALUATION                                          #
# ═══════════════════════════════════════════════════════════ #
h(1, '5. Evaluation (Evaluasi)')
h(2, '5.1 Perbandingan Metrik Evaluasi')
add_table(
    ['Model', 'Skenario', 'Accuracy', 'F1-Macro', 'ROC-AUC (OvR)'],
    [
        ('Random Forest', 'Baseline',        f'{acc_rf_b:.4f}',  f'{f1_rf_b:.4f}',  f'{roc_rf_b:.4f}'),
        ('XGBoost',       'Baseline',        f'{acc_xgb_b:.4f}', f'{f1_xgb_b:.4f}', f'{roc_xgb_b:.4f}'),
        ('Random Forest', 'Tuned (Pipeline)', f'{acc_rf_t:.4f}',  f'{f1_rf_t:.4f}',  f'{roc_rf_t:.4f}'),
        ('XGBoost',       'Tuned (Pipeline)', f'{acc_xgb_t:.4f}', f'{f1_xgb_t:.4f}', f'{roc_xgb_t:.4f}'),
    ]
)

h(2, '5.2 Confusion Matrix (4 Model)')
doc.add_picture(cm_buf, width=Inches(6.0))
doc.add_paragraph()

h(2, '5.3 Feature Importance – Model Terbaik (Tuned)')
doc.add_picture(fi_buf, width=Inches(6.0))
doc.add_page_break()

h(2, '5.4 Classification Report')
report_names = ['KRR/Rendah (0)', 'KRT/Sedang (1)', 'KRST/Tinggi (2)']

rep_str_rf_b  = classification_report(y_test, y_pred_rf_b,  target_names=report_names)
rep_str_xgb_b = classification_report(y_test, y_pred_xgb_b, target_names=report_names)
rep_str_rf_t  = classification_report(y_test, y_pred_rf_t,  target_names=report_names)
rep_str_xgb_t = classification_report(y_test, y_pred_xgb_t, target_names=report_names)

add_monospaced_block(f"Classification Report - Random Forest Baseline:\n\n{rep_str_rf_b}")
add_monospaced_block(f"Classification Report - XGBoost Baseline:\n\n{rep_str_xgb_b}")
add_monospaced_block(f"Classification Report - Random Forest Tuned (Pipeline):\n\n{rep_str_rf_t}")
add_monospaced_block(f"Classification Report - XGBoost Tuned (Pipeline):\n\n{rep_str_xgb_t}")

h(2, '5.5 Kurva ROC (Receiver Operating Characteristic)')
para(
    'Kurva ROC menggambarkan trade-off antara True Positive Rate (Sensitivitas) '
    'dan False Positive Rate pada berbagai threshold. Strategi One-vs-Rest (OvR) digunakan '
    'untuk menangani klasifikasi multikelas: setiap kelas dievaluasi secara independen '
    'terhadap gabungan kelas lainnya. Area di bawah kurva (AUC) mendekati 1.0 '
    'menandakan performa diskriminasi yang sangat baik.'
)
doc.add_picture(roc_curve_buf, width=Inches(6.2))
doc.add_paragraph()

h(2, '5.6 Explainable AI (SHAP)')
para('Analisis SHAP (Shapley Additive exPlanations) untuk interpretasi model tuned pada skala global dan kelas Risiko Tinggi.')

para('SHAP Random Forest (Tuned)', bold=True)
doc.add_picture(shap_rf_buf, width=Inches(6.0))
if shap_rf_tinggi_buf:
    doc.add_picture(shap_rf_tinggi_buf, width=Inches(6.0))

para('SHAP XGBoost (Tuned)', bold=True)
doc.add_picture(shap_xgb_buf, width=Inches(6.0))
if shap_xgb_tinggi_buf:
    doc.add_picture(shap_xgb_tinggi_buf, width=Inches(6.0))

doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  BAB 6: DEPLOYMENT                                          #
# ═══════════════════════════════════════════════════════════ #
h(1, '6. Deployment (Implementasi)')
h(2, '6.1 Model Rekomendasi')
para(
    f'Berdasarkan seluruh hasil evaluasi, model yang direkomendasikan untuk '
    f'deployment adalah: {best_name} (ROC-AUC = {best_roc:.4f}, F1-Macro = {best_f1:.4f}).'
)
h(2, '6.2 Rencana Implementasi')
for item in [
    f'Simpan pipeline {best_name} (termasuk SMOTE step) menggunakan joblib/pickle.',
    'Buat API inferensi (Flask/FastAPI) yang menerima input fitur klinis pasien.',
    'Catatan: SMOTE dalam pipeline hanya aktif saat training, tidak saat inferensi.',
    'Integrasikan dengan sistem informasi kesehatan (Puskesmas/SIMRS).',
]:
    bullet(item)

h(2, '6.3 Keterbatasan Model')
for item in [
    'Dataset dari survei cross-sectional (SKI 2023), bukan data longitudinal klinis langsung.',
    'Label risiko dihasilkan dari sistem pakar KSPR, bukan dari diagnosis dokter klinis.',
    'N_ITER masih dapat ditingkatkan lebih lanjut untuk pencarian hyperparameter yang lebih optimal.',
]:
    bullet(item)
doc.add_page_break()

# ═══════════════════════════════════════════════════════════ #
#  LAMPIRAN                                                   #
# ═══════════════════════════════════════════════════════════ #
h(1, 'Lampiran: Alur Kerja CRISP-DM Lengkap')
para('Dokumentasi lengkap setiap fase CRISP-DM – Versi 5 (Metodologi Pipeline yang Benar).')

crisp_dm = [
    ('1. Business Understanding', [
        ('Tujuan Analitik', 'Klasifikasi multikelas risiko kehamilan: Rendah (0), Sedang (1), Tinggi (2).'),
        ('Kriteria Sukses', 'ROC-AUC ≥ 0.90, F1-macro ≥ 0.80, tidak ada data leakage.'),
    ]),
    ('2. Data Understanding', [
        ('Volume Data',    f'{n_rows_raw:,} observasi, {n_cols_raw} kolom variabel.'),
        ('Target Variabel', '"label_risiko" – kategorik ordinal 3 kelas.'),
    ]),
    ('3. Data Preparation', [
        ('Penghapusan Fitur Leakage', 'Fitur persalinan/caesar + kolom ID unik di-drop.'),
        ('Feature & Target Split',    f'X: {n_features} fitur,  y: label_risiko.'),
        ('SMOTE Strategy',            'Melalui ImbPipeline – hanya pada training fold CV.'),
    ]),
    ('4. Modeling', [
        ('Metode Tuning',     f'RandomizedSearchCV + ImbPipeline + StratifiedKFold (5-fold).'),
        ('RF Best Params',    str(rf_best)),
        ('XGB Best Params',   str(xgb_best)),
        ('Perbaikan vs v2',   'SMOTE masuk ke dalam pipeline → CV score lebih akurat.'),
    ]),
    ('5. Evaluation', [
        ('Model Terbaik', f'{best_name} – ROC-AUC = {best_roc:.4f}, F1-Macro = {best_f1:.4f}'),
    ]),
    ('6. Deployment', [
        ('Model Produksi', f'{best_name} direkomendasikan untuk deployment.'),
    ]),
]

for phase_title, items in crisp_dm:
    h(2, phase_title)
    add_table(
        ['Sub-fase / Aspek', 'Detail'],
        [(k, v) for k, v in items]
    )

# -- SAVE dengan fallback jika file sedang dibuka di Word --
try:
    doc.save(OUTPUT_FILE)
    saved_to = OUTPUT_FILE
except PermissionError:
    fallback = OUTPUT_FILE.replace('.docx', f'_{time.strftime("%H%M%S")}.docx')
    doc.save(fallback)
    saved_to = fallback
    log(f"[WARNING] File utama sedang dibuka. Disimpan ke: {saved_to}")
log(f"Dokumen Word tersimpan: {saved_to}")
print()
print("=" * 65)
print("RANGKUMAN HASIL AKHIR VERSI 5 (ImbPipeline CV)")
print("=" * 65)
print(f"  RF  Baseline  ->  Acc={acc_rf_b:.4f}  F1={f1_rf_b:.4f}  ROC={roc_rf_b:.4f}")
print(f"  XGB Baseline  ->  Acc={acc_xgb_b:.4f}  F1={f1_xgb_b:.4f}  ROC={roc_xgb_b:.4f}")
print(f"  RF  Tuned     ->  Acc={acc_rf_t:.4f}  F1={f1_rf_t:.4f}  ROC={roc_rf_t:.4f}")
print(f"  XGB Tuned     ->  Acc={acc_xgb_t:.4f}  F1={f1_xgb_t:.4f}  ROC={roc_xgb_t:.4f}")
print(f"  Model Terbaik : {best_name}  (ROC-AUC = {best_roc:.4f})")
print("=" * 65)
if os.path.exists(CACHE_FILE):
    print(f"  Cache tersedia di: {CACHE_FILE}")
    print("  Run ulang berikutnya akan skip training & tuning (langsung dari cache).")
print("=" * 65)
