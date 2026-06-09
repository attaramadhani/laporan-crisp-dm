import os, sys, time, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon
import shap

# --- CONFIGURATION ---
DATA_FILE = 'final_dataset_kspr_attala.csv'
CACHE_FILE = 'v5_cache.pkl' # Gunakan cache file yang ada di root directory
RANDOM_STATE = 42
N_JOBS = 10

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if not os.path.exists(DATA_FILE) or not os.path.exists(CACHE_FILE):
    log(f"ERROR: Pastikan file {DATA_FILE} dan {CACHE_FILE} ada di folder ini.")
    sys.exit(1)

# --- 1. LOAD DATA ---
log("Loading data...")
df = pd.read_csv(DATA_FILE)
drop_cols = ['label_risiko']
id_cols = ['id_anggota_rt', 'id_rumah_tangga', 'id_provinsi', 'id_kabupaten']
for c in id_cols:
    if c in df.columns: drop_cols.append(c)
if 'metode_persalinan_sesar' in df.columns: drop_cols.append('metode_persalinan_sesar')
if 'operasi_caesar' in df.columns: drop_cols.append('operasi_caesar')

X = df.drop(drop_cols, axis=1)
y = df['label_risiko']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# --- 2. LOAD CACHE MODEL TERBAIK ---
log("Loading cache...")
cache = joblib.load(CACHE_FILE)
rf_tuned = cache['rf_tuned']
xgb_tuned = cache['xgb_tuned']
cm_rf_t = cache['cm_rf_t']
cm_xgb_t = cache['cm_xgb_t']

# --- 3. MENGHITUNG SPECIFICITY PER KELAS ---
log("Menghitung Specificity per kelas...")
def calc_specificity(cm):
    spec = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        spec.append(tn / (tn + fp))
    return spec

spec_rf = calc_specificity(cm_rf_t)
spec_xgb = calc_specificity(cm_xgb_t)

print("\n--- SPECIFICITY PER KELAS ---")
print("Kelas 0 (Rendah): RF={:.4f}, XGB={:.4f}".format(spec_rf[0], spec_xgb[0]))
print("Kelas 1 (Sedang): RF={:.4f}, XGB={:.4f}".format(spec_rf[1], spec_xgb[1]))
print("Kelas 2 (Tinggi): RF={:.4f}, XGB={:.4f}".format(spec_rf[2], spec_xgb[2]))
print("-----------------------------\n")

# --- 4. SHAP DEPENDENCE PLOT ---
log("Generating SHAP Dependence Plot untuk fitur usia ibu (umur_ibu_tahun)...")
X_test_sample = X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE)
xgb_clf_step = xgb_tuned.named_steps['clf']
explainer_xgb = shap.TreeExplainer(xgb_clf_step)
shap_values_xgb = explainer_xgb.shap_values(X_test_sample)

if isinstance(shap_values_xgb, list) and len(shap_values_xgb) > 2:
    shap_vals = shap_values_xgb[2]
elif hasattr(shap_values_xgb, 'shape') and len(shap_values_xgb.shape) == 3:
    shap_vals = shap_values_xgb[:, :, 2]
else:
    shap_vals = shap_values_xgb

plt.figure(figsize=(8, 6))
shap.dependence_plot("umur_ibu_tahun", shap_vals, X_test_sample, show=False)
plt.title("XGBoost SHAP Dependence Plot: Usia Ibu (Risiko Sangat Tinggi)", pad=20)
plt.tight_layout()
plt.savefig("fig_shap_dep.png", dpi=150)
plt.close()
log("Berhasil menyimpan fig_shap_dep.png!")

# --- 5. UJI STATISTIK WILCOXON & 10-FOLD CV (STD DEV) ---
log("Melakukan 10-Fold CV untuk uji statistik Wilcoxon & Standar Deviasi (estimasi 3-4 menit)...")
cv_wilcoxon = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# Note: rf_tuned and xgb_tuned are imblearn pipelines (which includes SMOTE)
rf_cv_scores = cross_val_score(rf_tuned, X, y, cv=cv_wilcoxon, scoring='f1_macro', n_jobs=N_JOBS)
xgb_cv_scores = cross_val_score(xgb_tuned, X, y, cv=cv_wilcoxon, scoring='f1_macro', n_jobs=N_JOBS)

stat_w, p_val_w = wilcoxon(rf_cv_scores, xgb_cv_scores)
rf_cv_mean, rf_cv_std = np.mean(rf_cv_scores), np.std(rf_cv_scores)
xgb_cv_mean, xgb_cv_std = np.mean(xgb_cv_scores), np.std(xgb_cv_scores)

print("\n=== HASIL UJI REVIEWER (WILCOXON & STD DEV) ===")
print(f"Random Forest (Tuned) 10-Fold CV F1-Macro : {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
print(f"XGBoost (Tuned)       10-Fold CV F1-Macro : {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")
print(f"P-Value Uji Wilcoxon Signed-Rank          : {p_val_w:.5e}")
wilcoxon_res = "SIGNIFIKAN" if p_val_w < 0.05 else "TIDAK SIGNIFIKAN"
print(f"Kesimpulan                                : Perbedaan {wilcoxon_res} secara statistik.")
print("=================================================")
log("Semua proses revisi selesai.")
