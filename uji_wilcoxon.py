import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings('ignore')

print("=== Script Uji Wilcoxon (RF vs XGBoost) ===")
print("Memuat dataset final_dataset_kspr_attala.csv...")

try:
    df = pd.read_csv('final_dataset_kspr_attala.csv')
except FileNotFoundError:
    print("ERROR: File 'final_dataset_kspr_attala.csv' tidak ditemukan di folder ini.")
    print("Harap pastikan file tersebut berada dalam satu folder dengan script ini.")
    exit()

target = 'Kategori_KSPR'
if target not in df.columns:
    print(f"ERROR: Kolom target '{target}' tidak ditemukan.")
    exit()

X = df.drop(columns=[target])
y = df[target]

print(f"Dataset berhasil dimuat: {X.shape[0]} baris, {X.shape[1]} fitur.")

# Hyperparameter terbaik dari hasil tuning Anda
rf_best_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 25, 'bootstrap': True, 'random_state': 42, 'n_jobs': -1}
xgb_best_params = {'subsample': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.5, 'n_estimators': 150, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.2, 'gamma': 0.1, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False, 'eval_metric': 'mlogloss'}

print("\nMenyiapkan model dengan hiperparameter terbaik...")
rf_pipe = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('clf', RandomForestClassifier(**rf_best_params))
])

xgb_pipe = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('clf', XGBClassifier(**xgb_best_params))
])

# Evaluasi cross-validation dengan StratifiedKFold
# Kita gunakan 10 folds agar ada sampel uji statistik yang cukup (N=10)
n_splits = 10
cv_strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nMenjalankan {n_splits}-Fold Cross-Validation untuk Random Forest (Mohon tunggu)...")
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv_strat, scoring='f1_macro', n_jobs=1)

print(f"Menjalankan {n_splits}-Fold Cross-Validation untuk XGBoost (Mohon tunggu)...")
xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv_strat, scoring='f1_macro', n_jobs=1)

print("\n=== HASIL F1-MACRO TIAP FOLD ===")
print("Fold | Random Forest | XGBoost")
print("---------------------------------")
for i in range(n_splits):
    print(f"{i+1:4} | {rf_scores[i]:.4f}        | {xgb_scores[i]:.4f}")

rf_mean = np.mean(rf_scores)
xgb_mean = np.mean(xgb_scores)
print("---------------------------------")
print(f"MEAN | {rf_mean:.4f}        | {xgb_mean:.4f}")

# Uji statistik Wilcoxon
print("\n=== UJI STATISTIK WILCOXON SIGNED-RANK ===")
stat, p_value = wilcoxon(rf_scores, xgb_scores)

print(f"Statistik T : {stat}")
print(f"P-value     : {p_value:.5f}")

if p_value < 0.05:
    print("Kesimpulan  : Perbedaan performa antara Random Forest dan XGBoost SIGNIFIKAN secara statistik (p < 0.05).")
else:
    print("Kesimpulan  : Perbedaan performa antara Random Forest dan XGBoost TIDAK SIGNIFIKAN secara statistik (p >= 0.05).")

print("Selesai.")
