import joblib, shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = 'final_dataset_kspr_attala.csv'
CACHE_FILE = 'v5_cache.pkl'

print("Loading data...")
df = pd.read_csv(DATA_FILE)
drop_cols = ['label_risiko']
id_cols = ['id_anggota_rt', 'id_rumah_tangga', 'id_provinsi', 'id_kabupaten']
for c in id_cols:
    if c in df.columns: drop_cols.append(c)
if 'metode_persalinan_sesar' in df.columns: drop_cols.append('metode_persalinan_sesar')
if 'operasi_caesar' in df.columns: drop_cols.append('operasi_caesar')

X = df.drop(drop_cols, axis=1)
y = df['label_risiko']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_test_sample = X_test.sample(min(300, len(X_test)), random_state=42)

print("Loading cache...")
cache = joblib.load(CACHE_FILE)
xgb_tuned = cache['xgb_tuned']
xgb_clf_step = xgb_tuned.named_steps['clf']

print("Calculating SHAP...")
explainer_xgb = shap.TreeExplainer(xgb_clf_step)
shap_values_xgb = explainer_xgb.shap_values(X_test_sample)

if isinstance(shap_values_xgb, list) and len(shap_values_xgb) > 2:
    shap_vals = shap_values_xgb[2]
elif hasattr(shap_values_xgb, 'shape') and len(shap_values_xgb.shape) == 3:
    shap_vals = shap_values_xgb[:, :, 2]
else:
    shap_vals = shap_values_xgb

print("Plotting...")
plt.figure(figsize=(8, 6))
shap.dependence_plot("umur_ibu_tahun", shap_vals, X_test_sample, show=False)
plt.title("XGBoost SHAP Dependence Plot: Usia Ibu (Risiko Sangat Tinggi)", pad=20)
plt.tight_layout()
plt.savefig("fig_shap_dep.png", dpi=150)
print("Berhasil menyimpan fig_shap_dep.png!")
