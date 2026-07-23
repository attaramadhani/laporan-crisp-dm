import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from PIL import Image

# --- CONFIGURATION ---
DATA_FILE = 'dataset_ski_2023.csv'
CACHE_FILE = 'model_cache.pkl'
OUTPUT_DIR = 'figures_final'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# --- FEATURE TRANSLATION DICTIONARY ---
feature_translation = {
    'id_anggota_rt': 'RT Member ID',
    'id_rumah_tangga': 'Household ID',
    'id_provinsi': 'Province ID',
    'id_kabupaten': 'Regency ID',
    'tipe_desa_kota': 'Urban/Rural Residence',
    'umur_ibu_tahun': 'Maternal Age (years)',
    'pendidikan': 'Maternal Education Level',
    'pekerjaan': 'Maternal Occupation',
    'umur_hamil_pertama': 'Age at First Pregnancy',
    'total_hamil_gravida': 'Gravida (Total Pregnancies)',
    'total_lahir_paritas': 'Parity (Prior Births)',
    'total_keguguran_abortus': 'Total Miscarriages',
    'pernah_melahirkan_periode_ini': 'Recent Birth (in Period)',
    'status_hamil_kembar': 'Multiple Pregnancy (Twins)',
    'periksa_kehamilan_medis': 'ANC Access (Medical)',
    'usia_kandungan_periksa_pertama_bulan': 'Gestational Age at First ANC (months)',
    'faskes_anc_tersering': 'Most Frequent ANC Facility',
    'anc_ukur_tinggi_badan': 'ANC Height Check',
    'anc_timbang_berat': 'ANC Weight Monitored',
    'anc_tensi_darah': 'ANC Blood Pressure Check',
    'anc_tes_hb': 'ANC Hemoglobin Test',
    'penolong_persalinan': 'Birth Attendant',
    'tempat_persalinan': 'Place of Delivery',
    'metode_persalinan_sesar': 'Cesarean Delivery Method',
    'terima_gizi_karena_anemia': 'Received Anemia Nutrition',
    'terima_gizi_karena_anemia_2': 'Received Anemia Nutrition (Phase 2)',
    'konsumsi_tablet_tambah_darah': 'Iron Pill Consumption',
    'aman_tidak_ada_faktor_risiko': 'Safe (No Risk Factors)',
    'muntah_diare_saat_hamil': 'Severe Vomiting/Diarrhea',
    'demam_saat_hamil': 'Gestational Fever',
    'hipertensi_saat_hamil': 'Gestational Hypertension',
    'janin_kurang_gerak': 'Reduced Fetal Movement',
    'pendarahan_saat_hamil': 'Gestational Bleeding',
    'ketuban_pecah_dini_saat_hamil': 'Premature Rupture of Membranes',
    'sakit_kencing_saat_hamil': 'Urinary Pain during Pregnancy',
    'batuk_lama_saat_hamil': 'Chronic Cough during Pregnancy',
    'sesak_napas_saat_hamil': 'Dyspnea during Pregnancy',
    'nyeri_dada_saat_hamil': 'Chest Pain during Pregnancy',
    'bengkak_kaki_saat_hamil': 'Swollen Feet during Pregnancy',
    'kejang_saat_hamil': 'Gestational Convulsions',
    'komplikasi_lain_saat_hamil': 'Other Pregnancy Complications',
    'tidak_ada_komplikasi_hamil': 'No Gestational Complications',
    'posisi_janin_sungsang': 'Breech Fetal Position',
    'pendarahan_saat_bersalin': 'Delivery Bleeding',
    'kejang_saat_bersalin': 'Delivery Convulsions',
    'ketuban_pecah_dini_saat_bersalin': 'PROM during Delivery',
    'partus_lama': 'Prolonged Labor',
    'lilitan_tali_pusar': 'Nuchal Cord',
    'plasenta_previa': 'Placenta Previa',
    'plasenta_tertinggal': 'Retained Placenta',
    'hipertensi_saat_bersalin': 'Delivery Hypertension',
    'komplikasi_lain_saat_bersalin': 'Other Delivery Complications',
    'tidak_ada_komplikasi_bersalin': 'No Prior Delivery Complications',
    'pendarahan_nifas': 'Postpartum Hemorrhage',
    'cairan_berbau_nifas': 'Foul Postpartum Discharge',
    'bengkak_pusing_nifas': 'Postpartum Edema & Dizziness',
    'kejang_nifas': 'Postpartum Convulsions',
    'demam_nifas': 'Postpartum Fever',
    'payudara_bengkak_nifas': 'Mastitis/Breast Engorgement',
    'depresi_nifas': 'Postpartum Depression',
    'tidak_ada_komplikasi_nifas': 'No Postpartum Complications',
    'label_risiko': 'Risk Label'
}

# --- LOAD DATA AND CACHE ---
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
drop_cols = ['label_risiko', 'id_anggota_rt', 'id_rumah_tangga', 'id_provinsi', 'id_kabupaten']
if 'metode_persalinan_sesar' in df.columns: drop_cols.append('metode_persalinan_sesar')
if 'operasi_caesar' in df.columns:          drop_cols.append('operasi_caesar')

X = df.drop(drop_cols, axis=1)
y = df['label_risiko']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

print("Loading cache file...")
cache = joblib.load(CACHE_FILE)

# Helper function to pad PIL images to same height
def pad_to_height(img, target_h):
    if img.size[1] == target_h:
        return img
    delta = target_h - img.size[1]
    padded = Image.new("RGB", (img.size[0], target_h), (255, 255, 255))
    padded.paste(img, (0, delta // 2))
    return padded

def enlarge_figure_texts(fig, tick_size=18, label_size=20, text_size=18, legend_size=14):
    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        try:
            if ax.xaxis and ax.xaxis.label:
                ax.xaxis.label.set_fontsize(label_size)
                ax.xaxis.label.set_fontweight('bold')
        except Exception:
            pass
        try:
            if ax.yaxis and ax.yaxis.label:
                ax.yaxis.label.set_fontsize(label_size)
                ax.yaxis.label.set_fontweight('bold')
        except Exception:
            pass
        for text in ax.texts:
            text.set_fontsize(text_size)
        try:
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(legend_size)
        except Exception:
            pass

# ==============================================================================
# FIGURE 2: Confusion Matrices
# ==============================================================================
def generate_figure_2():
    print("Generating Figure 2...")
    cm_rf_t = cache['cm_rf_t']
    cm_xgb_t = cache['cm_xgb_t']
    cm_xgb_t_norm = cm_xgb_t.astype('float') / cm_xgb_t.sum(axis=1)[:, np.newaxis]
    
    classes = ['Low Risk', 'High Risk', 'Very High Risk']
    
    # Setup plotting style for clarity and size
    plt.close('all')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
    
    # (a) Tuned Random Forest (Absolute counts)
    sns.heatmap(cm_rf_t, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 20, 'weight': 'bold'}, cbar=True)
    axes[0].set_title("(a) Tuned Random Forest\n(Absolute Counts)", fontsize=18, fontweight='bold', pad=15)
    axes[0].set_xlabel("Predicted Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[0].set_ylabel("Actual Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[0].tick_params(axis='both', labelsize=14)
    
    # (b) Tuned XGBoost (Absolute counts)
    sns.heatmap(cm_xgb_t, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 20, 'weight': 'bold'}, cbar=True)
    axes[1].set_title("(b) Tuned XGBoost\n(Absolute Counts)", fontsize=18, fontweight='bold', pad=15)
    axes[1].set_xlabel("Predicted Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[1].set_ylabel("Actual Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[1].tick_params(axis='both', labelsize=14)
    
    # (c) Tuned XGBoost (Normalized percentages)
    sns.heatmap(cm_xgb_t_norm, annot=True, fmt='.1%', cmap='Blues', ax=axes[2],
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 20, 'weight': 'bold'}, cbar=True)
    axes[2].set_title("(c) Tuned XGBoost\n(Normalized Percentages)", fontsize=18, fontweight='bold', pad=15)
    axes[2].set_xlabel("Predicted Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[2].set_ylabel("Actual Label", fontsize=15, fontweight='bold', labelpad=10)
    axes[2].tick_params(axis='both', labelsize=14)
    
    plt.tight_layout()
    fig = plt.gcf()
    enlarge_figure_texts(fig, tick_size=14, label_size=16, text_size=18)
    out_path = os.path.join(OUTPUT_DIR, "fig_2_confusion_matrices.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    # Restore defaults
    matplotlib.rcdefaults()

# ==============================================================================
# FIGURE 3: Multiclass Discrimination Analysis
# ==============================================================================
def generate_figure_3():
    print("Generating Figure 3...")
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_proba_rf_t = cache['y_proba_rf_t']
    y_proba_xgb_t = cache['y_proba_xgb_t']
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] # Blue, Orange, Red
    class_names = ['Low Risk', 'High Risk', 'Very High Risk']
    
    plt.close('all')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
    
    # Subplot (a): ROC-AUC curves for Tuned Random Forest
    for idx, (color, name) in enumerate(zip(colors, class_names)):
        fpr_rf, tpr_rf, _ = roc_curve(y_test_bin[:, idx], y_proba_rf_t[:, idx])
        auc_rf = auc(fpr_rf, tpr_rf)
        axes[0].plot(fpr_rf, tpr_rf, color=color, lw=3, linestyle='-',
                     label=f'{name} (AUC = {auc_rf:.4f})')
                     
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=8)
    axes[0].set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=8)
    axes[0].set_title('(a) ROC Curves (RF)', fontsize=16, fontweight='bold', pad=15)
    axes[0].legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Subplot (b): ROC-AUC curves for Tuned XGBoost
    for idx, (color, name) in enumerate(zip(colors, class_names)):
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin[:, idx], y_proba_xgb_t[:, idx])
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        axes[1].plot(fpr_xgb, tpr_xgb, color=color, lw=3, linestyle='-',
                     label=f'{name} (AUC = {auc_xgb:.4f})')
                     
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Classifier')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=8)
    axes[1].set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=8)
    axes[1].set_title('(b) ROC Curves (XGBoost)', fontsize=16, fontweight='bold', pad=15)
    axes[1].legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=12)
    
    # Subplot (c): Precision-Recall (PR) curves of the Tuned XGBoost model
    for idx, (color, name) in enumerate(zip(colors, class_names)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, idx], y_proba_xgb_t[:, idx])
        pr_auc = average_precision_score(y_test_bin[:, idx], y_proba_xgb_t[:, idx])
        axes[2].plot(recall, precision, color=color, lw=3, linestyle='-',
                     label=f'{name} (PR-AUC = {pr_auc:.4f})')
                     
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.02])
    axes[2].set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold', labelpad=8)
    axes[2].set_ylabel('Precision', fontsize=14, fontweight='bold', labelpad=8)
    axes[2].set_title('(c) PR Curves (XGBoost)', fontsize=16, fontweight='bold', pad=15)
    axes[2].legend(loc='lower left', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    fig = plt.gcf()
    enlarge_figure_texts(fig, tick_size=13, label_size=15, text_size=13, legend_size=15)
    out_path = os.path.join(OUTPUT_DIR, "fig_3_multiclass_discrimination.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    # Restore defaults
    matplotlib.rcdefaults()

# ==============================================================================
# FIGURE 4: SHAP Summary Plots (Tuned RF vs XGBoost)
# ==============================================================================
def generate_figure_4():
    print("Generating Figure 4...")
    rf_clf = cache['rf_tuned'].named_steps['clf']
    xgb_clf = cache['xgb_tuned'].named_steps['clf']
    
    explainer_rf = shap.TreeExplainer(rf_clf)
    explainer_xgb = shap.TreeExplainer(xgb_clf)
    
    X_test_sample = X_test.sample(min(120, len(X_test)), random_state=RANDOM_STATE)
    X_test_sample_en = X_test_sample.rename(columns=feature_translation)
    
    shap_values_rf = explainer_rf.shap_values(X_test_sample, check_additivity=False)
    shap_values_xgb = explainer_xgb.shap_values(X_test_sample, check_additivity=False)
    
    # Extract Class 2 (Very High Risk)
    if isinstance(shap_values_rf, list) and len(shap_values_rf) > 2:
        shap_rf_c2 = shap_values_rf[2]
    elif hasattr(shap_values_rf, 'shape') and len(shap_values_rf.shape) == 3:
        shap_rf_c2 = shap_values_rf[:, :, 2]
    else:
        shap_rf_c2 = shap_values_rf
        
    if isinstance(shap_values_xgb, list) and len(shap_values_xgb) > 2:
        shap_xgb_c2 = shap_values_xgb[2]
    elif hasattr(shap_values_xgb, 'shape') and len(shap_values_xgb.shape) == 3:
        shap_xgb_c2 = shap_values_xgb[:, :, 2]
    else:
        shap_xgb_c2 = shap_values_xgb

    # Temporarily update matplotlib parameters for SHAP rendering
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    def render_shap_dot_to_buf(shap_vals, X_en, title):
        plt.close('all')
        fig = plt.figure(figsize=(8.0, 5.8))
        shap.summary_plot(shap_vals, X_en, show=False)
        fig = plt.gcf()
        enlarge_figure_texts(fig, tick_size=14, label_size=16, text_size=14)
        plt.title(title, fontsize=18, fontweight='bold', pad=15)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        return buf

    print("Rendering RF SHAP summary plot...")
    buf_rf = render_shap_dot_to_buf(shap_rf_c2, X_test_sample_en, "(a) Tuned Random Forest")
    
    print("Rendering XGBoost SHAP summary plot...")
    buf_xgb = render_shap_dot_to_buf(shap_xgb_c2, X_test_sample_en, "(b) Tuned XGBoost")
    
    # Stitch side-by-side
    img_rf = Image.open(buf_rf)
    img_xgb = Image.open(buf_xgb)
    
    target_h = max(img_rf.size[1], img_xgb.size[1])
    img_rf = pad_to_height(img_rf, target_h)
    img_xgb = pad_to_height(img_xgb, target_h)
    
    gap = 40
    total_w = img_rf.size[0] + gap + img_xgb.size[0]
    combined = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    combined.paste(img_rf, (0, 0))
    combined.paste(img_xgb, (img_rf.size[0] + gap, 0))
    
    out_path = os.path.join(OUTPUT_DIR, "fig_4_shap_summary.png")
    combined.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}")
    
    # Restore default plt params
    matplotlib.rcdefaults()

# ==============================================================================
# FIGURE 5: SHAP Waterfall Plot
# ==============================================================================
def generate_figure_5():
    print("Generating Figure 5...")
    xgb_clf = cache['xgb_tuned'].named_steps['clf']
    y_pred_xgb_t = cache['y_pred_xgb_t']
    
    explainer_xgb = shap.TreeExplainer(xgb_clf)
    
    very_high_idx = np.where(y_test == 2)[0]
    tp_idx = [i for i in very_high_idx if y_pred_xgb_t[i] == 2][0]
    
    # Only calculate SHAP values for the specific patient to make it run instantly
    shap_values_single = explainer_xgb.shap_values(X_test.iloc[[tp_idx]])
    
    # Extract Class 2
    if isinstance(shap_values_single, list) and len(shap_values_single) > 2:
        shap_vals_single_c2 = shap_values_single[2][0]
    elif hasattr(shap_values_single, 'shape') and len(shap_values_single.shape) == 3:
        shap_vals_single_c2 = shap_values_single[0, :, 2]
    else:
        shap_vals_single_c2 = shap_values_single[0]
        
    if isinstance(explainer_xgb.expected_value, list):
        exp_val = explainer_xgb.expected_value[2]
    elif isinstance(explainer_xgb.expected_value, np.ndarray) and explainer_xgb.expected_value.size > 1:
        exp_val = explainer_xgb.expected_value[2]
    else:
        exp_val = explainer_xgb.expected_value
        
    X_test_en = X_test.rename(columns=feature_translation)
    
    # Update style for readability
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    plt.close('all')
    plt.figure(figsize=(9.0, 6.5))
    
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals_single_c2, 
        base_values=exp_val, 
        data=X_test_en.iloc[tp_idx], 
        feature_names=X_test_en.columns
    ), show=False)
    
    fig = plt.gcf()
    enlarge_figure_texts(fig, tick_size=14, label_size=16, text_size=14)
    
    plt.title('Figure 5. SHAP Waterfall Plot for a Very High-Risk Patient', fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "fig_5_shap_waterfall.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    # Restore default plt params
    matplotlib.rcdefaults()

# ==============================================================================
# FIGURE 6: XGBoost SHAP Dependence Plots
# ==============================================================================
def generate_figure_6():
    print("Generating Figure 6...")
    xgb_clf = cache['xgb_tuned'].named_steps['clf']
    explainer_xgb = shap.TreeExplainer(xgb_clf)
    
    X_test_sample = X_test.sample(min(120, len(X_test)), random_state=RANDOM_STATE)
    X_test_sample_en = X_test_sample.rename(columns=feature_translation)
    
    shap_values_xgb = explainer_xgb.shap_values(X_test_sample, check_additivity=False)
    
    # Extract Class 2 SHAP values
    if isinstance(shap_values_xgb, list) and len(shap_values_xgb) > 2:
        shap_vals_class2 = shap_values_xgb[2]
    elif hasattr(shap_values_xgb, 'shape') and len(shap_values_xgb.shape) == 3:
        shap_vals_class2 = shap_values_xgb[:, :, 2]
    else:
        shap_vals_class2 = shap_values_xgb
        
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    def render_dep_to_buf(feature, shap_vals, X_en, title):
        plt.close('all')
        # Let shap create the figure
        shap.dependence_plot(feature, shap_vals, X_en, show=False)
        fig = plt.gcf()
        enlarge_figure_texts(fig, tick_size=14, label_size=16, text_size=14)
        fig.set_size_inches(6.2, 5.0)
        plt.title(title, fontsize=18, fontweight='bold', pad=15)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        return buf

    print("Rendering dependency subplots...")
    buf_age = render_dep_to_buf('Maternal Age (years)', shap_vals_class2, X_test_sample_en, "(a) Maternal Age")
    buf_misc = render_dep_to_buf('Total Miscarriages', shap_vals_class2, X_test_sample_en, "(b) Miscarriage History")
    buf_parity = render_dep_to_buf('Parity (Prior Births)', shap_vals_class2, X_test_sample_en, "(c) Parity")
    
    img_age = Image.open(buf_age)
    img_misc = Image.open(buf_misc)
    img_parity = Image.open(buf_parity)
    
    target_h = max(img_age.size[1], img_misc.size[1], img_parity.size[1])
    img_age = pad_to_height(img_age, target_h)
    img_misc = pad_to_height(img_misc, target_h)
    img_parity = pad_to_height(img_parity, target_h)
    
    gap = 35
    total_w = img_age.size[0] + gap + img_misc.size[0] + gap + img_parity.size[0]
    combined = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    combined.paste(img_age, (0, 0))
    combined.paste(img_misc, (img_age.size[0] + gap, 0))
    combined.paste(img_parity, (img_age.size[0] + gap + img_misc.size[0] + gap, 0))
    
    out_path = os.path.join(OUTPUT_DIR, "fig_6_shap_dependence.png")
    combined.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}")
    
    # Restore default plt params
    matplotlib.rcdefaults()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    start_time = time.time()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print(f"All figures generated successfully in {time.time() - start_time:.2f} seconds.")
