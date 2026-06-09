# Pregnancy Risk Classification Using Machine Learning Algorithms (KSPR)

This project aims to build a pregnancy risk prediction model based on the **Poedji Rochjati Score Card (KSPR)** parameters using the 2023 Indonesian Health Survey (SKI) dataset. This research follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

## 🚀 Overview
This project conducts a comparative analysis between the **Random Forest** (parallel bagging) and **XGBoost** (sequential boosting) algorithms to classify maternal risk levels into three categories: **Low Risk, High Risk, and Very High Risk**.

### Key Updates (Version 5):
- **Anti-Leakage Pipeline (No Data Leakage)**: Implementation of `imblearn.pipeline.Pipeline` ensures that the SMOTE (Synthetic Minority Over-sampling Technique) process is exclusively executed on the training folds during Cross-Validation, preserving the purity of the validation data.
- **Advanced Hyperparameter Optimization**: Utilizing `RandomizedSearchCV` with comprehensive iterations (50 iterations for RF, 60 iterations for XGBoost) to find the best parameters.
- **Statistical Significance Testing**: Performing **10-Fold Cross-Validation** combined with the **Wilcoxon Signed-Rank Test** to scientifically prove the significance of performance differences between models.
- **Advanced Clinical Interpretability (XAI)**: Applying **SHAP (SHapley Additive exPlanations)** through Summary Plots and Dependence Plots to provide clinical transparency regarding the impact of features like maternal age on predictions.
- **Automated Word Report**: The script exports all metric analysis results (Accuracy, F1-Macro, ROC-AUC, Specificity), CRISP-DM summary, and various charts into a Microsoft Word (`.docx`) report.

## 📁 File & Script Structure
- `laporan_crisp_dm_v5.py`: The main Machine Learning script that executes the CRISP-DM pipeline.
- `analisis_statistik_v5.py`: A dedicated companion script for advanced statistical analysis computations (calculating 10-Fold CV Wilcoxon Test, Standard Deviation, Specificity evaluation, and rendering SHAP Dependence Plots).
- `.gitignore`: Security filter to prevent raw datasets, large cache files, and other confidential files from being uploaded to the public repository.

> **Privacy Note**: The original dataset (`final_dataset_kspr_attala.csv`) and model cache files (`.pkl`) are intentionally excluded from GitHub to protect health respondent privacy and save server quota.

## 🛠️ How to Run
1. Ensure Python 3.x is installed on your system.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap python-docx scipy joblib
   ```
3. Place your dataset file (`final_dataset_kspr_attala.csv`) in the same directory.
4. Run the main ML script:
   ```bash
   python laporan_crisp_dm_v5.py
   ```
5. To process additional metrics and SHAP plots from the advanced analysis, run:
   ```bash
   python analisis_statistik_v5.py
   ```

## 📈 Experimental Results
The final experiments demonstrate that **Tuned XGBoost** achieved the highest performance with:
- **ROC-AUC (OvR)**: 0.9946
- **F1-Macro**: 0.9207
- The evaluation proves that XGBoost precisely suppresses False Positives and produces a very high Specificity value without compromising the clinical protection of Very High-Risk patients.

---
*Developed for the purpose of scientific article publication and integration of maternal risk early warning systems.*

---
---

# Klasifikasi Risiko Kehamilan Menggunakan Algoritma Machine Learning (KSPR)

Proyek ini bertujuan untuk membangun model prediksi risiko kehamilan berdasarkan parameter **Skor Poedji Rochjati (KSPR)** menggunakan dataset Survei Kesehatan Indonesia (SKI) 2023. Penelitian ini mengikuti metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).

## 🚀 Gambaran Umum
Proyek ini melakukan analisis komparatif antara algoritma **Random Forest** (parallel bagging) dan **XGBoost** (sequential boosting) untuk mengklasifikasikan tingkat risiko maternal menjadi tiga kategori: **Risiko Rendah, Risiko Sedang, dan Risiko Sangat Tinggi**.

### Pembaruan Utama (Versi 5):
- **Pipeline Anti-Bocor (No Data Leakage)**: Implementasi `imblearn.pipeline.Pipeline` memastikan proses SMOTE (Synthetic Minority Over-sampling Technique) hanya dieksekusi pada data latih (*training fold*) saat Cross-Validation, menjaga kemurnian data validasi.
- **Optimasi Hyperparameter Tingkat Lanjut**: Menggunakan `RandomizedSearchCV` dengan iterasi komprehensif (50 iterasi untuk RF, 60 iterasi untuk XGBoost) untuk mencari parameter terbaik.
- **Uji Signifikansi Statistik (Baru)**: Melakukan komputasi **10-Fold Cross-Validation** yang digabungkan dengan **Wilcoxon Signed-Rank Test** untuk membuktikan signifikansi perbedaan performa antar model secara ilmiah.
- **Interpretabilitas Klinis Tingkat Lanjut (XAI)**: Menerapkan **SHAP (SHapley Additive exPlanations)** melalui *Summary Plots* dan *Dependence Plots* guna memberikan transparansi klinis mengenai pengaruh fitur seperti usia ibu terhadap prediksi.
- **Laporan Word Otomatis**: Skrip mengekspor seluruh hasil analisis metrik (Accuracy, F1-Macro, ROC-AUC, Specificity), ringkasan CRISP-DM, dan berbagai grafik ke dalam laporan Microsoft Word (`.docx`).

## 📁 Struktur File & Script
- `laporan_crisp_dm_v5.py`: Script *Machine Learning* utama (orisinal) yang mengeksekusi pipeline CRISP-DM.
- `analisis_statistik_v5.py`: Script pendamping khusus untuk komputasi analisis statistik tingkat lanjut (menghitung Uji Wilcoxon 10-Fold CV, kalkulasi Standar Deviasi, evaluasi Specificity, dan merender SHAP Dependence Plot).
- `.gitignore`: Filter keamanan agar dataset mentah, file *cache* berukuran besar, dan file rahasia lainnya tidak terunggah ke repositori publik.

> **Catatan Privasi**: Dataset asli (`final_dataset_kspr_attala.csv`) dan file *model cache* (`.pkl`) sengaja tidak disertakan di GitHub untuk melindungi privasi responden kesehatan dan menghemat kuota server.

## 🛠️ Cara Menjalankan
1. Pastikan Python 3.x telah terinstal di sistem Anda.
2. Install pustaka (*library*) yang dibutuhkan:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap python-docx scipy joblib
   ```
3. Tempatkan file dataset Anda (`final_dataset_kspr_attala.csv`) di direktori yang sama.
4. Jalankan script ML utama:
   ```bash
   python laporan_crisp_dm_v5.py
   ```
5. Untuk memproses metrik tambahan dan plot SHAP hasil analisis lanjutan, jalankan:
   ```bash
   python analisis_statistik_v5.py
   ```

## 📈 Hasil Eksperimen
Eksperimen akhir menunjukkan bahwa **XGBoost Tuned** mencapai performa tertinggi dengan:
- **ROC-AUC (OvR)**: 0.9946
- **F1-Macro**: 0.9207
- Evaluasi membuktikan bahwa XGBoost secara presisi menekan *False Positives* dan menghasilkan nilai *Specificity* yang sangat tinggi tanpa mengorbankan perlindungan klinis pasien Risiko Sangat Tinggi. 

---
*Dikembangkan untuk keperluan publikasi artikel ilmiah dan integrasi peringatan dini risiko kehamilan.*
