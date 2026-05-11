# Klasifikasi Risiko Kehamilan Menggunakan Algoritma Machine Learning (KSPR)

Proyek ini bertujuan untuk membangun model prediksi risiko kehamilan berdasarkan dataset **KSPR (Kartu Skor Poedji Rochjati)** dengan mengikuti metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).

## 🚀 Gambaran Umum
Proyek ini melakukan analisis komparatif antara algoritma **Random Forest** dan **XGBoost** untuk mengklasifikasikan tingkat risiko kehamilan menjadi tiga kategori: **Rendah, Sedang, dan Tinggi**.

### Fitur Utama:
- **Pipeline Anti-Bocor**: Implementasi `imblearn.Pipeline` untuk memastikan SMOTE hanya dilakukan pada data training saat Cross-Validation, mencegah *data leakage*.
- **Optimasi Hyperparameter**: Menggunakan `RandomizedSearchCV` untuk mencari parameter terbaik pada model RF dan XGBoost.
- **Interpretabilitas Model**: Menggunakan **SHAP (SHapley Additive exPlanations)** untuk menjelaskan bagaimana fitur-fitur klinis mempengaruhi prediksi model.
- **Laporan Otomatis**: Script ini secara otomatis menghasilkan laporan dalam format Microsoft Word (.docx) yang berisi visualisasi (Confusion Matrix, ROC Curve) dan hasil metrik evaluasi.

## 📊 Metodologi (CRISP-DM)
1. **Business Understanding**: Identifikasi faktor risiko klinis pada ibu hamil.
2. **Data Understanding**: Eksplorasi distribusi kelas (risiko rendah, sedang, tinggi).
3. **Data Preparation**: Penanganan data hilang, encoding, dan penanganan ketidakseimbangan kelas menggunakan **SMOTE**.
4. **Modeling**: Training model Baseline vs Tuned (Random Forest & XGBoost).
5. **Evaluation**: Evaluasi menggunakan Accuracy, F1-Score, dan AUC-ROC.
6. **Deployment**: Export model dan hasil laporan analisis.

## 📁 Dataset
> **Catatan Privasi**: Dataset asli (`final_dataset_kspr_attala.csv`) tidak disertakan dalam repositori ini untuk menjaga privasi data responden.

## 🛠️ Cara Menjalankan
1. Pastikan Python 3.x terinstall.
2. Install library yang dibutuhkan:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap python-docx
   ```
3. Jalankan script utama:
   ```bash
   python laporan_crisp_dm_v5.py
   ```

## 📈 Hasil Eksperimen
Model menghasilkan evaluasi performa yang detail termasuk:
- **Confusion Matrix** (untuk melihat akurasi prediksi per kelas).
- **ROC Curve** (untuk mengukur kemampuan diskriminasi model).
- **Feature Importance** (mengidentifikasi variabel paling berpengaruh).

---
*Dibuat untuk keperluan penelitian/tugas akhir mengenai analisis risiko kesehatan maternal.*
