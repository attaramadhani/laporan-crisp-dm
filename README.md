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
