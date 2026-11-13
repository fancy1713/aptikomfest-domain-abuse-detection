# Domain Abuse Detection — Pesta Data Nasional (PeDaS) APTIKOMFEST 2026

🥈 **Silver Award — Babak Penyisihan**

Model klasifikasi multi-kelas untuk mendeteksi kategori penyalahgunaan domain/URL (judi online, phishing, malware, spam, dll) berdasarkan data laporan/observasi URL.

**Kompetisi:** Pesta Data Nasional (PeDaS), APTIKOMFEST 2026
**Role:** Data Scientist / ML Engineer

---

## 📌 Problem Statement

Setiap baris data merupakan satu laporan dugaan penyalahgunaan domain. Tugasnya: mengklasifikasikan setiap laporan ke dalam salah satu dari **9 kategori** — *online gambling, phishing, malware, spam, brand, fakeshop, violence, pii exposure, other* — berdasarkan URL dan metadata terkait. Tantangan utama: distribusi kelas yang sangat tidak seimbang dan sinyal yang sebagian besar berupa teks (URL) tidak terstruktur.

## 🧠 Pendekatan

### 1. Exploratory Data Analysis (EDA)
- Analisis distribusi label/kategori dan pembersihan target.
- Analisis missing value menyeluruh per kolom.
- **Word Cloud** per kategori untuk mengidentifikasi pola kata kunci khas tiap jenis penyalahgunaan.

### 2. Feature Engineering
- **Keyword Signals** — fitur biner berbasis kemunculan kata kunci spesifik domain (mis. indikator judi online, phishing, malware) yang diekstraksi dari URL.
- **TF-IDF** — representasi teks dari URL untuk menangkap pola n-gram yang relevan.
- Kombinasi fitur numerik + TF-IDF menjadi satu **hybrid feature matrix** (`X_hybrid`) menggunakan sparse matrix (`scipy.sparse.hstack`).

### 3. Handling Class Imbalance
- **SMOTE + RandomOverSampler**, diterapkan **hanya pada data training per-fold** (bukan pada data validasi) untuk mencegah data leakage.
- Kelas dengan jumlah anggota terlalu sedikit (`< n_splits`) dikeluarkan sementara dari cross-validation, namun tetap dipakai penuh saat training model final.
- `sample_weight` berbasis `class_weight='balanced'` sebagai lapisan penyeimbang tambahan.

### 4. Modeling — Ensemble dengan Hyperparameter Tuning
- **XGBoost** dan **LightGBM**, masing-masing di-tuning menggunakan **Optuna** (objective function dengan `StratifiedKFold` untuk validasi).
- Parameter yang dituning: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, regularisasi (`reg_alpha`, `reg_lambda`), dll.
- Model final dilatih sebagai **ensemble XGBoost + LightGBM** pada seluruh data (9 kelas penuh).
- Evaluasi menggunakan **F1-Score** (cocok untuk multi-class imbalanced) dan `classification_report`.

### 5. Heuristic Post-Processing
- Lapisan aturan berbasis kata kunci eksplisit (mis. daftar kata kunci judi online, phishing, malware) diterapkan **setelah** prediksi ML untuk menangkap kasus-kasus jelas yang mungkin terlewat model statistik — dicek independen per kategori, bukan `elif` berantai, agar beberapa sinyal bisa overlap.

## 🏆 Hasil

- **Silver Award** pada babak penyisihan kompetisi.
- Pipeline berhasil menangani 9 kelas dengan distribusi sangat timpang tanpa mengorbankan recall pada kelas minoritas.

## 🛠️ Tech Stack

`Python` `XGBoost` `LightGBM` `Optuna` `Scikit-learn` `imbalanced-learn (SMOTE)` `TF-IDF` `WordCloud` `Pandas`

## 📁 Struktur Repo

```
aptikomfest-domain-abuse-detection/
├── notebooks/
│   └── TeamLearningByDoing3.ipynb   # EDA, feature engineering, modeling, evaluasi
└── README.md
```

> **Catatan:** dataset mentah adalah milik penyelenggara kompetisi (APTIKOMFEST) dan tidak diunggah ulang di repo ini.

## 📸 Visualisasi

> Tambahkan screenshot Word Cloud per kategori dan grafik evaluasi (confusion matrix / F1 per kelas).

---

*Dikerjakan sebagai bagian dari tim "LearningByDoing" pada Pesta Data Nasional (PeDaS), APTIKOMFEST 2026.*
