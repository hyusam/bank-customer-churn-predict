# Bank Customer Churn Prediction

Machine learning project untuk memprediksi kemungkinan nasabah bank melakukan churn, lengkap dengan exploratory data analysis, model interpretability (SHAP), dan demo interaktif berbasis Streamlit.

<img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
<img src="https://img.shields.io/badge/scikit--learn-ML-orange" alt="scikit-learn">
<img src="https://img.shields.io/badge/Streamlit-Demo-red" alt="Streamlit">

**[Live Demo](https://hyusam-bank-customer-churn-predict-appapp-2lefp7.streamlit.app/)**

---

## Ringkasan Proyek

Bank kehilangan nasabah (churn) berdampak langsung ke revenue, dan mendeteksi nasabah berisiko churn lebih awal memungkinkan tim retensi bertindak proaktif. Proyek ini membangun model klasifikasi untuk memprediksi probabilitas churn nasabah berdasarkan data demografis dan perilaku perbankan mereka.

Dataset yang digunakan terdiri dari 10.000 baris data nasabah dengan 14 fitur.

## Business Problem
 
- Bagaimana mengidentifikasi nasabah yang berisiko tinggi untuk churn?
- Fitur apa saja yang paling berpengaruh terhadap keputusan nasabah untuk churn?
- Bagaimana model dapat digunakan secara praktis oleh tim non-teknis (misal tim retensi/CRM)?

## Exploratory Data Analysis
 
- **Target imbalance**: mayoritas nasabah tidak churn, sehingga perlu penanganan khusus saat modeling (`class_weight='balanced'`, `scale_pos_weight`).
- **Age**: nasabah yang churn cenderung berusia lebih tua (puncak 40–50+ tahun), sedangkan nasabah yang bertahan didominasi usia 30–40 tahun.
- **Geography**: nasabah dari **Germany** memiliki tingkat churn tertinggi (32.4%), 2x lipat dibanding France dan Spain.
- **NumOfProducts**: nasabah dengan 3–4 produk hampir seluruhnya churn, sementara nasabah dengan 1 produk juga menunjukkan churn rate yang cukup tinggi.
- **IsActiveMember**: nasabah tidak aktif punya churn rate jauh lebih tinggi (26.9%) dibanding nasabah aktif (14.3%), keaktifan menekan churn secara signifikan.
- **Gender**: nasabah perempuan menunjukkan kecenderungan churn lebih tinggi (25.1%) dibanding laki-laki.

Detail visualisasi dan insight lengkap ada di [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Modeling
 
Tiga model dibandingkan menggunakan 5-fold Stratified Cross Validation (metrik F1, karena data imbalance):
 
| Model | Mean F1 (CV) |
|---|---|
| **Random Forest** | 0.599 |
| XGBoost | 0.584 |
| Logistic Regression | 0.488 |
 
Random Forest dipilih untuk hyperparameter tuning (`RandomizedSearchCV`), menghasilkan performa akhir di test set:
 
| Metrik | Skor |
|---|---|
| Accuracy | 84.25% |
| Precision | 60.36% |
| Recall | 65.85% |
| F1 Score | 62.98% |
| ROC-AUC | 85.49% |
 
> Recall lebih tinggi dari precision, model cenderung lebih agresif menandai nasabah berisiko churn, trade-off yang wajar untuk kasus retensi: lebih baik salah waspada daripada gagal mendeteksi nasabah yang benar-benar akan pergi.

### Feature Importance & Explainability
 
Fitur numerik (`Age`, `NumOfProducts`, `Balance`) mendominasi kontribusi terhadap prediksi model, jauh di atas fitur kategorikal seperti `Geography` dan `Gender`, meski secara deskriptif keduanya terlihat berbeda churn rate-nya di EDA. Analisis **SHAP** digunakan untuk memvalidasi arah pengaruh tiap fitur secara individual dan mendukung transparansi model.
 
Detail proses preprocessing, cross-validation, tuning, dan SHAP analysis ada di [`notebooks/02_preprocessing_modeling.ipynb`](notebooks/02_preprocessing_modeling.ipynb).

## Demo Aplikasi (Streamlit)
 
Aplikasi interaktif untuk mencoba model secara langsung:
 
- **Prediksi Manual**: input data satu nasabah lewat form, hasil probabilitas churn + visualisasi SHAP waterfall yang menjelaskan kontribusi tiap fitur terhadap prediksi tersebut.
- **Prediksi Batch**: upload CSV berisi banyak nasabah sekaligus, hasil bisa diunduh kembali dalam format CSV.
- **Threshold Slider**: atur ambang batas klasifikasi churn secara interaktif untuk melihat trade-off precision vs recall.
- **Model Performance Card**: ringkasan metrik evaluasi model langsung di dalam aplikasi.

### Cara Menjalankan
 
```bash
# 1. Clone repository
git clone https://github.com/hyusam/bank-customer-churn-predict.git
cd bank-customer-churn-predict/app
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Jalankan aplikasi
streamlit run app.py
```

Pastikan file model `final_pipeline_rftuned.pkl` tersedia di folder `streamlit_app/models/`.

## Struktur Proyek
 
```
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   └── 02_preprocessing_modeling.ipynb  # Preprocessing, modeling, tuning, SHAP
├── app/
│   ├── app.py                           # Aplikasi demo Streamlit
│   ├── requirements.txt
├── models/
│   └── final_pipeline_rftuned.pkl   # Model final (pipeline lengkap)
├── data/
│   ├── bank_customer_churn.csv
│   └── CLEAN_bank_customer_churn.csv
├── assets/                              # Visualisasi hasil EDA & modeling
├── README.md
└── requirements.txt
```

## Tech Stack
 
- **Data Processing & EDA**: pandas, numpy, matplotlib, seaborn
- **Modeling**: scikit-learn, XGBoost, imbalanced-learn (SMOTENC)
- **Interpretability**: SHAP
- **Deployment/Demo**: Streamlit

## Author
 
**Wahyu Iqsam** ([@hyusam](https://github.com/hyusam))

---