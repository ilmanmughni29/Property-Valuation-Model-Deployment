# Philadelphia OPA Property Valuation — Streamlit App

Aplikasi interaktif untuk mendeploy model XGBoost mass-appraisal dari notebook
`PHL_OPA_Analytics` (Section 4.6.1) menggunakan Streamlit.

## 1. Setup

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── xgboost_tuned_pipeline.pkl           
    └── xgboost_tuned_pipeline_metadata.pkl  
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Jalankan

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Deploy ke Streamlit Community Cloud (opsional)

1. Push folder ini ke repo GitHub (sertakan folder `models/` — perhatikan
   ukuran file `.pkl` Anda, ~6.7 MB, masih aman untuk GitHub & Streamlit Cloud
   free tier, batas defaultnya 1 GB per repo).
2. Buka [share.streamlit.io](https://share.streamlit.io), connect ke repo,
   pilih `app.py` sebagai entry point.
3. Streamlit Cloud otomatis install dari `requirements.txt`.
