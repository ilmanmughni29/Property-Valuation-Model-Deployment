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
    ├── xgboost_tuned_pipeline.pkl           <- copy dari notebook Anda
    └── xgboost_tuned_pipeline_metadata.pkl  <- copy dari notebook Anda
```

Copy kedua file `.pkl` yang sudah Anda hasilkan (dari cell 109 di notebook, tersimpan
di `...\Final Project\models\`) ke dalam folder `models/` di atas.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Jalankan

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Catatan penting

- **Kenapa harus ada fungsi `to_string_array` di `app.py`?**
  Pipeline Anda memakai `FunctionTransformer(to_string_array)` di dalam
  `ColumnTransformer` (cell "Preprocessing pipeline"). Saat di-`joblib.dump()`,
  Python hanya menyimpan *referensi* ke fungsi tersebut (nama + modul), bukan
  kodenya. Kalau file `.pkl` di-load di script lain yang tidak punya fungsi
  bernama sama, akan muncul error
  `AttributeError: Can't get attribute 'to_string_array'`.
  Karena itu, `app.py` mendefinisikan ulang fungsi yang **identik persis**
  dengan yang ada di notebook, sebelum memanggil `joblib.load()`. Kalau Anda
  mengubah fungsi ini di notebook dan retrain, update juga versi di `app.py`.

- **Fitur turunan (engineered features)** — seperti `log_total_livable_area`,
  `bath_bed_ratio`, `building_age`, `building_era`, `has_central_air`,
  `has_garage`, `has_fireplace`, `has_basement` — dihitung ulang di dalam
  `app.py` persis dengan formula di notebook (Section 2.5 & 4.1), berdasarkan
  input mentah yang diisi user. Kalau Anda menambah/mengubah fitur di notebook,
  update juga logika di bagian "Derived / engineered features" pada `app.py`.

- **Kode kategori mentah OPA** (`basements`, `garage_type`,
  `general_construction`, `topography`, `view_type`, `type_heater`,
  `parcel_shape`, `zoning`) — di UI disediakan sebagai input opsional dengan
  default `"Unknown"`. Ini aman karena pipeline training Anda memang dirancang
  untuk menangani nilai hilang/`"Unknown"` pada kolom-kolom ini
  (`SimpleImputer(fill_value="Unknown")` + `OneHotEncoder(handle_unknown="ignore")`
  / `OrdinalEncoder(unknown_value=-1)`). Jika user tahu kode aslinya (misal
  zoning `RSA5`), mereka bisa isi manual untuk hasil yang lebih akurat.

- **`category_code`** dipetakan dari deskripsi yang mudah dibaca user
  (Single Family, Multi Family, Mixed Use, Commercial, Industrial, Vacant
  Land) ke kode numerik standar OPA (1–6). Ini konsisten dengan komentar di
  notebook Anda sendiri (`category_code 5 = Industrial`, `6 = Vacant Land`).
  **Disarankan: verifikasi ulang mapping 1–4 terhadap kolom
  `category_code`/`category_code_description` di `df_raw` Anda**, untuk
  memastikan 100% cocok dengan data asli sebelum deployment produksi.

- **Batasan model** (dari Section 5.4 notebook Anda): akurasi menurun untuk
  properti luxury (>$1M) dan distressed (<$50K). App menampilkan warning
  otomatis untuk prediksi di rentang tersebut.

## Deploy ke Streamlit Community Cloud (opsional)

1. Push folder ini ke repo GitHub (sertakan folder `models/` — perhatikan
   ukuran file `.pkl` Anda, ~6.7 MB, masih aman untuk GitHub & Streamlit Cloud
   free tier, batas defaultnya 1 GB per repo).
2. Buka [share.streamlit.io](https://share.streamlit.io), connect ke repo,
   pilih `app.py` sebagai entry point.
3. Streamlit Cloud otomatis install dari `requirements.txt`.
