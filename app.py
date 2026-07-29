"""
Philadelphia OPA Property Valuation — Interactive Prediction App
==================================================================
Deploys the tuned XGBoost mass-appraisal pipeline (see
PHL_OPA_Analytics notebook, Section 4.6 / 4.6.1) as an interactive
Streamlit application.

Expected files (place next to this script):
    models/xgboost_tuned_pipeline.pkl           <- sklearn Pipeline (prep + model)
    models/xgboost_tuned_pipeline_metadata.pkl  <- dict with feature lists & metrics

Run with:
    streamlit run app.py
"""

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Philadelphia Property Valuation",
    page_icon="🏠",
    layout="wide",
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "xgboost_tuned_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "xgboost_tuned_pipeline_metadata.pkl"


# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: the training pipeline's ColumnTransformer uses a custom
# FunctionTransformer(to_string_array) inside its categorical sub-pipelines
# (see notebook Section 4.2, cell defining `cat_ohe_pipe` / `cat_ord_pipe`).
# joblib/pickle stores a *reference* to this function (its module + name),
# not its code. To unpickle the pipeline successfully in a standalone script,
# a function with this exact name must exist in this module before loading.
# If you retrain and the function's implementation ever changes, update it
# here too so it matches your notebook exactly.
# ──────────────────────────────────────────────────────────────────────────
def to_string_array(x):
    """Cast a column (possibly mixed int/str) to a uniform string dtype
    before one-hot/ordinal encoding. Must match the notebook's definition."""
    return x.astype(str)


# Constant used at training time for building_age = CURRENT_YEAR - year_built.
# The live app instead uses the real current year so the tool stays correct
# over time (see README note on this choice).
CURRENT_YEAR = dt.date.today().year

# Standard OPA category_code mapping (per the notebook's own cleaning logic,
# where category_code 5 = Industrial and 6 = Vacant Land are referenced
# explicitly). Codes 1-4 follow the same official OPA ordering.
CATEGORY_CODE_MAP = {
    "Single Family": 1,
    "Multi Family": 2,
    "Mixed Use": 3,
    "Commercial": 4,
    "Industrial": 5,
    "Vacant Land": 6,
}

# Building era bins — copied 1:1 from the notebook's feature engineering
# (Section 2.5) so the derived category matches training exactly.
ERA_BINS = [0, 1899, 1949, 1979, 1999, 2100]
ERA_LABELS = [
    "Historic (<1900)",
    "Early 20th (1900-49)",
    "Mid-century (1950-79)",
    "Modern (1980-99)",
    "Contemporary (2000+)",
]

CONDITION_OPTIONS = {
    "1 - New / Rehabbed": 1,
    "2 - Above Average": 2,
    "3 - Average": 3,
    "4 - Below Average": 4,
    "5 - Fair": 5,
    "6 - Poor": 6,
    "7 - Vacant / Sealed": 7,
}


# ──────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_metadata():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


pipeline, metadata = load_model_and_metadata()

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.title("🏠 Philadelphia Property Valuation")
st.caption(
    "Machine Learning-based mass appraisal tool for the City of Philadelphia's "
    "Office of Property Assessment (OPA) — powered by a tuned XGBoost regression model."
)

if pipeline is None:
    st.error(
        "Model files not found. Please place your trained pipeline files inside "
        "a `models/` folder next to `app.py`:\n\n"
        "- `models/xgboost_tuned_pipeline.pkl`\n"
        "- `models/xgboost_tuned_pipeline_metadata.pkl`\n\n"
        "These are produced by Section 4.6.1 of your training notebook "
        "(`joblib.dump(final_pipeline, ...)`)."
    )
    st.stop()

with st.expander("ℹ️ About this model", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", metadata.get("model_name", "XGBoost"))
    c2.metric("Test R²", f"{metadata.get('test_r2', 0):.3f}")
    c3.metric("Test MAPE", f"{metadata.get('test_mape_pct', 0):.1f}%")
    st.markdown(
        "This model predicts `log(1 + market_value)`; predictions are converted "
        "back to USD automatically. **Limitation:** accuracy is lower for luxury "
        "(> $1M) and distressed (< $50K) properties — treat those estimates as "
        "directional, not final assessments."
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────────────────────────────────
st.subheader("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📍 Location & Type**")
    category_label = st.selectbox("Property category", list(CATEGORY_CODE_MAP.keys()), index=0)
    geographic_ward = st.number_input(
        "Geographic ward", min_value=1, max_value=66, value=1, step=1,
        help="Philadelphia ward number (1-66)."
    )
    zoning = st.text_input(
        "Zoning code (optional)", value="",
        help="e.g. RSA5, CMX2, RM1, I1. Leave blank if unknown — "
             "the model treats unknown zoning as a neutral category."
    )

with col2:
    st.markdown("**📐 Size**")
    total_livable_area = st.number_input(
        "Total livable area (sq ft)", min_value=1, value=1200, step=50
    )
    total_area = st.number_input(
        "Total lot area (sq ft)", min_value=1, value=1500, step=50
    )
    frontage = st.number_input("Frontage (ft)", min_value=0.0, value=16.0, step=1.0)
    depth = st.number_input("Depth (ft)", min_value=0.0, value=90.0, step=1.0)

with col3:
    st.markdown("**🛏️ Rooms**")
    number_of_bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    number_of_bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=12.0, value=1.5, step=0.5)
    number_of_rooms = st.number_input("Total rooms", min_value=0, max_value=40, value=6, step=1)
    number_stories = st.number_input("Stories", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

st.divider()
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**🏗️ Construction & Condition**")
    year_built = st.number_input(
        "Year built", min_value=1650, max_value=CURRENT_YEAR, value=1925, step=1
    )
    exterior_condition = st.selectbox(
        "Exterior condition", list(CONDITION_OPTIONS.keys()), index=2
    )
    interior_condition = st.selectbox(
        "Interior condition", list(CONDITION_OPTIONS.keys()), index=2
    )
    general_construction = st.text_input(
        "General construction code (optional)", value="",
        help="Raw OPA code, e.g. A (Masonry), B (Masonry/Frame), C (Frame). "
             "Leave blank if unknown."
    )

with col5:
    st.markdown("**🔥 Amenities**")
    central_air = st.radio("Central air?", ["Yes", "No"], horizontal=True, index=1)
    garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=20, value=0, step=1)
    fireplaces = st.number_input("Fireplaces", min_value=0, max_value=10, value=0, step=1)
    has_basement_choice = st.radio("Has basement?", ["Yes", "No", "Unknown"], horizontal=True, index=0)
    off_street_open = st.number_input(
        "Off-street open spaces", min_value=0.0, value=0.0, step=1.0
    )

with col6:
    st.markdown("**💵 Last Sale (optional)**")
    sale_known = st.checkbox("Property has a recorded sale", value=False)
    if sale_known:
        sale_year = st.number_input(
            "Sale year", min_value=1900, max_value=CURRENT_YEAR, value=CURRENT_YEAR - 1, step=1
        )
        sale_month = st.number_input("Sale month", min_value=1, max_value=12, value=1, step=1)
    else:
        sale_year, sale_month = 0, 0
        st.caption("No sale on record — treated the same way as in training data.")

with st.expander("🔧 Advanced: raw OPA codes (optional, defaults to 'Unknown')"):
    st.caption(
        "These fields map directly to raw City of Philadelphia OPA category codes. "
        "If you don't know a property's exact code, leave it as 'Unknown' — the "
        "pipeline was trained to handle unknown/missing values for these fields "
        "gracefully (they simply contribute no signal, rather than causing an error)."
    )
    a1, a2, a3, a4 = st.columns(4)
    basements = a1.text_input("Basements code", value="Unknown")
    garage_type = a2.text_input("Garage type code", value="Unknown")
    topography = a3.text_input("Topography code", value="Unknown")
    view_type = a4.text_input("View type code", value="Unknown")
    type_heater = a1.text_input("Heater type code", value="Unknown")
    parcel_shape = a2.text_input("Parcel shape code", value="Unknown")

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict Market Value", type="primary", use_container_width=True)

if predict_clicked:
    # ---- Derived / engineered features (must mirror the training notebook) ----
    category_code = CATEGORY_CODE_MAP[category_label]

    log_total_livable_area = np.log1p(max(total_livable_area, 0))
    log_total_area = np.log1p(max(total_area, 0))

    livable_area_ratio = (total_livable_area / total_area) if total_area > 0 else np.nan
    livable_area_ratio = min(livable_area_ratio, 5) if livable_area_ratio is not None else np.nan

    bath_bed_ratio = number_of_bathrooms / (number_of_bedrooms if number_of_bedrooms != 0 else 1)
    bath_bed_ratio = float(np.clip(bath_bed_ratio, 0, 10))

    building_age = CURRENT_YEAR - year_built
    building_age = max(building_age, 0)

    building_era = pd.cut(
        [year_built], bins=ERA_BINS, labels=ERA_LABELS, right=True
    )[0]
    building_era = "Unknown" if pd.isna(building_era) else str(building_era)

    has_central_air = 1 if central_air == "Yes" else 0
    has_garage = 1 if garage_spaces > 0 else 0
    has_fireplace = 1 if fireplaces > 0 else 0
    has_basement = 1 if has_basement_choice == "Yes" else 0

    exterior_condition_val = CONDITION_OPTIONS[exterior_condition]
    interior_condition_val = CONDITION_OPTIONS[interior_condition]

    row = {
        # Numeric features
        "log_total_livable_area": log_total_livable_area,
        "log_total_area": log_total_area,
        "frontage": frontage,
        "depth": depth,
        "livable_area_ratio": livable_area_ratio,
        "number_of_bathrooms": number_of_bathrooms,
        "number_of_bedrooms": number_of_bedrooms,
        "number_of_rooms": number_of_rooms,
        "number_stories": number_stories,
        "bath_bed_ratio": bath_bed_ratio,
        "fireplaces": fireplaces,
        "garage_spaces": garage_spaces,
        "off_street_open": off_street_open,
        "building_age": building_age,
        "exterior_condition": exterior_condition_val,
        "interior_condition": interior_condition_val,
        "geographic_ward": geographic_ward,
        "has_central_air": has_central_air,
        "has_garage": has_garage,
        "has_fireplace": has_fireplace,
        "has_basement": has_basement,
        "sale_year": sale_year,
        "sale_month": sale_month,
        # OHE categorical features
        "basements": basements,
        "garage_type": garage_type,
        "general_construction": general_construction or "Unknown",
        "topography": topography,
        "view_type": view_type,
        "type_heater": type_heater,
        "parcel_shape": parcel_shape,
        "building_era": building_era,
        # Ordinal categorical features
        "zoning": zoning or "Unknown",
        "category_code": category_code,
    }

    all_features = metadata.get("all_features")
    X_input = pd.DataFrame([row])

    if all_features:
        missing = [f for f in all_features if f not in X_input.columns]
        if missing:
            st.error(f"Internal error: missing engineered features {missing}")
            st.stop()
        X_input = X_input[all_features]

    try:
        log_pred = pipeline.predict(X_input)[0]
        pred_usd = float(np.expm1(log_pred))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Prediction complete")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Market Value", f"${pred_usd:,.0f}")

    mape = metadata.get("test_mape_pct")
    if mape:
        lo, hi = pred_usd * (1 - mape / 100), pred_usd * (1 + mape / 100)
        m2.metric("Approx. range (± typical MAPE)", f"${lo:,.0f} – ${hi:,.0f}")
    m3.metric("Price per livable sq ft", f"${pred_usd / total_livable_area:,.0f}")

    if pred_usd > 1_000_000 or pred_usd < 50_000:
        st.warning(
            "⚠️ This estimate falls in the luxury (>$1M) or distressed (<$50K) "
            "value tier, where the model's historical error rate is notably "
            "higher (see Section 5.4 — Limitations in the notebook). Treat this "
            "prediction as directional only."
        )

    with st.expander("See model input (debug)"):
        debug_df = X_input.T.rename(columns={0: "value"}).astype(str)
        st.dataframe(debug_df)

st.divider()
st.caption(
    "Built on a tuned XGBoost pipeline trained on Philadelphia OPA property "
    "assessment data. For informational purposes only — not an official "
    "property assessment."
)
