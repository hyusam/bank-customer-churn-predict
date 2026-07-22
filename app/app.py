import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Bank Churn Predictor", page_icon="", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "final_pipeline_rftuned.pkl"
REQUIRED_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# Metrik hasil evaluasi model pada test set (lihat 02_preprocessing_modeling.ipynb)
MODEL_METRICS = {
    "Accuracy": 0.8425,
    "Precision": 0.6036,
    "Recall": 0.6585,
    "F1 Score": 0.6298,
    "ROC-AUC": 0.8549,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_shap_explainer(_fitted_model):
    return shap.TreeExplainer(_fitted_model)


model = load_model()
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_model = model.named_steps["model"]
explainer = get_shap_explainer(fitted_model)

st.title("Bank Customer Churn Predictor")
st.caption("Random Forest (tuned), prediksi probabilitas nasabah akan churn")

# ============================================================
# METRIK MODEL
# ============================================================
with st.expander("Performa Model (Test Set)", expanded=False):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.1%}")
    m2.metric("Precision", f"{MODEL_METRICS['Precision']:.1%}")
    m3.metric("Recall", f"{MODEL_METRICS['Recall']:.1%}")
    m4.metric("F1 Score", f"{MODEL_METRICS['F1 Score']:.1%}")
    m5.metric("ROC-AUC", f"{MODEL_METRICS['ROC-AUC']:.1%}")
    st.caption(
        "Recall lebih tinggi dari precision, model cenderung lebih agresif menandai nasabah "
        "berisiko churn, cocok untuk strategi retensi (lebih baik salah waspada daripada lolos)."
    )

# ============================================================
# SIDEBAR - THRESHOLD
# ============================================================
st.sidebar.header("⚙️ Pengaturan")

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

def reset_threshold():
    st.session_state.threshold = 0.50

threshold = st.sidebar.slider(
    "Threshold klasifikasi churn",
    min_value=0.0, max_value=1.0, step=0.01,
    key="threshold",
    help="Probabilitas ≥ threshold ini akan diberi label 'Churn'. "
         "Turunkan threshold untuk menaikkan recall (tangkap lebih banyak nasabah berisiko), "
         "naikkan untuk menaikkan precision (kurangi alarm palsu)."
)
st.sidebar.button("↺ Reset ke 0.50", on_click=reset_threshold)
st.sidebar.caption(
    f"""
    **Threshold saat ini: `{threshold:.2f}`**
    - Threshold rendah → recall ↑, precision ↓ (lebih agresif menandai churn)
    - Threshold tinggi → precision ↑, recall ↓ (lebih selektif)
    """
)

tab1, tab2 = st.tabs(["Prediksi Manual", "Prediksi Batch (CSV)"])

# ============================================================
# TAB 1: MANUAL INPUT
# ============================================================
with tab1:
    st.subheader("Input Data Nasabah")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (tahun)", min_value=0, max_value=20, value=3)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)

    with col3:
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Ya", "Tidak"])
        is_active = st.selectbox("Active Member?", ["Ya", "Tidak"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=80000.0, step=1000.0)

    if st.button("Prediksi", type="primary"):
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Ya" else 0,
            "IsActiveMember": 1 if is_active == "Ya" else 0,
            "EstimatedSalary": estimated_salary,
        }])

        proba = model.predict_proba(input_df)[:, 1][0]
        label = "Churn" if proba >= threshold else "Tidak Churn"

        st.session_state["last_input_df"] = input_df
        st.session_state["last_proba"] = proba
        st.session_state["last_label"] = label

    # Tampilkan hasil terakhir (persist walau expander SHAP dibuka/tutup)
    if "last_proba" in st.session_state:
        proba = st.session_state["last_proba"]
        label = st.session_state["last_label"]
        input_df = st.session_state["last_input_df"]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if label == "Churn":
                st.error(f"### ⚠️ {label}")
            else:
                st.success(f"### ✅ {label}")
            st.metric("Probabilitas Churn", f"{proba:.1%}")

        with res_col2:
            st.progress(min(float(proba), 1.0))
            st.caption(f"Probabilitas {proba:.1%} dibandingkan threshold {threshold:.1%}")

        # ---------------- SHAP explainability ----------------
        st.divider()
        st.subheader("🔎 Kenapa model memprediksi ini?")

        X_transformed = fitted_preprocessor.transform(input_df)
        feature_names = fitted_preprocessor.get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        shap_values_raw = explainer.shap_values(X_transformed_df)

        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1][0]
            base_value = explainer.expected_value[1]
        elif np.ndim(shap_values_raw) == 3:
            shap_values = shap_values_raw[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_raw[0]
            base_value = explainer.expected_value

        fig, ax = plt.subplots(figsize=(5, 3.5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=X_transformed_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plot_col, _ = st.columns([2, 1])
        with plot_col:
            st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption(
            "Batang merah mendorong prediksi ke arah **Churn**, batang biru mendorong ke arah "
            "**Tidak Churn**. Semakin panjang batang, semakin besar pengaruh fitur tersebut "
            "pada prediksi nasabah ini."
        )

# ============================================================
# TAB 2: BATCH CSV
# ============================================================
with tab2:
    st.subheader("Upload CSV untuk Prediksi Batch")
    st.caption(f"Kolom yang dibutuhkan: `{', '.join(REQUIRED_COLS)}`")

    template_df = pd.DataFrame([{
        "CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 35,
        "Tenure": 3, "Balance": 50000.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 80000.0
    }])
    st.download_button(
        "⬇️ Download Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_prediksi_churn.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        missing_cols = [c for c in REQUIRED_COLS if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
        else:
            X_batch = batch_df[REQUIRED_COLS].copy()

            probas = model.predict_proba(X_batch)[:, 1]
            labels = ["Churn" if p >= threshold else "Tidak Churn" for p in probas]

            result_df = batch_df.copy()
            result_df["Churn_Probability"] = probas.round(4)
            result_df["Prediction"] = labels

            st.success(f"Prediksi selesai untuk {len(result_df)} baris.")

            c1, c2 = st.columns(2)
            c1.metric("Total Diprediksi Churn", int((result_df["Prediction"] == "Churn").sum()))
            c2.metric("Total Diprediksi Tidak Churn", int((result_df["Prediction"] == "Tidak Churn").sum()))

            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                data=csv_out,
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv"
            )