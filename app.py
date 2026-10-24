"""Streamlit dashboard: Employee Attrition Prediction (educational prototype)."""
from __future__ import annotations
import io
import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from preprocess import TARGET, clean_dataset, validate_dataset
from train_model import train_and_evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "data", "employee_attrition.csv")
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")
st.title("Employee Attrition Prediction System")
st.caption("Educational HR analytics prototype. Do not use predictions to make employment decisions.")
st.info("Predictions show patterns learned from historical data; they are not proof that an individual employee will leave.")

@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv(DEFAULT_DATA)

with st.sidebar:
    st.header("Data input")
    uploaded = st.file_uploader("Upload a compatible HR CSV", type=["csv"])
    use_demo = st.checkbox("Use bundled demo dataset", value=True)
    st.caption("CSV must include an Attrition column containing Yes/No labels.")

try:
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    elif use_demo:
        df_raw = load_default()
        source_name = "Bundled IBM HR demo dataset"
    else:
        st.warning("Upload a CSV or enable the bundled demo dataset.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

errors = validate_dataset(df_raw)
if errors:
    for error in errors:
        st.error(error)
    st.stop()

df = clean_dataset(df_raw)
if df.empty or TARGET not in df.columns:
    st.error("No usable data remains after validation.")
    st.stop()

# Train in-memory for uploaded data; use cached model on bundled data if available.
try:
    with st.spinner("Training and evaluating models..."):
        results, bundle, train_n, test_n = train_and_evaluate(df, output_dir=MODEL_DIR)
except Exception as exc:
    st.error(f"Training could not be completed: {exc}")
    st.stop()

metrics_df = pd.DataFrame([{k: r[k] for k in ["Model", "Accuracy", "Precision", "Recall", "F1-score"]} for r in results])
tab_overview, tab_explore, tab_eval, tab_predict, tab_data = st.tabs(
    ["Overview", "Explore", "Model evaluation", "Demo prediction", "Data quality"]
)
with tab_overview:
    st.subheader("Dataset overview")
    yes_count = int((df[TARGET] == "Yes").sum())
    no_count = int((df[TARGET] == "No").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees in data", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Attrition: Yes", f"{yes_count:,}")
    c4.metric("Attrition: No", f"{no_count:,}")
    st.write(f"**Data source:** {source_name}")
    st.write(f"**Training rows:** {train_n:,} · **Test rows:** {test_n:,} · **Selected model:** {bundle['best_model']}")
    st.subheader("Sample records")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Attrition distribution")
    counts = df[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Employees")
    st.pyplot(fig)
    plt.close(fig)

with tab_explore:
    st.subheader("Explore employee data")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if cat_cols:
        col = st.selectbox("Categorical column", cat_cols, index=cat_cols.index(TARGET) if TARGET in cat_cols else 0)
        vc = df[col].astype(str).value_counts().head(20)
        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close(fig)
    if num_cols:
        coln = st.selectbox("Numeric column", num_cols, index=num_cols.index("MonthlyIncome") if "MonthlyIncome" in num_cols else 0)
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=coln, hue=TARGET if TARGET in df.columns else None, kde=False, ax=ax)
        ax.set_title(f"{coln} distribution")
        st.pyplot(fig)
        plt.close(fig)
    if "Department" in df.columns:
        st.subheader("Department and attrition")
        dept = pd.crosstab(df["Department"], df[TARGET], normalize="index").mul(100).round(1)
        st.dataframe(dept, use_container_width=True)

with tab_eval:
    st.subheader("Held-out test-set evaluation")
    st.caption("Models are evaluated on a held-out test split. Results may vary with different datasets.")
    st.dataframe(metrics_df.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-score"]}),
                 use_container_width=True)
    selected_name = st.selectbox("Confusion matrix model", [r["Model"] for r in results])
    selected = next(r for r in results if r["Model"] == selected_name)
    cm = selected["Confusion Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"],
                yticklabels=["Actual No", "Actual Yes"], ax=ax)
    ax.set_title(f"Confusion matrix — {selected_name}")
    st.pyplot(fig)
    plt.close(fig)
    st.write("**Selected model:**", bundle["best_model"])
    st.download_button("Download evaluation results (JSON)",
                       data=json.dumps({"results": results, "best_model": bundle["best_model"],
                                        "train_rows": train_n, "test_rows": test_n}, indent=2),
                       file_name="evaluation_results.json", mime="application/json")

with tab_predict:
    st.subheader("Demo prediction")
    st.warning("Use only for learning and aggregate research—not for decisions about real employees.")
    features = bundle["features"]
    input_values = {}
    numeric_cols = df[features].select_dtypes(include=["number", "bool"]).columns.tolist()
    for col in features:
        series = df[col]
        if col in numeric_cols:
            val = float(series.median()) if series.notna().any() else 0.0
            if pd.api.types.is_integer_dtype(series.dtype):
                input_values[col] = int(st.number_input(col, value=int(val)))
            else:
                input_values[col] = st.number_input(col, value=val)
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            input_values[col] = st.selectbox(col, options=options) if options else ""
    if st.button("Predict demo outcome", type="primary"):
        try:
            sample = pd.DataFrame([input_values], columns=features)
            prediction = bundle["pipeline"].predict(sample)[0]
            if hasattr(bundle["pipeline"], "predict_proba"):
                probability = float(bundle["pipeline"].predict_proba(sample)[0][1])
            else:
                probability = None
            st.success(f"Model output: {'Attrition = Yes' if int(prediction) == 1 else 'Attrition = No'}")
            if probability is not None:
                st.metric("Estimated model score for Attrition = Yes", f"{probability:.1%}")
            st.caption("This is a model score, not a reliable personal forecast or a causal explanation.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_data:
    st.subheader("Data quality checks")
    missing = df_raw.isna().sum().sort_values(ascending=False)
    q1, q2, q3 = st.columns(3)
    q1.metric("Original rows", f"{len(df_raw):,}")
    q2.metric("Duplicate rows", f"{int(df_raw.duplicated().sum()):,}")
    q3.metric("Missing cells", f"{int(df_raw.isna().sum().sum()):,}")
    st.write("Missing values by column")
    st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
    st.download_button("Download cleaned dataset", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_employee_attrition.csv", mime="text/csv")

st.divider()
st.caption("Dataset is a benchmark sample. Verify licensing and suitability before external redistribution. No employment decision should rely on this prototype.")
