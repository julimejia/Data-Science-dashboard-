
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple EDA", layout="wide")

st.title("📊 Simple EDA on a CSV (Streamlit)")

# --- File loading (local or upload) ---
st.sidebar.header("Load data")

use_uploader = st.sidebar.checkbox("Upload file instead of local path", value=False)

if use_uploader:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to start.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    # Change this to your file name if needed for Streamlit Cloud repo
    csv_path = st.sidebar.text_input("Local CSV path", value="agro_colombia.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")

# --- Sidebar filters (basic) ---
st.sidebar.header("Filters")

# Choose columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if cat_cols:
    cat_filter_col = st.sidebar.selectbox("Filter by a categorical column (optional)", ["(none)"] + cat_cols)
    if cat_filter_col != "(none)":
        options = df[cat_filter_col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Values for {cat_filter_col}", options, default=options[: min(5, len(options))])
        if selected:
            df = df[df[cat_filter_col].isin(selected)]

# --- Main view ---
tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Plots"])

with tab1:
    st.subheader("Data preview")
    st.dataframe(df.head(50), use_container_width=True)

with tab2:
    st.subheader("Basic summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.write("*Missing values per column*")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing.to_frame("missing_count"), use_container_width=True)

    st.write("*Describe (numeric)*")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    st.write("*Top categories (first 5 categorical columns)*")
    for col in cat_cols[:5]:
        st.write(f"*{col}*")
        st.dataframe(df[col].value_counts(dropna=False).head(10).to_frame("count"), use_container_width=True)

with tab3:
    st.subheader("Quick plots")

    if numeric_cols:
        col = st.selectbox("Choose a numeric column for histogram", numeric_cols)
        bins = st.slider("Bins", 10, 80, 30)

        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.markd
