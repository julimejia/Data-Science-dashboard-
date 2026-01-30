import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple EDA", layout="wide")
st.title("📊 Simple EDA on a CSV (Streamlit)")

# -----------------------
# Upload CSV from user's PC
# -----------------------
st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload a CSV from your PC", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file using the sidebar to start.")
    st.stop()

# Optional: encoding handling (common for Latin CSVs)
enc = st.sidebar.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)

try:
    df = pd.read_csv(uploaded, encoding=enc)
except Exception as e:
    st.error(f"Could not read CSV. Try another encoding. Error: {e}")
    st.stop()

st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if cat_cols:
    cat_filter_col = st.sidebar.selectbox(
        "Filter by a categorical column (optional)",
        ["(none)"] + cat_cols
    )
    if cat_filter_col != "(none)":
        options = df[cat_filter_col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(
            f"Values for {cat_filter_col}",
            options,
            default=options[: min(5, len(options))]
        )
        if selected:
            df = df[df[cat_filter_col].isin(selected)]

# -----------------------
# Main tabs
# -----------------------
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
        st.dataframe(df[col].value_counts(dropna=False).head(10).to_frame("count"),
                     use_container_width=True)

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

        st.markdown("---")
        st.write("*Correlation heatmap (numeric columns)*")

        # Recompute numeric_cols after filtering (just in case)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            im = ax2.imshow(corr.values)
            ax2.set_xticks(range(len(corr.columns)))
            ax2.set_yticks(range(len(corr.columns)))
            ax2.set_xticklabels(corr.columns, rotation=90)
            ax2.set_yticklabels(corr.columns)
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")
    else:
        st.info("No numeric columns detected, so plots are limited.")
