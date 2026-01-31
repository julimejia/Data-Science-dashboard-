import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Advanced EDA", layout="wide")
st.title("🚀 EDA Interactivo Pro")

# --- CARGA DE DATOS ---
st.sidebar.header("1. Configuración de Datos")
uploaded = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded is None:
    st.info("👋 Sube un CSV para comenzar.")
    st.stop()

enc = st.sidebar.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"])

try:
    df = pd.read_csv(uploaded, encoding=enc)
    
    # 🔥 SOLUCIÓN AL ERROR: Renombrar columnas duplicadas
    if df.columns.duplicated().any():
        st.warning("⚠️ Se detectaron nombres de columnas duplicados. Se han renombrado automáticamente.")
        cols = pd.Series(df.columns)
        for i, col in enumerate(df.columns):
            if cols.duplicated()[i]:
                cols[i] = f"{col}_{i}"
        df.columns = cols

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- PREPARACIÓN DE COLUMNAS ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- TABS PRINCIPALES ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Vista Previa", "📊 Estadísticas", "📈 Análisis Cuantitativo", "🎭 Análisis Cualitativo"])

with tab1:
    st.subheader("Exploración de Datos Crudos")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Resumen General")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Numéricas", len(numeric_cols))
    col4.metric("Categóricas", len(cat_cols))

    st.markdown("---")
    st.write("### Matriz de Correlación (Heatmap)")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Se necesitan al menos 2 columnas numéricas para correlación.")

with tab3:
    st.subheader("📈 Análisis Cuantitativo")
    if numeric_cols:
        col_x = st.selectbox("Eje X (Principal)", numeric_cols)
        col_y = st.selectbox("Eje Y (Para Relaciones)", ["Ninguno"] + numeric_cols)
        
        c1, c2 = st.columns(2)
        with c1:
            # Gráfico de Densidad con Histograma
            fig_dist = px.histogram(df, x=col_x, marginal="rug", title=f"Distribución de {col_x}",
                                  color_discrete_sequence=['#636EFA'], opacity=0.7)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with c2:
            # Gráfico de Violín para ver la dispersión
            fig_viol = px.violin(df, y=col_x, box=True, points="all", title=f"Rango y Outliers de {col_x}")
            st.plotly_chart(fig_viol, use_container_width=True)

        if col_y != "Ninguno":
            st.write(f"**Dispersión: {col_x} vs {col_y}**")
            # Usamos scatter pero con densidad de color (Density Contour)
            fig_scat = px.scatter(df, x=col_x, y=col_y, opacity=0.5, 
                                 trendline="ols" if 'statsmodels' in globals() else None,
                                 render_mode='webgl') # WebGL es más rápido para muchos datos
            st.plotly_chart(fig_scat, use_container_width=True)

with tab4:
    st.subheader("🎭 Análisis Cualitativo")
    if cat_cols:
        cat_sel = st.selectbox("Variable Categórica", cat_cols)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Gráfico de barras horizontal (más legible)
            top_n = df[cat_sel].value_counts().nlargest(15).reset_index()
            fig_bar = px.bar(top_n, x='count', y=cat_sel, orientation='h',
                            title=f"Top 15 {cat_sel}", color='count',
                            color_continuous_scale='Blues')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_right:
            # Gráfico Sunburst (Proporciones)
            fig_sun = px.sunburst(df, path=[cat_sel], title=f"Composición de {cat_sel}")
            st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info("No hay variables cualitativas.")
