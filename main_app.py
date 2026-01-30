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
    st.info("👋 Por favor, sube un archivo CSV en la barra lateral para comenzar.")
    st.stop()

enc = st.sidebar.selectbox("Codificación (Encoding)", ["utf-8", "latin-1", "cp1252"])

try:
    df = pd.read_csv(uploaded, encoding=enc)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
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
    st.subheader("📈 Análisis Cuantitativo Avanzado")
    
    if numeric_cols:
        col_x = st.selectbox("Variable X", numeric_cols, key="nx")
        col_y = st.selectbox("Variable Y (Comparativa)", ["Ninguna"] + numeric_cols, key="ny")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Histograma interactivo con selectores de densidad
            st.write(f"**Distribución y Outliers de {col_x}**")
            fig_hist = px.histogram(df, x=col_x, marginal="violin", # Cambiado a violín para más detalle
                                   color_discrete_sequence=['#00CC96'],
                                   nbins=50)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.write(f"**Análisis de Acumulación (ECDF)**")
            # El ECDF ayuda a ver qué porcentaje de datos está por debajo de cierto valor
            fig_ecdf = px.ecdf(df, x=col_x)
            st.plotly_chart(fig_ecdf, use_container_width=True)

        if col_y != "Ninguna":
            st.write(f"**Relación entre {col_x} y {col_y}**")
            try:
                # Intentamos graficar con línea de tendencia
                fig_scatter = px.scatter(df, x=col_x, y=col_y, 
                                       trendline="ols", 
                                       hover_data=cat_cols[:2],
                                       opacity=0.6)
            except ImportError:
                # Si falla statsmodels, graficamos sin tendencia
                st.warning("⚠️ Instala 'statsmodels' para ver la línea de tendencia. Mostrando solo puntos.")
                fig_scatter = px.scatter(df, x=col_x, y=col_y, 
                                       hover_data=cat_cols[:2],
                                       opacity=0.6)
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No hay columnas numéricas para este análisis.")

with tab2:
    # Mejora visual de la matriz de correlación
    st.write("### 🔥 Mapa de Calor de Relaciones")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        # Usamos un mapa de calor con escala de colores divergente (RdBu)
        fig_corr = px.imshow(corr, 
                            text_auto=".2f", # Muestra solo 2 decimales
                            color_continuous_scale='RdBu_r', 
                            zmin=-1, zmax=1) # Forzamos escala de -1 a 1
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader("Análisis de Variables Categóricas")
    
    if cat_cols:
        cat_to_plot = st.selectbox("Selecciona una categoría para analizar", cat_cols)
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            # Tabla de frecuencias
            counts = df[cat_to_plot].value_counts().reset_index()
            counts.columns = [cat_to_plot, 'conteo']
            st.write(f"**Frecuencias de {cat_to_plot}**")
            st.dataframe(counts, use_container_width=True)
            
        with c2:
            # Gráfico de Treemap (mejor que un Pie chart para muchas categorías)
            st.write(f"**Jerarquía / Proporción de {cat_to_plot}**")
            fig_tree = px.treemap(counts, path=[cat_to_plot], values='conteo', color='conteo',
                                  color_continuous_scale='Viridis')
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("---")
        # Comparativa cualitativa vs cuantitativa
        if numeric_cols:
            st.write(f"**Distribución de {numeric_cols[0]} por {cat_to_plot}**")
            fig_violin = px.violin(df, y=numeric_cols[0], x=cat_to_plot, color=cat_to_plot, box=True, points="all")
            st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("No hay columnas categóricas.")
