import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuración de página con tema profesional
st.set_page_config(page_title="Enterprise Analytics", layout="wide")

# --- ESTILOS PARA EVITAR PANTALLA EN BLANCO ---
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 4px 4px 0px 0px; padding: 10px 20px;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Intelligence Dashboard Pro")

# --- CARGA Y LIMPIEZA DE DATOS ---
with st.sidebar:
    st.header("📂 Configuración")
    uploaded = st.file_uploader("Subir CSV", type=["csv"])
    if not uploaded:
        st.info("Sube un archivo para comenzar")
        st.stop()
    
    # Manejo de Encoding y Carga
    enc = st.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"])
    df = pd.read_csv(uploaded, encoding=enc)
    
    # Limpieza de duplicados y nulos básicos
    df.columns = [str(c).replace(' ', '_') for c in df.columns]
    if df.columns.duplicated().any():
        df.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(df.columns, df.columns.duplicated()))]
    
    # Muestreo preventivo para evitar el bug visual de memoria
    max_rows = st.slider("Filas para análisis rápido", 100, len(df), min(2000, len(df)))
    df_sample = df.head(max_rows).copy()

# Clasificación de columnas
num_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_sample.select_dtypes(exclude=[np.number]).columns.tolist()

# --- TABS: ORGANIZACIÓN POR OBJETIVOS ---
t_preview, t_quant, t_qual, t_expert = st.tabs([
    "📋 Vista Previa", 
    "📈 Cuantitativo (Números)", 
    "🎭 Cualitativo (Categorías)", 
    "🛠️ Explorador Libre"
])

# 1. VISTA PREVIA
with t_preview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filas", len(df))
    col2.metric("Columnas", len(df.columns))
    col3.metric("Nulos", df.isna().sum().sum())
    
    st.dataframe(df_sample.head(50), use_container_width=True)

# 2. ANÁLISIS CUANTITATIVO (Fijado y estable)
with t_quant:
    st.subheader("Análisis de Variables Numéricas")
    if num_cols:
        target_num = st.selectbox("Selecciona métrica principal", num_cols)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Histograma con densidad
            fig_dist = px.histogram(df_sample, x=target_num, marginal="box", 
                                   title=f"Distribución de {target_num}",
                                   color_discrete_sequence=['#0083B8'])
            st.plotly_chart(fig_dist, use_container_width=True)
            
        
        with c2:
            st.write("**Estadísticas Clave**")
            stats = df_sample[target_num].describe()
            st.write(stats)
            
            # Scatter rápido contra índice
            fig_index = px.scatter(df_sample, y=target_num, opacity=0.4, title="Dispersión por Índice")
            st.plotly_chart(fig_index, use_container_width=True)
    else:
        st.warning("No se detectaron columnas numéricas.")

# 3. ANÁLISIS CUALITATIVO (Fijado y estable)
with t_qual:
    st.subheader("Análisis de Categorías")
    if cat_cols:
        target_cat = st.selectbox("Selecciona categoría", cat_cols)
        
        c1, c2 = st.columns(2)
        with c1:
            # Conteo de frecuencia
            top_data = df_sample[target_cat].value_counts().nlargest(10).reset_index()
            fig_bar = px.bar(top_data, x=target_cat, y='count', 
                             title=f"Top 10: {target_cat}",
                             color='count', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with c2:
            # Composición porcentual
            fig_pie = px.pie(top_data, names=target_cat, values='count', hole=0.4,
                             title=f"Proporción de {target_cat}")
            st.plotly_chart(fig_pie, use_container_width=True)
            
    else:
        st.warning("No se detectaron columnas categóricas.")

# 4. EXPLORADOR LIBRE (Simplificado para evitar bugs)
with t_expert:
    st.subheader("🛠️ Constructor de Reportes a Medida")
    st.caption("Usa esta sección para cruzar variables libremente.")
    
    with st.expander("Configurar Gráfico Automático", expanded=True):
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            x_ax = st.selectbox("Eje X", df_sample.columns)
            y_ax = st.selectbox("Eje Y (Numérico)", num_cols)
        with col_ex2:
            tipo = st.radio("Formato", ["Barras", "Líneas", "Puntos"], horizontal=True)
        with col_ex3:
            dividir = st.selectbox("Agrupar por Color", ["Ninguno"] + cat_cols)

    # Lógica simplificada
    color_param = dividir if dividir != "Ninguno" else None
    
    if tipo == "Barras":
        fig_custom = px.bar(df_sample, x=x_ax, y=y_ax, color=color_param, barmode="group")
    elif tipo == "Líneas":
        fig_custom = px.line(df_sample, x=x_ax, y=y_ax, color=color_param)
    else:
        fig_custom = px.scatter(df_sample, x=x_ax, y=y_ax, color=color_param, opacity=0.6)

    st.plotly_chart(fig_custom, use_container_width=True)
    

st.markdown("---")
st.caption("Reporte generado dinámicamente | Enterprise Suite 2026")
