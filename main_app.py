import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Enterprise Analytics Hub", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Enterprise Analytics Hub")
st.markdown("Generación de reportes dinámicos y análisis de datos avanzado.")

# --- CARGA DE DATOS ---
with st.sidebar:
    st.header("📂 Gestión de Archivos")
    uploaded = st.file_uploader("Subir Dataset (CSV)", type=["csv"])
    
    if uploaded:
        enc = st.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"])
        df = pd.read_csv(uploaded, encoding=enc)
        
        # Limpieza automática de nombres
        df.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(df.columns, df.columns.duplicated()))]
        
        st.success("✅ Archivo cargado")
        
        st.header("🛠️ Pre-procesamiento")
        if st.checkbox("Eliminar filas con Nulos"):
            df = df.dropna()
            st.caption(f"Filas restantes: {len(df)}")
            
        # CORRECCIÓN DEL ERROR: min() asegura que el default no exceda el máximo
        max_rows = st.number_input("Límite de filas para análisis", 1, len(df), min(5000, len(df)))
        df_view = df.head(max_rows).copy()

if not uploaded:
    st.info("💡 Por favor, cargue un archivo CSV para desbloquear el panel de control.")
    st.stop()

# --- VARIABLES DISPONIBLES ---
num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_view.select_dtypes(exclude=[np.number]).columns.tolist()
all_cols = df_view.columns.tolist()

# --- PANEL DE MÉTRICAS ---
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Registros", f"{len(df):,}")
with col2: st.metric("Columnas", len(all_cols))
with col3: st.metric("Numéricas", len(num_cols))
with col4: st.metric("Categóricas", len(cat_cols))

# --- TABS EMPRESARIALES ---
t_report, t_visual, t_stats = st.tabs(["📝 Reporte Ejecutivo", "📊 Visualización Avanzada", "🧬 Análisis Estadístico"])

with t_report:
    st.subheader("Vista Previa del Reporte")
    # Buscador multivariable
    search = st.text_input("🔍 Filtro inteligente de tabla (Busca en cualquier celda)")
    if search:
        df_view = df_view[df_view.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)]
    
    st.dataframe(df_view, use_container_width=True)
    
    # Botones de exportación
    csv_data = df_view.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Exportar reporte actual (CSV)", csv_data, "reporte_generado.csv", "text/csv")

with t_visual:
    st.subheader("Configuración de Gráficos de Alto Impacto")
    
    with st.container():
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            g_type = st.selectbox("Tipo de Visualización", ["Barras Agrupadas", "Líneas de Tendencia", "Dispersión (Scatter)", "Área de Relleno", "Boxplot Comparativo", "Treemap"])
        with c2:
            agg_func = st.selectbox("Agregación (Eje Y)", ["Sumar", "Promediar", "Contar", "Mínimo", "Máximo"])
        with c3:
            palette = st.selectbox("Esquema de Color", px.colors.named_colorscales())

    cc1, cc2, cc3, cc4 = st.columns(4)
    x_axis = cc1.selectbox("Variable X", all_cols)
    y_axis = cc2.selectbox("Variable Y", num_cols if g_type != "Treemap" else num_cols)
    color_by = cc3.selectbox("Dividir por (Color)", ["Ninguno"] + cat_cols)
    facet_by = cc4.selectbox("Facetado (Subplots)", ["Ninguno"] + cat_cols)

    # Lógica de Agregación
    agg_map = {"Sumar": "sum", "Promediar": "mean", "Contar": "count", "Mínimo": "min", "Máximo": "max"}
    
    # Construcción de Gráfico
    color_val = color_by if color_by != "Ninguno" else None
    facet_val = facet_by if facet_by != "Ninguno" else None

    if g_type == "Barras Agrupadas":
        fig = px.bar(df_view, x=x_axis, y=y_axis, color=color_val, facet_col=facet_val, barmode="group", template="plotly_white", color_continuous_scale=palette)
    elif g_type == "Líneas de Tendencia":
        fig = px.line(df_view, x=x_axis, y=y_axis, color=color_val, facet_col=facet_val, template="plotly_white")
    elif g_type == "Treemap":
        fig = px.treemap(df_view, path=[x_axis] + ([color_by] if color_by != "Ninguno" else []), values=y_axis, color_continuous_scale=palette)
    elif g_type == "Boxplot Comparativo":
        fig = px.box(df_view, x=x_axis, y=y_axis, color=color_val, notched=True)
    else:
        fig = px.scatter(df_view, x=x_axis, y=y_axis, color=color_val, size=y_axis if st.checkbox("Tamaño basado en Y") else None)

    st.plotly_chart(fig, use_container_width=True)

with t_stats:
    st.subheader("Diagnóstico Estadístico del Dataset")
    
    c_st1, c_st2 = st.columns([2, 1])
    
    with c_st1:
        st.write("**Matriz de Correlación de Pearson**")
        if len(num_cols) > 1:
            corr = df_view[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
            
    
    with c_st2:
        st.write("**Distribución y Sesgo (Skewness)**")
        target_num = st.selectbox("Analizar columna", num_cols)
        skew = df_view[target_num].skew()
        st.metric("Índice de Sesgo", round(skew, 2))
        
        # Interpretación del sesgo
        if skew > 0.5: st.warning("Sesgo Positivo: La cola está a la derecha.")
        elif skew < -0.5: st.warning("Sesgo Negativo: La cola está a la izquierda.")
        else: st.success("Distribución balanceada.")
        
        fig_dist = px.histogram(df_view, x=target_num, marginal="box", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_dist, use_container_width=True)
        

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Enterprise Data Tool v2.0 - Generación Automática de Insights")
