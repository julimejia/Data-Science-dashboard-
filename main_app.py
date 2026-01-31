import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Ultra Custom EDA", layout="wide")
st.title("🛠️ Master Data Lab: Personalización Total")

# --- CARGA Y LIMPIEZA ---
st.sidebar.header("📂 Origen de Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    # Limpieza de columnas duplicadas
    df.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(df.columns, df.columns.duplicated()))]
    
    # --- BARRA LATERAL: CONFIGURACIÓN GLOBAL ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Estética y Filtros")
    
    # Filtro de filas
    max_rows = st.sidebar.number_input("Filas máximas a procesar", 10, len(df), 5000)
    df_view = df.head(max_rows)

    # Buscador de texto global
    search_term = st.sidebar.text_input("🔍 Buscar en los datos (Filtro rápido)")
    if search_term:
        df_view = df_view[df_view.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    # Selector de Tema
    theme = st.sidebar.selectbox("Tema Visual", ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"])
    color_scale = st.sidebar.selectbox("Paleta de Colores", px.colors.named_colorscales())

    # --- PANEL DE CONTROL DE GRÁFICOS ---
    st.header("📈 Creador de Visualizaciones Personalizadas")
    
    num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_view.select_dtypes(exclude=[np.number]).columns.tolist()
    all_cols = df_view.columns.tolist()

    with st.expander("⚙️ Configuración del Gráfico", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            chart_type = st.selectbox("Tipo de Gráfico", 
                ["Dispersión (Scatter)", "Líneas", "Barras", "Área", "Histograma", "Caja (Box)", "Violín", "Embudo (Funnel)"])
        with c2:
            x_val = st.selectbox("Eje X (Categoría o Tiempo)", all_cols)
        with c3:
            y_val = st.selectbox("Eje Y (Valor)", num_cols)
        with c4:
            color_val = st.selectbox("Color por (Opcional)", ["Ninguno"] + all_cols)

    # --- SUBPLOTS Y DIMENSIONES ---
    with st.expander("🧩 Subplots y Personalización Avanzada"):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            facet_col = st.selectbox("Dividir en Subplots (Columnas)", ["Ninguno"] + cat_cols)
        with cc2:
            log_y = st.checkbox("Escala Logarítmica en Y")
        with cc3:
            chart_height = st.slider("Altura del gráfico", 400, 1000, 600)

    # --- RENDERIZADO LÓGICO ---
    color_arg = color_val if color_val != "Ninguno" else None
    facet_arg = facet_col if facet_col != "Ninguno" else None
    
    fig = None

    if chart_type == "Dispersión (Scatter)":
        fig = px.scatter(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, 
                         template=theme, color_continuous_scale=color_scale, trendline="lowess" if len(df_view)<1000 else None)
    
    elif chart_type == "Líneas":
        fig = px.line(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, template=theme)
    
    elif chart_type == "Barras":
        fig = px.bar(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, template=theme, barmode="group")
    
    elif chart_type == "Área":
        fig = px.area(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, template=theme)
    
    elif chart_type == "Histograma":
        fig = px.histogram(df_view, x=x_val, color=color_arg, facet_col=facet_arg, template=theme, marginal="rug")
    
    elif chart_type == "Caja (Box)":
        fig = px.box(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, template=theme, points="all")
    
    elif chart_type == "Violín":
        fig = px.violin(df_view, x=x_val, y=y_val, color=color_arg, facet_col=facet_arg, template=theme, box=True, points="all")
    
    elif chart_type == "Embudo (Funnel)":
        fig = px.funnel(df_view, x=y_val, y=x_val, color=color_arg, template=theme)

    if fig:
        fig.update_layout(height=chart_height)
        if log_y: fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DE DATOS FILTRADOS ---
    st.markdown("---")
    st.subheader("📋 Datos Seleccionados")
    st.write(f"Mostrando {len(df_view)} filas filtradas.")
    st.dataframe(df_view, use_container_width=True)

    # Botón de descarga
    csv = df_view.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar este subset como CSV", data=csv, file_name="subset_data.csv", mime="text/csv")

else:
    st.warning("👈 Por favor, sube un archivo CSV para empezar a crear.")
