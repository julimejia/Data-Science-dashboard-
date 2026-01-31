import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Advanced EDA Pro", layout="wide")
st.title("🚀 Explorador de Datos Interactivo")

# --- CARGA Y LIMPIEZA ---
st.sidebar.header("1. Configuración")
uploaded = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])

if uploaded:
    enc = st.sidebar.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"])
    df = pd.read_csv(uploaded, encoding=enc)
    
    # Limpieza de duplicados en columnas
    if df.columns.duplicated().any():
        df.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(df.columns, df.columns.duplicated()))]

    # --- FILTROS DE CONTROL DE DATOS ---
    st.sidebar.markdown("---")
    st.sidebar.header("2. Control de Visualización")
    
    # Slider para cantidad de datos
    n_rows = st.sidebar.slider("Filas a visualizar", 
                                min_value=min(10, len(df)), 
                                max_value=len(df), 
                                value=min(1000, len(df)))
    df_sample = df.head(n_rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Estadísticas", "📈 Análisis Numérico", "🎭 Análisis Categórico"])

    with tab1:
        st.subheader(f"Resumen de las primeras {n_rows} filas")
        col1, col2, col3 = st.columns(3)
        col1.metric("Columnas Totales", df.shape[1])
        col2.metric("Nulos Totales", df_sample.isna().sum().sum())
        col3.metric("Memoria (MB)", round(df_sample.memory_usage().sum() / 1024**2, 2))
        st.dataframe(df_sample.describe().T, use_container_width=True)

    with tab2:
        st.subheader("Análisis Cuantitativo Dinámico")
        
        # Opciones de control
        c_alt1, c_alt2, c_alt3 = st.columns(3)
        with c_alt1:
            x_axis = st.selectbox("Eje X", numeric_cols, key="x_num")
        with c_alt2:
            y_axis = st.selectbox("Eje Y", numeric_cols, key="y_num")
        with c_alt3:
            z_axis = st.selectbox("Eje Z (Para 3D)", ["Ninguno"] + numeric_cols)

        # Checkboxes para personalizar el gráfico
        st.write("**Opciones de gráfico:**")
        col_check1, col_check2, col_check3 = st.columns(3)
        show_trend = col_check1.checkbox("Mostrar línea de tendencia (OLS)")
        show_marginal = col_check2.checkbox("Mostrar distribuciones marginales", value=True)
        use_color = col_check3.checkbox("Colorear por categoría")

        color_col = None
        if use_color and cat_cols:
            color_col = st.selectbox("Categoría para color", cat_cols)

        if z_axis == "Ninguno":
            # Gráfico 2D
            fig = px.scatter(df_sample, x=x_axis, y=y_axis, 
                             color=color_col,
                             trendline="ols" if show_trend else None,
                             marginal_x="box" if show_marginal else None,
                             marginal_y="violin" if show_marginal else None,
                             template="plotly_white",
                             title=f"Relación: {x_axis} vs {y_axis}")
        else:
            # Gráfico 3D
            fig = px.scatter_3d(df_sample, x=x_axis, y=y_axis, z=z_axis,
                                color=color_col,
                                title=f"Análisis 3D: {x_axis}, {y_axis}, {z_axis}")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Análisis Cualitativo e Impacto")
        if cat_cols:
            selected_cat = st.selectbox("Selecciona Categoría", cat_cols)
            limit_cats = st.slider("Máximo de categorías a mostrar", 5, 50, 15)
            
            # Preparar datos
            cat_counts = df_sample[selected_cat].value_counts().nlargest(limit_cats).reset_index()
            
            col_l, col_r = st.columns(2)
            
            with col_l:
                # Gráfico de barras interactivo
                fig_bar = px.bar(cat_counts, x=selected_cat, y='count', 
                                 color='count', color_continuous_scale='Viridis',
                                 title=f"Top {limit_cats} de {selected_cat}")
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_r:
                # Gráfico circular tipo Donut
                fig_pie = px.pie(cat_counts, names=selected_cat, values='count', 
                                 hole=0.4, title="Distribución porcentual")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # Bonus: Boxplot condicionado
            st.markdown("---")
            st.write(f"### ¿Cómo afecta '{selected_cat}' a las variables numéricas?")
            num_target = st.selectbox("Elige variable numérica para comparar", numeric_cols)
            fig_facet = px.box(df_sample, x=selected_cat, y=num_target, color=selected_cat)
            st.plotly_chart(fig_facet, use_container_width=True)
        else:
            st.info("No hay columnas categóricas.")

else:
    st.info("💡 Sube un archivo para desbloquear los gráficos dinámicos.")
