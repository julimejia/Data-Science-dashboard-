import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq # Necesitas instalar: pip install groq

# Configuración profesional
st.set_page_config(page_title="Enterprise AI Dashboard", layout="wide")

# --- LÓGICA DE IA (Groq) ---
def analizar_con_ia(api_key, context_data, user_question):
    try:
        client = Groq(api_key=api_key)
        # Creamos un prompt con el contexto de los datos
        prompt = f"""
        Actúa como un experto Analista de Datos senior. 
        Aquí tienes un resumen de los datos cargados:
        {context_data}
        
        Pregunta del usuario: {user_question}
        
        Por favor, sé conciso, profesional y da insights basados solo en la estructura y estadísticas enviadas.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error con la IA: {str(e)}"

# --- SIDEBAR: CONFIGURACIÓN Y API ---
with st.sidebar:
    st.header("🔑 Configuración")
    uploaded = st.file_uploader("Subir CSV", type=["csv"])
    
    st.divider()
    st.subheader("🤖 Configuración IA")
    groq_api_key = st.text_input("Introduce tu Groq API Key", type="password")
    st.info("Consigue tu llave en: console.groq.com")

    if not uploaded:
        st.stop()

# --- CARGA Y PROCESAMIENTO ---
df = pd.read_csv(uploaded)
df.columns = [str(c).replace(' ', '_') for c in df.columns]
df_sample = df.head(2000)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- DASHBOARD PRINCIPAL ---
tabs = st.tabs(["📊 Cuantitativo", "🎭 Cualitativo", "🛠️ Explorador", "🤖 Consultoría IA"])

dataset_rows, dataset_cols = df.shape
missing_total = int(df.isna().sum().sum())
missing_pct = round(100 * missing_total / df.size, 2)

with tabs[0]:
    st.subheader("Análisis cuantitativo de variables numéricas")
    st.write(
        "Este panel explica la forma, la dispersión y las relaciones entre las variables numéricas. "
        "Los gráficos ayudan a identificar asimetría, valores atípicos y correlaciones relevantes."
    )

    st.metric("Registros", dataset_rows)
    st.metric("Columnas numéricas", len(num_cols))
    st.metric("Porcentaje de datos faltantes", f"{missing_pct}%")

    if num_cols:
        sel_num = st.selectbox("Variable numérica para descripción detallada", num_cols, key="num_main")
        num_series = df_sample[sel_num].dropna()
        if not num_series.empty:
            stats = num_series.describe().rename({
                '25%': 'Q1',
                '50%': 'Mediana',
                '75%': 'Q3'
            })
            st.markdown(f"**Estadísticas clave para `{sel_num}`:**")
            st.write(
                f"Media: {stats['mean']:.3f}, mediana: {stats['Mediana']:.3f}, "
                f"desviación estándar: {stats['std']:.3f}, rango: {stats['min']:.3f} – {stats['max']:.3f}."
            )

            fig = px.histogram(
                df_sample,
                x=sel_num,
                marginal="violin",
                nbins=45,
                opacity=0.8,
                color_discrete_sequence=['#0072B2'],
                title=f"Distribución de {sel_num} con densidad y valores extremos"
            )
            fig.update_layout(yaxis_title="Frecuencia", xaxis_title=sel_num)
            st.plotly_chart(fig, use_container_width=True)

            fig_box = px.box(
                df_sample,
                y=sel_num,
                points="suspectedoutliers",
                color_discrete_sequence=['#D55E00'],
                title=f"Boxplot de {sel_num} para identificar outliers"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.write(
                "El histograma junto con el violín permite ver la forma completa de la distribución, "
                "mientras que el boxplot facilita la detección de valores atípicos."
            )
        else:
            st.warning("La variable seleccionada no tiene datos disponibles después de limpiar valores nulos.")

        if len(num_cols) > 1:
            st.markdown("### Comparación bivariada entre variables numéricas")
            x_var = st.selectbox("Eje X", num_cols, index=0, key="x_var")
            y_var = st.selectbox("Eje Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="y_var")
            color_by = None
            if cat_cols:
                color_by = st.selectbox(
                    "Color por categoría (opcionales)",
                    ["Ninguno"] + cat_cols,
                    index=0,
                    key="color_by"
                )
                if color_by == "Ninguno":
                    color_by = None

            fig_scatter = px.scatter(
                df_sample,
                x=x_var,
                y=y_var,
                color=color_by,
                trendline="ols",
                opacity=0.7,
                title=f"Relación entre {x_var} y {y_var}"
            )
            fig_scatter.update_layout(xaxis_title=x_var, yaxis_title=y_var)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.write(
                "Este scatter con línea de tendencia muestra si existe una relación lineal o agrupamientos inesperados. "
                "Agregar una categoría como color ayuda a descubrir segmentos distintos dentro de los datos."
            )
        else:
            st.info("Se necesitan al menos dos variables numéricas para realizar comparaciones bivariadas.")
    else:
        st.warning("No se han detectado variables numéricas en este conjunto de datos.")

with tabs[1]:
    st.subheader("Análisis cualitativo de variables categóricas")
    st.write(
        "Explora cómo se distribuyen las categorías, cuáles son las más frecuentes y cómo influyen "
        "en el comportamiento de las variables numéricas."
    )

    if cat_cols:
        sel_cat = st.selectbox("Variable categórica principal", cat_cols, key="cat_main")
        counts = df_sample[sel_cat].value_counts().nlargest(12).reset_index()
        counts.columns = [sel_cat, 'count']

        fig_cat = px.bar(
            counts,
            x=sel_cat,
            y='count',
            color='count',
            color_continuous_scale='Blues',
            title=f"Frecuencia de las principales categorías en {sel_cat}"
        )
        fig_cat.update_layout(xaxis_title=sel_cat, yaxis_title="Número de registros")
        st.plotly_chart(fig_cat, use_container_width=True)
        st.write(
            "Este gráfico muestra la relevancia relativa de cada etiqueta. "
            "Si hay muchas categorías, aquí aparecen las que dominan el dataset."
        )

        if num_cols:
            sel_num_cat = st.selectbox(
                "Variable numérica para comparar por categoría",
                num_cols,
                key="num_vs_cat"
            )
            fig_box_cat = px.box(
                df_sample,
                x=sel_cat,
                y=sel_num_cat,
                points="outliers",
                title=f"Distribución de {sel_num_cat} por {sel_cat}",
                color=sel_cat
            )
            fig_box_cat.update_layout(xaxis_title=sel_cat, yaxis_title=sel_num_cat, showlegend=False)
            st.plotly_chart(fig_box_cat, use_container_width=True)
            st.write(
                "El boxplot por categoría exhibe diferencias en la mediana, rango intercuartílico y valores extremos "
                "entre grupos categóricos."
            )
        else:
            st.info("Para comparar categorías con valores numéricos se necesita al menos una columna numérica.")
    else:
        st.warning("No se han detectado variables categóricas en este conjunto de datos.")

with tabs[2]:
    st.subheader("Explorador de datos completo")
    st.write(
        "Revisa la estructura general del dataset, la presencia de datos faltantes y las correlaciones clave. "
        "Esta vista es útil para la inspección rápida de calidad y dependencias entre variables."
    )

    st.markdown("### Resumen de la estructura del dataset")
    st.write(f"El conjunto de datos contiene **{dataset_rows} filas** y **{dataset_cols} columnas**.")
    st.write(f"Hay **{missing_total} valores faltantes** en total, lo que representa el **{missing_pct}%** del dataset.")

    st.markdown("### Estadísticas descriptivas generales")
    st.dataframe(df_sample.describe(include='all').transpose(), use_container_width=True)

    missing_df = pd.DataFrame({
        "Columna": df.columns,
        "Valores faltantes": df.isna().sum().values,
        "% faltantes": (df.isna().mean() * 100).round(2).values
    }).sort_values("% faltantes", ascending=False)
    st.markdown("### Análisis de valores faltantes")
    st.dataframe(missing_df, use_container_width=True)

    if len(num_cols) > 1:
        st.markdown("### Matriz de correlación entre variables numéricas")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Correlación de variables numéricas"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.write(
            "La matriz de correlación destaca parejas de variables con relaciones fuertes positivas o negativas. "
            "Estos patrones son útiles para descubrir redundancias y posibles factores explicativos."
        )
    else:
        st.info("Se necesitan al menos dos variables numéricas para mostrar una matriz de correlación.")

# --- NUEVA TAB: CONSULTORÍA IA ---
with tabs[3]:
    st.subheader("Analista Virtual (Llama 3 via Groq)")
    
    if not groq_api_key:
        st.warning("⚠️ Por favor, introduce tu API Key de Groq en la barra lateral para usar el analista.")
    else:
        resumen_stats = {
            "columnas": list(df.columns),
            "tipos": df.dtypes.astype(str).to_dict(),
            "estadisticas": df.describe().to_dict(),
            "nulos": df.isna().sum().to_dict()
        }
        
        user_query = st.text_area(
            "¿Qué te gustaría saber sobre estos datos?", 
            placeholder="Ej: Haz un resumen de las tendencias principales y posibles anomalías."
        )
        
        if st.button("🚀 Analizar con IA"):
            with st.spinner("La IA está procesando tus datos..."):
                respuesta = analizar_con_ia(groq_api_key, resumen_stats, user_query)
                st.markdown("### 💡 Insights de la IA:")
                st.write(respuesta)
                st.toast("Análisis completado", icon='✅')

st.markdown("---")
st.caption("Intelligence Dashboard v3.0 | Power by Groq & Plotly")
