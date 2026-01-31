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

# (Las tabs anteriores se mantienen simplificadas...)

with tabs[0]:
    st.subheader("Análisis de Números")
    sel_num = st.selectbox("Variable numérica", num_cols)
    fig = px.histogram(df_sample, x=sel_num, marginal="box", color_discrete_sequence=['#00D4FF'])
    st.plotly_chart(fig, use_container_width=True)
    

with tabs[1]:
    st.subheader("Análisis de Categorías")
    sel_cat = st.selectbox("Variable categórica", cat_cols)
    counts = df_sample[sel_cat].value_counts().nlargest(10).reset_index()
    fig_cat = px.bar(counts, x=sel_cat, y='count', color='count')
    st.plotly_chart(fig_cat, use_container_width=True)

with tabs[2]:
    st.subheader("Explorador de Datos")
    st.dataframe(df_sample.describe(), use_container_width=True)

# --- NUEVA TAB: CONSULTORÍA IA ---
with tabs[3]:
    st.subheader("Analista Virtual (Llama 3 via Groq)")
    
    if not groq_api_key:
        st.warning("⚠️ Por favor, introduce tu API Key de Groq en la barra lateral para usar el analista.")
    else:
        # Generar contexto para la IA
        resumen_stats = {
            "columnas": list(df.columns),
            "tipos": df.dtypes.astype(str).to_dict(),
            "estadisticas": df.describe().to_dict(),
            "nulos": df.isna().sum().to_dict()
        }
        
        user_query = st.text_area("¿Qué te gustaría saber sobre estos datos?", 
                                 placeholder="Ej: Haz un resumen de las tendencias principales y posibles anomalías.")
        
        if st.button("🚀 Analizar con IA"):
            with st.spinner("La IA está procesando tus datos..."):
                respuesta = analizar_con_ia(groq_api_key, resumen_stats, user_query)
                st.markdown("### 💡 Insights de la IA:")
                st.write(respuesta)
                
                # Feedback visual
                st.toast("Análisis completado", icon='✅')

st.markdown("---")
st.caption("Intelligence Dashboard v3.0 | Power by Groq & Plotly")
