import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import matplotlib.pyplot as pt
import seaborn as sb
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title= 'Bank Churn Model',
    layout="wide"  
)



@st.cache_data
def cargar_datos():
    return pd.read_csv('Data/bank_churn_limpio.csv')

data = cargar_datos()

nombresEs = {
    'age': 'Edad',
    'products_number': 'Nº de Productos',
    'estimated_salary': 'Salario Estimado',
    'credit_score': 'Puntaje crediticios',
    'NewBalance': 'Saldo',
    'active_member': 'Miembro Activo',
    'tenure': 'Antigüedad',
    'credit_card': '¿Posee Tarjeta?',
    'gender': 'Género',
    'country_France': 'Frances',
    'country_Germany': 'Alemán',
    'country_Spain': 'Español',
    'limite_credito': 'Limite de Credito'
}

@st.cache_resource
def cargarModelo():
    data = jl.load('Modelo/modelo_banco_final.pkl')
    return data['modelo'], data['columnas'], data['umbral']

modelo, columnas, umbral = cargarModelo()

pt.style.use('dark_background')
colorInicio = "#83648ADE"
colorFin = "#8E07A9"
degradado = LinearSegmentedColormap.from_list("custom_grad", [colorInicio, colorFin])

pt.rcParams.update({
    "figure.facecolor":  (0, 0, 0, 0),
    "axes.facecolor":    (0, 0, 0, 0),
    "savefig.facecolor": (0, 0, 0, 0),
})

st.markdown("""
    <style>
    .stApp {
        background: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid #2d2d2d;
        padding: 2rem 1rem
    }
    
    [data-testid="stSidebar"] [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(142, 7, 169, 0.2) !important;
        border-radius: 20px !important;
        padding: 30px !important;
    }

    div[data-testid="stSlider"] > div [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #8E07A9 0%, #0d958e 100%) !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #8E07A9 !important;
        border: 2px solid white !important;
        box-shadow: 0 0 10px #8E07A9 !important;
    }

    div[data-testid="stThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
        color: #e0e0e0 !important;
        font-weight: bold;
    }

    .stButton > button {
        background: linear-gradient(90deg, #8E07A9, #6a0580) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6em 2em !important;
        width: 100% !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(142, 7, 169, 0.3) !important;
        transition: 0.3s all ease;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(142, 7, 169, 0.5) !important;
        background: #0d958e !important;
    }
    
    .stHeading h2, .stHeading h1 {
        background: linear-gradient(90deg, #83648ade, #8E07A9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    
    .introduccion {
        background: rgba(142, 7, 169, 0.05);
        border-left: 4px solid #8E07A9;
        padding: 20px;
        border-radius: 8px;
        color: #b0b0b0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Bank Churn Model')
st.markdown("## Luciano Y.")
st.markdown('Despliega la flecha para probar el modelo predictivo')

st.markdown(f"""
    <div class="introduccion">
        <p>
            Este es un proyecto de 'Data Science' diseñado para predecir un problema real: la fuga de clientes 'Churn'.
        A través de un flujo completo que va desde la limpieza y normalización de datos hasta el análisis exploratorio 'EDA',
        el entrenamiento de un modelo predictivo que analiza comportamientos clave. El sistema detecta patrones en variables como la edad,
        el número de productos para generar una alerta temprana.
        Lo más interesante no es que "adivina", sino que ayuda a entender por qué un cliente podría irse,
        permitiendo que el banco genere estrategias y así generando valor para el banco.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider() 
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric(label="Total de Clientes Analizados", value=f"{len(data):,}")
with col_kpi2:
    tasa_fuga = (data['churn'].mean() * 100)
    st.metric(label="Precisión del Modelo", value="85.0%")
with col_kpi3:

    st.metric(label="Tasa de Fuga Histórica", value=f"{tasa_fuga:.1f}%") 
st.divider()

st.sidebar.header("Simulador de Predicción")

with st.sidebar.form(key="formulario_simulador"):

    edad = st.slider("Edad", 18, 90, 35)
    
    balance = st.slider("Saldo en Cuenta", 0, 250000, 15000, step=1000)

    score = st.slider("Puntaje Crediticio", 300, 850, 600)
    tenure = st.slider("Antigüedad (Años)", 0, 10, 5)
    
    col_form1, col_form2 = st.columns(2)
    with col_form1:
        productos = st.selectbox("Nº de Productos", [1, 2, 3, 4])
        genero = st.radio("Género", ["Mujer", "Hombre"])
    with col_form2:
        miembro = st.radio("¿Miembro activo?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        tarjeta = st.radio("¿Tiene Tarjeta?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")

    submit_button = st.form_submit_button(label="Calcular Riesgo")

if submit_button:
    
    gen_val = 1 if genero == "Hombre" else 0

    input_data = {
        'age': edad,
        'NewBalance': np.arcsinh(balance),
        'credit_score': score,
        'products_number': productos,
        'active_member': miembro,
        'tenure': tenure,
        'credit_card': tarjeta,
        'gender': gen_val
    }
    
    dataInput = pd.DataFrame([input_data])
    dataInput = dataInput.reindex(columns=columnas, fill_value=0)
    
    prob = modelo.predict_proba(dataInput)[0][1]

    st.sidebar.divider()
    st.sidebar.write(f"### Probabilidad de fuga: {prob*100:.1f}%")
    if prob > umbral:
        st.sidebar.error("Riesgo de fuga")
    else:
        st.sidebar.success("Cliente estable")





st.header("Graficas importantes")

with st.container():
    columna1, columna2, columna3 = st.columns(3)
    
    with columna1:
        colores = {0: "#8E07A9", 1: "#0d958e"}
        st.subheader("Grafico: Correlacion de variables")
        columnasInteres = ['age', 'credit_score', 'NewBalance', 'estimated_salary', 'products_number','tenure', 'churn']
        dataMuestra = data.sample(n=min(200, len(data)), random_state=42)
        dataGrafico = dataMuestra[columnasInteres].rename(columns=nombresEs)
        
        pairplot = sb.pairplot(dataGrafico, hue='churn', palette=colores, diag_kind='kde', plot_kws={'alpha': 0.4}, height=2, aspect=1.198)
        st.pyplot(pairplot.figure)
        pt.close(pairplot.figure)
        st.caption("**Insight:** Se nota que la relacion entre la edad y las otras variables sigue un patron claro de color 'Churn' en el rango de 50-70 años")
        st.caption("Mismo patron en el numero de productos")
        
    with columna2:
        st.subheader("Grafico: Importancia de las variables")
        importanciasValores = modelo.feature_importances_
        importanciasSerie = pd.Series(importanciasValores, index=columnas).sort_values(ascending=False)
        importanciasSerie.index = [nombresEs.get(col, col) for col in importanciasSerie.index]
        fig, ax = pt.subplots(figsize=(10, 9.39))
        norm = pt.Normalize(importanciasSerie.values.min(), importanciasSerie.values.max())
        coloresImportancia = [degradado(i) for i in np.linspace(1, 0, len(importanciasSerie))]
        sb.barplot(x=importanciasSerie.values, y=importanciasSerie.index, ax=ax, hue=importanciasSerie.index, palette=coloresImportancia, legend=False)
        st.pyplot(fig)
        pt.close(fig)
        st.caption("**Insight:** Las variables con más peso para el modelo predictivo son 'Edad', 'Numero de productos contratados' y 'El salario estimado'")
        
    with columna3:   
        st.subheader("Grafico: Churn por pais")
        
        dataPais = data.copy()
        
        Paises = {
            'France': 'Francia',
            'Germany': 'Alemania',
            'Spain': 'España'
        }
        
        resPais = dataPais.groupby('country')['churn'].mean().sort_values()
        
        dataPais['country'] = dataPais['country'].map(Paises).fillna(dataPais['country'])
        
        fig, ax = pt.subplots(figsize=(10, 8))
        
        coloresPaises = [degradado(0.1), degradado(0.5), degradado(1.0)]
        
        sb.barplot(data=dataPais, x='country', y='churn', hue='country', palette=coloresPaises, ax=ax)
        st.pyplot(fig)
        pt.close(fig)
        st.caption("**Insight:** Alemania es el pais con una tasa de ida 'Churn' más alta")
        
        