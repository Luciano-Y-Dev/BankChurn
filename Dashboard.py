import base64
from pathlib import Path

import joblib as jl
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bank Churn — Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE_DIR / "Data/bank_churn_limpio.csv")


@st.cache_resource
def cargar_modelo():
    modelo_serializado = jl.load(BASE_DIR / "Modelo/modelo_banco_final.pkl")
    return (
        modelo_serializado["modelo"],
        modelo_serializado["columnas"],
        modelo_serializado["umbral"],
    )


@st.cache_data
def datos_edad():
    riesgo_edad = data.groupby("age")["churn"].mean().sort_index()
    return riesgo_edad.rolling(window=5, center=True, min_periods=1).mean()


@st.cache_data
def datos_paises():
    return data.groupby("country")["churn"].mean().sort_values(ascending=False)


@st.cache_data
def importancias_modelo():
    return pd.Series(modelo.feature_importances_, index=columnas).sort_values(
        ascending=False
    )


def numero(valor):
    return f"{valor:,.0f}".replace(",", ".")


def renderizar_html(texto, unsafe_allow_html=True):
    compacto = "".join(linea.strip() for linea in texto.splitlines())
    st.markdown(compacto, unsafe_allow_html=unsafe_allow_html)


def svg_riesgo(probabilidad):
    progreso = max(0, min(probabilidad, 1)) * 264
    return f"""
    <svg class="risk-ring" viewBox="0 0 104 104" role="img" aria-label="Probabilidad {probabilidad:.1%}">
        <circle class="ring-track" cx="52" cy="52" r="42"></circle>
        <circle class="ring-value" cx="52" cy="52" r="42" pathLength="264"
            stroke-dasharray="{progreso:.2f} 264"></circle>
        <path class="ring-arrow" d="M45 42h17v17M45 59l17-17"></path>
    </svg>
    """


def grafico_edad(edad_actual):
    serie = datos_edad()
    x_min, x_max = 18, 90
    y_max = max(0.5, float(serie.max()) * 1.12)
    left, right, top, bottom = 42, 570, 18, 158

    def coordenada(edad, valor):
        x = left + (edad - x_min) / (x_max - x_min) * (right - left)
        y = bottom - valor / y_max * (bottom - top)
        return x, y

    puntos = [coordenada(edad, valor) for edad, valor in serie.items()]
    linea = " ".join(f"{x:.1f},{y:.1f}" for x, y in puntos)
    area = f"{left},{bottom} {linea} {right},{bottom}"
    indice_cercano = np.abs(serie.index.to_numpy() - edad_actual).argmin()
    marcador_x, marcador_y = coordenada(
        edad_actual, float(serie.get(edad_actual, serie.iloc[indice_cercano]))
    )
    grid = "".join(
        f'<line class="chart-grid" x1="{left}" y1="{bottom - valor / y_max * (bottom - top):.1f}" x2="{right}" y2="{bottom - valor / y_max * (bottom - top):.1f}"></line>'
        for valor in (0, 0.25, 0.5)
    )
    etiquetas = "".join(
        f'<text class="axis-label" x="{coordenada(valor, 0)[0]:.1f}" y="180" text-anchor="middle">{valor}</text>'
        for valor in (18, 30, 40, 50, 60, 70, 90)
    )
    return f"""
    <svg class="chart-svg" viewBox="0 0 600 194" role="img" aria-label="Tasa histórica de fuga por edad; edad seleccionada {edad_actual}">
        <defs>
            <linearGradient id="ageArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#8b8d91" stop-opacity=".20"></stop>
                <stop offset="100%" stop-color="#8b8d91" stop-opacity="0"></stop>
            </linearGradient>
        </defs>
        {grid}
        <polygon points="{area}" fill="url(#ageArea)"></polygon>
        <polyline class="age-line" points="{linea}"></polyline>
        <line class="current-line" x1="{marcador_x:.1f}" y1="12" x2="{marcador_x:.1f}" y2="{bottom}"></line>
        <circle class="current-dot-outer" cx="{marcador_x:.1f}" cy="{marcador_y:.1f}" r="8"></circle>
        <circle class="current-dot" cx="{marcador_x:.1f}" cy="{marcador_y:.1f}" r="3.5"></circle>
        {etiquetas}
    </svg>
    """


def grafico_paises():
    nombres = {"France": "Francia", "Germany": "Alemania", "Spain": "España"}
    maximo = float(datos_paises().max())
    filas = []
    for indice, (pais, valor) in enumerate(datos_paises().items()):
        ancho = 100 * valor / maximo
        clase = "bar-primary" if indice == 0 else "bar-secondary"
        filas.append(
            f"""
            <div class="country-row">
                <div class="country-meta"><span>{nombres.get(pais, pais)}</span><strong>{valor:.1%}</strong></div>
                <div class="bar-track"><div class="country-bar {clase}" style="width:{ancho:.1f}%"></div></div>
            </div>
            """
        )
    return "".join(filas)


def grafico_importancias():
    nombres = {
        "age": "Edad",
        "products_number": "Productos",
        "estimated_salary": "Salario estimado",
        "credit_score": "Score crediticio",
        "NewBalance": "Saldo transformado",
        "balance": "Saldo",
        "active_member": "Miembro activo",
        "tenure": "Antigüedad",
        "limite_credito": "Límite de crédito",
    }
    principales = importancias_modelo().head(6)
    maximo = float(principales.max())
    filas = []
    for indice, (variable, valor) in enumerate(principales.items()):
        ancho = 100 * valor / maximo
        clase = "bar-primary" if indice == 0 else "bar-secondary"
        filas.append(
            f"""
            <div class="importance-row">
                <span class="importance-name">{nombres.get(variable, variable)}</span>
                <div class="bar-track"><div class="importance-bar {clase}" style="width:{ancho:.1f}%"></div></div>
                <strong>{valor:.0%}</strong>
            </div>
            """
        )
    return "".join(filas)


data = cargar_datos()
modelo, columnas, umbral = cargar_modelo()

with st.sidebar:
    renderizar_html(
        """
        <div class="sidebar-brand">
            <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M5 10l11-7 11 7-11 7-11-7zm0 12l11 7 11-7M5 16l11 7 11-7"></path></svg>
            <span>churn<span class="brand-dot">.</span></span>
        </div>
        <div class="sidebar-heading"><h2>Perfil del cliente</h2><span class="live-dot"></span></div>
        """,
        unsafe_allow_html=True,
    )
    edad = st.slider("Edad", 18, 90, 35, help="Marca simultáneamente la posición del cliente en el análisis histórico por edad.")
    balance = st.slider("Saldo en cuenta", 0, 250000, 15000, step=1000)
    score = st.slider("Puntaje crediticio", 300, 850, 600)
    pais_col, productos_col = st.columns(2)
    with pais_col:
        pais = st.selectbox("País", ["Spain", "France", "Germany"], format_func=lambda x: {"Spain": "España", "France": "Francia", "Germany": "Alemania"}[x])
    with productos_col:
        productos = st.selectbox("Productos", [1, 2, 3, 4], help="Número de productos bancarios contratados.")
    tenure = st.slider("Antigüedad · años", 0, 10, 5)
    salario_col, limite_col = st.columns(2)
    with salario_col:
        salario = st.number_input("Salario", 0, 250000, 50000, step=5000)
    with limite_col:
        limite = st.number_input("Límite", 0, 250000, 15000, step=1000)
    genero = st.segmented_control("Género", ["Mujer", "Hombre"], default="Mujer")
    activo_col, tarjeta_col = st.columns(2)
    with activo_col:
        miembro = st.toggle("Miembro activo", value=True)
    with tarjeta_col:
        tarjeta = st.toggle("Tarjeta", value=True)
    renderizar_html('<div class="sidebar-status"><span class="live-dot"></span> Actualización en vivo</div>', unsafe_allow_html=True)

input_data = {
    "age": edad,
    "NewBalance": np.arcsinh(balance),
    "balance": balance,
    "credit_score": score,
    "products_number": productos,
    "active_member": int(miembro),
    "tenure": tenure,
    "credit_card": int(tarjeta),
    "gender": 1 if genero == "Hombre" else 0,
    "country_France": int(pais == "France"),
    "country_Germany": int(pais == "Germany"),
    "country_Spain": int(pais == "Spain"),
    "limite_credito": limite,
    "estimated_salary": salario,
}
data_input = pd.DataFrame([input_data]).reindex(columns=columnas, fill_value=0)
prob = float(modelo.predict_proba(data_input)[0][1])
clase_nativa = int(modelo.predict(data_input)[0])
riesgo_alto = prob > umbral
tasa_historica = float(data["churn"].mean())
delta = prob - tasa_historica

font_data = base64.b64encode((BASE_DIR / "assets/Manrope.ttf").read_bytes()).decode()

renderizar_html(
    f"""
    <style>
    @font-face {{
        font-family: Manrope;
        src: url(data:font/ttf;base64,{font_data}) format("truetype");
        font-weight: 200 800;
        font-display: swap;
    }}
    :root {{
        --ground: #111212;
        --surface: #1c1d1e;
        --surface-raised: #232426;
        --line: #303134;
        --ink: #f3f2ed;
        --muted: #9b9c9e;
        --accent: #f3eb73;
        --paper: #e6e5df;
        --row-top: calc((100dvh - 190px) * .47);
        --row-bottom: calc((100dvh - 190px) * .53);
    }}
    * {{ box-sizing: border-box; }}
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
        width: 100%; height: 100dvh; overflow: hidden;
        background: var(--ground); color: var(--ink);
        font-family: Manrope, sans-serif;
    }}
    [data-testid="stHeader"] {{ display: block; height: 0; background: transparent; }}
    [data-testid="stDecoration"], [data-testid="stToolbarActions"], [data-testid="stHeaderActionElements"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], footer {{ display: none; }}
    [data-testid="collapsedControl"] {{ display: block; position: fixed; top: 12px; left: 12px; z-index: 1000000; }}
    [data-testid="collapsedControl"] button {{ color: #b9bab9; background: #1c1d1e; border-radius: 8px; }}
    [data-testid="stAppViewContainer"] > .main .block-container {{
        max-width: none; height: 100dvh; padding: 20px 24px 18px; overflow: hidden;
    }}
    .main [data-testid="stVerticalBlock"] {{ gap: 0; }}
    .main [data-testid="stHorizontalBlock"] {{ gap: 14px; margin: 0; }}
    .main [data-testid="stHorizontalBlock"] + div {{ margin-top: 14px; }}
    .main [data-testid="stColumn"] > div {{ height: 100%; }}
    .main [data-testid="stMarkdownContainer"] {{ height: 100%; }}
    .main [data-testid="stMarkdownContainer"]:has(style) {{ display: none; }}
    .dashboard-header {{
        height: 52px; margin-bottom: 16px; display: flex; align-items: center;
        justify-content: space-between;
    }}
    .dashboard-title {{ margin: 0; font-size: clamp(20px, 2vw, 29px); line-height: 1; letter-spacing: -.035em; font-weight: 650; }}
    .dashboard-title span {{ color: #77797c; font-weight: 400; }}
    .header-meta {{ display: flex; align-items: center; gap: 9px; font-size: 11px; color: #b9bab9; }}
    .live-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: inline-block; flex: none; }}
    .bento-card {{
        position: relative; height: 100%; min-height: 0; border-radius: 16px;
        background: var(--surface); padding: clamp(17px, 1.6vw, 25px); overflow: hidden;
        box-shadow: 0 10px 28px rgba(0,0,0,.16);
    }}
    .top-card {{ height: var(--row-top); }}
    .bottom-card {{ height: var(--row-bottom); }}
    .card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; }}
    .card-title {{ margin: 0; color: var(--ink); font-size: 13px; line-height: 1.25; font-weight: 650; letter-spacing: -.01em; white-space: nowrap; }}
    .card-meta {{ color: var(--muted); font-size: 10px; white-space: nowrap; }}
    .risk-card {{ background: var(--accent); color: #181914; box-shadow: 0 14px 34px rgba(0,0,0,.23); }}
    .risk-card .card-title {{ color: #181914; font-size: 14px; }}
    .risk-top {{ display: flex; align-items: flex-start; justify-content: space-between; }}
    .risk-badge {{ display: inline-flex; align-items: center; margin-top: 15px; padding: 6px 9px; border-radius: 7px; background: rgba(24,25,20,.09); font-size: 9px; font-weight: 800; letter-spacing: .08em; }}
    .risk-ring {{ width: clamp(72px, 7vw, 104px); height: auto; transform: rotate(-90deg); }}
    .risk-ring circle {{ fill: none; stroke-width: 8; }}
    .ring-track {{ stroke: rgba(24,25,20,.14); }}
    .ring-value {{ stroke: #20211c; stroke-linecap: round; transition: stroke-dasharray .5s cubic-bezier(.16,1,.3,1); }}
    .ring-arrow {{ fill: none; stroke: #20211c; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; transform: rotate(90deg); transform-origin: center; }}
    .risk-value {{ margin: auto 0 0; font-size: clamp(54px, 5.6vw, 88px); line-height: .95; letter-spacing: -.065em; font-weight: 550; font-variant-numeric: tabular-nums; }}
    .risk-value sup {{ font-size: .35em; letter-spacing: -.02em; vertical-align: top; position: relative; top: .22em; margin-left: 3px; }}
    .risk-foot {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 8px; margin-top: 18px; padding-top: 13px; border-top: 1px solid rgba(24,25,20,.17); font-size: 10px; }}
    .risk-foot strong {{ font-size: 12px; }}
    .risk-card {{ display: flex; flex-direction: column; }}
    .paper-card {{ background: var(--paper); color: #20211f; }}
    .paper-card .card-title {{ color: #20211f; }}
    .paper-card .card-meta {{ color: #656660; }}
    .historic-value {{ margin: 12px 0 0; font-size: clamp(46px, 4.2vw, 66px); font-weight: 500; letter-spacing: -.06em; line-height: 1; font-variant-numeric: tabular-nums; }}
    .historic-stats {{ position: absolute; left: clamp(17px, 1.6vw, 25px); right: clamp(17px, 1.6vw, 25px); bottom: 22px; padding-top: 14px; border-top: 1px solid #c4c4bd; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .historic-stat span {{ display: block; color: #676862; font-size: 9px; margin-bottom: 5px; }}
    .historic-stat strong {{ display: block; font-size: 18px; letter-spacing: -.03em; font-weight: 650; }}
    .paper-card {{ display: block; }}
    .chart-card {{ display: flex; flex-direction: column; }}
    .chart-svg {{ display: block; width: 100%; flex: 1; min-height: 0; overflow: visible; }}
    .chart-grid {{ stroke: #303134; stroke-width: 1; }}
    .age-line {{ fill: none; stroke: #b8b9b5; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }}
    .current-line {{ stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 5; opacity: .65; }}
    .current-dot-outer {{ fill: var(--accent); opacity: .18; }}
    .current-dot {{ fill: var(--accent); }}
    .axis-label {{ fill: #8f9092; font-family: Manrope, sans-serif; font-size: 10px; }}
    .chart-legend {{ display: flex; align-items: center; gap: 14px; color: var(--muted); font-size: 9px; margin-top: 4px; }}
    .legend-key {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend-line {{ width: 14px; height: 2px; background: #b8b9b5; }}
    .legend-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }}
    .country-list {{ display: flex; flex-direction: column; justify-content: space-around; flex: 1; padding: 3px 0 8px; }}
    .country-meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 11px; color: #c8c8c4; }}
    .country-meta strong {{ color: var(--ink); font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .bar-track {{ height: 7px; border-radius: 4px; background: #2b2c2e; overflow: hidden; }}
    .country-bar, .importance-bar {{ height: 100%; border-radius: 4px; transform-origin: left; animation: grow .7s cubic-bezier(.16,1,.3,1); }}
    .bar-primary {{ background: #deded7; }}
    .bar-secondary {{ background: #65676a; }}
    .importance-list {{ display: flex; flex-direction: column; justify-content: space-between; flex: 1; padding-bottom: 5px; }}
    .importance-row {{ display: grid; grid-template-columns: minmax(86px, .85fr) 1.4fr 32px; align-items: center; gap: 12px; }}
    .importance-name {{ font-size: 10px; color: #c0c1bd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .importance-row strong {{ text-align: right; color: #d6d6d1; font-size: 10px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .profile-card {{ display: flex; flex-direction: column; }}
    .profile-lead {{ margin: 4px 0 12px; }}
    .profile-age {{ display: block; font-size: clamp(36px, 3.2vw, 48px); font-weight: 500; line-height: 1; letter-spacing: -.055em; }}
    .profile-country {{ display: block; color: var(--muted); font-size: 11px; margin-top: 6px; }}
    .profile-list {{ margin: 0; padding: 0; }}
    .profile-row {{ display: flex; justify-content: space-between; gap: 8px; padding: 8px 0; border-top: 1px solid var(--line); font-size: 10px; }}
    .profile-row dt {{ color: var(--muted); }}
    .profile-row dd {{ margin: 0; color: var(--ink); font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
    @keyframes grow {{ from {{ transform: scaleX(.05); }} to {{ transform: scaleX(1); }} }}
    @keyframes rise {{ from {{ transform: translateY(8px); filter: blur(3px); }} to {{ transform: translateY(0); filter: blur(0); }} }}
    .risk-value {{ animation: rise .55s cubic-bezier(.16,1,.3,1); }}
    section[data-testid="stSidebar"] {{
        width: 310px; min-width: 310px; background: #171819; border-right: 1px solid #292a2c;
    }}
    section[data-testid="stSidebar"] > div {{ width: 310px; }}
    [data-testid="stSidebarContent"] {{ overflow: hidden; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 7px; }}
    [data-testid="stSidebar"] .block-container {{ padding: 22px 20px 14px; }}
    .sidebar-brand {{ display: flex; align-items: center; gap: 9px; height: 40px; margin-bottom: 21px; font-size: 24px; font-weight: 750; letter-spacing: -.05em; }}
    .sidebar-brand svg {{ width: 28px; height: 28px; fill: none; stroke: var(--accent); stroke-width: 2.1; stroke-linejoin: round; }}
    .brand-dot {{ color: var(--accent); }}
    .sidebar-heading {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
    .sidebar-heading h2 {{ margin: 0; color: #efeee9; font-size: 13px; font-weight: 650; letter-spacing: -.01em; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: #a7a8a7; font-size: 10px; font-weight: 500; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ margin-bottom: 3px; }}
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {{ background: var(--accent) !important; }}
    [data-testid="stSidebar"] [role="slider"] {{ background: var(--accent) !important; border: 3px solid #171819 !important; box-shadow: 0 0 0 1px var(--accent) !important; width: 15px !important; height: 15px !important; }}
    [data-testid="stSidebar"] [data-testid="stThumbValue"] {{ color: #e7e6e0 !important; background: #2a2b2c !important; font-size: 10px !important; }}
    [data-testid="stSidebar"] [data-testid="stTickBarMin"], [data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ display: none; }}
    [data-testid="stSidebar"] div[data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="input"] {{ background: #222325; border: 1px solid #303134; border-radius: 9px; min-height: 36px; }}
    [data-testid="stSidebar"] input {{ color: #ecebe6; font-size: 11px; caret-color: var(--accent); }}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{ color: #ecebe6; font-size: 11px; }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{ color: #aaa; }}
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {{ background: #222325; border: 1px solid #303134; border-radius: 9px; padding: 3px; }}
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {{ min-height: 29px; color: #aaa; font-size: 10px; }}
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-selected="true"] {{ background: #3a3b3c; color: #fff; }}
    [data-testid="stSidebar"] [data-baseweb="switch"] > div {{ background: #3b3c3e; }}
    [data-testid="stSidebar"] [data-baseweb="switch"] > div:has(span[aria-checked="true"]) {{ background: var(--accent); }}
    .sidebar-status {{ display: flex; align-items: center; justify-content: center; gap: 7px; color: #8f908f; font-size: 9px; margin-top: 12px; padding-top: 13px; border-top: 1px solid #2a2b2d; }}
    button:focus-visible, input:focus-visible, [role="slider"]:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 3px !important; }}
    ::selection {{ background: var(--accent); color: #171819; }}
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: #444548; border-radius: 4px; }}
    @media (max-width: 1300px) {{
        .card-meta {{ display: none; }}
        .card-title {{ font-size: 11px; }}
        .risk-card .card-title {{ font-size: 12px; }}
    }}
    @media (max-width: 1100px) {{
        section[data-testid="stSidebar"] {{ width: 280px; min-width: 280px; }}
        section[data-testid="stSidebar"] > div {{ width: 280px; }}
        [data-testid="stSidebar"] .block-container {{ padding-left: 15px; padding-right: 15px; }}
        [data-testid="stAppViewContainer"] > .main .block-container {{ padding-left: 16px; padding-right: 16px; }}
        .bento-card {{ padding: 17px; }}
        .importance-row {{ grid-template-columns: 90px 1fr 30px; gap: 8px; }}
    }}
    @media (max-height: 740px) {{
        [data-testid="stSidebarContent"] {{ overflow-y: auto; }}
        .card-head {{ margin-bottom: 10px; }}
        .risk-foot {{ margin-top: 10px; padding-top: 9px; }}
        .profile-row {{ padding: 7px 0; }}
    }}
    @media (max-width: 800px) {{
        section[data-testid="stSidebar"] {{ position: fixed; left: 0; top: 0; bottom: 0; z-index: 999999; }}
        .dashboard-header, .main [data-testid="stMarkdownContainer"]:has(.dashboard-header) {{ display: none; }}
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{ height: auto; min-height: 100dvh; overflow: auto; }}
        [data-testid="stAppViewContainer"] > .main .block-container {{ height: auto; overflow: visible; }}
        .main [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
        .main [data-testid="stColumn"] {{ flex: 1 1 100% !important; width: 100% !important; }}
        .bento-card {{ height: 320px; margin-bottom: 14px; }}
        .risk-card {{ height: 300px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

renderizar_html(
    f"""
    <header class="dashboard-header">
        <h1 class="dashboard-title">Bank Churn <span>/ Risk intelligence</span></h1>
        <div class="header-meta" title="Clase nativa del modelo: {clase_nativa}. La decisión visual usa el umbral almacenado de {umbral:.0%}."><span class="live-dot"></span> Modelo activo · {numero(len(data))} perfiles</div>
    </header>
    """,
    unsafe_allow_html=True,
)

risk_label = "RIESGO ALTO" if riesgo_alto else "RIESGO BAJO"
pais_es = {"Spain": "España", "France": "Francia", "Germany": "Alemania"}[pais]

col_riesgo, col_edad, col_historico = st.columns([1.08, 1.52, 0.9])
with col_riesgo:
    renderizar_html(
        f"""
        <section class="bento-card top-card risk-card">
            <div class="risk-top">
                <div><h2 class="card-title">Probabilidad de fuga</h2><span class="risk-badge">{risk_label} · UMBRAL {umbral:.0%}</span></div>
                {svg_riesgo(prob)}
            </div>
            <div class="risk-value">{prob * 100:.1f}<sup>%</sup></div>
            <div class="risk-foot"><span>Frente al histórico</span><strong>{delta:+.1%}</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
with col_edad:
    renderizar_html(
        f"""
        <section class="bento-card top-card chart-card">
            <div class="card-head"><h2 class="card-title">Fuga por edad</h2><span class="card-meta">{edad} años · perfil actual</span></div>
            {grafico_edad(edad)}
            <div class="chart-legend"><span class="legend-key"><i class="legend-line"></i>Media histórica</span><span class="legend-key"><i class="legend-dot"></i>Edad actual</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
with col_historico:
    renderizar_html(
        f"""
        <section class="bento-card top-card paper-card">
            <div class="card-head"><h2 class="card-title">Histórico</h2><span class="card-meta">Portfolio</span></div>
            <div class="historic-value">{tasa_historica:.1%}</div>
            <div class="historic-stats">
                <div class="historic-stat"><span>CLIENTES</span><strong>{numero(len(data))}</strong></div>
                <div class="historic-stat"><span>PRECISIÓN</span><strong>85%</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

col_paises, col_importancia, col_perfil = st.columns([1.08, 1.52, 0.9])
with col_paises:
    renderizar_html(
        f"""
        <section class="bento-card bottom-card chart-card">
            <div class="card-head"><h2 class="card-title">Fuga por país</h2><span class="card-meta">Tasa media</span></div>
            <div class="country-list">{grafico_paises()}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
with col_importancia:
    renderizar_html(
        f"""
        <section class="bento-card bottom-card chart-card">
            <div class="card-head"><h2 class="card-title">Peso en el modelo</h2><span class="card-meta">Importancia relativa</span></div>
            <div class="importance-list">{grafico_importancias()}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
with col_perfil:
    renderizar_html(
        f"""
        <section class="bento-card bottom-card profile-card">
            <div class="card-head"><h2 class="card-title">Perfil</h2><span class="card-meta">En vivo</span></div>
            <div class="profile-lead"><strong class="profile-age">{edad}</strong><span class="profile-country">años · {pais_es}</span></div>
            <dl class="profile-list">
                <div class="profile-row"><dt>Productos</dt><dd>{productos}</dd></div>
                <div class="profile-row"><dt>Score</dt><dd>{score}</dd></div>
                <div class="profile-row"><dt>Saldo</dt><dd>{numero(balance)}</dd></div>
                <div class="profile-row"><dt>Vinculación</dt><dd>{"Activo" if miembro else "Inactivo"}</dd></div>
            </dl>
        </section>
        """,
        unsafe_allow_html=True,
    )
