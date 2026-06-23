# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Dependencias opcionales (el dashboard funciona aunque falten) ---
try:
    from sklearn.cluster import KMeans
    HAY_SKLEARN = True
except Exception:
    HAY_SKLEARN = False

try:
    import statsmodels.api as _sm  # necesario para las líneas de tendencia
    HAY_STATSMODELS = True
except Exception:
    HAY_STATSMODELS = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAY_SCIPY = True
except Exception:
    HAY_SCIPY = False


CARPETA_SALIDA = "salida"

st.set_page_config(
    page_title="Análisis de la mortalidad por COVID-19 en EE.UU.",
    layout="wide",
    page_icon="🦠"
)
#====================================================
# CARGA DE DATOS
#====================================================
@st.cache_data
def cargar():
    feat = pd.read_csv(
        os.path.join(CARPETA_SALIDA,"condados_features.csv")
    )
    var = pd.read_csv(
        os.path.join(CARPETA_SALIDA,"pca_varianza.csv")
    )
    corr = pd.read_csv(
        os.path.join(CARPETA_SALIDA,"correlaciones.csv")
    )
    covid = pd.read_csv(
        "time_series_covid_19_deaths_US.csv"
    )
    loadings=pd.read_csv(
        os.path.join(CARPETA_SALIDA,"pca_loadings.csv")
    )
    return feat,var,corr,covid, loadings 
try:
    feat,var,corr,covid, loadings = cargar()
except FileNotFoundError:
    st.error(
        "No se encontraron todos los archivos necesarios."
    )
    st.stop()

#====================================================
# DETECTAR COLUMNAS DE FECHA
#====================================================

columnas_fecha = []
for c in covid.columns:
    if "/" in c:
        columnas_fecha.append(c)

#====================================================
# DETECTAR K DE LLE
#====================================================
ks_lle = sorted({
    int(re.search(r"k(\d+)",c).group(1))
    for c in feat.columns
    if c.startswith("LLE1_k")
})
#====================================================
# TITULO
#====================================================
st.title("🦠Análisis de la mortalidad por COVID-19 en EE.UU.🦠")
# Periodo del conjunto de datos
primera_fecha = columnas_fecha[0]
ultima_fecha = columnas_fecha[-1]
st.info(f"📅 Período de registro de los datos: **{primera_fecha}** al **{ultima_fecha}**")

# Indicadores del CSV original
total_condados = len(covid)
total_estados = covid["Province_State"].nunique()
poblacion_total = covid["Population"].sum()
muertes_totales = covid[ultima_fecha].sum()
tasa_nacional = (muertes_totales / poblacion_total) * 100000

# Tarjetas resumen
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "🏘️ Condados",
    f"{total_condados:,}"
)
c2.metric(
    "🗺️ Estados",
    total_estados
)
c3.metric(
    "👥 Población",
    f"{poblacion_total/1_000_000:.1f} M",
    help="Población total representada en el conjunto de datos."
)
c4.metric(
    "☠️ Muertes",
    f"{muertes_totales/1_000:.0f} mil",
    help="Muertes totales representadas en el conjunto de datos."
)
c5.metric(
    "🇺🇸 Mortalidad",
    f"{tasa_nacional:.0f}",
    help="Muertes acumuladas por COVID-19 por cada 100,000 habitantes durante el periodo del estudio"
)
st.subheader("Vista previa del conjunto de datos")
columnas = feat.loc[:, :"Tasa_Final"].columns
st.dataframe(
    feat[columnas].head(),
    use_container_width=True
)
#====================================================
# DATAFRAME DE TRABAJO, REGIÓN Y COLORES CONSISTENTES
#====================================================
# Las pestañas de PCA, LLE y Correlaciones trabajan sobre este DataFrame.
df = feat.copy()

# Región censal: agrupar 50+ estados en 4 regiones hace LEGIBLES los scatter.
REGIONES = {
    # Noreste
    "Connecticut": "Noreste", "Maine": "Noreste", "Massachusetts": "Noreste",
    "New Hampshire": "Noreste", "Rhode Island": "Noreste", "Vermont": "Noreste",
    "New Jersey": "Noreste", "New York": "Noreste", "Pennsylvania": "Noreste",
    # Medio Oeste
    "Illinois": "Medio Oeste", "Indiana": "Medio Oeste", "Michigan": "Medio Oeste",
    "Ohio": "Medio Oeste", "Wisconsin": "Medio Oeste", "Iowa": "Medio Oeste",
    "Kansas": "Medio Oeste", "Minnesota": "Medio Oeste", "Missouri": "Medio Oeste",
    "Nebraska": "Medio Oeste", "North Dakota": "Medio Oeste", "South Dakota": "Medio Oeste",
    # Sur
    "Delaware": "Sur", "Florida": "Sur", "Georgia": "Sur", "Maryland": "Sur",
    "North Carolina": "Sur", "South Carolina": "Sur", "Virginia": "Sur",
    "District of Columbia": "Sur", "West Virginia": "Sur", "Alabama": "Sur",
    "Kentucky": "Sur", "Mississippi": "Sur", "Tennessee": "Sur", "Arkansas": "Sur",
    "Louisiana": "Sur", "Oklahoma": "Sur", "Texas": "Sur",
    # Oeste
    "Arizona": "Oeste", "Colorado": "Oeste", "Idaho": "Oeste", "Montana": "Oeste",
    "Nevada": "Oeste", "New Mexico": "Oeste", "Utah": "Oeste", "Wyoming": "Oeste",
    "Alaska": "Oeste", "California": "Oeste", "Hawaii": "Oeste", "Oregon": "Oeste",
    "Washington": "Oeste",
}
if "Estado" in df.columns:
    df["Region"] = df["Estado"].map(REGIONES).fillna("Otro")

# Paleta fija → el mismo color significa la misma región en TODAS las pestañas.
COLOR_REGION = {
    "Noreste": "#4C78A8",
    "Medio Oeste": "#F58518",
    "Sur": "#E45756",
    "Oeste": "#54A24B",
    "Otro": "#BAB0AC",
}
ORDEN_REGION = ["Noreste", "Medio Oeste", "Sur", "Oeste", "Otro"]


def varianza_pct(pc_name):
    """% de varianza explicada de una componente 'PCk' (None si no se puede)."""
    try:
        num = int(re.sub(r"\D", "", pc_name))
        fila = var.loc[var["Componente"] == num, "Var_Explicada"]
        if not len(fila):
            return None
        v = float(fila.iloc[0])
        return v * 100 if v <= 1 else v
    except Exception:
        return None
#====================================================
# TABS
#====================================================
tab_estadisticas,\
tab_dist,\
tab_rank,\
tab_e_t,\
tab_pca,\
tab_lle,\
tab_corr = st.tabs(
[
    "📈 Estadísticas",
    "📉 Distribución",
    "🏆 Rankings",
    "🗺️ Espacio-Temporal",
    "📉 PCA",
    "🌀 LLE",
    "🔗 Correlaciones"
]
)
#====================================================
# ESTADÍSTICAS DESCRIPTIVAS
#====================================================
from scipy.stats import skew

with tab_estadisticas:
    st.header("Estadísticos descriptivos por estado")
    ultima_fecha = columnas_fecha[-1]
    
    df_covid = (
        covid
        .groupby("Province_State")
        .agg({
            ultima_fecha: "sum",
            "Population": "sum"
        })
        .reset_index()
    )
    df_covid = df_covid.rename(columns={
        "Province_State": "Estado",
        ultima_fecha: "Muertes",
        "Population": "Poblacion"
    })
    df_covid = df_covid[df_covid["Poblacion"] > 0]
    df_covid["Tasa_Mortalidad"] = (
        df_covid["Muertes"] /
        df_covid["Poblacion"]
    ) * 100000
    variables_num = [
        "Muertes",
        "Poblacion",
        "Tasa_Mortalidad"
    ]
    variable = st.selectbox(
        "Selecciona una variable",
        variables_num
    )
    serie = df_covid[variable]

    cv = 100 * serie.std() / serie.mean()
    asimetria = skew(serie)
    iqr = serie.quantile(0.75) - serie.quantile(0.25)

    c1, c2, c3 = st.columns(3)
    c1.metric("Media", f"{serie.mean():,.2f}")
    c2.metric("Mediana", f"{serie.median():,.2f}")
    c3.metric("Desv. Est.", f"{serie.std():,.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Coef. Variación", f"{cv:.2f}%")
    c5.metric("Asimetría", f"{asimetria:.2f}")
    c6.metric("IQR", f"{iqr:,.2f}")
    
    st.subheader("Interpretación")
    if variable == "Muertes":
        st.markdown(f"""
    **Análisis de las muertes acumuladas por COVID-19**
    La media de **{serie.mean():,.2f}** defunciones indica el promedio de muertes registradas por estado, mientras que la mediana de **{serie.median():,.2f}** sugiere que la mitad de los estados presenta un número de defunciones inferior a este valor.
    La desviación estándar (**{serie.std():,.2f}**) y el coeficiente de variación (**{cv:.2f}%**) muestran que existe {'alta' if cv>=50 else 'moderada' if cv>=20 else 'baja'} variabilidad en el número de fallecimientos entre los estados, lo que refleja que el impacto de la pandemia no fue uniforme en todo el país.
    La asimetría de **{asimetria:.2f}** {'indica que unos pocos estados concentraron una cantidad considerablemente mayor de defunciones que la mayoría.' if asimetria>0 else 'indica que algunos estados presentan valores inferiores respecto al resto.'}
    El IQR de **{iqr:,.2f}** representa el rango donde se encuentra el 50 % central de los estados, mostrando la dispersión de los valores típicos y reduciendo la influencia de estados con cifras extremadamente altas.
    """)
    elif variable == "Poblacion":
        st.markdown(f"""
    **Análisis de la población estatal**
    La media de **{serie.mean():,.2f}** habitantes representa el tamaño poblacional promedio de los estados incluidos en el conjunto de datos, mientras que la mediana (**{serie.median():,.2f}**) evidencia que la distribución poblacional no es completamente uniforme.
    El coeficiente de variación (**{cv:.2f}%**) muestra una {'alta' if cv>=50 else 'moderada' if cv>=20 else 'baja'} diferencia en el tamaño poblacional entre los estados, lo cual es esperado debido a que entidades como California, Texas o Florida concentran una población considerablemente mayor que otras.
    La asimetría de **{asimetria:.2f}** {'indica que pocos estados poseen poblaciones muy superiores al promedio.' if asimetria>0 else 'sugiere una concentración de estados con poblaciones relativamente altas.'}
    El IQR de **{iqr:,.2f}** refleja la dispersión del 50 % central de las poblaciones estatales.
    """)
    else:
        st.markdown(f"""
    **Análisis de la tasa de mortalidad por COVID-19**
    La tasa promedio de mortalidad es de **{serie.mean():,.2f}** fallecimientos por cada 100,000 habitantes, mientras que la mediana (**{serie.median():,.2f}**) representa el valor central entre los estados.
    El coeficiente de variación (**{cv:.2f}%**) indica que la tasa de mortalidad presenta una {'alta' if cv>=50 else 'moderada' if cv>=20 else 'baja'} variabilidad, lo que sugiere que el riesgo de fallecimiento asociado a la pandemia fue diferente entre las entidades.
    La asimetría de **{asimetria:.2f}** {'muestra que algunos estados registraron tasas de mortalidad considerablemente superiores al resto, posiblemente influenciadas por factores demográficos, sanitarios o de acceso a servicios de salud.' if asimetria>0 else 'indica que la mayoría de los estados presenta tasas relativamente elevadas respecto a unos pocos con valores bajos.'}
    El IQR de **{iqr:,.2f}** describe la variabilidad de la mitad central de las tasas de mortalidad y permite evaluar la dispersión sin que los valores extremos tengan una influencia significativa.
    """)
#====================================================
# DISTRIBUCIÓN DE LA TASA DE MORTALIDAD
#====================================================
with tab_dist:
    st.header("Distribución de la tasa de mortalidad")
    fig = px.histogram(
        df_covid,
        x="Tasa_Mortalidad",
        nbins=12,
        marginal="box",
        color_discrete_sequence=["#4C78A8"],
        labels={
            "Tasa_Mortalidad": "Tasa por cada 100,000 habitantes"
        }
    )
    fig.update_layout(
        xaxis_title="Tasa de mortalidad (por cada 100,000 habitantes)",
        yaxis_title="Número de estados"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    q1 = df_covid["Tasa_Mortalidad"].quantile(0.25)
    q3 = df_covid["Tasa_Mortalidad"].quantile(0.75)
    st.info(f"""
    - La mayor concentración de estados presenta tasas de mortalidad cercanas al intervalo comprendido entre **{q1:.1f}** y **{q3:.1f}** fallecimientos por cada 100,000 habitantes.
    - El 50 % central de los estados se encuentra dentro de este rango, como se observa en el boxplot.
    - La mediana divide a los estados en dos grupos de igual tamaño, indicando que aproximadamente la mitad presenta tasas inferiores y la otra mitad superiores.
    - La distribución permite identificar si existen estados con tasas de mortalidad considerablemente más altas o más bajas que la mayoría.
    """)
#====================================================
# RANKINGS
#====================================================
with tab_rank:
    col1, col2 = st.columns(2)
    #====================================================
    # TOP 10 TASA PROMEDIO DE MORTALIDAD
    #====================================================
    with col1:
        st.subheader("Top 10 estados con mayor tasa promedio de mortalidad")
        datos_tasa = (
            feat
            .groupby("Estado")["Tasa_Final"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            datos_tasa,
            x="Tasa_Final",
            y="Estado",
            color="Tasa_Final",
            orientation="h",
            labels={
                "Tasa_Final": "Tasa por cada 100 000 habitantes",
                "Estado": "Estado"
            }
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    #====================================================
    # TOP 10 ESTADOS CON MAYOR POBLACIÓN
    #====================================================
    with col2:
        st.subheader("Top 10 estados con mayor población")
        datos_pob = (
            feat
            .groupby("Estado")["Poblacion"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            datos_pob,
            x="Poblacion",
            y="Estado",
            color="Poblacion",
            orientation="h",
            labels={
                "Poblacion": "Población",
                "Estado": "Estado"
            }
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

#====================================================
# FUNCIONES PARA SERIES DE TIEMPO
#====================================================
@st.cache_data
def obtener_series_estado(covid):
    fechas = [c for c in covid.columns if "/" in c]
    estado = covid.groupby("Province_State")[fechas].sum()
    return estado

@st.cache_data
def obtener_series_condado(covid):

    fechas = [c for c in covid.columns if "/" in c]

    return covid, fechas


series_estado = obtener_series_estado(covid)

covid_condados, fechas = obtener_series_condado(covid)


# =========================================================
# TAB PCA
# =========================================================
with tab_pca:
    st.subheader("Diagnóstico de la reducción de dimensionalidad")
    st.caption(
        "Las variables se estandarizan (media 0, varianza 1) antes del PCA, "
        "porque PCA es sensible a la escala de cada variable."
    )

    n90 = var.loc[var["Var_Acumulada"] >= 0.90, "Componente"]
    n90 = int(n90.iloc[0]) if len(n90) else None

    fig = px.line(
        var, x="Componente", y="Var_Acumulada", markers=True,
        title="Varianza acumulada"
    )
    fig.add_hline(y=0.90, line_dash="dash", annotation_text="90 %")
    if n90:
        fig.add_vline(x=n90, line_dash="dot", annotation_text=f"{n90} componentes")
    st.plotly_chart(fig, use_container_width=True, key="pca_var_acumulada")

    if n90:
        st.info(
            f"Se necesitan **{n90} componentes** para explicar el **90 %** de la varianza. "
            "Esto indica que la información del conjunto de datos no puede resumirse en una "
            "o dos dimensiones, por lo que el análisis se complementa después con **LLE**."
        )

    # Solo trabajamos con las tres primeras componentes
    pcs = [c for c in ["PC1", "PC2", "PC3"] if c in df.columns]

    # ------------------------------------------------------------------
    # 1) Proyección de los condados
    # ------------------------------------------------------------------
    st.subheader("Proyección de los condados en el espacio PCA")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    eje_x = c1.selectbox("Eje X", pcs, index=0)
    eje_y = c2.selectbox("Eje Y", pcs, index=1 if len(pcs) > 1 else 0)
    modo_color = c3.radio(
        "Colorear por", ["Región", "Tasa de mortalidad"],
        horizontal=True, key="pca_color"
    )

    if modo_color == "Región" and "Region" in df.columns:
        fig = px.scatter(
            df, x=eje_x, y=eje_y, color="Region",
            color_discrete_map=COLOR_REGION, category_orders={"Region": ORDEN_REGION},
            hover_data=["Condado", "Estado", "Tasa_Final"], opacity=0.7,
            title=f"{eje_x} vs {eje_y}",
        )
    else:
        fig = px.scatter(
            df, x=eje_x, y=eje_y, color="Tasa_Final",
            color_continuous_scale="Reds",
            hover_data=["Condado", "Estado", "Tasa_Final"], opacity=0.7,
            title=f"{eje_x} vs {eje_y}",
        )

    # Tres condados extremos, marcados con un diamante
    sub = df[[eje_x, eje_y, "Condado", "Estado", "Tasa_Final"]].dropna()
    extremos = pd.DataFrame()
    if len(sub):
        cx0 = sub[eje_x] - sub[eje_x].mean()
        cy0 = sub[eje_y] - sub[eje_y].mean()
        sub = sub.assign(Distancia=np.sqrt(cx0 ** 2 + cy0 ** 2))
        extremos = sub.nlargest(3, "Distancia")
        fig.add_scatter(
            x=extremos[eje_x], y=extremos[eje_y],
            mode="markers+text", text=extremos["Condado"],
            textposition="top center",
            marker=dict(size=13, color="black", symbol="diamond",
                        line=dict(width=1, color="white")),
            name="Condados extremos", showlegend=False,
        )
    st.plotly_chart(fig, use_container_width=True)

    # Caption de varianza mejorado
    vx, vy = varianza_pct(eje_x), varianza_pct(eje_y)
    if vx is not None and vy is not None:
        total = vx + vy
        st.caption(
            f"📊 **{eje_x}** explica el **{vx:.1f} %** de la varianza y **{eje_y}** el "
            f"**{vy:.1f} %**: juntos resumen **{total:.1f} %** del total, dejando fuera "
            f"**{100 - total:.1f} %**. Por eso esta vista 2D es una aproximación, no la "
            f"imagen completa. Los ◆ marcan los condados más extremos, que más influyen "
            f"en estas componentes."
        )
    else:
        st.caption("Los ◆ marcan los condados más extremos.")


    # ------------------------------------------------------------------
    # 3) Loadings: PC1, PC2 y PC3 a lo largo del tiempo
    # ------------------------------------------------------------------
    st.subheader("¿Qué momento de la pandemia resume cada componente?")
    load_t = loadings.copy()
    load_t["Fecha"] = pd.to_datetime(load_t["Variable"], format="%m/%d/%y", errors="coerce")
    load_t = load_t.sort_values("Fecha")
    pcs_load = [c for c in ["PC1", "PC2", "PC3"] if c in loadings.columns]
    if pcs_load:
        largo = load_t.melt(
            id_vars="Fecha", value_vars=pcs_load,
            var_name="Componente", value_name="Loading"
        )
        fig = px.line(
            largo, x="Fecha", y="Loading", color="Componente",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Peso (loading) de cada componente a lo largo del tiempo",
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        # Franjas que marcan las grandes olas de la pandemia
        olas = [
            ("2020-03-01", "2020-05-31", "gray",   0.10, "Primera ola",  "top left"),
            ("2020-11-01", "2021-02-28", "red",    0.07, "Invierno 2020", "top right"),
        ]
        for x0, x1, color, op, etiqueta, pos in olas:
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=color, opacity=op, line_width=0,
                annotation_text=etiqueta, annotation_position=pos,
                annotation_font_size=11,
            )

        fig.update_traces(line_width=2.5)
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Componente",
            xaxis_title="Fecha",
            yaxis_title="Loading (peso)",
            margin=dict(t=60),
        )
        st.plotly_chart(fig, use_container_width=True)



    # ------------------------------------------------------------------
    # 4) Interpretación: correlación PC vs variables originales (heatmap)
    # ------------------------------------------------------------------
    st.subheader("¿Qué representa cada componente?")
    vars_interp = [v for v in ["Tasa_Final", "Latitud", "Longitud", "Poblacion", "Densidad"]
                   if v in df.columns]
    if vars_interp and pcs:
        filas = []
        for pc in pcs:
            fila = {"Componente": pc}
            for v in vars_interp:
                fila[v] = df[pc].corr(df[v])
            filas.append(fila)
        corr_pc = pd.DataFrame(filas).set_index("Componente")
        fig = px.imshow(
            corr_pc, text_auto=".1f", color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Correlación entre componentes y variables originales",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cercano a **1**: relación positiva fuerte · cercano a **−1**: relación "
            "negativa fuerte · cercano a **0**: poca relación. Una correlación alta con "
            "Tasa_Final significa que la componente resume la magnitud de la mortalidad; "
            "con Latitud/Longitud, que captura estructura geográfica."
        )

# =========================================================
# TAB LLE
# =========================================================
with tab_lle:

    st.subheader("Embedding no lineal (LLE)")

    st.info(
        "LLE (Locally Linear Embedding) reduce la dimensionalidad preservando la "
        "vecindad local entre los condados. Si dos puntos aparecen cercanos, "
        "significa que tuvieron trayectorias de mortalidad similares durante la "
        "pandemia, aunque la relación entre ellas no sea lineal."
    )

    # Se utiliza el embedding con k = 15
    cx = "LLE1_k15"
    cy = "LLE2_k15"

    fig = px.scatter(
        df,
        x=cx,
        y=cy,
        color="Tasa_Final",
        color_continuous_scale="Viridis",
        hover_data=[
            "Condado",
            "Estado",
            "Tasa_Final"
        ],
        opacity=0.75,
        title="LLE (k = 15)"
    )

    fig.update_layout(
        xaxis_title="LLE 1",
        yaxis_title="LLE 2",
        coloraxis_colorbar_title="Tasa final"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Cada punto representa un condado. Los puntos cercanos tuvieron trayectorias "
        "de mortalidad similares durante la pandemia. Se observa un gradiente de "
        "color, donde los condados con mayor tasa de mortalidad tienden a "
        "concentrarse hacia un extremo del embedding, mientras que los de menor "
        "mortalidad se ubican en la zona opuesta. Esto indica que LLE preserva "
        "patrones locales relevantes de la evolución temporal de la mortalidad."
    )
# =========================================================
# TAB CORRELACIÓN
# =========================================================
with tab_corr:
    st.subheader("Pearson vs Spearman")
    st.markdown(
        "Comparamos dos formas de medir la relación entre variables: **Pearson** "
        "detecta relaciones lineales y **Spearman** relaciones monótonas, siendo "
        "más robusta a valores atípicos. Donde ambas difieren, hay no-linealidad u "
        "outliers."
    )

    # Matrices Pearson y Spearman lado a lado
    vars_num = [v for v in ["Tasa_Final", "Latitud", "Longitud", "Poblacion", "Densidad"]
                if v in df.columns and df[v].notna().any()]
    cizq, cder = st.columns(2)
    with cizq:
        m = df[vars_num].corr(method="pearson")
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Matriz Pearson (lineal)")
        st.plotly_chart(fig, use_container_width=True)
    with cder:
        m = df[vars_num].corr(method="spearman")
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Matriz Spearman (monótona)")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB ESPACIO-TIEMPO
# =========================================================
with tab_e_t:
    st.header("Mapa de la tasa de mortalidad por COVID-19")
    # -----------------------------------------
    # TASA NACIONAL DE MORTALIDAD
    # -----------------------------------------
    datos_original = pd.read_csv("time_series_covid_19_deaths_US.csv")
    ultima_fecha = datos_original.columns[-1]
    tasa_estatal = (
        datos_original
        .groupby("Province_State")
        .agg({
            ultima_fecha: "sum",
            "Population": "sum"
        })
        .reset_index()
    )

    tasa_estatal["Tasa_Estatal_Real"] = (
        tasa_estatal[ultima_fecha]
        / tasa_estatal["Population"]
    ) * 100000

    tasa_estatal = tasa_estatal.rename(columns={
        "Province_State": "Estado",
        ultima_fecha: "Muertes",
        "Population": "Poblacion"
    })


    abreviaturas = {
    "Alabama":"AL",
    "Alaska":"AK",
    "Arizona":"AZ",
    "Arkansas":"AR",
    "California":"CA",
    "Colorado":"CO",
    "Connecticut":"CT",
    "Delaware":"DE",
    "Florida":"FL",
    "Georgia":"GA",
    "Hawaii":"HI",
    "Idaho":"ID",
    "Illinois":"IL",
    "Indiana":"IN",
    "Iowa":"IA",
    "Kansas":"KS",
    "Kentucky":"KY",
    "Louisiana":"LA",
    "Maine":"ME",
    "Maryland":"MD",
    "Massachusetts":"MA",
    "Michigan":"MI",
    "Minnesota":"MN",
    "Mississippi":"MS",
    "Missouri":"MO",
    "Montana":"MT",
    "Nebraska":"NE",
    "Nevada":"NV",
    "New Hampshire":"NH",
    "New Jersey":"NJ",
    "New Mexico":"NM",
    "New York":"NY",
    "North Carolina":"NC",
    "North Dakota":"ND",
    "Ohio":"OH",
    "Oklahoma":"OK",
    "Oregon":"OR",
    "Pennsylvania":"PA",
    "Rhode Island":"RI",
    "South Carolina":"SC",
    "South Dakota":"SD",
    "Tennessee":"TN",
    "Texas":"TX",
    "Utah":"UT",
    "Vermont":"VT",
    "Virginia":"VA",
    "Washington":"WA",
    "West Virginia":"WV",
    "Wisconsin":"WI",
    "Wyoming":"WY",
    "District of Columbia":"DC"
}
    tasa_estatal["Codigo"] = tasa_estatal["Estado"].map(abreviaturas)

    # -----------------------------------------
    # MAPA ANIMADO POR FECHA
    # -----------------------------------------
    # Se construye una "foto" del mapa por cada mes. Como las muertes son
    # acumuladas, tomamos la última fecha disponible de cada mes.
    fechas_dt = pd.to_datetime(columnas_fecha, format="%m/%d/%y")
    ref_fechas = pd.DataFrame({"col": columnas_fecha, "fecha": fechas_dt})
    ref_fechas["periodo"] = ref_fechas["fecha"].dt.to_period("M")
    fechas_animacion = ref_fechas.groupby("periodo")["col"].last().tolist()

    poblacion_estado = datos_original.groupby("Province_State")["Population"].sum()

    marcos = []
    for col in fechas_animacion:
        muertes_col = datos_original.groupby("Province_State")[col].sum()
        tmp = pd.DataFrame({
            "Estado": muertes_col.index,
            "Muertes": muertes_col.values,
            "Poblacion": poblacion_estado.reindex(muertes_col.index).values,
        })
        tmp = tmp[tmp["Poblacion"] > 0].copy()
        tmp["Tasa"] = tmp["Muertes"] / tmp["Poblacion"] * 100000
        tmp["Fecha"] = pd.to_datetime(col, format="%m/%d/%y").strftime("%Y-%m")
        marcos.append(tmp)

    df_animado = pd.concat(marcos, ignore_index=True)
    df_animado["Codigo"] = df_animado["Estado"].map(abreviaturas)
    df_animado = df_animado.sort_values("Fecha")

    fig = px.choropleth(
        df_animado,
        locations="Codigo",
        locationmode="USA-states",
        color="Tasa",
        scope="usa",
        color_continuous_scale="Reds",
        hover_name="Estado",
        animation_frame="Fecha",
        # Escala de color FIJA: hace comparables todas las fechas y deja ver
        # cómo el mapa se "enciende" a lo largo del tiempo.
        range_color=(0, df_animado["Tasa"].max()),
        hover_data={
            "Muertes": ":,",
            "Poblacion": ":,",
            "Tasa": ":.2f",
            "Codigo": False
        }
    )
    fig.update_layout(
        height=700,
        coloraxis_colorbar_title="Tasa por<br>100,000"
    )
    # Velocidad de la animación (ms por cuadro). Opcional.
    try:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    except (IndexError, KeyError, TypeError):
        pass

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.caption(
        "Usa el botón ▶ para reproducir la evolución, o arrastra la barra de "
        "fechas para moverte mes a mes. La escala de color es fija para que "
        "puedas comparar entre fechas."
    )

    # -----------------------------------------
    # EVOLUCIÓN NACIONAL DE MUERTES
    # -----------------------------------------
    st.subheader("Evolución nacional de muertes acumuladas")
    # Suma las muertes de todos los condados para cada fecha
    serie_nacional = covid_condados[fechas].sum(axis=0)
    datos_nacional = pd.DataFrame({
        "Fecha": pd.to_datetime(fechas),
        "Muertes": serie_nacional.values
    })
    fig = px.line(
        datos_nacional,
        x="Fecha",
        y="Muertes",
        title="Estados Unidos - Muertes acumuladas"
    )
    fig.update_traces(line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # INCREMENTO MENSUAL DE MUERTES
    # -----------------------------------------
    st.subheader("Incremento mensual de muertes")
    datos_mes = (
        datos_nacional
        .set_index("Fecha")
        .resample("ME")   # Usa "M" si tu versión de pandas no acepta "ME"
        .last()
    )
    datos_mes["Incremento"] = datos_mes["Muertes"].diff()
    # Elimina el primer registro (NaN)
    datos_mes = datos_mes.dropna()
    fig = px.line(
        datos_mes,
        x=datos_mes.index,
        y="Incremento",
        title="Incremento mensual de muertes",
        markers=True,
        labels={
            "x": "Mes",
            "Incremento": "Incremento de muertes"
        }
    )
    fig.update_traces(line_width=3)
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Incremento mensual"
    )
    st.plotly_chart(fig, use_container_width=True)
