
# -*- coding: utf-8 -*-
"""
Dashboard interactivo COVID-19

Incluye:
✔ Estadísticos descriptivos
✔ Histogramas
✔ Boxplots
✔ Rankings
✔ Series de tiempo
✔ Comparación entre estados
✔ PCA
✔ LLE
✔ Correlaciones
✔ Mapa
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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
# SIDEBAR
#====================================================

st.sidebar.header("Filtros Globales")
estados = sorted(
    feat["Estado"].dropna().unique()
)
sel_estados = st.sidebar.multiselect(
    "Estados",
    estados,
    default=[]

)
color_por = st.sidebar.selectbox(
    "Colorear por",
    [
        "Estado",
        "Tasa_Final",
        "Poblacion",
        "Latitud"
    ]
)
if len(sel_estados)==0:
    df = feat.copy()
else:
    df = feat[
        feat["Estado"].isin(sel_estados)
    ]

#====================================================
# TITULO
#====================================================
st.title("🦠Análisis de la mortalidad por COVID-19 en EE.UU.🦠")

# Periodo del conjunto de datos
primera_fecha = columnas_fecha[0]
ultima_fecha = columnas_fecha[-1]

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
    f"{len(df):,}",
    help="Número de condados analizados."
)
c2.metric(
    "🗺️ Estados",
    df["Estado"].nunique(),
    help="Número de estados representados."
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
columnas = df.loc[:, :"Tasa_Final"].columns
st.dataframe(
    df[columnas].head(),
    use_container_width=True
)

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

    df_estado = (
        covid
        .groupby("Province_State")
        .agg({
            ultima_fecha: "sum",
            "Population": "sum"
        })
        .reset_index()
    )

    df_estado = df_estado.rename(columns={
        "Province_State": "Estado",
        ultima_fecha: "Muertes",
        "Population": "Poblacion"
    })

    df_estado["Tasa_Mortalidad"] = (
        df_estado["Muertes"] /
        df_estado["Poblacion"]
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

    serie = df_estado[variable]

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

    st.divider()

    resumen = pd.DataFrame({
        "Estadístico": [
            "Mínimo",
            "Q1",
            "Mediana",
            "Q3",
            "Máximo",
            "Rango",
            "Varianza"
        ],
        "Valor": [
            serie.min(),
            serie.quantile(0.25),
            serie.median(),
            serie.quantile(0.75),
            serie.max(),
            serie.max() - serie.min(),
            serie.var()
        ]
    })

    st.dataframe(
        resumen.round(2),
        use_container_width=True
    )


#====================================================
# DISTRIBUCIÓN DE LA TASA DE MORTALIDAD
#====================================================

with tab_dist:
    st.header("Distribución de la tasa de mortalidad")
    fig = px.histogram(
        df_estado,
        x="Tasa_Mortalidad",
        nbins=12,
        marginal="box",
        title="Distribución de la tasa de mortalidad por estado"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

#====================================================
# RANKINGS
#====================================================

with tab_rank:
    st.header("Rankings")
    ranking = st.selectbox(
        "Selecciona el ranking",
        [
            "Top 10 Estados con mayor tasa promedio",
            "Top 10 Estados con mayor población",
        ]
    )

    #------------------------------------------------
    # ESTADOS MAYOR TASA
    #------------------------------------------------
    if ranking=="Top 10 Estados con mayor tasa promedio":
        datos = (
            df
            .groupby("Estado")["Tasa_Final"]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            datos,
            x="Tasa_Final",
            y="Estado",
            color="Tasa_Final",
            orientation="h",
            title="Estados con mayor tasa promedio"
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.dataframe(
            datos,
            use_container_width=True
        )


    #------------------------------------------------
    # ESTADOS MAS POBLADOS
    #------------------------------------------------
    elif ranking=="Top 10 Estados con mayor población":
        datos = (
            df
            .groupby("Estado")["Poblacion"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            datos,
            x="Poblacion",
            y="Estado",
            color="Poblacion",
            orientation="h",
            title="Estados con mayor población"
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.dataframe(
            datos,
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


#====================================================
# EVOLUCIÓN NACIONAL DE MUERTES
#====================================================

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


#====================================================
# INCREMENTO MENSUAL DE MUERTES
#====================================================
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

# =========================================================
# TAB PCA
# =========================================================
with tab_pca:
    st.subheader("Diagnóstico de varianza (¿la estructura es lineal?)")
    n90 = var.loc[var["Var_Acumulada"] >= 0.90, "Componente"]
    n90 = int(n90.iloc[0]) if len(n90) else None

    cizq, cder = st.columns(2)
    with cizq:
        fig = px.bar(var, x="Componente", y="Var_Explicada",
                     title="Varianza explicada por componente (scree)")
        st.plotly_chart(fig, use_container_width=True)
    with cder:
        fig = px.line(var, x="Componente", y="Var_Acumulada", markers=True,
                      title="Varianza acumulada")
        fig.add_hline(y=0.90, line_dash="dash", annotation_text="90%")
        if n90:
            fig.add_vline(x=n90, line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)

    if n90:
        st.info(f"Se necesitan **{n90} componentes** para explicar el 90% de la "
                f"varianza. Que no baste 1–2 indica que la estructura **no es "
                f"puramente lineal** → se justifica LLE.")

    st.subheader("Proyección de condados en el espacio de componentes")
    pcs = [c for c in df.columns if re.fullmatch(r"PC\d+", c)]
    cx, cy = st.columns(2)
    eje_x = cx.selectbox("Eje X", pcs, index=0)
    eje_y = cy.selectbox("Eje Y", pcs, index=1 if len(pcs) > 1 else 0)
    fig = px.scatter(
        df,
        x=eje_x,
        y=eje_y,
        color=color_por,
        hover_data=["Condado", "Estado", "Tasa_Final"],
        opacity=0.7,
        title=f"{eje_x} vs {eje_y}",
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# TAB LLE
# =========================================================
with tab_lle:
    st.subheader("Embedding no lineal (LLE)")
    if not ks_lle:
        st.warning("No hay columnas de LLE en los datos.")
    else:
        k = st.select_slider("n_neighbors (vecinos locales)", options=ks_lle,
                             value=ks_lle[len(ks_lle) // 2])
        cx, cy = f"LLE1_k{k}", f"LLE2_k{k}"
        fig = px.scatter(
            df, x=cx, y=cy, color=color_por,
            hover_data=["Condado", "Estado", "Tasa_Final"],
            opacity=0.7, title=f"LLE (k={k}) — patrones locales / no lineales",
        )
        fig.update_layout(xaxis_title="LLE 1", yaxis_title="LLE 2")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("LLE preserva la vecindad local: condados con trayectorias de "
                   "mortalidad parecidas quedan cerca, aunque la relación global "
                   "sea no lineal. Cambia *k* para ver qué tan estable es la estructura.")


# =========================================================
# TAB CORRELACIÓN
# =========================================================
with tab_corr:
    st.subheader("Pearson (lineal) vs Spearman (monótona, robusta a outliers)")

    st.dataframe(
        corr.style.format({
            "Pearson_r": "{:.3f}", "Pearson_p": "{:.2e}",
            "Spearman_rho": "{:.3f}", "Spearman_p": "{:.2e}",
        }),
        use_container_width=True,
    )

    # Matrices de correlación calculadas en vivo sobre el subconjunto filtrado
    vars_num = [v for v in ["Tasa_Final", "Latitud", "Longitud", "Poblacion", "Densidad"]
                if v in df.columns and df[v].notna().any()]
    cizq, cder = st.columns(2)
    with cizq:
        m = df[vars_num].corr(method="pearson")
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Matriz Pearson")
        st.plotly_chart(fig, use_container_width=True)
    with cder:
        m = df[vars_num].corr(method="spearman")
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Matriz Spearman")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dispersión: tasa de mortalidad vs variable")
    opciones = [v for v in ["Latitud", "Densidad", "Poblacion", "Longitud"] if v in vars_num]
    var_sel = st.selectbox("Variable", opciones)
    datos = df.dropna(subset=[var_sel, "Tasa_Final"])
    fig = px.scatter(datos, x=var_sel, y="Tasa_Final", trendline="ols",
                     opacity=0.5, hover_data=["Condado", "Estado"],
                     title=f"Tasa_Final vs {var_sel}")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("La distribución de muertes por COVID es muy sesgada y con outliers, "
               "por eso Spearman es el contraste honesto frente a Pearson.")


# =========================================================
# TAB MAPA
# =========================================================

with tab_e_t:

    st.header("Mapa de mortalidad por estado")

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
    # MAPA
    # -----------------------------------------
    fig = px.choropleth(
        tasa_estatal,
        locations="Codigo",
        locationmode="USA-states",
        color="Tasa_Estatal_Real",
        scope="usa",
        color_continuous_scale="Reds",
        hover_name="Estado",
        hover_data={
            "Muertes": ":,",
            "Poblacion": ":,",
            "Tasa_Estatal_Real": ":.2f"
        }
    )
    fig.update_layout(
    height=700,
    coloraxis_colorbar_title="Tasa por<br>100,000"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )