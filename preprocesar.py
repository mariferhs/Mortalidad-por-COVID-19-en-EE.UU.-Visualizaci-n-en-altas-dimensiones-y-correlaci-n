# -*- coding: utf-8 -*-
"""
preprocesar.py
==============
Calcula PCA, LLE y correlaciones (Pearson / Spearman) sobre el dataset
COVID-19 (muertes en EE.UU.) y guarda los resultados como CSV para que el
dashboard de Streamlit los lea sin recalcular nada pesado.

Se ejecuta UNA vez (o cada que cambien los datos o los parametros):

    python preprocesar.py

Entradas:
  - El dataset ORIGINAL en formato ancho (time_series_covid_19_deaths_US.csv)

Salidas (en la carpeta CARPETA_SALIDA):
  - condados_features.csv  -> metadata + PC1..PC5 + LLE (varios k) + tasa final
  - pca_varianza.csv       -> varianza explicada y acumulada por componente
  - correlaciones.csv      -> Pearson y Spearman (con p-values) de la tasa vs cada variable
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from scipy.stats import pearsonr, spearmanr

# =========================================================
# CONFIGURACION  (edita estas rutas/nombres a tu gusto)
# =========================================================
RUTA_COVID = "time_series_covid_19_deaths_US.csv"   # dataset original (formato ancho)
RUTA_AREAS = "areas_condados.csv"                   # tu archivo de areas por condado
CARPETA_SALIDA = "salida"

# --- Como se llama lo relevante en TU archivo de areas ---
COL_FIPS_AREAS = "FIPS"      # columna con el codigo FIPS en tu archivo de areas
COL_AREA       = "Area"      # columna con el area del condado
AREA_EN_KM2    = True        # True si el area ya viene en km2; False si viene en millas2
#   (si AREA_EN_KM2=False, se convierte: 1 milla2 = 2.58999 km2)

# --- Parametros de los metodos ---
N_COMPONENTES_PCA = 10       # cuantos PCs calcular (para la grafica de varianza)
N_PCS_GUARDAR     = 5        # cuantos scores de PC guardar para graficar/explorar
KS_LLE            = [8, 15, 30]   # valores de n_neighbors de LLE a precalcular
LLE_COMPONENTES   = 2
SEMILLA           = 42

# Columnas que NO son fechas (tras renombrar). Todo lo demas se trata como fecha.
COLUMNAS_BASE = [
    "ID", "iso2", "iso3", "code3", "FIPS",
    "Condado", "Estado", "Pais",
    "Latitud", "Longitud", "Combined_Key", "Poblacion",
]


# =========================================================
# 1. CARGA Y LIMPIEZA  (misma logica de tu pipeline)
# =========================================================
def cargar_y_limpiar(ruta):
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()

    # Quitar registros que no son condados reales
    df = df[~df["Admin2"].isin(["Unassigned", "Out of State", "Out of Country"])]
    df = df.dropna(subset=["Admin2", "Province_State", "Lat", "Long_", "Population"])
    df = df[(df["Lat"] != 0) & (df["Long_"] != 0) & (df["Population"] > 0)]

    df = df.rename(columns={
        "UID": "ID", "Admin2": "Condado", "Province_State": "Estado",
        "Country_Region": "Pais", "Lat": "Latitud", "Long_": "Longitud",
        "Population": "Poblacion",
    })
    return df.reset_index(drop=True)


# =========================================================
# 2. MATRIZ DE TASA DE MORTALIDAD POR 100k (county x fecha)
# =========================================================
def construir_matriz_tasa(df):
    """
    Cada celda = (muertes acumuladas en esa fecha / poblacion) * 100000.
    Esta es la 'tasa normalizada' que justifica usar mas componentes en PCA
    y abre la puerta a LLE (estructura no lineal).
    """
    columnas_fecha = [c for c in df.columns if c not in COLUMNAS_BASE]
    muertes = df[columnas_fecha].to_numpy(dtype=float)        # acumuladas
    poblacion = df["Poblacion"].to_numpy(dtype=float)[:, None]

    matriz_tasa = muertes / poblacion * 100_000.0
    tasa_final = matriz_tasa[:, -1]                           # ultima fecha = tasa acumulada final
    return matriz_tasa, tasa_final, columnas_fecha


# =========================================================
# 4. PCA
# =========================================================
def correr_pca(matriz_tasa_estandarizada, columnas_fecha):
    n_comp = min(
        N_COMPONENTES_PCA,
        matriz_tasa_estandarizada.shape[1]
    )
    pca = PCA(
        n_components=n_comp,
        random_state=SEMILLA
    )
    scores = pca.fit_transform(
        matriz_tasa_estandarizada
    )
    var = pca.explained_variance_ratio_
    tabla_var = pd.DataFrame({
        "Componente": np.arange(1, n_comp + 1),
        "Var_Explicada": var,
        "Var_Acumulada": np.cumsum(var)
    })
    n_guardar = min(
        N_PCS_GUARDAR,
        n_comp
    )
    cols_pc = {
        f"PC{i+1}": scores[:, i]
        for i in range(n_guardar)
    }
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )
    loadings.insert(
        0,
        "Variable",
        columnas_fecha
    )
    return cols_pc, tabla_var, loadings


# =========================================================
# 5. LLE  (para varios n_neighbors)
# =========================================================
def correr_lle(matriz_estandarizada):
    cols = {}
    for k in KS_LLE:
        print(f"  - LLE con n_neighbors={k} ...")
        try:
            lle = LocallyLinearEmbedding(
                n_neighbors=k,
                n_components=LLE_COMPONENTES,
                method="standard",
                random_state=SEMILLA,
            )
            emb = lle.fit_transform(matriz_estandarizada)
        except Exception as e:
            print(f"    [AVISO] LLE k={k} fallo con 'standard' ({e}); intento 'modified'.")
            lle = LocallyLinearEmbedding(
                n_neighbors=k, n_components=LLE_COMPONENTES,
                method="modified", random_state=SEMILLA,
            )
            emb = lle.fit_transform(matriz_estandarizada)
        cols[f"LLE1_k{k}"] = emb[:, 0]
        cols[f"LLE2_k{k}"] = emb[:, 1]
    return cols


# =========================================================
# 6. CORRELACIONES Pearson + Spearman  (1 renglon por condado)
# =========================================================
def correr_correlaciones(meta):
    objetivo = "Tasa_Final"
    candidatas = ["Latitud", "Longitud", "Poblacion"]
    filas = []
    for var in candidatas:
        x = meta[objetivo]
        y = meta[var]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        pr, pp = pearsonr(x[mask], y[mask])
        sr, sp = spearmanr(x[mask], y[mask])
        filas.append({
            "Variable": var,
            "Pearson_r": pr, "Pearson_p": pp,
            "Spearman_rho": sr, "Spearman_p": sp,
            "N": int(mask.sum()),
        })
    return pd.DataFrame(filas)


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("1) Cargando y limpiando dataset...")
    df = cargar_y_limpiar(RUTA_COVID)
    print(f"   Condados despues de limpieza: {len(df)}")

    print("2) Construyendo matriz de tasa por 100k...")
    matriz_tasa, tasa_final, columnas_fecha = construir_matriz_tasa(df)
    print(f"   Matriz: {matriz_tasa.shape[0]} condados x {matriz_tasa.shape[1]} fechas")

    # Metadata base (1 renglon por condado)
    meta = df[["ID", "FIPS", "Condado", "Estado", "Latitud", "Longitud", "Poblacion"]].copy()
    meta["Tasa_Final"] = tasa_final

    # Estandarizar por columna (cada fecha) antes de PCA/LLE
    X = StandardScaler().fit_transform(matriz_tasa)

    print("4) PCA...")
    cols_pc, tabla_var, loadings = correr_pca(
    X,
    columnas_fecha
)
    cruce_90 = tabla_var.loc[tabla_var["Var_Acumulada"] >= 0.90, "Componente"]
    if len(cruce_90):
        print(f"   Se alcanza 90% de varianza con {int(cruce_90.iloc[0])} componentes.")

    print("5) LLE...")
    cols_lle = correr_lle(X)

    # Unir todo en un solo features
    for nombre, valores in {**cols_pc, **cols_lle}.items():
        meta[nombre] = valores

    print("6) Correlaciones Pearson + Spearman...")
    tabla_corr = correr_correlaciones(meta)
    print(tabla_corr.to_string(index=False))

    # Guardar
    # Guardar
    f_feat = os.path.join(CARPETA_SALIDA, "condados_features.csv")
    f_var = os.path.join(CARPETA_SALIDA, "pca_varianza.csv")
    f_corr = os.path.join(CARPETA_SALIDA, "correlaciones.csv")
    f_load = os.path.join(CARPETA_SALIDA, "pca_loadings.csv")

    meta.to_csv(
        f_feat,
        index=False,
        encoding="utf-8-sig"
    )

    tabla_var.to_csv(
        f_var,
        index=False,
        encoding="utf-8-sig"
    )

    tabla_corr.to_csv(
        f_corr,
        index=False,
        encoding="utf-8-sig"
    )

    loadings.to_csv(
        f_load,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nListo. Archivos generados:")

    for f in (f_feat, f_var, f_corr, f_load):
        print("  -", f)

if __name__ == "__main__":
    main()
