# -*- coding: utf-8 -*-
"""
generar_datos_demo.py
=====================
Crea CSVs FALSOS en la carpeta 'salida/' con la misma estructura que produce
preprocesar.py, para que puedas abrir el dashboard y ver como va quedando
SIN necesitar todavia tus datos reales.

    python generar_datos_demo.py
    streamlit run dashboard.py

Cuando ya tengas tus datos reales, corre preprocesar.py y sobrescribe estos.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CARPETA_SALIDA = "salida"
N = 350  # condados falsos
rng = np.random.default_rng(42)
os.makedirs(CARPETA_SALIDA, exist_ok=True)

estados = ["California", "Texas", "Florida", "New York", "Illinois",
           "Pennsylvania", "Ohio", "Georgia", "Arizona", "Washington"]

# --- metadata geografica plausible (EE.UU. continental) ---
df = pd.DataFrame({
    "ID": np.arange(N),
    "FIPS": rng.integers(1001, 56045, N),
    "Condado": [f"Condado_{i}" for i in range(N)],
    "Estado": rng.choice(estados, N),
    "Latitud": rng.uniform(25, 48, N),
    "Longitud": rng.uniform(-123, -70, N),
    "Poblacion": rng.lognormal(10.5, 1.2, N).astype(int) + 500,
})
df["Densidad"] = rng.lognormal(3.0, 1.3, N)  # hab/km2

# Tasa final: ligeramente correlada con latitud (negativa) + sesgo + outliers
base = 250 - 2.5 * df["Latitud"] + rng.normal(0, 40, N)
df["Tasa_Final"] = np.clip(base, 5, None) * rng.lognormal(0, 0.4, N)

# --- PCA: 5 componentes + tabla de varianza ---
for i in range(1, 6):
    df[f"PC{i}"] = rng.normal(0, 1, N) * (6 - i)

var = np.array([0.42, 0.23, 0.14, 0.07, 0.05, 0.035, 0.025, 0.02, 0.015, 0.01])
pca_var = pd.DataFrame({
    "Componente": np.arange(1, 11),
    "Var_Explicada": var,
    "Var_Acumulada": np.cumsum(var),
})

# --- LLE: para k = 8, 15, 30 (con cierta estructura por estado) ---
centros = {e: rng.normal(0, 5, 2) for e in estados}
for k in (8, 15, 30):
    c = np.array([centros[e] for e in df["Estado"]])
    ruido = rng.normal(0, 1 + k * 0.03, (N, 2))
    df[f"LLE1_k{k}"] = c[:, 0] + ruido[:, 0]
    df[f"LLE2_k{k}"] = c[:, 1] + ruido[:, 1]

# --- correlaciones reales sobre los datos falsos (para que cuadre la tabla) ---
filas = []
for v in ["Latitud", "Longitud", "Poblacion", "Densidad"]:
    pr, pp = pearsonr(df["Tasa_Final"], df[v])
    sr, sp = spearmanr(df["Tasa_Final"], df[v])
    filas.append({"Variable": v, "Pearson_r": pr, "Pearson_p": pp,
                  "Spearman_rho": sr, "Spearman_p": sp, "N": N})
corr = pd.DataFrame(filas)

# --- guardar ---
df.to_csv(f"{CARPETA_SALIDA}/condados_features.csv", index=False, encoding="utf-8-sig")
pca_var.to_csv(f"{CARPETA_SALIDA}/pca_varianza.csv", index=False, encoding="utf-8-sig")
corr.to_csv(f"{CARPETA_SALIDA}/correlaciones.csv", index=False, encoding="utf-8-sig")

print(f"Datos de prueba creados en '{CARPETA_SALIDA}/'. Ahora corre: streamlit run dashboard.py")
