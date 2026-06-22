# Dashboard COVID-19 · PCA · LLE · Correlación
Reyna Alvarez Brandon Yire
Hernandez Sosol Maria Fernanda


Tres piezas:

| Archivo | Qué hace |
|---|---|
| `preprocesar.py` | Limpia, calcula PCA + LLE + correlaciones y guarda CSVs en `salida/`. Se corre **una vez**. |
| `dashboard.py` | App de Streamlit que lee esos CSVs y los muestra interactivos. |
| `requirements.txt` | Dependencias. |

## 1. Instalar

```bash
pip install -r requirements.txt
```

## 2. Preparar archivos de entrada

En la misma carpeta necesitas:

1. **`time_series_covid_19_deaths_US.csv`** — el dataset ORIGINAL en formato ancho
   (el mismo que limpiaste; no la versión larga/Looker). El script reaplica tu
   misma limpieza para reconstruir la matriz `condado × 494 fechas`.

2. **`areas_condados.csv`** — tu archivo de áreas. Solo necesita dos columnas:

   ```
   FIPS,Area
   1001,1539.6
   1003,4117.5
   ...
   ```

   - `FIPS` = código FIPS del condado (sirve para el cruce; maneja `1001` o `01001`).
   - `Area` = área del condado. Si está en **millas²** pon `AREA_EN_KM2 = False`
     arriba en `preprocesar.py` y se convierte sola a km².
   - Si tus columnas se llaman distinto, ajusta `COL_FIPS_AREAS` y `COL_AREA`.
   - Acepta `.csv` o `.xlsx`.

   > La densidad sale de `Población / Área` (hab/km²). Sin este archivo, todo lo
   > demás funciona, pero las correlaciones con densidad quedan vacías.

## 3. Correr

```bash
python preprocesar.py      # genera salida/condados_features.csv, pca_varianza.csv, correlaciones.csv
streamlit run dashboard.py # abre el dashboard en el navegador
```

## Notas metodológicas

- **PCA** se hace sobre la **tasa de mortalidad por 100k estandarizada** (no sobre
  muertes acumuladas crudas). Con datos crudos, PC1 explica >95% por puro tamaño y
  no hay historia; con la tasa normalizada aparecen ~5 componentes para 90%, que es
  justo el diagnóstico que te llevó a LLE.
- **LLE** se precalcula para varios `n_neighbors` (`KS_LLE` en el script). El
  dashboard te deja cambiar `k` con un slider para ver qué tan estable es la
  estructura local.
- **Correlaciones** se calculan sobre **un renglón por condado** (la tasa acumulada
  final vs latitud, densidad, población, longitud). Pearson mide relación lineal;
  Spearman es el contraste robusto ante la fuerte asimetría y los outliers típicos
  de muertes por COVID.

