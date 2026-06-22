# Dashboard COVID-19: Análisis Exploratorio y métodos de visualización en altas dimensiones y correlación

## Autores

- Reyna Alvarez Brandon Yire
- Hernández Sosol María Fernanda

## Descripción

Este proyecto desarrolla un dashboard interactivo en **Streamlit** para analizar la evolución de la mortalidad por COVID-19 en Estados Unidos mediante técnicas de análisis exploratorio de datos y reducción de dimensionalidad.

El sistema permite visualizar información por estados y condados utilizando:

- Mapas interactivos
- Series de tiempo
- Rankings
- Análisis Exploratorio de Datos (EDA)
- Análisis de Componentes Principales (PCA)
- Locally Linear Embedding (LLE)
- Análisis de correlaciones

El preprocesamiento de los datos se realiza una sola vez mediante un script independiente, mientras que el dashboard únicamente consume los archivos generados para mejorar el rendimiento.

---

## Estructura del proyecto

```text
Proyecto/
│
├── dashboard.py
├── preprocesar.py
├── requirements.txt
├── time_series_covid_19_deaths_US.csv
└── salida/
    ├── condados_features.csv
    ├── pca_varianza.csv
    ├── pca_loadings.csv
    └── correlaciones.csv

```

---

## Archivos principales

| Archivo | Descripción |
|----------|-------------|
| `preprocesar.py` | Limpia los datos, calcula las variables derivadas, ejecuta PCA, LLE y correlaciones, y guarda los resultados en la carpeta `salida`. |
| `dashboard.py` | Dashboard interactivo desarrollado con Streamlit. |
| `requirements.txt` | Lista de dependencias necesarias para ejecutar el proyecto. |

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Archivos de entrada

El proyecto requiere los siguientes archivos:

### 1. Dataset COVID-19

`time_series_covid_19_deaths_US.csv`

Corresponde al dataset original de series de tiempo en formato ancho.

## Ejecución

### 1. Generar los archivos procesados

```bash
python preprocesar.py
```

### 2. Ejecutar el dashboard

```bash
streamlit run dashboard.py
```

---

## Funcionalidades

- Visualización de series de tiempo.
- Mapas interactivos.
- Rankings de estados.
- Análisis Exploratorio de Datos (EDA).
- Análisis de Componentes Principales (PCA).
- Locally Linear Embedding (LLE).
- Análisis de correlaciones.

---

## Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Statsmodels
- OpenPyXL

---

## Fuente de datos

Johns Hopkins University COVID-19 Data Repository.


