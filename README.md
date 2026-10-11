# Experimentación con MLflow · Telco Customer Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Script de entrenamiento parametrizado que predice qué clientes de una empresa de telecomunicaciones cancelan su servicio (*churn*) con una regresión logística. Cada ejecución queda registrada en MLflow Tracking con sus parámetros, métricas, artefactos y el modelo entrenado, para comparar las corridas desde la interfaz de MLflow.

Sebastián Estrada García · TC5061 Operaciones de Aprendizaje Automático · Tecnológico de Monterrey

## Instalación

Requiere Python 3.12.

```bash
git clone https://github.com/majikatinvictus/tc5061-experimentacion-mlflow.git
cd tc5061-experimentacion-mlflow
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt  # o: make requirements
```

El archivo `requirements.txt` fija la versión de cada librería e instala el paquete del proyecto (`telco_churn`). Los datos ya vienen incluidos en `data/raw/`.

## Ejecución

Todos los comandos se corren desde la raíz del proyecto. Sin argumentos, el script entrena con los valores por defecto:

```bash
python -m telco_churn.modeling.train
```

Los hiperparámetros se cambian desde la línea de comandos, sin editar el código:

```bash
python -m telco_churn.modeling.train --C 0.1 --class-weight balanced
python -m telco_churn.modeling.train --C 0.1 --l1-ratio 1 --solver saga
```

| Argumento | Por defecto | Qué controla |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización: valores pequeños simplifican más el modelo |
| `--l1-ratio` | `0.0` | Tipo de regularización: 0 = L2, 1 = L1 (puede anular variables); mayor que 0 requiere `--solver saga` |
| `--class-weight` | `none` | `balanced` da más peso a los clientes que se van, que son la minoría |
| `--solver` | `lbfgs` | Algoritmo de optimización: `lbfgs` solo admite L2; `saga` admite L1 y L2 |
| `--max-iter` | `1000` | Máximo de iteraciones del optimizador |
| `--semilla` | `42` | Semilla de la división de los datos, de la validación cruzada y del optimizador |
| `--proporcion-prueba` | `0.2` | Proporción de clientes que se apartan para la prueba final |
| `--datos` | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Ruta del CSV original |
| `--experimento` | `telco-churn-regresion-logistica` | Experimento de MLflow donde se registra la corrida |

La ayuda completa está en `python -m telco_churn.modeling.train --help`.

## Corridas del experimento

`make experimentos` ejecuta las ocho corridas comparadas en la entrega, todas en el experimento `telco-churn-regresion-logistica`. Cada una cambia una cosa respecto al modelo base, y las últimas combinan lo que mejor funcionó:

| # | Argumentos | Qué se prueba |
|---|---|---|
| 1 | *(ninguno)* | Modelo base |
| 2 | `--C 0.001` | Regularización muy fuerte |
| 3 | `--C 100` | Regularización muy débil |
| 4 | `--class-weight balanced` | Más peso a los clientes que se van |
| 5 | `--l1-ratio 1 --solver saga` | Regularización L1 |
| 6 | `--C 0.1 --class-weight balanced` | Clases balanceadas con más regularización |
| 7 | `--C 0.1 --l1-ratio 1 --solver saga --class-weight balanced` | Clases balanceadas con L1 |
| 8 | *(ninguno)* | Repetición de la corrida 1 para comprobar la reproducibilidad |

## Ver los resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # o: make ui
```

En http://127.0.0.1:5001 se abre el experimento `telco-churn-regresion-logistica`; para contrastar corridas, se seleccionan y se usa **Compare**.

Cada corrida registra:

- **Parámetros:** los hiperparámetros, la semilla, la proporción de prueba, los pliegues de la validación cruzada, el archivo de datos y el tamaño de cada conjunto.
- **Métricas:** F1 y ROC-AUC de la validación cruzada (`cv_*`, sobre entrenamiento); F1, recall, precisión, ROC-AUC y exactitud en el conjunto de prueba (`prueba_*`); y cuántos coeficientes distintos de cero usa el modelo.
- **Artefactos:** la matriz de confusión, la curva ROC y el reporte de clasificación, en la carpeta `evaluacion/`.
- **Modelo:** el pipeline completo (preprocesamiento y regresión logística) con su firma y un ejemplo de entrada.

## Reproducibilidad

- **Semilla:** controla la división estratificada de los datos, los pliegues de la validación cruzada y el optimizador `saga`. Con los mismos argumentos, el resultado es idéntico; las corridas 1 y 8 lo comprueban.
- **Dependencias:** `requirements.txt` fija la versión exacta de cada librería, y `pyproject.toml` exige Python 3.12.
- **Datos:** el CSV original no se modifica; la limpieza se aplica en el código (`telco_churn/dataset.py`).
- **Registros locales:** `mlflow.db` y `mlartifacts/` se crean en la raíz del proyecto al correr el script y no se versionan.

## Datos

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (BlastChar, Kaggle): 7,043 clientes, 19 variables que los describen y la variable `Churn`, que indica si se fueron. La limpieza convierte `TotalCharges` a número (los 11 valores en blanco son clientes con 0 meses y se rellenan con 0), pasa `SeniorCitizen` de 0/1 a No/Yes y elimina el identificador `customerID`.

## Organización del proyecto

```
├── LICENSE
├── Makefile            <- make requirements, make train, make experimentos, make ui
├── README.md
├── pyproject.toml      <- metadatos del paquete y configuración de ruff
├── requirements.txt    <- dependencias con versiones fijas
├── data
│   └── raw             <- CSV original, sin modificar
├── notebooks           <- análisis exploratorio (solo exploración)
└── telco_churn         <- código fuente del proyecto
    ├── config.py       <- rutas y valores por defecto
    ├── dataset.py      <- carga y limpieza de los datos
    ├── features.py     <- escalado y codificación de las variables
    ├── plots.py        <- matriz de confusión y curva ROC
    └── modeling
        ├── evaluate.py <- validación cruzada y métricas de prueba
        └── train.py    <- argumentos, entrenamiento y registro en MLflow
```

La estructura se generó con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) y se simplificó a lo que usa el proyecto.
