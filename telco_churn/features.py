"""Preparación de las variables para el modelo."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.config import NUMERICAS


def crear_preprocesamiento(X: pd.DataFrame) -> ColumnTransformer:
    """Escala las numéricas y convierte las categóricas en columnas de ceros y unos."""
    categoricas = [columna for columna in X.columns if columna not in NUMERICAS]
    return ColumnTransformer(
        [
            ("numericas", StandardScaler(), NUMERICAS),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )
