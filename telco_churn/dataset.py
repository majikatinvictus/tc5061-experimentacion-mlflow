"""Carga del CSV original y limpieza con las decisiones del análisis exploratorio."""

from pathlib import Path

import pandas as pd

from telco_churn.config import COLUMNA_ID, OBJETIVO


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV tal como viene de Kaggle."""
    return pd.read_csv(ruta)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos, rellena faltantes y quita el identificador; no modifica el original."""
    limpio = df.copy()
    # Los 11 cargos totales en blanco son clientes con 0 meses: aún no se les ha cobrado nada
    limpio["TotalCharges"] = pd.to_numeric(
        limpio["TotalCharges"].str.strip(), errors="coerce"
    ).fillna(0)
    # Misma convención Yes/No que el resto de las variables de sí o no
    limpio["SeniorCitizen"] = limpio["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return limpio.drop(columns=COLUMNA_ID)


def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables del cliente (X) de la variable a predecir (y: 1 = se va)."""
    X = df.drop(columns=OBJETIVO)
    y = (df[OBJETIVO] == "Yes").astype(int)
    return X, y
