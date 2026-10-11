"""Métricas del modelo: validación cruzada sobre entrenamiento y evaluación en prueba."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def validacion_cruzada(
    modelo: Pipeline, X: pd.DataFrame, y: pd.Series, pliegues: int, semilla: int
) -> dict:
    """Estima el desempeño con validación cruzada estratificada, sin tocar el conjunto de prueba."""
    division = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=semilla)
    metricas = {"f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc"}
    resultados = cross_validate(modelo, X, y, cv=division, scoring=metricas)
    return {
        "cv_f1_promedio": resultados["test_f1"].mean(),
        "cv_f1_desviacion": resultados["test_f1"].std(),
        "cv_roc_auc_promedio": resultados["test_roc_auc"].mean(),
    }


def coeficientes_no_nulos(modelo: Pipeline) -> int:
    """Cuántas variables usa realmente el modelo (L1 puede dejar coeficientes en cero)."""
    return int((modelo.named_steps["modelo"].coef_ != 0).sum())


def calcular_metricas(y_real: pd.Series, prediccion, probabilidad) -> dict:
    """Métricas en el conjunto de prueba; F1 es la principal por el desbalance de clases."""
    return {
        "prueba_f1": f1_score(y_real, prediccion, zero_division=0),
        "prueba_recall": recall_score(y_real, prediccion, zero_division=0),
        "prueba_precision": precision_score(y_real, prediccion, zero_division=0),
        "prueba_roc_auc": roc_auc_score(y_real, probabilidad),
        "prueba_accuracy": accuracy_score(y_real, prediccion),
    }
