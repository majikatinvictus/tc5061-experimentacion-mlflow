"""Gráficas de evaluación que se guardan como artefactos de cada corrida."""

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

ETIQUETAS = ["Se queda", "Se va"]


def graficar_matriz_confusion(y_real: pd.Series, prediccion: np.ndarray) -> Figure:
    """Matriz de confusión con el número de clientes y el porcentaje sobre su grupo real."""
    figura = Figure(figsize=(5.5, 4.6), layout="constrained")
    eje = figura.subplots()
    pantalla = ConfusionMatrixDisplay.from_predictions(
        y_real, prediccion, display_labels=ETIQUETAS, colorbar=False, cmap="Blues", ax=eje
    )
    matriz = pantalla.confusion_matrix
    for (i, j), texto in np.ndenumerate(pantalla.text_):
        texto.set_text(f"{matriz[i, j]:,}\n{matriz[i, j] / matriz[i].sum():.0%}")
    eje.set_xlabel("Lo que predijo el modelo")
    eje.set_ylabel("Lo que realmente pasó")
    eje.set_title("Matriz de confusión · conjunto de prueba", loc="left")
    return figura


def graficar_curva_roc(y_real: pd.Series, probabilidad: np.ndarray, auc: float) -> Figure:
    """Curva ROC del modelo frente a un modelo que adivina al azar."""
    falsas_alarmas, detectados, _ = roc_curve(y_real, probabilidad)
    figura = Figure(figsize=(6, 5), layout="constrained")
    eje = figura.subplots()
    eje.plot([0, 1], [0, 1], color="gray", linewidth=1, label="Al azar (AUC = 0.500)")
    eje.plot(falsas_alarmas, detectados, linewidth=2, label=f"Modelo (AUC = {auc:.3f})")
    eje.set_xlabel("Proporción de falsas alarmas")
    eje.set_ylabel("Proporción de clientes que se van detectados")
    eje.set_title("Curva ROC · conjunto de prueba", loc="left")
    eje.legend(loc="lower right")
    return figura
