"""Entrena la regresión logística y registra la corrida en MLflow.

python -m telco_churn.modeling.train
python -m telco_churn.modeling.train --C 0.1 --class-weight balanced
"""

import argparse
from pathlib import Path
import tempfile
import warnings

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.config import (
    DATOS_CRUDOS,
    EXPERIMENTO,
    MLFLOW_ARTEFACTOS,
    MLFLOW_TRACKING_URI,
    PLIEGUES_CV,
    PROPORCION_PRUEBA,
    SEMILLA,
)
from telco_churn.dataset import cargar_datos, limpiar_datos, separar_variables
from telco_churn.features import crear_preprocesamiento
from telco_churn.modeling.evaluate import (
    calcular_metricas,
    coeficientes_no_nulos,
    validacion_cruzada,
)
from telco_churn.plots import graficar_curva_roc, graficar_matriz_confusion

# Aviso de MLflow sobre columnas enteras con faltantes; tras la limpieza no hay faltantes
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def leer_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    """Lee los hiperparámetros y la configuración de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m telco_churn.modeling.train",
        description="Entrena una regresión logística para predecir qué clientes cancelan "
        "su servicio y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    modelo = parser.add_argument_group("hiperparámetros del modelo")
    modelo.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización: valores pequeños simplifican más el modelo.",
    )
    modelo.add_argument(
        "--l1-ratio",
        type=float,
        default=0.0,
        help="Tipo de regularización: 0 = L2, 1 = L1 (puede anular variables), "
        "intermedios = combinación. Mayor que 0 requiere --solver saga.",
    )
    modelo.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Peso de cada clase: 'balanced' da más peso a los clientes que se "
        "van, que son la minoría.",
    )
    modelo.add_argument(
        "--solver",
        choices=["lbfgs", "saga"],
        default="lbfgs",
        help="Algoritmo de optimización: lbfgs solo admite L2; saga admite L1 y L2.",
    )
    modelo.add_argument(
        "--max-iter", type=int, default=1000, help="Máximo de iteraciones del optimizador."
    )

    ejecucion = parser.add_argument_group("ejecución")
    ejecucion.add_argument(
        "--semilla",
        type=int,
        default=SEMILLA,
        help="Semilla de la división de los datos, de los pliegues de la "
        "validación cruzada y del optimizador.",
    )
    ejecucion.add_argument(
        "--proporcion-prueba",
        type=float,
        default=PROPORCION_PRUEBA,
        help="Proporción de clientes que se apartan para la prueba final.",
    )
    ejecucion.add_argument(
        "--datos", type=Path, default=DATOS_CRUDOS, help="Ruta del CSV original."
    )
    ejecucion.add_argument(
        "--experimento",
        default=EXPERIMENTO,
        help="Experimento de MLflow donde se registra la corrida.",
    )

    args = parser.parse_args(argumentos)
    if args.C <= 0:
        parser.error("--C debe ser mayor que 0")
    if not 0 <= args.l1_ratio <= 1:
        parser.error("--l1-ratio debe estar entre 0 y 1")
    if args.l1_ratio > 0 and args.solver != "saga":
        parser.error("--l1-ratio mayor que 0 requiere --solver saga")
    if not 0 < args.proporcion_prueba < 1:
        parser.error("--proporcion-prueba debe estar entre 0 y 1")
    return args


def obtener_hiperparametros(args: argparse.Namespace) -> dict:
    """Hiperparámetros del modelo: se usan para entrenarlo y para registrarlos en MLflow."""
    return {
        "C": args.C,
        "l1_ratio": args.l1_ratio,
        "class_weight": args.class_weight,
        "solver": args.solver,
        "max_iter": args.max_iter,
    }


def dividir_datos(X: pd.DataFrame, y: pd.Series, proporcion_prueba: float, semilla: int):
    """División estratificada: entrenamiento y prueba conservan la proporción de cancelaciones."""
    return train_test_split(X, y, test_size=proporcion_prueba, stratify=y, random_state=semilla)


def crear_modelo(X: pd.DataFrame, hiperparametros: dict, semilla: int) -> Pipeline:
    """Pipeline con el preprocesamiento y la regresión logística."""
    peso_clases = None if hiperparametros["class_weight"] == "none" else "balanced"
    regresion = LogisticRegression(
        **{**hiperparametros, "class_weight": peso_clases}, random_state=semilla
    )
    return Pipeline([("preprocesamiento", crear_preprocesamiento(X)), ("modelo", regresion)])


def preparar_experimento(nombre: str) -> None:
    """Conecta con la base de MLflow del proyecto y crea el experimento la primera vez."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(nombre) is None:
        mlflow.create_experiment(nombre, artifact_location=MLFLOW_ARTEFACTOS.as_uri())
    mlflow.set_experiment(nombre)


def guardar_artefactos(
    carpeta: Path,
    y_prueba: pd.Series,
    prediccion: np.ndarray,
    probabilidad: np.ndarray,
    auc: float,
) -> list[Path]:
    """Guarda la matriz de confusión, la curva ROC y el reporte de clasificación."""
    archivos = {
        "matriz_confusion.png": graficar_matriz_confusion(y_prueba, prediccion),
        "curva_roc.png": graficar_curva_roc(y_prueba, probabilidad, auc),
    }
    for nombre, figura in archivos.items():
        figura.savefig(carpeta / nombre, dpi=150)
    reporte = carpeta / "reporte_clasificacion.txt"
    reporte.write_text(
        classification_report(y_prueba, prediccion, digits=3, target_names=["Se queda", "Se va"])
    )
    return [carpeta / nombre for nombre in archivos] + [reporte]


def registrar_corrida(
    nombre: str,
    parametros: dict,
    metricas: dict,
    modelo: Pipeline,
    X_entrenamiento: pd.DataFrame,
    y_prueba: pd.Series,
    prediccion: np.ndarray,
    probabilidad: np.ndarray,
) -> str:
    """Registra parámetros, métricas, artefactos y el modelo en una corrida de MLflow."""
    with mlflow.start_run(run_name=nombre) as corrida:
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        with tempfile.TemporaryDirectory() as carpeta:
            for archivo in guardar_artefactos(
                Path(carpeta), y_prueba, prediccion, probabilidad, metricas["prueba_roc_auc"]
            ):
                mlflow.log_artifact(str(archivo), artifact_path="evaluacion")
        firma = infer_signature(X_entrenamiento, modelo.predict(X_entrenamiento))
        mlflow.sklearn.log_model(
            modelo, name="modelo", signature=firma, input_example=X_entrenamiento.head(3)
        )
    return corrida.info.run_id


def imprimir_resumen(nombre: str, metricas: dict, run_id: str) -> None:
    """Muestra en la terminal las métricas principales de la corrida."""
    print(f"Corrida registrada: {nombre}  (run_id {run_id})")
    print(
        f"  Validación cruzada  F1 {metricas['cv_f1_promedio']:.3f} "
        f"± {metricas['cv_f1_desviacion']:.3f}"
    )
    print(
        f"  Prueba              F1 {metricas['prueba_f1']:.3f} · "
        f"recall {metricas['prueba_recall']:.3f} · "
        f"precisión {metricas['prueba_precision']:.3f} · "
        f"ROC-AUC {metricas['prueba_roc_auc']:.3f}"
    )


def main() -> None:
    args = leer_argumentos()

    X, y = separar_variables(limpiar_datos(cargar_datos(args.datos)))
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(
        X, y, args.proporcion_prueba, args.semilla
    )

    hiperparametros = obtener_hiperparametros(args)
    modelo = crear_modelo(X_entrenamiento, hiperparametros, args.semilla)

    # La validación cruzada solo usa entrenamiento; la prueba se reserva para la evaluación final
    metricas = validacion_cruzada(
        modelo, X_entrenamiento, y_entrenamiento, PLIEGUES_CV, args.semilla
    )
    modelo.fit(X_entrenamiento, y_entrenamiento)
    prediccion = modelo.predict(X_prueba)
    probabilidad = modelo.predict_proba(X_prueba)[:, 1]
    metricas |= calcular_metricas(y_prueba, prediccion, probabilidad)
    metricas["coeficientes_no_nulos"] = coeficientes_no_nulos(modelo)

    parametros = {
        **hiperparametros,
        "semilla": args.semilla,
        "proporcion_prueba": args.proporcion_prueba,
        "pliegues_cv": PLIEGUES_CV,
        "datos": args.datos.name,
        "registros_entrenamiento": len(X_entrenamiento),
        "registros_prueba": len(X_prueba),
    }
    nombre = f"C={args.C:g}, l1_ratio={args.l1_ratio:g}, class_weight={args.class_weight}"

    preparar_experimento(args.experimento)
    run_id = registrar_corrida(
        nombre, parametros, metricas, modelo, X_entrenamiento, y_prueba, prediccion, probabilidad
    )
    imprimir_resumen(nombre, metricas, run_id)


if __name__ == "__main__":
    main()
