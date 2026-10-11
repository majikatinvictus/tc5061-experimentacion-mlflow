"""Rutas y constantes del proyecto: ningún otro módulo escribe una ruta a mano."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Datos
DATOS_CRUDOS = RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
COLUMNA_ID = "customerID"
OBJETIVO = "Churn"
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Valores por defecto de la ejecución
SEMILLA = 42
PROPORCION_PRUEBA = 0.20
PLIEGUES_CV = 5

# MLflow: base de seguimiento y artefactos en la raíz del proyecto
MLFLOW_TRACKING_URI = f"sqlite:///{RAIZ / 'mlflow.db'}"
MLFLOW_ARTEFACTOS = RAIZ / "mlartifacts"
EXPERIMENTO = "telco-churn-regresion-logistica"
