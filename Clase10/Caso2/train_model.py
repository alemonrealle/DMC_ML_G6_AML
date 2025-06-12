import pandas as pd
from pycaret.anomaly import *
import mlflow
import mlflow.pycaret

# Cargar datos
df = pd.read_csv("transactions.csv")

# Convertir variables categóricas
df["merchant_category"] = df["merchant_category"].astype("category")
df["day_of_week"] = df["day_of_week"].astype("category")

# Excluir columnas no útiles para el entrenamiento
data = df.drop(columns=["transaction_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("outlier_detection_tx")

# Configurar PyCaret para detección de anomalías
s = setup(
    data=data,
    session_id=42,
    log_experiment=True,
    experiment_name="outlier_detection_tx",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenar el mejor modelo (Isolation Forest, KNN, etc.)
best_model = create_model("iforest")

# Log en MLflow
mlflow.pycaret.log_model(best_model, "best_outlier_model")

# Guardar localmente
save_model(best_model, "outlier_model")

print("✅ Entrenamiento de modelo de outliers finalizado.")


#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py


