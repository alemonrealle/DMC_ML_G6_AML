import pandas as pd
from pycaret.classification import *
import mlflow
import mlflow.pycaret

# Cargar datos
df = pd.read_csv("user_churn.csv")

# Preparar data para entrenamiento
df_model = df.drop(columns=["user_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("clasificacion_churn")

# Configurar PyCaret
s = setup(
    data=df_model,
    target="churned",
    session_id=303,
    log_experiment=True,
    experiment_name="clasificacion_churn",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenar y seleccionar el mejor modelo
best_model = compare_models()

# Registrar modelo en MLflow
mlflow.pycaret.log_model(best_model, "mejor_modelo_churn")

# Guardar localmente
save_model(best_model, "churn_model")

print("✅ Modelo de churn entrenado y registrado.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py
