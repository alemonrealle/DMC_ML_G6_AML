import pandas as pd
from pycaret.regression import *
import mlflow
import mlflow.pycaret

# Cargar dataset
df = pd.read_csv("house_prices.csv")

# Eliminar identificador
df_model = df.drop(columns=["property_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("prediccion_precio_propiedades")

# Setup de PyCaret
s = setup(
    data=df_model,
    target="price_usd",
    session_id=202,
    log_experiment=True,
    experiment_name="prediccion_precio_propiedades",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenamiento automático
best_model = compare_models()

# Registro en MLflow
mlflow.pycaret.log_model(best_model, "mejor_modelo_precio_propiedades")

# Guardar localmente
save_model(best_model, "house_price_model")

print("✅ Modelo de regresión entrenado y registrado.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py
