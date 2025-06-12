import pandas as pd
from pycaret.classification import *
import mlflow
import mlflow.sklearn

# Cargar dataset
df = pd.read_csv(r"C:\Users\amonreal\source\repos\DMC_ML_G6_AML\Clase10\Caso3\conversion_users.csv")

# Excluir columnas no predictivas
df_model = df.drop(columns=["user_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("propension_compra")

# Configurar PyCaret
s = setup(
    data=df_model,
    target="converted_product",
    session_id=123,
    log_experiment=True,  # activa logging automático en MLflow
    experiment_name="propension_compra",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenar modelos y seleccionar el mejor
best_model = compare_models()

# Registrar modelo en MLflow manualmente (opcional, por si no quieres depender del logging automático)
with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "propension_model")

# Guardar modelo localmente
save_model(best_model, "propension_model")

print("✅ Entrenamiento y registro completado.")
