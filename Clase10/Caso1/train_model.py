import pandas as pd
from pycaret.clustering import *
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pycaret.clustering import *
from pycaret.clustering import setup,create_model,evaluate_model,save_model

# Cargar datos
df = pd.read_csv(r"C:\Users\amonreal\source\repos\DMC_ML_G6_AML\Clase10\Caso1\customer_segmentation.csv")

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("clustering_cust")

# Configuración de PyCaret


# Configuración de PyCaret
setup(
    data=df,
    drop_columns=["customer_id"],  # excluimos ID
    session_id=123,
    log_experiment=True,
    experiment_name="clustering_cust",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Comparar modelos de clustering automáticamente
best_model = create_model('kmeans')

# Registrar el mejor modelo en MLflow
#mlflow.pycaret.log_model(best_model, "best_clustering_model")

#Registrar manualmente el mlflow
#with mlflow.start_run():
mlflow.sklearn.log_model(best_model,"best_clustering_model")

# Guardar el modelo localmente
save_model(best_model, "cluster_model")

print("Entrenamiento y registro completado.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py