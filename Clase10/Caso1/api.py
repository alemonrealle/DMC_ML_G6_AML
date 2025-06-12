from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.clustering import load_model, predict_model
import pandas as pd

# Inicializar FastAPI
app = FastAPI(title="API de Segmentación de Clientes")

# Cargar modelo previamente entrenado
model = load_model("cluster_model")

# Clase de entrada
class Cliente(BaseModel):
    annual_spend: float
    transactions_per_year: int
    avg_basket_size: float
    days_since_last_purchase: int

# Endpoint POST para predicción
@app.post("/predict")
def predict(cliente: Cliente):
    data = pd.DataFrame([cliente.dict()])
    prediction = predict_model(model, data=data)
    cluster = int(prediction["Cluster"][0])
    return {
        "input": cliente.dict(),
        "cluster": cluster
    }
