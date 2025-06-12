from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.anomaly import load_model, predict_model
import pandas as pd

# Crear instancia de FastAPI
app = FastAPI(title="API de Detección de Transacciones Sospechosas")

# Cargar modelo
model = load_model("outlier_model")

# Clase de entrada
class Transaccion(BaseModel):
    amount_usd: float
    merchant_category: str
    hour: int
    day_of_week: str
    is_foreign: int  # 0 o 1

# Endpoint de detección
@app.post("/detect_outlier")
def detect(transaccion: Transaccion):
    # Convertir entrada a DataFrame
    data = pd.DataFrame([transaccion.dict()])
    data["merchant_category"] = data["merchant_category"].astype("category")
    data["day_of_week"] = data["day_of_week"].astype("category")

    # Predecir con modelo cargado
    prediction = predict_model(model, data=data)
    outlier_flag = int(prediction["Anomaly"][0])

    return {
        "input": transaccion.dict(),
        "es_sospechosa": bool(outlier_flag),
        "anomaly_flag": outlier_flag
    }


#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
#curl -X POST http://localhost:8000/detect_outlier -H "Content-Type: application/json" \
#-d '{"amount_usd": 3000, "merchant_category": "Tech", "hour": 3, "day_of_week": "Sunday", "is_foreign": 1}'
