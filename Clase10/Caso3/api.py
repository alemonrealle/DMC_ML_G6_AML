from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inicializar API
app = FastAPI(title="API de Propensión a Compra")

# Cargar modelo entrenado
model = load_model("propension_model")

# Clase de entrada
class Cliente(BaseModel):
    age: int
    account_balance_usd: float
    has_mobile_app: int  # 0 o 1
    visits_last_30d: int
    interacted_last_campaign: int  # 0 o 1

# Endpoint de predicción
@app.post("/predict_conversion")
def predict(cliente: Cliente):
    data = pd.DataFrame([cliente.dict()])
    resultado = predict_model(model, data=data)
    prediccion = int(resultado["prediction_label"][0])
    score = float(resultado["prediction_score"][0])
    return {
        "input": cliente.dict(),
        "comprará": bool(prediccion),
        "score_probabilidad": round(score, 3)
    }

#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
