from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inicializar FastAPI
app = FastAPI(title="API de Taller 4 - Aprobacion de Crédito")

# Cargar modelo
model = load_model("aprobacion_credito")

# Entrada esperada
class Cliente(BaseModel):
    age: int
    monthly_income_usd: float
    app_usage_score: float	
    digital_profile_strength: float
    num_contacts_uploaded: int
    residence_risk_zone: str
    political_event_last_month: int
    threshold: float  # nuevo: umbral personalizado (0.0 - 1.0)

@app.post("/predict_aprobacion")
def predict(cliente: Cliente):
    input_data = pd.DataFrame([cliente.dict()])
    threshold = input_data.pop("threshold").iloc[0]
    input_data["residence_risk_zone"] = input_data["residence_risk_zone"].astype("category")
    
    resultado = predict_model(model, data=input_data)
    score = float(resultado["prediction_score"][0])
    decision = int(score >= threshold)

    return {
        "input": cliente.dict(),
        "score_probabilidad": round(score, 4),
        "threshold_usado": threshold,
        "aprobado": bool(decision)
    }


#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
