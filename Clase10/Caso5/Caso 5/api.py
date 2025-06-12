from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inicializar API
app = FastAPI(title="API de Predicción de Churn")

# Cargar modelo entrenado
model = load_model("churn_model")

# Definir clase de entrada
class Usuario(BaseModel):
    subscription_age_months: int
    monthly_usage_hours: float
    num_support_tickets: int
    payment_method: str
    last_feedback_score: int

# Endpoint
@app.post("/predict_churn")
def predict(usuario: Usuario):
    data = pd.DataFrame([usuario.dict()])
    data["payment_method"] = data["payment_method"].astype("category")
    result = predict_model(model, data=data)
    pred = int(result["prediction_label"][0])
    score = round(float(result["prediction_score"][0]), 3)
    return {
        "input": usuario.dict(),
        "churn_probable": bool(pred),
        "score_probabilidad": score
    }

#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
