from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import pandas as pd

# Crear instancia de FastAPI
app = FastAPI(title="API de Predicción de Precio de Propiedades")

# Cargar modelo entrenado
model = load_model("house_price_model")

# Clase de entrada
class Propiedad(BaseModel):
    area_m2: float
    bedrooms: int
    bathrooms: int
    parking_spaces: int
    district: str
    age_years: float

# Endpoint para predicción
@app.post("/predict_price")
def predict(propiedad: Propiedad):
    data = pd.DataFrame([propiedad.dict()])
    data["district"] = data["district"].astype("category")
    resultado = predict_model(model, data=data)
    prediccion = round(float(resultado["prediction_label"][0]), 2)
    return {
        "input": propiedad.dict(),
        "precio_estimado_usd": prediccion
    }

#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
