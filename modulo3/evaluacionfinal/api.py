from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model,setup,predict_model
# Cargar modelo previamente entrenado
model = load_model("modelo_respuesta_cliente_mlflow")
# Crear instancia de API
app = FastAPI()
# Esquema del cliente que recibiremos
class Cliente(BaseModel):
    edad: int
    genero: str	
    ingreso_mensual: float	
    nivel_educacion: str		
    usa_app: int		
    usa_web: int		
    satisfaccion: int		
    num_productos: int		
    reclamos_ult_6m: int		
    tasa_credito: float	
    region: str	

#@app.post("/predecir_cluster")
@app.post("/predict")
def predict(cliente:Cliente):
    data = pd.DataFrame([cliente.dict()])
    pred = predict_model(model,data=data)
    #Esto es la respuesta del api
    return {
        "score":float(pred['prediction_score'][0]),
        "prediccion":int(pred['prediction_label'][0])
    }


