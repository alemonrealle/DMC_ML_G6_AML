
import pandas as pd
from pycaret.classification import *
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
df = pd.read_csv(r"C:\Users\amonreal\source\repos\DMC_ML_G6_AML\modulo3\evaluacionfinal\hbr_caso_cliente_responde_oferta.csv")

df['region'] = df['region'].astype('category')
df['nivel_educacion'] = df['nivel_educacion'].astype('category')
df['genero'] = df['genero'].astype('category')

# 2. Configurar PyCaret para clasificación
#setup(data=df, normalize=True, session_id=123)


modelo =  setup(data=df,target='respondio_oferta',
                ignore_features = ['cliente_id'],
                remove_multicollinearity = True,multicollinearity_threshold = 0.8,
               # normalize=True,
                session_id=123,normalize=True,fix_imbalance=True,
                transformation=True,
                transformation_method='yeo-johnson',                
                categorical_features=['genero','nivel_educacion','region'])



# 5. Comparar modelos por F1 (conversion es evento raro)
best = compare_models(sort='F1')

# 6. Ajuste fino y visualización
final = tune_model(best, optimize='F1')
#plot_model(final, plot='pr')  # Precision-Recall
#plot_model(final, plot='confusion_matrix')

# 7. Interpretabilidad con SHAP
#cleainterpret_model(final)

#Comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model,"modelo_respuesta_cliente_mlflow")

