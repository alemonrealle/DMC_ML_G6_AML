from pycaret.classification import * 
#import mlflow
import pandas as pd
#activar el mlflow
#mlflow.set_experiment("banking-autoML")
df = pd.read_csv(r"C:\Users\amonreal\source\repos\DMC_ML_G6_AML\Clase 8\Ejercicio1\churn_bank_automl.csv")
#Iniciarmos el pipeline de pycaret
s = setup(data=df,
          target='cerrara_cuenta',
          session_id=123,
          log_experiment=False,
          experiment_name="banking-autoML",
          log_plots=True
          )
#Comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model,"modelo_bank_mlflow")