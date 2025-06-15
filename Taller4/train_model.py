import pandas as pd
from pycaret.classification import *
import mlflow
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Configuración inicial
df = pd.read_csv("fintech_credit_approval.csv")
df_model = df.drop(columns=["user_id"])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("aprobacion_credito")

# Setup PyCaret
s = setup(
    data=df_model,
    categorical_features=["residence_risk_zone"],
    fix_imbalance=True,
    target="approved",
    session_id=606,
    log_experiment=False,
    experiment_name="aprobacion_credito",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenamiento
best_model = compare_models()

# Registro MLflow
with mlflow.start_run():
    # Evaluación
    predictions = predict_model(best_model)
    report = classification_report(predictions['approved'], predictions['prediction_label'])
    mlflow.log_text(report, "classification_report.txt")
    
    # SHAP Integration
    try:
        pipeline = get_config('pipeline')
        X_train = get_config('X_train')
        X_test = get_config('X_test')
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        
        explainer = shap.Explainer(best_model, X_train_transformed[:100])  # Muestra reducida
        shap_values = explainer(X_test_transformed)
        
        feature_names = X_train.columns.tolist()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
        mlflow.log_figure(plt.gcf(), "shap_summary_plot.png")
        plt.close()
        
        # Plot específico para riesgo político
        if 'political_event_last_month' in feature_names:
            idx = feature_names.index('political_event_last_month')
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values.values, X_test_transformed, feature_names=feature_names, show=False)
            mlflow.log_figure(plt.gcf(), "shap_political_risk.png")
            plt.close()
            
    except Exception as e:
        print(f"Error en SHAP: {str(e)}")
        mlflow.log_text(f"SHAP Error: {str(e)}", "shap_error.txt")
    
    # Guardar modelo
    save_model(best_model, "aprobacion_credito_model")
    mlflow.log_artifact("aprobacion_credito_model.pkl")
    
    # Importancia de características
    if hasattr(best_model, 'feature_importances_'):
        try:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            mlflow.log_text(feature_importance.to_csv(index=False), "feature_importance.csv")
        except Exception as e:
            print(f"Error al registrar importancia: {str(e)}")

print("✅ Entrenamiento completado exitosamente!")
