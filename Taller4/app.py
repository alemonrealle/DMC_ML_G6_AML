import streamlit as st
import requests

# Configuración de página
st.set_page_config(page_title="Predicción de Aprobación", layout="centered")
st.title("📈 Predicción de Aprobación de Crédito")
st.markdown("Simula si el cliente será aprobado o no.")

# Inputs del cliente
age = st.slider("🎂 Edad", 25, 70, 40)
monthly_income_usd = st.number_input("💵 Ingreso mensual (USD)", min_value=0.0, step=1000.0, value=15000.0)
app_usage_score = st.slider("📆 Score de uso de la app", 1, 20, 1)
digital_profile_strength = st.slider("📄 Score de perfil digital", 1.0, 100.0, 0.1)
num_contacts_uploaded = st.slider("💼 Numero de contactos", 1, 
                                  100, 1)
residence_risk_zone = st.selectbox("📬 Zona de residencia",["Baja", "Media", "Alta"])
political_event_last_month = st.radio("📬 Hubo disturbios/elecciones en su región", ["No", "Sí"])
#threshold = st.slider("📆 Umbral", 0, 1, 0.1)


# Threshold slider
threshold = st.slider("🎚 Umbral de aceptación (threshold)", 0.0, 1.0, 0.5, step=0.01)

# Botón de predicción
if st.button("🔍 Evaluar Probabilidad"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "monthly_income_usd": monthly_income_usd,
                "app_usage_score": app_usage_score,
                "digital_profile_strength": digital_profile_strength,
                "num_contacts_uploaded": num_contacts_uploaded,
                "residence_risk_zone": residence_risk_zone,
                "political_event_last_month": 1 if political_event_last_month == "Sí" else 0,
                "threshold": threshold
            }

            r = requests.post("http://localhost:8000/predict_aprobacion", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_probabilidad"]
                aprobado = resultado["aprobado"]

                st.markdown(f"### 🔢 Score de aceptación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                if aprobado:
                    st.success("✅ El cliente probablemente **será aprobado**.")
                else:
                    st.warning("⚠️ El cliente probablemente **será rechazaso**.")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#uvicorn api:app --reload --port 8000
#streamlit run app.py
