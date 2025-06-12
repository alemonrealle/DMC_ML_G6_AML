import streamlit as st
import requests

# Configuración
st.set_page_config(page_title="Predicción de Churn", layout="centered")
st.title("🚪 Predicción de Churn de Clientes")
st.write("Completa los datos del cliente para estimar si se dará de baja próximamente.")

# Inputs del usuario
age_months = st.slider("📆 Antigüedad de suscripción (meses)", 1, 60, 24)
usage_hours = st.number_input("⏱ Horas promedio de uso mensual", min_value=0.0, step=1.0, value=30.0)
tickets = st.slider("📨 Tickets de soporte (últimos 6 meses)", 0, 10, 1)
payment = st.selectbox("💳 Método de pago", ["CreditCard", "PayPal", "Crypto", "Others"])
feedback = st.slider("⭐ Última calificación de satisfacción (0-10)", 0, 10, 8)

# Botón de predicción
if st.button("🔍 Evaluar riesgo de churn"):
    with st.spinner("Consultando modelo..."):
        payload = {
            "subscription_age_months": age_months,
            "monthly_usage_hours": usage_hours,
            "num_support_tickets": tickets,
            "payment_method": payment,
            "last_feedback_score": feedback
        }

        try:
            r = requests.post("http://localhost:8000/predict_churn", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                if resultado["churn_probable"]:
                    st.error(f"⚠️ Alto riesgo de churn. Score: {resultado['score_probabilidad']}")
                else:
                    st.success(f"✅ Cliente estable. Score: {resultado['score_probabilidad']}")
            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")

#uvicorn api:app --reload --port 8000
#streamlit run app.py
