import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Propensión a Compra", layout="centered")
st.title("📈 Propensión a Compra de Producto Financiero")
st.write("Simula el perfil de un cliente para predecir si comprará el nuevo producto.")

# Inputs
age = st.slider("🎂 Edad del cliente", min_value=18, max_value=70, value=35)
balance = st.number_input("💰 Saldo en cuenta (USD)", min_value=0.0, step=100.0, value=3000.0)
app = st.radio("📱 ¿Usa la app móvil?", ["No", "Sí"])
visits = st.number_input("🔁 Visitas al sitio en últimos 30 días", min_value=0, step=1, value=4)
interacted = st.radio("📧 ¿Interactuó con campaña anterior?", ["No", "Sí"])

# Botón de predicción
if st.button("🔍 Predecir"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "account_balance_usd": balance,
                "has_mobile_app": 1 if app == "Sí" else 0,
                "visits_last_30d": visits,
                "interacted_last_campaign": 1 if interacted == "Sí" else 0
            }

            r = requests.post("http://localhost:8000/predict_conversion", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                if resultado["comprará"]:
                    st.success(f"✅ Es probable que compre. Score: {resultado['score_probabilidad']}")
                else:
                    st.warning(f"⚠️ Poca probabilidad de compra. Score: {resultado['score_probabilidad']}")
            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#python train_model.py
#uvicorn api:app --reload --port 8000
#streamlit run app.py
