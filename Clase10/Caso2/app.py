import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Detección de Transacciones Sospechosas", layout="centered")
st.title("💳 Detección de Transacciones Sospechosas")
st.write("Completa los datos de la transacción para evaluar si es atípica o no.")

# Inputs del formulario
amount = st.number_input("💰 Monto de la transacción (USD)", min_value=0.0, step=1.0, value=50.0)
merchant_category = st.selectbox("🏪 Categoría del comercio", ["Retail", "Food", "Travel", "Tech", "Health"])
hour = st.slider("🕒 Hora de la transacción", min_value=0, max_value=23, value=12)
day_of_week = st.selectbox("📅 Día de la semana", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
is_foreign = st.radio("🌍 ¿Transacción en el extranjero?", ["No", "Sí"])

# Botón de predicción
if st.button("🔍 Evaluar transacción"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "amount_usd": amount,
                "merchant_category": merchant_category,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_foreign": 1 if is_foreign == "Sí" else 0
            }
            response = requests.post("http://localhost:8000/detect_outlier", json=payload)
            if response.status_code == 200:
                resultado = response.json()
                if resultado["es_sospechosa"]:
                    st.error("⚠️ Transacción clasificada como *sospechosa* o *atípica*.")
                else:
                    st.success("✅ Transacción normal.")
            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#uvicorn api:app --reload --port 8000
#streamlit run app.py
