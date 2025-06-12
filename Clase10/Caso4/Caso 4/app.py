import streamlit as st
import requests

# Configuración
st.set_page_config(page_title="Predicción de Precio de Propiedad", layout="centered")
st.title("🏡 Estimador de Precio de Propiedad")
st.write("Completa los datos de la propiedad para estimar su valor de mercado.")

# Inputs del usuario
area = st.number_input("📏 Área construida (m2)", min_value=10.0, step=1.0, value=120.0)
bed = st.slider("🛏 Dormitorios", min_value=1, max_value=6, value=3)
bath = st.slider("🛁 Baños", min_value=1, max_value=4, value=2)
parking = st.selectbox("🚗 Espacios de estacionamiento", [0, 1, 2])
district = st.selectbox("📍 Distrito", ["Miraflores", "San Isidro", "Barranco", "Surco", "La Molina", "San Borja"])
age = st.slider("🏗 Antigüedad del inmueble (años)", min_value=0, max_value=60, value=15)

# Botón de predicción
if st.button("🔍 Estimar precio"):
    with st.spinner("Consultando modelo..."):
        payload = {
            "area_m2": area,
            "bedrooms": bed,
            "bathrooms": bath,
            "parking_spaces": parking,
            "district": district,
            "age_years": age
        }

        try:
            response = requests.post("http://localhost:8000/predict_price", json=payload)
            if response.status_code == 200:
                resultado = response.json()
                st.success(f"💲 Precio estimado: USD {resultado['precio_estimado_usd']:,.2f}")
            else:
                st.error("❌ Error al procesar la solicitud.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")

#uvicorn api:app --reload --port 8000
#streamlit run app.py
