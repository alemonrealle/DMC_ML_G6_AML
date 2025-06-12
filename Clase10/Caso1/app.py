import streamlit as st
import requests

# Título y descripción
st.set_page_config(page_title="Segmentación de Clientes", layout="centered")
st.title("📊 Segmentación de Clientes")
st.write("Ingrese la información de un cliente para predecir a qué clúster pertenece.")

# Campos de entrada
spend = st.number_input("💰 Gasto anual (USD)", min_value=0.0, step=10.0, value=800.0)
transactions = st.number_input("🔁 Transacciones por año", min_value=0, step=1, value=30)
basket = st.number_input("🛒 Tamaño promedio de compra", min_value=0.0, step=0.1, value=4.0)
days = st.number_input("📅 Días desde la última compra", min_value=0, step=1, value=45)

# Botón de predicción
if st.button("📌 Predecir Clúster"):
    with st.spinner("Consultando el modelo..."):
        payload = {
            "annual_spend": spend,
            "transactions_per_year": transactions,
            "avg_basket_size": basket,
            "days_since_last_purchase": days
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                resultado = response.json()
                st.success(f"🔷 El cliente pertenece al clúster: {resultado['cluster']}")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ No se pudo conectar al API: {e}")
