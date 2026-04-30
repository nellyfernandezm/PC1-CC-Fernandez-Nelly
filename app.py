import streamlit as st
import joblib
import numpy as np
import psycopg2
import pandas as pd


USER = st.secrets["postgres"]["USER"]
PASSWORD = st.secrets["postgres"]["PASSWORD"]
HOST = st.secrets["postgres"]["HOST"]
PORT = st.secrets["postgres"]["PORT"]
DBNAME = st.secrets["postgres"]["DBNAME"]



# =========================================
# 🔌 CONEXIÓN A SUPABASE
# =========================================
conn = psycopg2.connect(
    host=HOST,
    database=DBNAME,
    user=USER,
    password=PASSWORD,
    port=PORT
)

# =========================================
#  CARGAR MODELO
# =========================================
model = joblib.load("components/model.pkl")
scaler = joblib.load("components/scaler.pkl")
feature_names = joblib.load("components/features.pkl")

st.title("Predicción de Breast cancer")
st.write("Ingresa los valores de las variables más importantes:")

inputs = []

# =========================================
#  INPUTS DINÁMICOS
# =========================================
for feature in feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# =========================================
#  PREDICCIÓN + INSERT
# =========================================
if st.button("Predecir"):

    data = np.array(inputs).reshape(1, -1)
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)
    prob = model.predict_proba(data_scaled)[0]

    # Resultado
    prediction = int(pred[0])
    probability = float(prob[prediction])

    # Mostrar resultado
    if prediction == 1:
        st.success(f"Benigno ✅ (prob: {probability:.2f})")
    else:
        st.error(f"Maligno ⚠️ (prob: {probability:.2f})")

    # =========================================
    # 🗄️ INSERT EN BD
    # =========================================
    cursor = conn.cursor()

    # Mapear valores (en orden)
    mean_radius = inputs[0]
    mean_texture = inputs[1]
    mean_perimeter = inputs[2]
    mean_area = inputs[3]
    mean_smoothness = inputs[4]

    cursor.execute("""
        INSERT INTO pc_ml_cloud (
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            prediction,
            probability
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        prediction,
        probability
    ))

    conn.commit()
    cursor.close()

    st.info("Resultado guardado en Supabase")

# =========================================
#  MOSTRAR HISTORIAL
# =========================================
st.subheader("Historial de predicciones")

query = "SELECT * FROM pc_ml_cloud ORDER BY created_time DESC LIMIT 10"

df = pd.read_sql(query, conn)

st.dataframe(df)