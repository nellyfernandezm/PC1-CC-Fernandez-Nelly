# Databricks notebook source
!pip install toml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import toml
import numpy as np
import joblib
import psycopg2
from sklearn.datasets import load_breast_cancer

# COMMAND ----------

import toml
import numpy as np
import joblib
import psycopg2
from sklearn.datasets import load_breast_cancer

# =========================================
# 1️⃣ Leer secretos
# =========================================
secrets = toml.load("secrets.toml")["postgres"]

USER = secrets["USER"]
PASSWORD = secrets["PASSWORD"]
HOST = secrets["HOST"]
PORT = secrets["PORT"]
DBNAME = secrets["DBNAME"]

# =========================================
# 2️⃣ Cargar modelo
# =========================================
model = joblib.load("components/model.pkl")
scaler = joblib.load("components/scaler.pkl")
feature_names = joblib.load("components/features.pkl")

# =========================================
# 3️⃣ Estadísticas reales del dataset
# =========================================
data_real = load_breast_cancer()
X_real = data_real.data
all_features = list(data_real.feature_names)

indices = [all_features.index(f) for f in feature_names]
X_selected = X_real[:, indices]

means = X_selected.mean(axis=0)
stds = X_selected.std(axis=0)

# =========================================
# 4️⃣ Generar datos balanceados
# =========================================
np.random.seed(42)

data = []

for i in range(20):

    # 🔹 mitad normal
    if i < 10:
        record = np.random.normal(means, stds)

    # 🔹 mitad "forzada" a valores benignos
    else:
        record = np.random.normal(means, stds)

        # empujar valores hacia zona benigna
        for j, f in enumerate(feature_names):
            if "radius" in f or "perimeter" in f or "area" in f:
                record[j] *= 0.7   # más pequeños → benigno
            if "smoothness" in f:
                record[j] *= 0.9

    data.append(record.tolist())

# =========================================
# 5️⃣ Insertar en DB
# =========================================
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor()

    for record in data:

        features_scaled = scaler.transform([record])

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        prediction = int(pred)
        probability = float(prob[prediction])

        # Map dinámico
        feature_map = dict(zip(feature_names, record))

        mean_radius = feature_map.get("mean radius", 0)
        mean_texture = feature_map.get("mean texture", 0)
        mean_perimeter = feature_map.get("mean perimeter", 0)
        mean_area = feature_map.get("mean area", 0)
        mean_smoothness = feature_map.get("mean smoothness", 0)

        # Debug (MUY útil)
        print("Pred:", prediction, "Prob:", probability)

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

    connection.commit()
    cursor.close()
    connection.close()

    print("✅ Datos variados insertados correctamente.")

except Exception as e:
    print(f"❌ Error: {e}")