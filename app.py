import os
import pickle
import numpy as np
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
# model = joblib.load("crop_yield_model.pkl")

MODEL_PATH = "crop_yield_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model file not found. Build failed to download it.")

model = joblib.load(MODEL_PATH)



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = pd.DataFrame([{
            "rainfall_mm": float(request.form["rainfall"]),
            "temperature_c": float(request.form["temperature"]),
            "humidity_percent": float(request.form["humidity"]),
            "soil_ph": float(request.form["soil_ph"]),
            "nitrogen": float(request.form["nitrogen"]),
            "phosphorus": float(request.form["phosphorus"]),
            "potassium": float(request.form["potassium"]),
            "sunlight_hours": float(request.form["sunlight"]),
            "area_hectares": float(request.form["area"]),
            "crop_type": request.form["crop"]
        }])

        prediction = round(model.predict(data)[0], 2)

    return render_template("index.html", prediction=prediction)

# IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


# ---------------- PREDICTION----------------
@app.route("/index")
def prediction():
    return render_template("index.html")

# ---------------- HISTORY ----------------
@app.route("/history")
def history():
    return render_template("history.html")


# ---------------- DATASET ----------------
@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


# ---------------- SETTINGS ----------------
@app.route("/settings")
def settings():
    return render_template("settings.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
