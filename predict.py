import joblib
import pandas as pd

# Load trained model
model = joblib.load("crop_yield_model.pkl")

def predict_yield(
    rainfall_mm,
    temperature_c,
    humidity_percent,
    soil_ph,
    nitrogen,
    phosphorus,
    potassium,
    sunlight_hours,
    area_hectares,
    crop_type
):
    input_data = pd.DataFrame([{
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "soil_ph": soil_ph,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "sunlight_hours": sunlight_hours,
        "area_hectares": area_hectares,
        "crop_type": crop_type
    }])

    return round(model.predict(input_data)[0], 2)


if __name__ == "__main__":
    result = predict_yield(
        rainfall_mm=1200,
        temperature_c=28,
        humidity_percent=75,
        soil_ph=6.5,
        nitrogen=80,
        phosphorus=45,
        potassium=60,
        sunlight_hours=7.5,
        area_hectares=2.0,
        crop_type="Maize"
    )

    print(f"Predicted Crop Yield: {result} tonnes")
