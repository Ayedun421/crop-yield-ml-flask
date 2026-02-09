
CROP YIELD PREDICTION USING MACHINE LEARNING

Files:
- crop_yield_data.csv : Large dataset (10,000 rows)
- train_model.py      : Model training script
- crop_yield_model.pkl: Generated after training


How to run:

1. pip install pandas numpy scikit-learn joblib
2. pip install "numpy<2" --force-reinstall
3. pip install flask
4. pip install gunicorn
5. pip freeze > requirements.txt
6. python train_model.py
7. python predict.py
8. python app.py
9. Open browser: http://127.0.0.1:5000







Dataset List:

rainfall_mm
temperature_c
humidity_percent
soil_ph
nitrogen
phosphorus
potassium
sunlight_hours
area_hectares
crop_type
yield_tonnes


