from datetime import datetime, timezone
from pathlib import Path
import csv
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path('diabetes_logistic_regression_pipeline.joblib')
LOG_PATH = Path('monitoring_log.csv')
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
ZERO_MEANS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Model tidak ditemukan: {MODEL_PATH.resolve()}')
model = joblib.load(MODEL_PATH)
app = Flask(__name__)

@app.get('/')
def home():
    return jsonify({'status': 'aktif', 'endpoint': 'POST /predict'})

@app.post('/predict')
def predict():
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or 'data' not in payload:
        return jsonify({'error': "Kirim JSON berbentuk {'data': [[...fitur...]]}"}), 400
    try:
        frame = pd.DataFrame(payload['data'], columns=FEATURES)
        frame = frame.apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        return jsonify({'error': f'Format data tidak valid: {exc}'}), 400
    if frame.empty or frame.isna().any().any():
        return jsonify({'error': 'Data harus berisi baris dan tidak boleh kosong/NaN.'}), 400
    # Samakan perlakuan nilai 0 dengan proses cleaning saat pelatihan.
    frame[ZERO_MEANS_MISSING] = frame[ZERO_MEANS_MISSING].replace(0, float('nan'))
    predictions = model.predict(frame).astype(int)
    probabilities = model.predict_proba(frame)[:, 1]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    # Simpan ringkasan, bukan data kesehatan mentah, untuk menjaga privasi.
    log_row = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'row_count': len(frame),
        'predicted_0': int((predictions == 0).sum()),
        'predicted_1': int((predictions == 1).sum()),
        'mean_probability_1': round(float(probabilities.mean()), 6),
        'latency_ms': latency_ms,
    }
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)
    return jsonify({
        'prediction': predictions.tolist(),
        'probability_diabetes': [round(float(p), 4) for p in probabilities],
        'latency_ms': latency_ms,
    })

if __name__ == '__main__':
    # Server bawaan Flask hanya untuk uji lokal/pembelajaran.
    app.run(host='127.0.0.1', port=5000, debug=False)
