from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from pathlib import Path

import os
template_dir = os.path.abspath('static')
app = Flask(__name__, static_folder='static', template_folder=template_dir)

# Load the AI model
model = joblib.load(Path("models/anomaly_model.pkl"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    # Expecting: { "click_speed": 100, "fill_time": 5, "mouse_moves": 20 }
    features = np.array([[
        data['click_speed'], 
        data['fill_time'], 
        data['mouse_moves']
    ]])
    
    # Predict: -1 = anomaly, 1 = normal
    prediction = model.predict(features)[0]
    is_suspicious = bool(prediction == -1)
    
    return jsonify({
        "status": "success",
        "is_suspicious": is_suspicious,
        "message": "Suspicious activity detected!" if is_suspicious else "User behavior normal."
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
