from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5000"}})  # Adjust as needed

model = tf.keras.models.load_model('cancer_model.h5')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler (no fitting required)

# Features expected for prediction
expected_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]


@app.route('/')
def home():
    return render_template('cancer.html')  # Render the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.json
        print(f"Received data: {data}")  # Log the received data for debugging

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Ensure all expected features are present in the received data
        for feature in expected_columns:
            if feature not in data:
                data[feature] = 0  # Default value if the feature is missing
        
        print(f"Modified data (with missing features filled): {data}")

        # Convert input data into a list
        user_input = [float(data[feature]) for feature in expected_columns]

        # Convert input to numpy array and scale
        user_input_np = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(user_input_np)

        # Make prediction using the model
        prediction_prob = model.predict(input_scaled)[0][0]
        diagnosis = 'Malignant' if prediction_prob > 0.5 else 'Benign'

        # Return the diagnosis and probability as JSON
        return jsonify({
            'diagnosis': diagnosis,
            'probability': float(prediction_prob)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")  # Log any error for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
