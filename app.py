# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:29:56 2024

@author: muslim
"""

import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model from the pickle file
with open('diabetes_model.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def home():
    return 'Welcome to the Diabetes Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Extract features from the JSON data
        features = [data.get('Pregnancies'), data.get('Glucose'), data.get('BloodPressure'),
                    data.get('SkinThickness'), data.get('Insulin'), data.get('BMI'),
                    data.get('DiabetesPedigreeFunction'), data.get('Age')]

        # Convert features to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)

        # Interpret prediction
        result = 'The person has diabetes' if prediction[0] == 1 else 'The person does not have diabetes'

        return jsonify({'result': result}), 200

    except Exception as e:
        # If an error occurs, return an error response
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)