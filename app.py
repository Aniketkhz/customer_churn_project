from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model and feature columns
model, feature_columns = joblib.load('model/churn_artifacts.pkl')

@app.route('/')
def home():
    return "Customer Churn Prediction API is running ✅"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input from request
    input_data = request.get_json()

    # Convert TotalCharges to numeric
    input_data['TotalCharges'] = pd.to_numeric(input_data.get('TotalCharges', 0), errors='coerce').fillna(0)

    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)

    # Add missing columns with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[feature_columns]

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    # Return JSON response
    return jsonify({
        "churn": int(prediction[0]),
        "churn_probability": float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)