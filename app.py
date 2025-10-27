from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the model and artifacts
try:
    artifacts = joblib.load('model/churn_artifacts.pkl')
    model = artifacts['model']
    feature_columns = artifacts['feature_columns']
    metrics = artifacts.get('metrics', {})
    print("✅ Model loaded successfully!")
    print(f"📊 Model Accuracy: {metrics.get('accuracy', 'N/A')}")
except FileNotFoundError:
    print("❌ Model file not found! Please run train.py first.")
    model = None
    feature_columns = None
    metrics = {}
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None
    feature_columns = None
    metrics = {}

@app.route('/')
def home():
    """Health check endpoint"""
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train the model first.",
            "timestamp": datetime.now().isoformat()
        }), 500
    
    return jsonify({
        "message": "Customer Churn Prediction API is running ✅",
        "status": "healthy",
        "model_loaded": True,
        "model_metrics": metrics,
        "features_count": len(feature_columns) if feature_columns else 0,
        "endpoints": {
            "/": "Health check",
            "/predict": "Make churn prediction (POST)",
            "/model-info": "Get model information",
            "/sample-input": "Get sample input format"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Get model information and metrics"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": "XGBoost Classifier",
        "features_count": len(feature_columns),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/sample-input')
def sample_input():
    """Provide sample input format"""
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.7,
        "TotalCharges": "848.4"
    }
    
    return jsonify({
        "sample_input": sample,
        "required_fields": "All fields in sample_input are typically required",
        "notes": {
            "TotalCharges": "Can be string or numeric",
            "SeniorCitizen": "0 or 1",
            "tenure": "Number of months",
            "MonthlyCharges": "Numeric value",
            "categorical_fields": "Use exact values as shown in sample"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make churn prediction"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first.",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Get JSON input from request
        input_data = request.get_json()
        
        # Validate input
        if not input_data:
            return jsonify({
                "error": "No JSON data provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Log prediction request (optional - remove in production if needed)
        print(f"🔍 Prediction request received at {datetime.now()}")
        
        # Create a copy to avoid modifying original data
        processed_data = input_data.copy()
        
        # Convert TotalCharges to numeric
        if 'TotalCharges' in processed_data:
            try:
                processed_data['TotalCharges'] = pd.to_numeric(
                    processed_data['TotalCharges'], 
                    errors='coerce'
                )
                if pd.isna(processed_data['TotalCharges']):
                    processed_data['TotalCharges'] = 0
            except:
                processed_data['TotalCharges'] = 0
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([processed_data])
        
        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Add missing columns with 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Remove extra columns that weren't in training
        input_df = input_df[[col for col in input_df.columns if col in feature_columns]]
        
        # Reorder columns to match training
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        
        # Prepare resp
        result = {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(prediction_proba[0]),
            "risk_level": get_risk_level(float(prediction_proba[0])),
            "confidence": float(max(model.predict_proba(input_df)[0])),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Prediction successful: {result['churn_prediction']} (prob: {result['churn_probability']:.3f})")
        
        return jsonify(result)
        
    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print(f"❌ KeyError: {error_msg}")
        return jsonify({
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }), 400
        
    except ValueError as e:
        error_msg = f"Invalid data format: {str(e)}"
        print(f"❌ ValueError: {error_msg}")
        return jsonify({
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }), 400
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ Unexpected error: {error_msg}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }), 500

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/predict", "/model-info", "/sample-input"],
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the HTTP method. /predict requires POST method.",
        "timestamp": datetime.now().isoformat()
    }), 405

if __name__ == '__main__':
    print("🚀 Starting Customer Churn Prediction API...")
    print(f"📊 Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    if feature_columns:
        print(f"📊 Features: {len(feature_columns)} columns")
    print("🌐 API will be available at: http://127.0.0.1:5000/")
    print("📖 Visit http://127.0.0.1:5000/ for API information")
    
    app.run(debug=True, host='0.0.0.0', port=5000)