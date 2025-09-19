import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json

print("🚀 Starting Customer Churn Model Training...")

# Load data
try:
    data = pd.read_csv('data/customer_churn.csv')
    print(f"✅ Data loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    raise FileNotFoundError("❌ Dataset not found! Please ensure 'data/customer_churn.csv' exists.")

# Drop customerID if it exists
if 'customerID' in data.columns:
    data.drop(columns=['customerID'], inplace=True)
    print("✅ CustomerID column dropped")

# Convert TotalCharges to numeric if present
if 'TotalCharges' in data.columns:
    print("🔄 Converting TotalCharges to numeric...")
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    print("✅ TotalCharges converted successfully")
else:
    print("⚠️ Column 'TotalCharges' not found in dataset, skipping...")

# Encode target column Churn (Yes→1, No→0)
if 'Churn' in data.columns:
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print("✅ Target variable 'Churn' encoded")
    
    # Check for class distribution
    churn_dist = data['Churn'].value_counts()
    print(f"📊 Class Distribution - No Churn: {churn_dist[0]}, Churn: {churn_dist[1]}")
else:
    raise KeyError("❌ Column 'Churn' not found in dataset. Please check your CSV.")

# Store original categorical columns for reference
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
print(f"📝 Categorical columns found: {categorical_columns}")

# One-hot encode categorical variables
print("🔄 Performing one-hot encoding...")
data = pd.get_dummies(data, drop_first=True)
print(f"✅ One-hot encoding complete! New shape: {data.shape}")

# Split into features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

print(f"📊 Features: {X.shape[1]} columns")
print(f"📊 Target distribution: {y.value_counts().to_dict()}")

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train/test split complete!")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Train an XGBoost classifier
print("🤖 Training XGBoost model...")
model = XGBClassifier(
    eval_metric='logloss', 
    random_state=42, 
    n_jobs=4, 
    use_label_encoder=False,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

# Make predictions
print("🔍 Evaluating model performance...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate detailed metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Store metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'test_samples': len(y_test)
}

# Print detailed evaluation results
print("\n" + "="*50)
print("📊 MODEL EVALUATION RESULTS")
print("="*50)
print(f"🎯 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall:    {recall:.4f}")
print(f"🎯 F1-Score:  {f1:.4f}")
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred))
print("📋 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model, feature columns, and metrics
artifacts = {
    'model': model,
    'feature_columns': list(X.columns),
    'metrics': metrics,
    'feature_importance': feature_importance.to_dict('records')
}

joblib.dump(artifacts, 'model/churn_artifacts.pkl')
print(f"\n✅ Training complete! Artifacts saved to: model/churn_artifacts.pkl")

# Save metrics to JSON for easy access
with open('model/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save feature importance
feature_importance.to_csv('model/feature_importance.csv', index=False)

print("✅ Additional files saved:")
print("   - model/model_metrics.json")
print("   - model/feature_importance.csv")
print("\n🎉 All done! Your model is ready for deployment.")