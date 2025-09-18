import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load dataset
data = pd.read_csv('data/customer_churn.csv')

# Drop customerID if it exists
if 'customerID' in data.columns:
    data.drop(columns=['customerID'], inplace=True)

# Convert TotalCharges to numeric if present
if 'TotalCharges' in data.columns:
    print("Converting TotalCharges to numeric")
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
else:
    print("⚠️ Column 'TotalCharges' not found in dataset, skipping...")

# Encode target column Churn (Yes→1, No→0)
if 'Churn' in data.columns:
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
else:
    raise KeyError("Column 'Churn' not found in dataset. Please check your CSV.")

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split into features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train an XGBoost classifier
model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=4, use_label_encoder=False)
model.fit(X_train, y_train)

# Print classification report and confusion matrix
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation Results:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model and feature columns
joblib.dump((model, X.columns), 'model/churn_artifacts.pkl')

print("\n✅ Training complete! Model saved at: model/churn_artifacts.pkl")
