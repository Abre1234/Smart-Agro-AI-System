"""
Training script for Smart Agro AI Crop Recommendation Model
Generates model artifacts for deployment
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

print("=" * 60)
print("SMART AGRO AI - MODEL TRAINING")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv("data/Crop_recommendation.csv")
print(f"   Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Feature Engineering
print("\n2. Creating enhanced features...")
df_enhanced = df.copy()

# NPK ratio features
df_enhanced['NPK_sum'] = df_enhanced['N'] + df_enhanced['P'] + df_enhanced['K']
df_enhanced['NP_ratio'] = df_enhanced['N'] / (df_enhanced['P'] + 1)
df_enhanced['NK_ratio'] = df_enhanced['N'] / (df_enhanced['K'] + 1)
df_enhanced['PK_ratio'] = df_enhanced['P'] / (df_enhanced['K'] + 1)

# Environmental interaction features
df_enhanced['temp_humidity_interaction'] = df_enhanced['temperature'] * df_enhanced['humidity']
df_enhanced['temp_rainfall_interaction'] = df_enhanced['temperature'] * df_enhanced['rainfall']
df_enhanced['humidity_rainfall_interaction'] = df_enhanced['humidity'] * df_enhanced['rainfall']

print(f"   Enhanced features created: {df_enhanced.shape[1]} total features")

# Prepare data
print("\n3. Preparing data...")
X = df_enhanced.drop(['label'], axis=1)
y = df_enhanced['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Number of crop classes: {len(le.classes_)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Scale features
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   Features scaled successfully")

# Train model
print("\n5. Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train_scaled, y_train)
print("   Model training completed")

# Evaluate model
print("\n6. Evaluating model...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Test F1-Score:  {test_f1:.4f}")

# Save artifacts
print("\n7. Saving model artifacts...")
os.makedirs('artifacts', exist_ok=True)

artifact_names = [
    ("model.joblib", model),
    ("scaler.joblib", scaler),
    ("label_encoder.joblib", le),
    ("feature_names.joblib", X.columns.tolist()),
]

for folder in ("artifacts", "api/artifacts"):
    os.makedirs(folder, exist_ok=True)
    for name, obj in artifact_names:
        joblib.dump(obj, f"{folder}/{name}")

print("   [ok] model.joblib")
print("   [ok] scaler.joblib")
print("   [ok] label_encoder.joblib")
print("   [ok] feature_names.joblib")
print("   [ok] copied to api/artifacts for Vercel")

# Test prediction
print("\n8. Testing prediction...")
sample_data = pd.DataFrame({
    'N': [90],
    'P': [42],
    'K': [43],
    'temperature': [20.8],
    'humidity': [82.0],
    'ph': [6.5],
    'rainfall': [202.9]
})

# Add enhanced features
sample_enhanced = sample_data.copy()
sample_enhanced['NPK_sum'] = sample_enhanced['N'] + sample_enhanced['P'] + sample_enhanced['K']
sample_enhanced['NP_ratio'] = sample_enhanced['N'] / (sample_enhanced['P'] + 1)
sample_enhanced['NK_ratio'] = sample_enhanced['N'] / (sample_enhanced['K'] + 1)
sample_enhanced['PK_ratio'] = sample_enhanced['P'] / (sample_enhanced['K'] + 1)
sample_enhanced['temp_humidity_interaction'] = sample_enhanced['temperature'] * sample_enhanced['humidity']
sample_enhanced['temp_rainfall_interaction'] = sample_enhanced['temperature'] * sample_enhanced['rainfall']
sample_enhanced['humidity_rainfall_interaction'] = sample_enhanced['humidity'] * sample_enhanced['rainfall']

sample_scaled = scaler.transform(sample_enhanced)
prediction = model.predict(sample_scaled)[0]
crop_name = le.inverse_transform([prediction])[0]
probabilities = model.predict_proba(sample_scaled)[0]

print(f"   Sample input: N=90, P=42, K=43, Temp=20.8°C")
print(f"   Predicted crop: {crop_name}")
print(f"   Confidence: {probabilities[prediction]*100:.2f}%")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nYou can now:")
print("1. Run the API: python -m uvicorn api.predict:app --reload")
print("2. Test locally: http://localhost:8000")
print("3. Deploy to Vercel")
print("=" * 60)
