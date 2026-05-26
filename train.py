"""
Training script for Smart Agro AI Crop Recommendation Model
Uses Random Forest (scikit-learn only) for reliable Vercel deployment.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("SMART AGRO AI - MODEL TRAINING")
print("=" * 60)

print("\n1. Loading data...")
df = pd.read_csv("data/Crop_recommendation.csv")
print(f"   Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

print("\n2. Creating enhanced features...")
df_enhanced = df.copy()
df_enhanced["NPK_sum"] = df_enhanced["N"] + df_enhanced["P"] + df_enhanced["K"]
df_enhanced["NP_ratio"] = df_enhanced["N"] / (df_enhanced["P"] + 1)
df_enhanced["NK_ratio"] = df_enhanced["N"] / (df_enhanced["K"] + 1)
df_enhanced["PK_ratio"] = df_enhanced["P"] / (df_enhanced["K"] + 1)
df_enhanced["temp_humidity_interaction"] = (
    df_enhanced["temperature"] * df_enhanced["humidity"]
)
df_enhanced["temp_rainfall_interaction"] = (
    df_enhanced["temperature"] * df_enhanced["rainfall"]
)
df_enhanced["humidity_rainfall_interaction"] = (
    df_enhanced["humidity"] * df_enhanced["rainfall"]
)

print("\n3. Preparing data...")
X = df_enhanced.drop(["label"], axis=1)
y = df_enhanced["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n5. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

print("\n6. Evaluating model...")
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
test_f1 = f1_score(y_test, model.predict(X_test_scaled), average="weighted")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"   Test F1-Score: {test_f1:.4f}")

print("\n7. Saving model artifacts...")
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

print("   [ok] saved to artifacts/ and api/artifacts/")
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
