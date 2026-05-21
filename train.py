import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = Path("data") / "Crop_recommendation.csv"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_model(X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "n_classes": len(encoder.classes_),
        "n_features": X.shape[1],
        "n_samples": len(X),
    }

    return model, encoder, metrics


def save_artifacts(model, encoder):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved encoder to: {ENCODER_PATH}")


if __name__ == "__main__":
    X, y = load_data()
    model, encoder, metrics = train_model(X, y)
    save_artifacts(model, encoder)

    print("\nTraining complete")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
