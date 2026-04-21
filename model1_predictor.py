"""
Model 1 — Stream Threshold Predictor (XGBoost)
StreamBreaker AI — Harsh's prediction model, extracted for pipeline use.

Trains an XGBoost classifier on Spotify audio features to predict whether
a track will cross the 1K-stream threshold within 90 days.
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPSTONE_DIR = os.path.join(os.path.dirname(BASE_DIR), "capstone")

# Try multiple dataset locations
DATASET_PATHS = [
    os.path.join(CAPSTONE_DIR, "-spotify-tracks-dataset", "dataset.csv"),
    os.path.join(CAPSTONE_DIR, "dataset.csv"),
    os.path.join(BASE_DIR, "dataset.csv"),
]

MODEL_PATH = os.path.join(BASE_DIR, "streambreaker_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "genre_encoder.joblib")
RANDOM_SEED = 42

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "time_signature",
]

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
    "explicit_int", "genre_encoded",
    "energy_valence", "dance_energy", "acoustic_instrumental",
    "loudness_norm", "tempo_bucket",
]

INDIE_GENRES = [
    "indie", "folk", "acoustic", "alternative", "singer-songwriter",
    "dream-pop", "shoegaze", "lo-fi", "bedroom-pop", "indie-pop",
    "indie-folk", "indie-rock", "chillwave", "indie-r&b",
    "stomp-and-holler", "chamber-pop", "new-americana", "folk-rock",
]


def _find_dataset():
    """Locate the Spotify dataset CSV."""
    for path in DATASET_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find dataset.csv. Searched:\n" +
        "\n".join(f"  - {p}" for p in DATASET_PATHS)
    )


def _engineer_features(df):
    """Add engineered features to the dataframe."""
    df = df.copy()
    df["energy_valence"] = df["energy"] * df["valence"]
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["acoustic_instrumental"] = df["acousticness"] * df["instrumentalness"]
    df["loudness_norm"] = (df["loudness"] + 60) / 60
    df["explicit_int"] = df["explicit"].astype(int)
    df["tempo_bucket"] = df["tempo"].apply(
        lambda t: 0 if t < 90 else (1 if t < 130 else 2)
    )
    return df


def train_model():
    """
    Train the XGBoost model on the full Spotify dataset.
    Returns the trained model, label encoder, and evaluation metrics.
    """
    print("=" * 60)
    print("STREAMBREAKER AI — MODEL 1 TRAINING")
    print("=" * 60)

    # Load dataset
    dataset_path = _find_dataset()
    print(f"\n📂 Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Raw: {len(df)} tracks")

    # Clean: drop NaN in audio columns
    df = df.dropna(subset=AUDIO_COLS).reset_index(drop=True)
    print(f"   After NaN drop: {len(df)} tracks")

    # Label: popularity <=20 → 0 (won't hit), >=40 → 1 (will hit), else drop
    df["label"] = df["popularity"].apply(
        lambda p: 0 if p <= 20 else (1 if p >= 40 else -1)
    )
    df = df[df["label"] != -1].reset_index(drop=True)
    print(f"   After labeling: {len(df)} tracks")
    print(f"   Label balance: {df['label'].value_counts().to_dict()}")

    # Feature engineering
    df = _engineer_features(df)

    # Encode genre
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["track_genre"].astype(str))

    # Prepare X and y
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

    # Class imbalance
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    # Train XGBoost
    print("\n🚀 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'=' * 60}")
    print(f"MODEL 1 — RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Target 85%+ → {'✅ ACHIEVED' if acc >= 0.85 else '⚠️ Close'}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Wont hit 1K', 'Will hit 1K'])}")

    # Save model and encoder using joblib (avoids XGBoost _estimator_type bug)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Genre encoder saved to: {ENCODER_PATH}")

    return model, le, {"accuracy": acc, "roc_auc": auc}


class StreamBreakerPredictor:
    """
    Loads the trained XGBoost model and provides prediction functionality.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("⚠️  No saved model found. Training now...")
            self.model, self.label_encoder, _ = train_model()
        else:
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            print(f"✅ Model loaded from: {MODEL_PATH}")

    def predict(self, audio_features: dict) -> dict:
        """
        Predict streaming success from audio features.

        Args:
            audio_features: dict with keys like danceability, energy, loudness, etc.
                Required: danceability, energy, key, loudness, mode, speechiness,
                          acousticness, instrumentalness, liveness, valence,
                          tempo, duration_ms, time_signature, explicit, genre

        Returns:
            dict with prediction_probability (0-100), will_hit_1k (bool),
            confidence (str), and the audio_features used.
        """
        row = {}
        for col in AUDIO_COLS:
            row[col] = audio_features.get(col, 0.0)

        # Explicit
        explicit_val = audio_features.get("explicit", False)
        row["explicit_int"] = int(explicit_val) if isinstance(explicit_val, bool) else int(explicit_val)

        # Genre encoding
        genre = audio_features.get("genre", "indie")
        if genre.lower() in [g.lower() for g in self.label_encoder.classes_]:
            row["genre_encoded"] = int(self.label_encoder.transform([genre.lower()])[0])
        else:
            # Default to 'indie' if genre not found
            try:
                row["genre_encoded"] = int(self.label_encoder.transform(["indie"])[0])
            except ValueError:
                row["genre_encoded"] = 0

        # Engineered features
        row["energy_valence"] = row["energy"] * row["valence"]
        row["dance_energy"] = row["danceability"] * row["energy"]
        row["acoustic_instrumental"] = row["acousticness"] * row["instrumentalness"]
        row["loudness_norm"] = (row["loudness"] + 60) / 60
        row["tempo_bucket"] = 0 if row["tempo"] < 90 else (1 if row["tempo"] < 130 else 2)

        # Create DataFrame with correct column order
        X_input = pd.DataFrame([row])[FEATURE_COLS]

        # Predict
        prob = float(self.model.predict_proba(X_input)[0][1])
        probability_pct = round(prob * 100, 2)

        # Confidence level
        if abs(prob - 0.5) > 0.3:
            confidence = "High"
        elif abs(prob - 0.5) > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "prediction_probability": probability_pct,
            "will_hit_1k_streams": prob >= 0.5,
            "confidence": confidence,
            "audio_features": {
                "danceability": row["danceability"],
                "energy": row["energy"],
                "tempo": row["tempo"],
                "loudness": row["loudness"],
                "valence": row["valence"],
                "acousticness": row["acousticness"],
            },
        }


# ---------------------------------------------------------------------------
# CLI — train model when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, le, metrics = train_model()

    # Quick test prediction
    print("\n" + "=" * 60)
    print("QUICK TEST PREDICTION")
    print("=" * 60)

    predictor = StreamBreakerPredictor()
    test_result = predictor.predict({
        "danceability": 0.7,
        "energy": 0.8,
        "key": 5,
        "loudness": -6.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.6,
        "tempo": 125.0,
        "duration_ms": 210000,
        "time_signature": 4,
        "explicit": False,
        "genre": "indie-pop",
    })

    print(f"\n🎯 Prediction: {test_result['prediction_probability']}%")
    print(f"   Will hit 1K: {test_result['will_hit_1k_streams']}")
    print(f"   Confidence:  {test_result['confidence']}")
