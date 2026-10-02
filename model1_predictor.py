"""
Model 1 — Release Traction Predictor (XGBoost)
StreamBreaker AI — Harsh's prediction model, extracted for pipeline use.

Estimates the chance that a new release reaches a Spotify popularity score of
POPULARITY_THRESHOLD or more, from its audio features and the size of the
artist's Spotify audience. Genre is not an input.

What the number is, and is not:
  - Trained on releases from January to mid-August 2025 (see prepare_model1_data.py).
  - Spotify popularity is not a stream count, so this is not a "1K streams" prediction.
  - The source data leaves out tracks with zero popularity, so real-world
    chances are lower than the model's.
  - Follower counts were measured after release, which flatters the result a little.
Honest test results are saved next to the model in streambreaker_metrics.json.
"""

import json
import math
import os
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPSTONE_DIR = os.path.join(os.path.dirname(BASE_DIR), "capstone")

# Built by prepare_model1_data.py
DATASET_PATHS = [
    os.path.join(CAPSTONE_DIR, "hf-spotify-huge", "model1_training.parquet"),
    os.path.join(BASE_DIR, "model1_training.parquet"),
]
DATASET_NAME = "Spotify Huge Track Analysis Dataset (Gildas Le Drogoff, CC BY-NC 4.0)"

MODEL_PATH = os.path.join(BASE_DIR, "streambreaker_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "streambreaker_metrics.json")
RANDOM_SEED = 42

POPULARITY_THRESHOLD = 10       # a release "gains traction" at this Spotify popularity or above
MAX_TRAINING_TRACKS = 1_500_000
TEST_SHARE = 0.2
VALIDATION_SHARE = 0.15         # of what is left after the test tracks are set aside
SMALL_ARTIST_FOLLOWERS = 1_000  # the app's main audience; reported separately

# Follower counts at which the "typical for your audience size" curve is stored
FOLLOWER_CURVE_POINTS = [0, 1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
OUTLOOK_ABOVE_RATIO = 1.25      # score this many times the typical rate, or more
OUTLOOK_BELOW_RATIO = 0.8       # score this many times the typical rate, or less

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
    "explicit_int",
    "energy_valence", "dance_energy", "acoustic_instrumental",
    "loudness_norm", "tempo_bucket",
    "log_followers",
]


def _find_dataset() -> str:
    """Locate the prepared training table."""
    for path in DATASET_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find model1_training.parquet. Run prepare_model1_data.py first. Searched:\n" +
        "\n".join(f"  - {p}" for p in DATASET_PATHS)
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the model's engineered features added."""
    df = df.copy()
    df["energy_valence"] = df["energy"] * df["valence"]
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["acoustic_instrumental"] = df["acousticness"] * df["instrumentalness"]
    df["loudness_norm"] = (df["loudness"] + 60) / 60
    df["explicit_int"] = df["explicit"].astype(int)
    df["tempo_bucket"] = df["tempo"].apply(
        lambda t: 0 if t < 90 else (1 if t < 130 else 2)
    )
    df["log_followers"] = np.log10(df["artist_followers"].astype(float) + 1)
    return df


def label_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with label 1 at or above the popularity threshold, else 0. Every track is kept."""
    df = df.copy()
    df["label"] = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)
    return df


def split_by_artist(df: pd.DataFrame, test_share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that no artist has tracks on both sides (an artist's songs sound alike and score alike)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["lead_artist"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def typical_curve(labelled: pd.DataFrame) -> list[dict]:
    """
    How often tracks reach the threshold at each audience size.

    Fitted so the rate never falls as followers rise. In the raw data it does
    fall at the very bottom: accounts with 0-4 followers are often playlist-fed
    background-music catalogues that do better than artists with 5-99, which is
    not something a new artist with no audience can expect.
    """
    log_followers = np.log10(labelled["artist_followers"].astype(float) + 1)
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(log_followers, labelled["label"])
    rates = fit.predict(np.log10(np.array(FOLLOWER_CURVE_POINTS, dtype=float) + 1))
    return [{"followers": n, "rate": round(float(rate), 4)} for n, rate in zip(FOLLOWER_CURVE_POINTS, rates)]


def typical_rate(curve: list[dict], artist_followers: float) -> float:
    """Read the curve at this follower count, interpolating on a log scale so there are no jumps."""
    xs = [math.log10(point["followers"] + 1) for point in curve]
    ys = [point["rate"] for point in curve]
    return float(np.interp(math.log10(max(0.0, float(artist_followers)) + 1), xs, ys))


def outlook_for(probability: float, typical: float) -> str:
    """Compare a track's score with what is typical for artists of the same size."""
    if typical <= 0:
        return "typical"
    ratio = probability / typical
    if ratio >= OUTLOOK_ABOVE_RATIO:
        return "above_typical"
    if ratio <= OUTLOOK_BELOW_RATIO:
        return "below_typical"
    return "typical"


def _as_flag(value) -> int:
    """Read a yes/no value that may arrive as a bool, a number or text ("False" must not count as yes)."""
    if isinstance(value, str):
        return int(value.strip().lower() in ("true", "1", "yes", "y"))
    return int(bool(value))


def _score(model: XGBClassifier, tracks: pd.DataFrame) -> dict:
    """Accuracy, majority-guess baseline and ranking quality on a set of labelled tracks."""
    y_true = tracks["label"]
    y_prob = model.predict_proba(tracks[FEATURE_COLS])[:, 1]
    positive_rate = float(y_true.mean())
    return {
        "tracks": int(len(tracks)),
        "positive_rate": round(positive_rate, 4),
        "accuracy": round(float(accuracy_score(y_true, y_prob >= 0.5)), 4),
        "baseline_accuracy": round(max(positive_rate, 1 - positive_rate), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def train_model():
    """
    Train the XGBoost model on recent releases and test it on artists it never saw.
    Returns the trained model and the evaluation metrics.
    """
    print("=" * 60)
    print("STREAMBREAKER AI — MODEL 1 TRAINING")
    print("=" * 60)

    # Load dataset
    dataset_path = _find_dataset()
    print(f"\n📂 Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"   Prepared table: {len(df)} tracks (one row per track)")
    release_window = [str(df["release_date"].min())[:10], str(df["release_date"].max())[:10]]

    if len(df) > MAX_TRAINING_TRACKS:
        df = df.sample(MAX_TRAINING_TRACKS, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"   Sampled down to: {len(df)} tracks")

    # Label: at or above the popularity threshold → 1, everything else → 0
    df = engineer_features(label_tracks(df))
    print(f"   Label balance: {df['label'].value_counts().to_dict()}")

    # Three-way split by artist: train / validation (early stopping) / test (reported)
    train_val, test = split_by_artist(df, TEST_SHARE, RANDOM_SEED)
    train, val = split_by_artist(train_val, VALIDATION_SHARE, RANDOM_SEED + 1)
    shared_artists = len(set(train["lead_artist"]) & set(test["lead_artist"]))
    print(f"\n   Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"   Artists in both train and test: {shared_artists}")

    # Train XGBoost — no class re-weighting, so the output stays a usable probability.
    # The score is not allowed to fall as followers rise (see typical_curve).
    print("\n🚀 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        early_stopping_rounds=30,
        tree_method="hist",
        monotone_constraints=tuple(1 if col == "log_followers" else 0 for col in FEATURE_COLS),
    )

    model.fit(
        train[FEATURE_COLS], train["label"],
        eval_set=[(val[FEATURE_COLS], val["label"])],
        verbose=100,
    )

    # Evaluate on the held-out artists
    overall = _score(model, test)
    small_artists = _score(model, test[test["artist_followers"] < SMALL_ARTIST_FOLLOWERS])

    metrics = {
        "trained_on": date.today().isoformat(),
        "dataset": DATASET_NAME,
        "release_window": release_window,
        "popularity_threshold": POPULARITY_THRESHOLD,
        "train_tracks": int(len(train)),
        "test_tracks": overall["tracks"],
        "artists_shared_between_train_and_test": shared_artists,
        "positive_rate": overall["positive_rate"],
        "accuracy": overall["accuracy"],
        "baseline_accuracy": overall["baseline_accuracy"],
        "roc_auc": overall["roc_auc"],
        "brier": overall["brier"],
        "small_artists": {"max_followers": SMALL_ARTIST_FOLLOWERS, **small_artists},
        "typical_curve": typical_curve(train),
    }

    print(f"\n{'=' * 60}")
    print(f"MODEL 1 — RESULTS (artists the model never saw)")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {overall['accuracy']:.4f}  (always guessing the majority: {overall['baseline_accuracy']:.4f})")
    print(f"ROC-AUC:   {overall['roc_auc']:.4f}")
    print(f"Under {SMALL_ARTIST_FOLLOWERS} followers: accuracy {small_artists['accuracy']:.4f} "
          f"(majority {small_artists['baseline_accuracy']:.4f}), ROC-AUC {small_artists['roc_auc']:.4f}")

    # Save model using joblib (avoids XGBoost _estimator_type bug) and the metrics beside it
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Metrics saved to: {METRICS_PATH}")

    return model, metrics


def load_metrics() -> Optional[dict]:
    """The saved evaluation of the current model, or None if it has not been trained here."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


class StreamBreakerPredictor:
    """
    Loads the trained XGBoost model and provides prediction functionality.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
            print("⚠️  No saved model found. Training now...")
            self.model, self.metrics = train_model()
        else:
            self.model = joblib.load(MODEL_PATH)
            self.metrics = load_metrics()
            print(f"✅ Model loaded from: {MODEL_PATH}")

    def predict(self, audio_features: dict, artist_followers: int = 0) -> dict:
        """
        Estimate a release's chance of gaining traction.

        Args:
            audio_features: dict with keys like danceability, energy, loudness, etc.
                Used: danceability, energy, key, loudness, mode, speechiness,
                      acousticness, instrumentalness, liveness, valence,
                      tempo, duration_ms, explicit. Genre is ignored.
            artist_followers: the artist's Spotify follower count (0 = no audience yet)

        Returns:
            dict with prediction_probability (0-100), typical_probability (0-100,
            for artists with a similar audience), outlook, reaches_popularity_threshold
            (bool), confidence (str), and the audio_features used.

        Raises:
            ValueError: if any of the audio features above is missing.
        """
        missing = [col for col in AUDIO_COLS if audio_features.get(col) is None]
        if missing:
            raise ValueError(f"Missing audio features: {', '.join(missing)}")

        row = {col: audio_features[col] for col in AUDIO_COLS}
        row["explicit"] = _as_flag(audio_features.get("explicit", False))
        row["artist_followers"] = max(0, int(artist_followers or 0))

        # Create DataFrame with correct column order
        X_input = engineer_features(pd.DataFrame([row]))[FEATURE_COLS]

        # Predict
        prob = float(self.model.predict_proba(X_input)[0][1])
        typical = typical_rate(self.metrics["typical_curve"], row["artist_followers"])

        # Confidence level
        if abs(prob - 0.5) > 0.3:
            confidence = "High"
        elif abs(prob - 0.5) > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "prediction_probability": round(prob * 100, 2),
            "typical_probability": round(typical * 100, 2),
            "outlook": outlook_for(prob, typical),
            "reaches_popularity_threshold": prob >= 0.5,
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
    train_model()

    # Quick test prediction
    print("\n" + "=" * 60)
    print("QUICK TEST PREDICTION")
    print("=" * 60)

    predictor = StreamBreakerPredictor()
    test_song = {
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
        "explicit": False,
    }

    for followers in (100, 1_000, 10_000):
        test_result = predictor.predict(test_song, artist_followers=followers)
        print(f"\n🎯 {followers} followers → {test_result['prediction_probability']}% "
              f"(typical for that size: {test_result['typical_probability']}%, {test_result['outlook']})")
