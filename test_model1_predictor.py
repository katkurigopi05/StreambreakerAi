"""
Model 1 tests — training helpers and the saved predictor.
The helper tests use tiny hand-made tables; the predictor tests use the model
files checked into the repo, so nothing here needs the training dataset.

Run: pytest test_model1_predictor.py -v
"""

import pandas as pd
import pytest

import model1_predictor as m1

SONG = {
    "danceability": 0.65, "energy": 0.75, "key": 5, "loudness": -6.0, "mode": 1,
    "speechiness": 0.05, "acousticness": 0.15, "instrumentalness": 0.0, "liveness": 0.12,
    "valence": 0.55, "tempo": 125.0, "duration_ms": 210000, "time_signature": 4,
    "explicit": False, "genre": "indie-pop",
}


def make_tracks(rows: list[tuple]) -> pd.DataFrame:
    """rows of (lead_artist, artist_followers, popularity) with fixed audio values."""
    audio = {k: v for k, v in SONG.items() if k in m1.AUDIO_COLS}
    return pd.DataFrame([
        {"lead_artist": artist, "artist_followers": followers, "popularity": popularity,
         "explicit": 0, **audio}
        for artist, followers, popularity in rows
    ])


# ── Training helpers ─────────────────────────────────────────

def test_label_keeps_every_track_and_splits_at_the_threshold():
    tracks = make_tracks([("a", 10, 0), ("b", 10, m1.POPULARITY_THRESHOLD - 1),
                          ("c", 10, m1.POPULARITY_THRESHOLD), ("d", 10, 80)])

    labelled = m1.label_tracks(tracks)

    assert len(labelled) == len(tracks)  # no "ambiguous middle" is thrown away
    assert labelled["label"].tolist() == [0, 0, 1, 1]
    assert "label" not in tracks.columns  # input is not modified


def test_split_never_puts_one_artist_on_both_sides():
    tracks = make_tracks([(f"artist{i % 20}", 100, i % 30) for i in range(200)])

    train, test = m1.split_by_artist(tracks, test_share=0.25, seed=1)

    assert len(train) + len(test) == len(tracks)
    assert set(train["lead_artist"]).isdisjoint(test["lead_artist"])
    assert len(test) > 0


def test_features_include_audience_size_and_no_genre():
    features = m1.engineer_features(make_tracks([("a", 999, 5)]))

    assert features["log_followers"].iloc[0] == pytest.approx(3.0)
    assert set(m1.FEATURE_COLS) <= set(features.columns)
    assert not [c for c in m1.FEATURE_COLS if "genre" in c]


def test_typical_curve_never_falls_as_followers_rise():
    # raw data dips in the middle: tiny accounts all hit, mid-size ones all miss
    tracks = m1.label_tracks(make_tracks(
        [(f"tiny{i}", 2, 50) for i in range(30)] +
        [(f"mid{i}", 500, 0) for i in range(30)] +
        [(f"big{i}", 50_000, 50) for i in range(30)]
    ))

    curve = m1.typical_curve(tracks)

    rates = [point["rate"] for point in curve]
    assert [point["followers"] for point in curve] == m1.FOLLOWER_CURVE_POINTS
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_typical_rate_reads_the_curve_smoothly():
    curve = [{"followers": 0, "rate": 0.10}, {"followers": 99, "rate": 0.20}, {"followers": 9_999, "rate": 0.40}]

    assert m1.typical_rate(curve, 0) == pytest.approx(0.10)
    assert m1.typical_rate(curve, 99) == pytest.approx(0.20)
    assert m1.typical_rate(curve, 999) == pytest.approx(0.30)        # halfway between, on a log scale
    assert m1.typical_rate(curve, 5_000_000) == pytest.approx(0.40)  # past the end: last value
    assert abs(m1.typical_rate(curve, 999) - m1.typical_rate(curve, 1_000)) < 0.001  # no jump at round numbers


@pytest.mark.parametrize("probability, typical, expected", [
    (0.50, 0.30, "above_typical"),
    (0.31, 0.30, "typical"),
    (0.20, 0.30, "below_typical"),
    (0.30 * m1.OUTLOOK_ABOVE_RATIO, 0.30, "above_typical"),  # exactly on the cut-offs
    (0.30 * m1.OUTLOOK_BELOW_RATIO, 0.30, "below_typical"),
])
def test_outlook_compares_the_score_with_what_is_typical(probability, typical, expected):
    assert m1.outlook_for(probability, typical) == expected


# ── Saved model ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor():
    return m1.StreamBreakerPredictor()


def test_prediction_has_the_fields_the_pipeline_reads(predictor):
    result = predictor.predict(SONG, artist_followers=1_200)

    assert 0 <= result["prediction_probability"] <= 100
    assert 0 <= result["typical_probability"] <= 100
    assert result["outlook"] in {"above_typical", "typical", "below_typical"}
    assert isinstance(result["reaches_popularity_threshold"], bool)
    assert result["confidence"] in {"High", "Medium", "Low"}
    assert "will_hit_1k_streams" not in result  # the model never predicted stream counts


def test_genre_does_not_change_the_score(predictor):
    scores = {
        predictor.predict({**SONG, "genre": genre}, artist_followers=1_200)["prediction_probability"]
        for genre in ("rock", "acoustic", "indie-rock", "lo-fi", "not-a-real-genre")
    }

    assert len(scores) == 1


def test_a_bigger_audience_raises_the_score(predictor):
    small = predictor.predict(SONG, artist_followers=100)["prediction_probability"]
    large = predictor.predict(SONG, artist_followers=100_000)["prediction_probability"]

    assert large > small


def test_score_never_falls_as_followers_rise(predictor):
    counts = [0, 1, 5, 20, 100, 999, 1_000, 10_000, 100_000, 5_000_000]

    scores = [predictor.predict(SONG, artist_followers=n)["prediction_probability"] for n in counts]

    assert scores == sorted(scores)


def test_missing_followers_is_treated_as_no_audience(predictor):
    assert predictor.predict(SONG) == predictor.predict(SONG, artist_followers=0)


def test_missing_audio_feature_is_an_error_not_a_silent_zero(predictor):
    incomplete = {k: v for k, v in SONG.items() if k != "loudness"}

    with pytest.raises(ValueError, match="loudness"):
        predictor.predict(incomplete, artist_followers=100)


@pytest.mark.parametrize("not_explicit", [False, 0, "False", "false", "0", "no", ""])
def test_explicit_flag_given_as_text_is_read_correctly(predictor, not_explicit):
    clean = predictor.predict({**SONG, "explicit": False}, artist_followers=100)

    assert predictor.predict({**SONG, "explicit": not_explicit}, artist_followers=100) == clean


def test_saved_metrics_say_how_good_the_model_really_is():
    metrics = m1.load_metrics()

    assert metrics["popularity_threshold"] == m1.POPULARITY_THRESHOLD
    assert metrics["accuracy"] > metrics["baseline_accuracy"]  # beats always guessing the majority
    assert 0.5 < metrics["roc_auc"] < 1.0
    assert metrics["test_tracks"] > 10_000
    assert metrics["artists_shared_between_train_and_test"] == 0


def test_saved_metrics_report_small_artists_separately():
    small = m1.load_metrics()["small_artists"]

    assert small["max_followers"] == m1.SMALL_ARTIST_FOLLOWERS
    assert {"accuracy", "baseline_accuracy", "roc_auc", "tracks"} <= set(small)
    assert small["roc_auc"] > 0.5


def test_saved_typical_curve_never_falls():
    rates = [point["rate"] for point in m1.load_metrics()["typical_curve"]]

    assert len(rates) == len(m1.FOLLOWER_CURVE_POINTS)
    assert rates == sorted(rates)
