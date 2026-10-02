"""
Pipeline tests — how Model 1's output reaches Model 3 and the training-data script.
The LLM call is replaced, so no API key or network is needed.

Run: pytest test_pipeline.py -v
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from main import MarketingStrategyGenerator
from orchestrator import StreamBreakerPipeline
from prompts import create_marketing_prompt

SONG = {
    "danceability": 0.65, "energy": 0.75, "key": 5, "loudness": -6.0, "mode": 1,
    "speechiness": 0.05, "acousticness": 0.15, "instrumentalness": 0.0, "liveness": 0.12,
    "valence": 0.55, "tempo": 125.0, "duration_ms": 210000, "time_signature": 4,
    "explicit": False, "genre": "indie-pop",
}


@pytest.fixture
def strategy_calls():
    """Replace Model 3 and record what the pipeline hands it."""
    calls = []

    def fake_generate(self, **kwargs):
        calls.append(kwargs)
        return {"success": True, "strategy": "S", "metadata": {"model": "m", "tokens_used": 1, "cost_estimate": 0.0}}

    with patch.object(MarketingStrategyGenerator, "generate_strategy", fake_generate):
        yield calls


@pytest.fixture(scope="module")
def pipeline():
    return StreamBreakerPipeline(backend="openai", api_key="test-key")


def run(pipeline, followers):
    return pipeline.run(audio_features=SONG, lyrics="", budget=1000,
                        artist_profile={"spotify_followers": followers, "instagram_followers": 0})


def test_spotify_followers_reach_model_1(pipeline, strategy_calls):
    small = run(pipeline, 100)["model1_prediction"]["prediction_probability"]
    large = run(pipeline, 100_000)["model1_prediction"]["prediction_probability"]

    assert large > small


def test_model_3_is_told_what_is_typical_for_that_audience_size(pipeline, strategy_calls):
    prediction = run(pipeline, 1_200)["model1_prediction"]

    assert strategy_calls[0]["typical_probability"] == prediction["typical_probability"]
    assert strategy_calls[0]["outlook"] == prediction["outlook"]


def test_pipeline_carries_on_when_model_1_fails(pipeline, strategy_calls):
    with patch.object(pipeline.predictor, "predict", side_effect=RuntimeError("model file corrupt")):
        result = run(pipeline, 1_200)

    assert any("Model 1 error" in err for err in result["errors"])
    assert result["model1_prediction"]["outlook"] is None
    assert result["model1_prediction"]["typical_probability"] is None
    assert strategy_calls[0]["typical_probability"] is None  # Model 3 still runs, without the comparison


def test_prompt_describes_the_score_honestly():
    prompt = create_marketing_prompt(prediction_probability=31.0, budget=1000,
                                     typical_probability=26.0, outlook="above_typical")

    assert "1,000 streams" not in prompt
    assert "31.0%" in prompt
    assert "26.0%" in prompt
    assert "above average" in prompt.lower()


def test_prompt_still_works_without_the_comparison():
    prompt = create_marketing_prompt(prediction_probability=75.0, budget=1000)

    assert "75.0%" in prompt
    assert "1,000 streams" not in prompt


# ── Training-data script ─────────────────────────────────────

def test_training_table_has_one_row_per_track_led_by_the_biggest_artist(tmp_path):
    pytest.importorskip("duckdb")
    from prepare_model1_data import WINDOW_END, WINDOW_START, build_training_table

    in_window = date.fromisoformat(WINDOW_START)
    audio = {"duration_ms": 200000, "explicit": 0, "tempo": 120.0, "key": 1, "mode": 1, "danceability": 0.5,
             "energy": 0.5, "loudness": -7.0, "speechiness": 0.05, "acousticness": 0.2,
             "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.5}
    rows = [
        {"track_id": "t1", "artist_name": "Small", "artist_followers": 50, "track_popularity": 12, "album_release_date": in_window, **audio},
        {"track_id": "t1", "artist_name": "Big", "artist_followers": 9000, "track_popularity": 12, "album_release_date": in_window, **audio},
        {"track_id": "t2", "artist_name": "Solo", "artist_followers": 300, "track_popularity": 3, "album_release_date": in_window, **audio},
        {"track_id": "old", "artist_name": "Solo", "artist_followers": 300, "track_popularity": 40, "album_release_date": date(2019, 5, 1), **audio},
        {"track_id": "late", "artist_name": "Solo", "artist_followers": 300, "track_popularity": 40, "album_release_date": date.fromisoformat(WINDOW_END), **audio},
        {"track_id": "tie", "artist_name": "Zed", "artist_followers": 700, "track_popularity": 9, "album_release_date": in_window, **audio},
        {"track_id": "tie", "artist_name": "Abe", "artist_followers": 700, "track_popularity": 9, "album_release_date": in_window, **audio},
    ]
    source, output = tmp_path / "raw.parquet", tmp_path / "out.parquet"
    pd.DataFrame(rows).to_parquet(source)

    build_training_table(str(source), str(output))

    table = pd.read_parquet(output).set_index("track_id")
    assert sorted(table.index) == ["t1", "t2", "tie"]  # one row per track; too-old and too-new releases left out
    assert table.loc["tie", "lead_artist"] == "Abe"  # equal followers: same answer on every run
    assert table.loc["t1", "lead_artist"] == "Big"
    assert table.loc["t1", "artist_followers"] == 9000
    assert table.loc["t2", "popularity"] == 3
