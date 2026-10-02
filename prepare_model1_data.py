"""
Model 1 — Training data preparation
StreamBreaker AI

Turns the raw Hugging Face file into the one-row-per-track table that
model1_predictor.train_model() reads.

Source: "Spotify Huge Track Analysis Dataset" by Gildas Le Drogoff (CC BY-NC 4.0)
  https://huggingface.co/datasets/GildasLeDrogoff/spotify-huge-track-analysis-dataset
Download spotify-huge-audio-features.parquet (4.3 GB) into ../capstone/hf-spotify-huge/

Install: pip install -r requirements-dev.txt
Run:     python prepare_model1_data.py
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "capstone", "hf-spotify-huge")
RAW_PATH = os.path.join(DATA_DIR, "spotify-huge-audio-features.parquet")
TRAINING_TABLE_PATH = os.path.join(DATA_DIR, "model1_training.parquet")

# New releases only. The dataset was collected in mid-November 2025, so these
# tracks had been out for 3 to 10 months and their popularity reflects how the
# release went, not years of catalogue listening. January 1 is left out: it
# has about 1.6 times the usual daily releases, which points to tracks where
# only the release year was recorded.
WINDOW_START = "2025-01-02"
WINDOW_END = "2025-08-15"

AUDIO_SOURCE_COLS = [
    "duration_ms", "explicit", "tempo", "key", "mode", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
]


def build_training_table(raw_path: str = RAW_PATH, output_path: str = TRAINING_TABLE_PATH) -> int:
    """
    Write one row per track released inside the window and return the row count.

    The raw file has one row per (track, credited artist). A track's audio
    features and popularity are the same on each of its rows; the lead artist
    is the credited artist with the most followers (alphabetical on a tie, so
    the table comes out the same on every run).
    """
    import duckdb

    audio_select = ",\n                   ".join(f"min({col}) AS {col}" for col in AUDIO_SOURCE_COLS)
    quoted_output = output_path.replace("'", "''")  # COPY ... TO takes a literal, not a parameter
    query = f"""
        COPY (
            SELECT track_id,
                   first(artist_name ORDER BY artist_followers DESC, artist_name) AS lead_artist,
                   max(artist_followers)   AS artist_followers,
                   min(album_release_date) AS release_date,
                   min(track_popularity)   AS popularity,
                   {audio_select}
            FROM read_parquet(?)
            WHERE album_release_date >= CAST(? AS DATE) AND album_release_date < CAST(? AS DATE)
            GROUP BY track_id
            ORDER BY track_id
        ) TO '{quoted_output}' (FORMAT parquet)
    """
    con = duckdb.connect()
    try:
        con.execute(query, [raw_path, WINDOW_START, WINDOW_END])
        return con.execute("SELECT count(*) FROM read_parquet(?)", [output_path]).fetchone()[0]
    finally:
        con.close()


if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise SystemExit(f"Raw dataset not found at {RAW_PATH}\nSee the download note at the top of this file.")

    print(f"📂 Reading {RAW_PATH}")
    print(f"   Keeping releases from {WINDOW_START} up to {WINDOW_END}")
    n_tracks = build_training_table()
    print(f"✅ {n_tracks:,} tracks written to {TRAINING_TABLE_PATH}")
