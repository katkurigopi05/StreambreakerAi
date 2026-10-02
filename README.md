# 🎵 StreamBreaker AI
![StreamBreaker Banner](https://img.shields.io/badge/AI--Powered-Music%20Marketing-667eea?style=for-the-badge&logo=spotify&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/XGBoost-ready-brightgreen?style=for-the-badge)
![Profile Visitors](https://komarev.com/ghpvc/?username=katkurigopi05&label=Profile%20views&color=0e75b6&style=flat)

> Privacy note: this counter is served by a third-party badge service and may log standard request metadata required to provide the image count.

**StreamBreaker AI** is a multi-model machine learning pipeline designed to help independent artists estimate how likely a release is to gain traction and automatically generate data-driven marketing strategies for their music. 

This is our collaborative Capstone Project, utilizing acoustic feature analysis, natural language processing for lyrics, and Large Language Models (LLMs) to create an end-to-end orchestration platform.

---

## 👥 The Team & Architecture

The application is structured into four distinct modules orchestrated together:

- **Model 1: Prediction Engine (Harsh)**
  - Analyzes 12 Spotify audio features (e.g., danceability, energy, valence) together with the artist's Spotify follower count. Genre is not an input.
  - Uses an **XGBoost Classifier** to estimate the chance that a new release reaches a Spotify popularity score of 10 or more. Spotify popularity is not a stream count.
  - Tested on artists the model never saw: 71% accuracy, against 64% for always guessing "no". Details and limits are in [Model 1: data and accuracy](#model-1-data-and-accuracy).

- **Model 2: NLP Lyric Analyzer (Stephanie)**
  - Evaluates raw lyrics targeting emotional impact and catchiness.
  - Extracts metrics such as **Sentiment**, **Lexical Diversity**, **Hook Repetition**, and **Semantic Coherence** using `TextBlob`.

- **Model 3: LLM Marketing Strategist (Miguel)**
  - Synthesizes the prediction scores and NLP features to formulate an actionable marketing blueprint.
  - Supports multiple backends: **Ollama** (local/free execution), **Groq**, and **OpenAI**.

- **Model 4: Application Orchestrator & UI (Gopi Krishna)**
  - Connects the three models into a unified pipeline.
  - Wraps the infrastructure into a dynamic, beautiful **Streamlit** web application.

---

## 🚀 Features

- **Spotify Attribute Toggles**: Adjust sliders to mirror your song's BPM, energy, danceability, and more.
- **Copy & Paste Lyrics**: Receive real-time insights on your song's hook repetition and vocabulary richness.
- **Flexible AI Integration**: Bring your own API key for fast inference (OpenAI) or use local AI (Ollama) to keep costs at zero.
- **Actionable Advice**: Outputs budget allocations and platform-specific targeting strategies for IG/TikTok/Spotify.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/katkurigopi05/StreambreakerAi.git
   cd StreambreakerAi
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add API Keys (Optional but Recommended):**
   Create a `.streamlit/secrets.toml` file to securely store your OpenAI or Groq keys. This ensures the cloud deployment works right out of the box!
   ```toml
   OPENAI_API_KEY = "sk-proj-YOUR-API-KEY-HERE"
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment

StreamBreaker AI is fully optimized for **Streamlit Community Cloud**. 
- Simply link this repository to your Streamlit Cloud account.
- **Note:** Ensure you paste your `OPENAI_API_KEY` into the Streamlit Cloud *App Settings -> Secrets* section since local `.streamlit/` folders are safely ignored by git.

## 📄 License & Case Studies

Be sure to check out `CASE_STUDIES.md` and `TEAM_INTEGRATION.md` for our performance benchmarks, testing history, and simulated campaign trackings.

---
*Built for the future of independent artist promotion.*

## Running it

The app is Streamlit.

```bash
pip install streamlit scikit-learn joblib pandas numpy
streamlit run app.py          # full app
streamlit run app_demo.py     # lighter demo
```

`main.py` runs the pipeline headlessly, without the UI.

The individual models are importable on their own — `model0_audio.py`,
`model1_predictor.py` and the rest — which is the easiest way to inspect one
stage in isolation. `streambreaker_model.joblib` is the trained Model 1 and
`streambreaker_metrics.json` holds its test results; both are committed, so
prediction works without retraining.

A `.devcontainer` is included if you would rather not manage the environment
yourself.

`CASE_STUDIES.md`, `TEST_RESULTS.md` and `TEAM_INTEGRATION.md` carry the
evaluation and integration notes.

## Libraries & Methods

Scanned every `.py` file (10 files, 3,122 lines) — five distinct signal
sources actually feed the final prediction, which is what the "5-model"
description refers to.

**Audio** — `librosa` for feature extraction, `mutagen` for file metadata
(`model0_audio.py`).

**NLP / sentiment** — `textblob.TextBlob` for lyric sentiment
(`model2_nlp.py`).

**Classical ML** — `sklearn` (`GroupShuffleSplit`, `accuracy_score`,
`roc_auc_score`, `brier_score_loss`) and `xgboost.XGBClassifier` — gradient
boosting is the model that actually makes the prediction, not just a baseline.

**App** — `streamlit`, `plotly` for the interactive charts.

**Own modules, wired through an orchestrator** — `model1_predictor
.StreamBreakerPredictor`, `model2_nlp.LyricAnalyzer`,
`orchestrator.StreamBreakerPipeline`, `main.MarketingStrategyGenerator`, which
takes the model outputs and produces a text brief via `prompts
.create_marketing_prompt`.

## Model 1: data and accuracy

Model 1 is trained on the
[Spotify Huge Track Analysis Dataset](https://huggingface.co/datasets/GildasLeDrogoff/spotify-huge-track-analysis-dataset)
by Gildas Le Drogoff (CC BY-NC 4.0, non-commercial use). The dataset is not in
this repository.

- **What it predicts:** the chance that a new release reaches a Spotify
  popularity score of 10 or more. About 36% of the training releases did.
- **Training data:** 1.5 million tracks released from January to mid-August
  2025, sampled from 4.6 million, one row per track.
- **How it is tested:** on about 300,000 tracks by artists that do not appear
  in the training data at all. A threshold of 20 was also tried and gave a
  similar picture.

| | Accuracy | Always guessing "no" | ROC-AUC |
|---|---|---|---|
| All test tracks | 70.9% | 64.1% | 0.73 |
| Artists under 1,000 followers | 74.3% | 74.3% | 0.65 |

Known limits:

- Most of what the model knows comes from audience size. For artists under
  1,000 followers it is no more accurate than always guessing "no", although it
  still ranks tracks better than chance.
- The dataset leaves out tracks with zero popularity, so real-world chances are
  lower than the model's numbers.
- Follower counts were collected after release, so they partly reflect a
  track's own success.
- The score is built so that it never falls as followers rise. In the raw data
  it does fall at the very bottom, because accounts with 0-4 followers are
  often playlist-fed background-music catalogues.
- The app's audio upload estimates these features with `librosa`, using its own
  formulas, and they have not been checked against Spotify's values. Loudness
  in particular is measured differently there (relative to the loudest moment
  of the file, not Spotify's overall loudness) and is one of the features the
  model leans on most, so scores from uploaded audio are less reliable than
  scores from values typed in from Spotify.

To retrain, download `spotify-huge-audio-features.parquet` into
`../capstone/hf-spotify-huge/`, then:

```bash
pip install -r requirements-dev.txt
python prepare_model1_data.py   # one row per track, new releases only
python model1_predictor.py      # trains, tests, saves the model and metrics
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest test_app.py test_model1_predictor.py test_pipeline.py
```

These run without an API key, a network connection or the training dataset.
`test_integration.py` is separate: it calls a live LLM.
