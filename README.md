# 🎵 StreamBreaker AI
![StreamBreaker Banner](https://img.shields.io/badge/AI--Powered-Music%20Marketing-667eea?style=for-the-badge&logo=spotify&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/XGBoost-ready-brightgreen?style=for-the-badge)

**StreamBreaker AI** is a multi-model machine learning pipeline designed to help independent artists predict their streaming success and automatically generate data-driven marketing strategies for their music. 

This is our collaborative Capstone Project, utilizing acoustic feature analysis, natural language processing for lyrics, and Large Language Models (LLMs) to create an end-to-end orchestration platform.

---

## 👥 The Team & Architecture

The application is structured into four distinct modules orchestrated together:

- **Model 1: Prediction Engine (Harsh)**
  - Analyzes 14+ Spotify audio features (e.g., danceability, energy, valence).
  - Uses an **XGBoost Classifier** to predict if a track will hit the 1,000-stream threshold within its first 90 days.

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
