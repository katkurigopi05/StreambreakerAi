"""
StreamBreaker AI — Web Demo (Streamlit)
Model 4 — Orchestrator Web Interface by Gopi Krishna

Run: /opt/anaconda3/envs/anaconda-ml-ai/bin/streamlit run app.py
"""

import streamlit as st
import sys
import os

# Ensure project is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import StreamBreakerPipeline

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StreamBreaker AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: -8px;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e5ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        border-bottom: 2px solid #667eea;
        padding-bottom: 8px;
        margin-top: 24px;
    }

    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }

    .warning-banner {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #eef0ff 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INIT PIPELINE (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(backend="ollama", model=None, api_key=None):
    return StreamBreakerPipeline(backend=backend, model=model, api_key=api_key)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎵 StreamBreaker AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Music Marketing Strategy Platform — Capstone Project</p>', unsafe_allow_html=True)

# Team credits
with st.expander("👥 Team Members", expanded=False):
    cols = st.columns(5)
    with cols[0]:
        st.markdown("**Model 0** — Team  \n*Audio Feature Extractor*")
    with cols[1]:
        st.markdown("**Model 1** — Harsh  \n*XGBoost Prediction*")
    with cols[2]:
        st.markdown("**Model 2** — Stephanie  \n*NLP Lyric Analysis*")
    with cols[3]:
        st.markdown("**Model 3** — Miguel  \n*LLM Marketing Strategy*")
    with cols[4]:
        st.markdown("**Model 4** — Gopi Krishna  \n*Orchestration & Web App*")

st.divider()


# ─────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Track Configuration")

# LLM Backend
st.sidebar.markdown("### 🤖 LLM Backend")
llm_backend = st.sidebar.selectbox(
    "Backend",
    ["Ollama (Local)", "Groq (Fast & Free)", "OpenAI"],
    index=2,  # Set OpenAI as default for public cloud deployment
)

backend_map = {"Ollama (Local)": "ollama", "Groq (Fast & Free)": "groq", "OpenAI": "openai"}
selected_backend = backend_map[llm_backend]

api_key_input = ""
if selected_backend != "ollama":
    try:
        api_key_input = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key_input = os.getenv("OPENAI_API_KEY", "")
    
    if api_key_input:
        st.sidebar.caption("✅ Processing via Secure Cloud Backend")
    else:
        st.sidebar.error("⚠️ API Key not configured in backend secrets. App will not function properly.")
else:
    st.sidebar.caption("Using local Ollama — no API key needed")
    st.sidebar.caption("⏱️ First run may take 2-3 min (model loads into memory)")

# ── Model 0: Audio Upload ─────────────────────────────────
st.sidebar.markdown("### 🎵 Model 0: Audio Upload")
st.sidebar.caption("Upload a track to auto-fill all audio features below")

uploaded_audio = st.sidebar.file_uploader(
    "Upload", type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="200MB per file • MP3, WAV, M4A, OGG, FLAC"
)

# Default feature values
defaults = {
    "danceability": 0.65, "energy": 0.75, "valence": 0.55,
    "acousticness": 0.15, "speechiness": 0.05, "instrumentalness": 0.0,
    "liveness": 0.12, "loudness": -6.0, "tempo": 125, "duration_ms": 210000,
    "key": 5, "mode": 1, "time_signature": 4,
}
audio_metadata = None

if uploaded_audio is not None:
    try:
        from model0_audio import extract_features, get_file_metadata
        with st.sidebar.status("🔄 Analyzing audio with Model 0...", expanded=True) as status:
            st.write("Extracting audio features via librosa...")
            feats, err = extract_features(uploaded_audio, filename=uploaded_audio.name)
            if err:
                st.sidebar.error(f"Audio analysis failed: {err}")
            elif feats:
                defaults.update(feats)
                st.sidebar.success(f"✅ Auto-filled from: **{uploaded_audio.name}**")
                # Get metadata (title, artist, embedded lyrics)
                uploaded_audio.seek(0)
                audio_metadata = get_file_metadata(uploaded_audio, filename=uploaded_audio.name)
            status.update(label="✅ Audio analyzed!", state="complete")
    except ImportError:
        st.sidebar.warning("⚠️ librosa not installed. Using manual sliders.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Audio Features
st.sidebar.markdown("### 🎧 Audio Features")
st.sidebar.caption("Auto-filled from upload · or set manually")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    danceability = st.slider("Danceability", 0.0, 1.0, float(defaults["danceability"]), 0.05)
    energy = st.slider("Energy", 0.0, 1.0, float(defaults["energy"]), 0.05)
    valence = st.slider("Valence", 0.0, 1.0, float(defaults["valence"]), 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, float(defaults["acousticness"]), 0.05)

with col_b:
    speechiness = st.slider("Speechiness", 0.0, 1.0, float(defaults["speechiness"]), 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, float(defaults["instrumentalness"]), 0.01)
    liveness = st.slider("Liveness", 0.0, 1.0, float(defaults["liveness"]), 0.01)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, float(defaults["loudness"]), 0.5)

tempo = st.sidebar.slider("Tempo (BPM)", 60, 220, int(defaults["tempo"]), 5)
duration_sec = st.sidebar.slider("Duration (sec)", 60, 600, int(defaults["duration_ms"] / 1000), 10)
key = st.sidebar.selectbox("Key", list(range(12)), index=int(defaults["key"]),
                            format_func=lambda x: ["C", "C#", "D", "D#", "E", "F",
                                                     "F#", "G", "G#", "A", "A#", "B"][x])
mode = st.sidebar.selectbox("Mode", [0, 1], index=int(defaults["mode"]),
                             format_func=lambda x: "Major" if x == 1 else "Minor")
time_signature = st.sidebar.selectbox("Time Signature", [3, 4, 5], index=[3,4,5].index(int(defaults["time_signature"])))
explicit = st.sidebar.checkbox("Explicit Content", value=False)

genre = st.sidebar.selectbox(
    "Genre",
    ["indie", "indie-pop", "indie-rock", "indie-folk", "folk", "acoustic",
     "alternative", "singer-songwriter", "dream-pop", "lo-fi", "pop",
     "rock", "electronic", "hip-hop", "r-n-b"],
    index=1,
)

# Artist Profile
st.sidebar.markdown("### 👤 Artist Profile")
instagram = st.sidebar.number_input("Instagram Followers", 0, 1000000, 1200, 100)
spotify_listeners = st.sidebar.number_input("Spotify Monthly Listeners", 0, 500000, 350, 50)
youtube_subs = st.sidebar.number_input("YouTube Subscribers", 0, 1000000, 800, 100)
career_stage = st.sidebar.selectbox("Career Stage", ["emerging", "growing", "established"])

# Budget
st.sidebar.markdown("### 💰 Marketing Budget")
budget = st.sidebar.number_input("Budget (USD)", 100, 50000, 1500, 100)

# Lyrics
st.sidebar.markdown("### 📝 Song Lyrics")
lyrics = st.sidebar.text_area(
    "Paste lyrics (include [Verse], [Chorus] headers for best analysis)",
    height=200,
    value="""[Verse 1]
I've been walking through the city lights
Chasing shadows in the neon nights
Every corner turns to something new
But all I see is shades of you

[Chorus]
Take me higher, take me higher
Set my heart on fire, fire
We're burning brighter through the rain
Take me higher once again

[Verse 2]
The music plays beneath the stars
We dance like nothing leaves a scar
Your laughter echoes through the street
A melody I can't delete

[Chorus]
Take me higher, take me higher
Set my heart on fire, fire
We're burning brighter through the rain
Take me higher once again""",
)


# ─────────────────────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────────────────────
if st.sidebar.button("🚀 Run StreamBreaker AI", type="primary", use_container_width=True):

    # Load pipeline with selected backend
    pipeline = load_pipeline(
        backend=selected_backend,
        api_key=api_key_input if api_key_input else None,
    )

    # Prepare audio features
    audio_features = {
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": float(tempo),
        "duration_ms": duration_sec * 1000,
        "time_signature": time_signature,
        "explicit": explicit,
        "genre": genre,
    }

    artist_profile = {
        "instagram_followers": instagram,
        "spotify_listeners": spotify_listeners,
        "youtube_subscribers": youtube_subs,
        "genre": genre.replace("-", " ").title(),
    }

    # Run pipeline with progress
    progress = st.progress(0, text="Initializing pipeline...")

    progress.progress(10, text="📊 Running Model 1: XGBoost Prediction...")
    
    # Show special warning for Ollama
    ollama_warning = None
    if selected_backend == "ollama":
        ollama_warning = st.warning("⏳ **Loading local AI model...** This may take 1-3 minutes on the first run as the 18GB model loads into memory. Please wait.")
        progress.progress(40, text="🎤 Running Model 2: NLP Analysis...")
        progress.progress(70, text="🚀 Running Model 3: Local LLM (Loading model into memory...)")
        
    result = pipeline.run(
        audio_features=audio_features,
        lyrics=lyrics,
        budget=budget,
        artist_profile=artist_profile,
        career_stage=career_stage,
    )
    
    if ollama_warning:
        ollama_warning.empty()
        
    progress.progress(100, text="✅ Pipeline complete!")

    # ── RESULTS ───────────────────────────────────────────
    prediction = result["model1_prediction"]
    nlp = result["model2_nlp"]
    strategy = result["model3_strategy"]

    # Top metrics
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        pred_pct = prediction["prediction_probability"]
        color = "#10b981" if pred_pct >= 70 else "#f59e0b" if pred_pct >= 50 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color}">{pred_pct}%</div>
            <div class="metric-label">Stream Prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(nlp["sentiment"], "😐")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sentiment_emoji}</div>
            <div class="metric-label">Sentiment: {nlp['sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        hook = nlp["hook_repetition"]
        hook_color = "#10b981" if hook >= 0.7 else "#f59e0b" if hook >= 0.4 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {hook_color}">{hook:.0%}</div>
            <div class="metric-label">Hook Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #4f46e5">${budget:,}</div>
            <div class="metric-label">Budget</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Prediction & NLP Details
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">📊 Model 1: Prediction Details</div>',
                    unsafe_allow_html=True)

        if pred_pct >= 70:
            st.markdown('<div class="success-banner">✅ HIGH POTENTIAL — Invest in marketing!</div>',
                        unsafe_allow_html=True)
        elif pred_pct >= 50:
            st.markdown('<div class="warning-banner">⚠️ MODERATE — Proceed with caution</div>',
                        unsafe_allow_html=True)
        else:
            st.error("🔴 LOW POTENTIAL — Consider improving the track first")

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Prediction Probability** | {pred_pct}% |
        | **Will Hit 1K Streams** | {'✅ Yes' if prediction['will_hit_1k_streams'] else '❌ No'} |
        | **Confidence** | {prediction['confidence']} |
        | **Genre** | {genre.replace('-', ' ').title()} |
        """)

    with col_right:
        st.markdown('<div class="section-header">🎤 Model 2: Lyric Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        | Feature | Value | Interpretation |
        |---------|-------|----------------|
        | **Sentiment** | {nlp['sentiment']} ({nlp.get('sentiment_score', 'N/A')}) | {'Good for mainstream' if nlp['sentiment'] == 'positive' else 'Target niche audiences' if nlp['sentiment'] == 'negative' else 'Versatile targeting'} |
        | **Lexical Diversity** | {nlp['lexical_diversity']:.2f} | {'Rich vocabulary' if nlp['lexical_diversity'] >= 0.7 else 'Moderate complexity' if nlp['lexical_diversity'] >= 0.4 else 'Simple/catchy'} |
        | **Hook Repetition** | {nlp['hook_repetition']:.2f} | {'🔥 HIGH viral potential!' if nlp['hook_repetition'] >= 0.7 else 'Moderate catchiness' if nlp['hook_repetition'] >= 0.4 else 'Weak hook'} |
        | **Semantic Coherence** | {nlp['semantic_coherence']:.2f} | {'Strong theme' if nlp['semantic_coherence'] >= 0.6 else 'Mixed themes'} |
        | **Profanity** | {'⚠️ Yes' if nlp['profanity_detected'] else '✅ Clean'} | {'May limit playlist placement' if nlp['profanity_detected'] else 'Playlist-friendly'} |
        """)

    # Marketing Strategy
    st.markdown('<div class="section-header">🚀 Model 3: Marketing Strategy</div>',
                unsafe_allow_html=True)

    if strategy and strategy.get("success"):
        st.markdown(strategy["strategy"])

        st.divider()
        st.caption(
            f"✅ Generated using {strategy['metadata']['model']} | "
            f"Tokens: {strategy['metadata']['tokens_used']} | "
            f"Cost: ${strategy['metadata']['cost_estimate']:.4f} (local — free!)"
        )
    elif strategy:
        st.error(f"❌ Strategy generation failed: {strategy.get('error', 'Unknown error')}")
        st.info("💡 Make sure Ollama is running. Open the Ollama app or run `ollama serve` in terminal.")
    else:
        st.warning("Strategy not generated.")

    # Errors
    if result.get("errors"):
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for err in result["errors"]:
                st.warning(err)

else:
    # Default state
    st.info("👈 Configure track parameters in the sidebar and click **Run StreamBreaker AI**")

    st.markdown("### 🔄 How It Works")
    flow_cols = st.columns(5)
    with flow_cols[0]:
        st.markdown("""
        #### 🎵 Model 0
        **Audio Feature Extractor**
        - Extracts Spotify-style features
        - Analyzes raw audio files
        - Powered by librosa
        """)
    with flow_cols[1]:
        st.markdown("""
        #### 📊 Model 1
        **XGBoost Prediction**
        - Analyzes audio features
        - Predicts streaming success
        - Returns probability %
        """)
    with flow_cols[2]:
        st.markdown("""
        #### 🎤 Model 2
        **NLP Lyric Analysis**
        - Sentiment detection
        - Hook/catchiness score
        - Vocabulary richness
        """)
    with flow_cols[3]:
        st.markdown("""
        #### 🚀 Model 3
        **LLM Strategy Generator**
        - Uses real predictions
        - Cites industry benchmarks
        - Actionable marketing plan
        """)
    with flow_cols[4]:
        st.markdown("""
        #### 🎯 Model 4
        **Orchestrator (This App)**
        - Wires all models together
        - Unified pipeline
        - Interactive web interface
        """)
