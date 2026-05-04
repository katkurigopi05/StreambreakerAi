"""
StreamBreaker AI — Web Demo (Streamlit)
Model 4 — Orchestrator Web Interface by Gopi Krishna

Run: /opt/anaconda3/envs/anaconda-ml-ai/bin/streamlit run app.py
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

# ── Audio Upload ──────────────────────────────────────────
st.sidebar.markdown("### 🎵 Upload Audio")
st.sidebar.caption("Upload a track to auto-fill all audio features")

uploaded_audio = st.sidebar.file_uploader(
    "Upload", type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="200MB per file • MP3, WAV, M4A, OGG, FLAC"
)

# Default feature values
_FEAT_DEFAULTS = {
    "sb_danceability": 0.65, "sb_energy": 0.75, "sb_valence": 0.55,
    "sb_acousticness": 0.15, "sb_speechiness": 0.05, "sb_instrumentalness": 0.0,
    "sb_liveness": 0.12, "sb_loudness": -6.0, "sb_tempo": 125,
    "sb_duration": 210, "sb_key": 5, "sb_mode": 1, "sb_time_sig": 4,
    "sb_lyrics": """[Verse 1]
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
    "sb_last_audio": "",  # tracks last processed filename
}
# Seed session state on first load only
for k, v in _FEAT_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

audio_metadata = None

if uploaded_audio is not None:
    # Only re-extract if it's a NEW file
    if uploaded_audio.name != st.session_state["sb_last_audio"]:
        try:
            from model0_audio import extract_features, get_file_metadata
            with st.sidebar.spinner("🔄 Analyzing audio..."):
                feats, err = extract_features(uploaded_audio, filename=uploaded_audio.name)
                if err:
                    st.sidebar.error(f"Audio analysis failed: {err}")
                elif feats:
                    st.session_state["sb_danceability"]     = round(float(feats.get("danceability",     0.65)), 2)
                    st.session_state["sb_energy"]           = round(float(feats.get("energy",           0.75)), 2)
                    st.session_state["sb_valence"]          = round(float(feats.get("valence",          0.55)), 2)
                    st.session_state["sb_acousticness"]     = round(float(feats.get("acousticness",     0.15)), 2)
                    st.session_state["sb_speechiness"]      = round(float(feats.get("speechiness",      0.05)), 2)
                    st.session_state["sb_instrumentalness"] = round(float(feats.get("instrumentalness", 0.0)),  2)
                    st.session_state["sb_liveness"]         = round(float(feats.get("liveness",         0.12)), 2)
                    st.session_state["sb_loudness"]         = round(float(feats.get("loudness",         -6.0)), 1)
                    st.session_state["sb_tempo"]            = int(feats.get("tempo", 125))
                    st.session_state["sb_duration"]         = max(60, min(600, int(feats.get("duration_ms", 210000) / 1000)))
                    st.session_state["sb_key"]              = int(feats.get("key",  5))
                    st.session_state["sb_mode"]             = int(feats.get("mode", 1))
                    ts = int(feats.get("time_signature", 4))
                    st.session_state["sb_time_sig"]         = ts if ts in [3, 4, 5] else 4
                    st.session_state["sb_last_audio"]       = uploaded_audio.name
                    # Try to extract embedded lyrics
                    uploaded_audio.seek(0)
                    audio_metadata = get_file_metadata(uploaded_audio, filename=uploaded_audio.name)
                    if audio_metadata and audio_metadata.get("embedded_lyrics"):
                        st.session_state["sb_lyrics"] = audio_metadata["embedded_lyrics"]
                    else:
                        # Clear sample lyrics — prompt user to paste real ones
                        st.session_state["sb_lyrics"] = ""
                    # Force re-render so sliders pick up new session_state values
                    st.rerun()
        except ImportError as e:
            st.sidebar.error(f"⚠️ Missing library: {e}. Audio upload disabled.")
        except Exception as e:
            st.sidebar.error(f"Audio error: {e}")
    else:
        st.sidebar.caption(f"✅ Loaded: **{uploaded_audio.name}**")


# Audio Features — use key= so session_state drives slider values
st.sidebar.markdown("### 🎧 Audio Features")
st.sidebar.caption("Auto-filled from upload · or set manually")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    danceability     = st.slider("Danceability",     0.0,  1.0,  step=0.01, key="sb_danceability")
    energy           = st.slider("Energy",           0.0,  1.0,  step=0.01, key="sb_energy")
    valence          = st.slider("Valence",          0.0,  1.0,  step=0.01, key="sb_valence")
    acousticness     = st.slider("Acousticness",     0.0,  1.0,  step=0.01, key="sb_acousticness")

with col_b:
    speechiness      = st.slider("Speechiness",      0.0,  1.0,  step=0.01, key="sb_speechiness")
    instrumentalness = st.slider("Instrumentalness", 0.0,  1.0,  step=0.01, key="sb_instrumentalness")
    liveness         = st.slider("Liveness",         0.0,  1.0,  step=0.01, key="sb_liveness")
    loudness         = st.slider("Loudness (dB)",    -60.0, 0.0, step=0.5,  key="sb_loudness")

tempo          = st.sidebar.slider("Tempo (BPM)",      60,  220, step=1,  key="sb_tempo")
duration_sec   = st.sidebar.slider("Duration (sec)",   60,  600, step=5,  key="sb_duration")
key            = st.sidebar.selectbox("Key",  list(range(12)), index=int(st.session_state["sb_key"]),
                    format_func=lambda x: ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][x])
mode           = st.sidebar.selectbox("Mode", [0, 1], index=int(st.session_state["sb_mode"]),
                    format_func=lambda x: "Major" if x == 1 else "Minor")
time_signature = st.sidebar.selectbox("Time Signature", [3, 4, 5],
                    index=[3,4,5].index(int(st.session_state["sb_time_sig"])))
explicit       = st.sidebar.checkbox("Explicit Content", value=False)


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
    key="sb_lyrics",
)

if uploaded_audio is not None:
    if st.sidebar.button("✨ Extract Lyrics from Audio with Gemini", use_container_width=True):
        import google.generativeai as genai
        import os
        import tempfile
        
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("GEMINI_API_KEY not found in secrets.toml")
        else:
            with st.sidebar.spinner("Uploading and transcribing with Gemini 1.5 Flash... (takes 10-30s)"):
                try:
                    genai.configure(api_key=gemini_api_key)
                    # Write to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        uploaded_audio.seek(0)
                        tmp.write(uploaded_audio.read())
                        tmp_path = tmp.name
                    
                    # Upload to Gemini
                    audio_file = genai.upload_file(path=tmp_path)
                    
                    # Generate content
                    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                    prompt = "Listen to this audio track and transcribe the lyrics exactly as they are sung. Include headers like [Verse], [Chorus] where appropriate. If there are no vocals, just say '[Instrumental]'. Do not include any conversational text, only the lyrics."
                    response = model.generate_content([prompt, audio_file])
                    
                    st.session_state["sb_lyrics"] = response.text
                    os.remove(tmp_path)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Gemini API Error: {e}")


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
    m1, m2, m3, m4, m5 = st.columns(5)

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

    with m5:
        bpm_val = int(tempo)
        bpm_label = "🟢 Dance" if 100 <= bpm_val <= 140 else "🔵 Fast" if bpm_val > 140 else "🔴 Slow"
        bpm_color = "#10b981" if 100 <= bpm_val <= 140 else "#06b6d4" if bpm_val > 140 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {bpm_color}">{bpm_val}</div>
            <div class="metric-label">BPM · {bpm_label}</div>
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

    # ── 3D VISUALIZATIONS (Show/Hide) ───────────────────────
    st.markdown("---")
    with st.expander("📊 View Interactive Analytics & 3D Charts", expanded=False):

        st.markdown("""
        <div style='background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                    border-radius: 16px; padding: 20px 24px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0; font-size: 1.2rem; font-weight: 700;
                       letter-spacing: 0.05em;'>
                🎛️ Audio Intelligence Dashboard
            </h3>
            <p style='color: rgba(255,255,255,0.6); margin: 4px 0 0 0; font-size: 0.85rem;'>
                Interactive charts — drag, zoom, and hover to explore your track's data
            </p>
        </div>
        """, unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            feat_labels = ["Danceability", "Energy", "Valence", "Acousticness",
                           "Speechiness", "Instrumentalness", "Liveness"]
            feat_values = [danceability, energy, valence, acousticness,
                           speechiness, instrumentalness, liveness]
            feat_values_closed = feat_values + [feat_values[0]]
            feat_labels_closed = feat_labels + [feat_labels[0]]

            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=[0.65, 0.7, 0.6, 0.3, 0.08, 0.1, 0.15, 0.65],
                theta=feat_labels_closed,
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.08)',
                line=dict(color='rgba(16, 185, 129, 0.5)', width=1.5, dash='dot'),
                name='🎯 Avg Hit Song'
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=feat_values_closed,
                theta=feat_labels_closed,
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.25)',
                line=dict(color='#a78bfa', width=2.5),
                name='🎵 Your Track',
                marker=dict(size=6, color='#c4b5fd')
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor='rgba(255,255,255,0.03)',
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        tickfont=dict(color='rgba(255,255,255,0.4)', size=9),
                        gridcolor='rgba(255,255,255,0.1)',
                        linecolor='rgba(255,255,255,0.1)',
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='rgba(255,255,255,0.8)', size=11),
                        gridcolor='rgba(255,255,255,0.08)',
                        linecolor='rgba(255,255,255,0.1)',
                    )
                ),
                showlegend=True,
                legend=dict(font=dict(color='white', size=11), bgcolor='rgba(0,0,0,0)'),
                title=dict(text="🎯 Audio Feature Radar", font=dict(color='white', size=14)),
                height=420,
                paper_bgcolor='rgba(15,12,41,0.9)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with viz_col2:
            scatter_fig = go.Figure(data=[go.Scatter3d(
                x=[danceability],
                y=[energy],
                z=[valence],
                mode='markers+text',
                text=[f"  {pred_pct}% ▲"],
                textfont=dict(color='white', size=13),
                textposition='top center',
                marker=dict(
                    size=max(18, pred_pct / 3.5),
                    color=pred_pct,
                    colorscale=[[0,'#ef4444'],[0.4,'#f59e0b'],[0.7,'#10b981'],[1,'#06b6d4']],
                    colorbar=dict(
                        title=dict(text="Score %", font=dict(color='white')),
                        tickfont=dict(color='white'),
                        bgcolor='rgba(0,0,0,0)',
                        outlinecolor='rgba(255,255,255,0.2)',
                    ),
                    opacity=0.9,
                    line=dict(color='rgba(255,255,255,0.4)', width=2)
                ),
                hovertemplate=(
                    f"<b>🎵 Your Track</b><br>"
                    f"Danceability: <b>{danceability:.2f}</b><br>"
                    f"Energy: <b>{energy:.2f}</b><br>"
                    f"Valence: <b>{valence:.2f}</b><br>"
                    f"Stream Score: <b>{pred_pct}%</b><extra></extra>"
                )
            )])
            scatter_fig.update_layout(
                title=dict(text="🌐 3D Feature Space", font=dict(color='white', size=14)),
                scene=dict(
                    xaxis=dict(title="Danceability", range=[0,1],
                               backgroundcolor='rgba(255,255,255,0.02)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True,
                               tickfont=dict(color='rgba(255,255,255,0.6)')),
                    yaxis=dict(title="Energy", range=[0,1],
                               backgroundcolor='rgba(255,255,255,0.02)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True,
                               tickfont=dict(color='rgba(255,255,255,0.6)')),
                    zaxis=dict(title="Valence", range=[0,1],
                               backgroundcolor='rgba(255,255,255,0.02)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True,
                               tickfont=dict(color='rgba(255,255,255,0.6)')),
                    bgcolor='rgba(15,12,41,0.95)',
                ),
                height=420,
                paper_bgcolor='rgba(15,12,41,0.9)',
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Score dashboard bar chart
        bar_labels  = ["Stream Score", "Hook Score", "Lexical Diversity", "Semantic Coherence"]
        bar_values  = [pred_pct/100, nlp["hook_repetition"], nlp["lexical_diversity"], nlp["semantic_coherence"]]
        bar_colors  = ['rgba(102,126,234,0.85)', 'rgba(245,158,11,0.85)',
                       'rgba(16,185,129,0.85)', 'rgba(139,92,246,0.85)']
        bar_borders = ['#818cf8', '#fbbf24', '#34d399', '#a78bfa']

        bar_fig = go.Figure(data=[go.Bar(
            x=bar_labels,
            y=bar_values,
            marker=dict(
                color=bar_colors,
                line=dict(color=bar_borders, width=2),
                pattern=dict(shape=""),
            ),
            text=[f"<b>{v:.0%}</b>" for v in bar_values],
            textposition='outside',
            textfont=dict(color='white', size=13),
            width=0.5,
        )])
        bar_fig.update_layout(
            title=dict(text="📈 Combined Intelligence Score", font=dict(color='white', size=14)),
            yaxis=dict(
                title="Score", range=[0, 1.3], tickformat=".0%",
                gridcolor='rgba(255,255,255,0.06)',
                tickfont=dict(color='rgba(255,255,255,0.6)'),
                zerolinecolor='rgba(255,255,255,0.1)',
            ),
            xaxis=dict(tickfont=dict(color='rgba(255,255,255,0.85)', size=12)),
            height=330,
            paper_bgcolor='rgba(15,12,41,0.9)',
            plot_bgcolor='rgba(255,255,255,0.02)',
            showlegend=False,
            bargap=0.3,
            margin=dict(t=50, b=20, l=40, r=20),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # BPM Gauge
        bpm_zone_color = "#10b981" if 100 <= tempo <= 140 else "#06b6d4" if tempo > 140 else "#f59e0b"
        bpm_zone_label = "Dance Tempo" if 100 <= tempo <= 140 else "Fast / Hype" if tempo > 140 else "Slow / Ballad"
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=tempo,
            delta={"reference": 120, "suffix": " BPM", "font": {"color": "white"}},
            number={"suffix": " BPM", "font": {"color": "white", "size": 32}},
            title={"text": f"🥁 Tempo — {bpm_zone_label}", "font": {"color": "white", "size": 14}},
            gauge={
                "axis": {
                    "range": [60, 220],
                    "tickcolor": "rgba(255,255,255,0.5)",
                    "tickfont": {"color": "rgba(255,255,255,0.6)"},
                },
                "bar": {"color": bpm_zone_color, "thickness": 0.25},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [60, 100],  "color": "rgba(245,158,11,0.15)"},
                    {"range": [100, 140], "color": "rgba(16,185,129,0.15)"},
                    {"range": [140, 220], "color": "rgba(6,182,212,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 120,
                },
            },
        ))
        gauge_fig.update_layout(
            height=280,
            paper_bgcolor="rgba(15,12,41,0.9)",
            font={"color": "white"},
            margin=dict(t=60, b=20, l=40, r=40),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)


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
