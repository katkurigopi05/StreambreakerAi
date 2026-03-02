"""
Streamlit web demo for Marketing Strategy Generator
Run: streamlit run app_demo.py
"""

import streamlit as st
from main import MarketingStrategyGenerator

# Page config
st.set_page_config(
    page_title="StreamBreaker AI - Marketing Generator",
    page_icon="🎵",
    layout="wide"
)

# Title
st.title("🎵 StreamBreaker AI - Marketing Strategy Generator")
st.markdown("**Model 3 Demo** by Miguel Davila")
st.divider()

# Initialize generator
@st.cache_resource
def get_generator():
    return MarketingStrategyGenerator()

generator = get_generator()

# Sidebar inputs
st.sidebar.header("📊 Input Parameters")

# Model 1 output (Harsh)
st.sidebar.subheader("Model 1: Prediction")
prediction = st.sidebar.slider(
    "Probability of 1K streams (%)",
    min_value=0,
    max_value=100,
    value=87,
    step=1
)

# User input
st.sidebar.subheader("Artist Profile")
budget = st.sidebar.number_input(
    "Marketing Budget ($)",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)

genre = st.sidebar.selectbox(
    "Genre",
    ["Indie Pop", "Pop", "Indie Folk", "Electronic", "Hip Hop", "R&B", "Rock", "Experimental"]
)

followers = st.sidebar.number_input(
    "Instagram Followers",
    min_value=0,
    max_value=100000,
    value=1200,
    step=100
)

listeners = st.sidebar.number_input(
    "Spotify Monthly Listeners",
    min_value=0,
    max_value=50000,
    value=350,
    step=50
)

# Model 2 output (Stephanie)
st.sidebar.subheader("Model 2: Lyric Analysis")
hook_strength = st.sidebar.slider(
    "Hook Strength (0-10)",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.5
)

has_lyrics = st.sidebar.checkbox("Has Lyrics", value=True)

# Audio features
st.sidebar.subheader("Audio Features")
energy = st.sidebar.slider("Energy (0-10)", 0.0, 10.0, 7.5, 0.5)
danceability = st.sidebar.slider("Danceability (0-10)", 0.0, 10.0, 7.0, 0.5)

# Generate button
if st.sidebar.button("🚀 Generate Strategy", type="primary", use_container_width=True):
    
    with st.spinner("🤖 AI is generating your marketing strategy..."):
        
        # Call your model
        result = generator.generate_strategy(
            prediction_probability=prediction,
            budget=budget,
            genre=genre,
            instagram_followers=followers,
            spotify_listeners=listeners,
            has_fanbase=followers > 500,
            energy=energy,
            has_lyrics=has_lyrics,
            hook_strength=hook_strength,
            danceability=danceability
        )
        
        # Display results
        if result["success"]:
            # Success metrics at top
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", f"{prediction}%", 
                         "High Confidence" if prediction >= 70 else "Moderate")
            with col2:
                st.metric("Budget", f"${budget:,}")
            with col3:
                st.metric("API Cost", f"${result['metadata']['cost_estimate']:.4f}")
            
            st.divider()
            
            # Strategy output
            st.subheader("📋 Marketing Strategy")
            st.markdown(result["strategy"])
            
            # Metadata
            st.divider()
            st.caption(f"✅ Generated using {result['metadata']['model']} | "
                      f"Tokens: {result['metadata']['tokens_used']}")
            
        else:
            st.error(f"❌ Error: {result['error']}")

# Show info when not generated yet
else:
    st.info("👈 Adjust parameters in the sidebar and click **Generate Strategy** to see results")
    
    # Show example
    st.subheader("📖 How It Works")
    st.markdown("""
    This demo shows **Model 3** of the StreamBreaker AI system:
    
    1. **Model 1 (Harsh)** predicts streaming success → You simulate with the slider
    2. **Model 2 (Stephanie)** analyzes lyrics → You simulate with hook strength
    3. **Model 3 (YOU - Miguel)** generates marketing strategy → **This is your model!**
    4. **Model 4 (Reddy)** orchestrates everything in production
    
    Try different scenarios:
    - High prediction (90%) + high budget ($5000)
    - Low prediction (30%) + low budget ($500)
    - Moderate prediction (65%) + moderate budget ($1500)
    """)
