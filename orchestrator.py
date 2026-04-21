"""
Model 4 — Orchestrator Pipeline
StreamBreaker AI — Gopi Krishna's orchestration layer.

Wires together:
  Model 1 (XGBoost prediction) → Model 2 (NLP lyrics) → Model 3 (LLM marketing)
into a single unified pipeline.
"""

import os
import sys
import json

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model1_predictor import StreamBreakerPredictor
from model2_nlp import LyricAnalyzer
from main import MarketingStrategyGenerator


class StreamBreakerPipeline:
    """
    Main orchestration class that ties all models together.

    Usage:
        pipeline = StreamBreakerPipeline()
        result = pipeline.run(
            audio_features={...},
            lyrics="...",
            budget=1500,
            artist_profile={...}
        )
    """

    def __init__(self, backend="ollama", model=None, api_key=None):
        """Initialize all three models."""
        print("🎵 StreamBreaker AI — Initializing Pipeline...")

        # Model 1: XGBoost Predictor
        print("   Loading Model 1 (XGBoost Predictor)...")
        self.predictor = StreamBreakerPredictor()

        # Model 2: NLP Lyric Analyzer
        print("   Loading Model 2 (NLP Lyric Analyzer)...")
        self.nlp_analyzer = LyricAnalyzer()

        # Model 3: Marketing Strategy Generator (LLM)
        print("   Loading Model 3 (Marketing Strategy LLM)...")
        self.marketing_gen = MarketingStrategyGenerator(
            backend=backend, model=model, api_key=api_key
        )

        # Check status
        if self.marketing_gen.check_ollama_status():
            print(f"   ✅ LLM ready ({self.marketing_gen.backend}: {self.marketing_gen.model})")
        else:
            print(f"   ⚠️  LLM not available. Check backend config.")

        print("✅ Pipeline ready!\n")

    def run(
        self,
        audio_features: dict,
        lyrics: str = "",
        budget: int = 1000,
        artist_profile: dict = None,
        career_stage: str = "emerging",
    ) -> dict:
        """
        Run the full StreamBreaker AI pipeline.

        Args:
            audio_features: Dict with Spotify audio features
                Required keys: danceability, energy, key, loudness, mode,
                speechiness, acousticness, instrumentalness, liveness,
                valence, tempo, duration_ms, time_signature, explicit, genre

            lyrics: Song lyrics text (optional, but recommended)

            budget: Marketing budget in USD

            artist_profile: Dict with artist info (optional)
                Keys: instagram_followers, spotify_listeners,
                      youtube_subscribers, genre

            career_stage: "emerging", "growing", or "established"

        Returns:
            Unified result dict with all model outputs
        """
        if artist_profile is None:
            artist_profile = {}

        result = {
            "success": True,
            "model1_prediction": None,
            "model2_nlp": None,
            "model3_strategy": None,
            "errors": [],
        }

        # ── Step 1: Model 1 — Prediction ──────────────────────────
        print("━" * 60)
        print("📊 Step 1: Running Model 1 (XGBoost Prediction)...")
        try:
            prediction = self.predictor.predict(audio_features)
            result["model1_prediction"] = prediction
            print(f"   ✅ Prediction: {prediction['prediction_probability']}%")
            print(f"      Will hit 1K: {prediction['will_hit_1k_streams']}")
            print(f"      Confidence:  {prediction['confidence']}")
        except Exception as e:
            result["errors"].append(f"Model 1 error: {str(e)}")
            result["model1_prediction"] = {
                "prediction_probability": 50.0,
                "will_hit_1k_streams": False,
                "confidence": "Low",
            }
            print(f"   ⚠️  Model 1 failed: {e}. Using default 50%.")

        # ── Step 2: Model 2 — NLP Analysis ────────────────────────
        print("\n🎤 Step 2: Running Model 2 (NLP Lyric Analysis)...")
        try:
            if lyrics and lyrics.strip():
                nlp_result = self.nlp_analyzer.analyze(lyrics)
                result["model2_nlp"] = nlp_result
                print(f"   ✅ Sentiment:         {nlp_result['sentiment']} ({nlp_result['sentiment_score']})")
                print(f"      Lexical Diversity: {nlp_result['lexical_diversity']}")
                print(f"      Hook Repetition:   {nlp_result['hook_repetition']}")
                print(f"      Coherence:         {nlp_result['semantic_coherence']}")
                print(f"      Profanity:         {nlp_result['profanity_detected']}")
            else:
                result["model2_nlp"] = self.nlp_analyzer._empty_result()
                print("   ⚠️  No lyrics provided. Using defaults.")
        except Exception as e:
            result["errors"].append(f"Model 2 error: {str(e)}")
            result["model2_nlp"] = self.nlp_analyzer._empty_result()
            print(f"   ⚠️  Model 2 failed: {e}. Using defaults.")

        # ── Step 3: Model 3 — Marketing Strategy ──────────────────
        print(f"\n🚀 Step 3: Running Model 3 (Marketing Strategy LLM)...")
        print(f"   Using Ollama {self.marketing_gen.model} — this may take 30-60s...")

        pred_prob = result["model1_prediction"]["prediction_probability"]
        nlp = result["model2_nlp"]
        genre = audio_features.get("genre", artist_profile.get("genre", "Indie Pop"))

        # Map Spotify 0-1 features to the 0-10 scale expected by prompts
        energy_scaled = audio_features.get("energy", 0.5) * 10
        danceability_scaled = audio_features.get("danceability", 0.5) * 10

        try:
            strategy_result = self.marketing_gen.generate_strategy(
                prediction_probability=pred_prob,
                budget=budget,
                genre=genre,
                instagram_followers=artist_profile.get("instagram_followers", 500),
                spotify_listeners=artist_profile.get("spotify_listeners", 100),
                youtube_subscribers=artist_profile.get("youtube_subscribers", 0),
                has_fanbase=artist_profile.get("instagram_followers", 0) > 500,
                energy=energy_scaled,
                danceability=danceability_scaled,
                tempo=audio_features.get("tempo", 120.0),
                sentiment=nlp["sentiment"],
                lexical_diversity=nlp["lexical_diversity"],
                hook_repetition=nlp["hook_repetition"],
                semantic_coherence=nlp["semantic_coherence"],
                profanity_detected=nlp["profanity_detected"],
                career_stage=career_stage,
            )

            result["model3_strategy"] = strategy_result

            if strategy_result["success"]:
                print(f"   ✅ Strategy generated! ({strategy_result['metadata']['tokens_used']} tokens)")
            else:
                print(f"   ❌ Strategy failed: {strategy_result['error']}")
                result["errors"].append(f"Model 3 error: {strategy_result['error']}")

        except Exception as e:
            result["errors"].append(f"Model 3 error: {str(e)}")
            result["model3_strategy"] = {
                "success": False,
                "error": str(e),
                "strategy": None,
            }
            print(f"   ❌ Model 3 failed: {e}")

        # ── Summary ───────────────────────────────────────────────
        print("\n" + "━" * 60)
        print("📋 PIPELINE SUMMARY")
        print("━" * 60)
        print(f"   Prediction:  {pred_prob}% ({'✅' if pred_prob >= 50 else '⚠️'})")
        print(f"   Sentiment:   {nlp['sentiment']}")
        print(f"   Hook Score:  {nlp['hook_repetition']}")
        print(f"   Strategy:    {'✅ Generated' if result['model3_strategy'].get('success') else '❌ Failed'}")
        if result["errors"]:
            print(f"   Errors:      {len(result['errors'])}")
            for err in result["errors"]:
                print(f"     - {err}")
        print("━" * 60)

        return result


# ---------------------------------------------------------------------------
# CLI — test the full pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🎵 STREAMBREAKER AI — FULL PIPELINE TEST")
    print("=" * 60)

    pipeline = StreamBreakerPipeline()

    # Test with sample data
    test_audio = {
        "danceability": 0.7,
        "energy": 0.75,
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
    }

    test_lyrics = """
[Verse 1]
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
Take me higher once again
"""

    test_profile = {
        "instagram_followers": 1200,
        "spotify_listeners": 350,
        "youtube_subscribers": 800,
        "genre": "Indie Pop",
    }

    result = pipeline.run(
        audio_features=test_audio,
        lyrics=test_lyrics,
        budget=1500,
        artist_profile=test_profile,
        career_stage="emerging",
    )

    # Print strategy if generated
    if result["model3_strategy"] and result["model3_strategy"].get("success"):
        print("\n📋 MARKETING STRATEGY (first 500 chars):")
        print(result["model3_strategy"]["strategy"][:500])
        print("...")
