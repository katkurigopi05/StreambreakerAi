"""
Integration Test - Mock Data from Team Models
Tests Model 3 (Marketing Generator) with simulated inputs from Models 1 & 2
"""

from main import MarketingStrategyGenerator
import json

def test_scenario_1_high_confidence():
    """Test: High prediction, positive sentiment, strong hook"""
    print("="*60)
    print("TEST SCENARIO 1: High Confidence Track")
    print("="*60)
    
    # Mock Model 1 output (Harsh - XGBoost Prediction)
    model1_prediction = 87.5  # High probability of success
    
    # Mock Model 2 output (Stephanie - NLP Analysis)
    model2_features = {
        "sentiment": "positive",
        "lexical_diversity": 0.72,
        "hook_repetition": 0.85,
        "semantic_coherence": 0.78,
        "profanity_detected": False
    }
    
    # Mock User Input (via Gopi Krishna's web app)
    user_input = {
        "budget": 1500,
        "genre": "Indie Pop",
        "instagram_followers": 1200,
        "spotify_listeners": 350,
        "youtube_subscribers": 800
    }
    
    # Mock Audio Features (from Model 1's data collection)
    audio_features = {
        "energy": 7.5,
        "danceability": 7.0,
        "tempo": 125.0
    }
    
    # Generate Strategy
    generator = MarketingStrategyGenerator()
    result = generator.generate_strategy(
        prediction_probability=model1_prediction,
        budget=user_input["budget"],
        genre=user_input["genre"],
        instagram_followers=user_input["instagram_followers"],
        spotify_listeners=user_input["spotify_listeners"],
        youtube_subscribers=user_input["youtube_subscribers"],
        **model2_features,
        **audio_features
    )
    
    if result["success"]:
        print("\n✅ Strategy Generated Successfully\n")
        print(result["strategy"])
        print(f"\n💰 Cost: ${result['metadata']['cost_estimate']:.4f}")
        print(f"📊 Tokens: {result['metadata']['tokens_used']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*60 + "\n")


def test_scenario_2_low_confidence():
    """Test: Low prediction, negative sentiment, weak hook"""
    print("="*60)
    print("TEST SCENARIO 2: Low Confidence Track")
    print("="*60)
    
    model1_prediction = 35.0  # Low probability
    
    model2_features = {
        "sentiment": "negative",
        "lexical_diversity": 0.45,
        "hook_repetition": 0.30,
        "semantic_coherence": 0.55,
        "profanity_detected": True
    }
    
    user_input = {
        "budget": 500,
        "genre": "Experimental",
        "instagram_followers": 200,
        "spotify_listeners": 50,
        "youtube_subscribers": 100
    }
    
    audio_features = {
        "energy": 4.0,
        "danceability": 3.5,
        "tempo": 95.0
    }
    
    generator = MarketingStrategyGenerator()
    result = generator.generate_strategy(
        prediction_probability=model1_prediction,
        budget=user_input["budget"],
        genre=user_input["genre"],
        instagram_followers=user_input["instagram_followers"],
        spotify_listeners=user_input["spotify_listeners"],
        youtube_subscribers=user_input["youtube_subscribers"],
        **model2_features,
        **audio_features
    )
    
    if result["success"]:
        print("\n✅ Strategy Generated Successfully\n")
        print(result["strategy"])
        print(f"\n💰 Cost: ${result['metadata']['cost_estimate']:.4f}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*60 + "\n")


def test_json_output():
    """Test: JSON output for Gopi Krishna's web app integration"""
    print("="*60)
    print("TEST SCENARIO 3: JSON Output for Web App")
    print("="*60)
    
    generator = MarketingStrategyGenerator()
    result = generator.generate_strategy_json(
        prediction_probability=75.0,
        budget=2000,
        genre="Hip Hop",
        instagram_followers=5000,
        spotify_listeners=1200,
        youtube_subscribers=3000,
        sentiment="positive",
        lexical_diversity=0.68,
        hook_repetition=0.90,
        semantic_coherence=0.72,
        profanity_detected=True,
        energy=8.5,
        danceability=9.0,
        tempo=140.0
    )
    
    if result["success"]:
        print("\n✅ JSON Output Generated\n")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("\n🎵 StreamBreaker AI - Model 3 Integration Tests")
    print("Testing with mock data from Models 1 & 2\n")
    
    # Run all test scenarios
    test_scenario_1_high_confidence()
    test_scenario_2_low_confidence()
    test_json_output()
    
    print("\n✅ All integration tests complete!")
