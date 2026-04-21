# Team Integration Guide - Model 3 (Marketing Strategy Generator)

## Overview
Model 3 generates marketing strategies based on prediction probability (Model 1) and lyric analysis (Model 2).

---

## Input Requirements

### From Model 1 (Harsh - XGBoost Prediction)
**Need to know:**
- [ ] What is the exact output format? (Python dict? JSON? Float?)
- [ ] What range? (0-1 or 0-100?)
- [ ] Do you provide confidence intervals?
- [ ] Can I access the audio features you used? (danceability, energy, tempo)

**Expected format:**
```python
{
    "prediction_probability": 87.5,  # 0-100 scale
    "confidence_interval": [82.0, 93.0],  # optional
    "audio_features": {
        "danceability": 0.7,
        "energy": 0.8,
        "tempo": 125.0
    }
}
```

---

### From Model 2 (Stephanie - NLP Lyric Analysis)
**Need to know:**
- [ ] What are the exact feature names you output?
- [ ] What scales? (0-1? 0-10? 0-100?)
- [ ] Format of sentiment? ("positive"/"negative"/"neutral" or numbers?)
- [ ] Are all features always available?

**Expected format:**
```python
{
    "sentiment": "positive",  # or "negative", "neutral"
    "lexical_diversity": 0.72,  # 0-1 scale
    "hook_repetition": 0.85,  # 0-1 scale
    "semantic_coherence": 0.78,  # 0-1 scale
    "profanity_detected": False  # boolean
}
```

---

### From User Input (via Gopi Krishna - Model 4)
**Need to know:**
- [ ] How will user data be passed to me?
- [ ] What format? (REST API call? Python function?)
- [ ] Which fields are required vs optional?

**Expected format:**
```python
{
    "budget": 1500,
    "genre": "Indie Pop",
    "artist_profile": {
        "instagram_followers": 1200,
        "spotify_monthly_listeners": 350,
        "youtube_subscribers": 800
    }
}
```

---

## Output Format

### Text Output (Current)
```python
{
    "success": True,
    "strategy": "Full text marketing strategy...",
    "metadata": {
        "model": "gpt-3.5-turbo",
        "tokens_used": 1234,
        "cost_estimate": 0.0025
    }
}
```

### JSON Output (For Web App Integration)
```python
{
    "success": True,
    "recommendation": "invest",  # or "skip", "maybe"
    "confidence": "high",  # "very_high", "high", "moderate", "low"
    "prediction_probability": 87.5,
    "budget": 1500,
    "strategy_text": "Full text strategy...",
    "platforms": ["spotify", "instagram", "tiktok"],
    "budget_allocation": {
        "spotify": 600,
        "instagram": 500,
        "tiktok": 300,
        "reserve": 100
    },
    "metadata": {
        "model": "gpt-3.5-turbo",
        "tokens_used": 1234,
        "cost_estimate": 0.0025
    }
}
```

---

## Integration Methods

### Option 1: Python Function Call (Simplest)
```python
from miguel_model import MarketingStrategyGenerator

generator = MarketingStrategyGenerator()

# Gopi Krishna calls this with data from Models 1 & 2
result = generator.generate_strategy_json(
    prediction_probability=harsh_prediction,
    budget=user_budget,
    sentiment=stephanie_sentiment,
    lexical_diversity=stephanie_lexical,
    # ... etc
)
```

### Option 2: REST API Endpoint
```python
# POST /api/generate-strategy
{
    "model1_output": {...},
    "model2_output": {...},
    "user_input": {...}
}

# Response
{
    "success": true,
    "strategy": {...}
}
```

---

## Questions for Next Team Meeting

### For Harsh (Model 1):
1. What's your output format and scale?
2. When will your model be ready for integration?
3. Can you share example predictions?
4. Do you provide audio features in your output?

### For Stephanie (Model 2):
1. What's the exact scale for each feature (0-1? 0-10?)?
2. How is sentiment represented?
3. Can you share example NLP outputs?
4. What happens if lyrics aren't available?

### For Gopi Krishna (Model 4):
1. How should data flow? (Function calls? API?)
2. Where in the UI will my strategy appear?
3. How should errors be handled?
4. Do you need real-time generation or can we cache?
5. What's the deployment timeline?

---

## Testing Plan

### Week 3:
- [ ] Get example outputs from Harsh (Model 1)
- [ ] Get example outputs from Stephanie (Model 2)
- [ ] Create mock data for testing
- [ ] Test end-to-end with mock data

### Week 4:
- [ ] Integrate with Harsh's actual model
- [ ] Integrate with Stephanie's actual model
- [ ] Test with real predictions and lyrics

### Week 5:
- [ ] Full system integration with Gopi Krishna
- [ ] End-to-end testing
- [ ] Performance optimization

---

## Contact

**Miguel Davila** - Model 3 (Marketing Strategy Generator)
- GitHub: [Your repo]
- Email: d3e2j2m1p1@gmail.com

Ready to integrate! Just need the exact data formats from Models 1 & 2.
