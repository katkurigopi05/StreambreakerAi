"""
Prompt templates for generating marketing strategies based on prediction results
"""

# Industry benchmarks from 10 verified case studies
INDUSTRY_BENCHMARKS = """
VERIFIED CAMPAIGN DATA (10 case studies, $395-$3,867 budgets):

META ADS → SPOTIFY PERFORMANCE:
- Cost Per Stream: $0.004-$0.074 (average: $0.035)
- Average CPC: $0.27-$0.43
- Landing Page Conversion: 60%+
- Organic Multiplier: 8.3x (1 paid click → 8+ total streams)

ALGORITHMIC TRIGGER THRESHOLDS:
- Save Rate >40% by Day 6 → Algorithm tests on Day 7
- 10K streams + 10% save rate in 28 days → Discover Weekly
- 70%+ save rate → Spotify Radio activation
- Timeline: Algorithm typically kicks in Month 2-3

BUDGET-SPECIFIC EXPECTATIONS:
- $400-600: 5K-80K streams (quality dependent)
- $1000-2000: 40K-60K streams + algorithmic pickup
- $2000-4000: 680K-1M+ streams (with viral potential)

GEO-TARGETING ROI:
- Brazil: $0.32 CPC, high engagement
- Mexico: $0.43 CPC
- USA: $0.31 CPC
- Canada: $0.27 CPC (best value in Tier 1)
- Germany: Often identified by Spotify algorithm as strong market

FRONT-LOADING STRATEGY (PROVEN):
- Days 1-2: $100-150/day (build engagement signals fast)
- Days 3-6: $50-85/day (reach Day 6 checkpoint with 40%+ save rate)
- Day 7+: $30/day (reinforce algorithmic momentum)

GENRE-SPECIFIC CPS:
- Tech House (strong creative): $0.0075
- Alt-R&B (high save rate): $0.038
- Dance/EDM: $0.051
- Indie Pop/Rock: $0.022
- Singer-Songwriter: $0.049

COMPLEMENTARY STRATEGIES:
- Playlist Pitching (Groover/SubmitHub): 55% acceptance rate, $200-300 budget
- Pre-save Campaigns: $0.47 per pre-save, 70% landing page CTR
- Playlist-First Ads: Keeps older catalog alive, prevents song decay
"""

# Example output showing proper citation format
EXAMPLE_STRATEGY_OUTPUT = """
EXAMPLE OF PROPERLY CITED STRATEGY (Your outputs must follow this pattern):

**Investment Recommendation:**
Yes, invest in marketing. Based on Indie Pop benchmark CPS of $0.022 and your $600 budget, 
you should generate approximately 15,000-27,000 streams over 4 weeks (budget ÷ CPS × organic multiplier).
With 85% prediction confidence, the track has strong potential to trigger algorithmic pickup.

**Platform Prioritization:**

1. **Spotify (50% = $300):**
   • Meta Ads: $180 targeting US/Canada/Brazil geo-mix
     - Expected CPC: $0.27-$0.31 (benchmark data)
     - Expected clicks: ~600 at $0.30 avg
     - Direct streams: ~600 (60% landing page conversion)
     - With 8.3x organic multiplier: ~5,000 streams
     - Based on Indie Pop CPS of $0.022: ~8,200 streams total
   • Playlist Pitching: $120 via Groover/SubmitHub
     - 55% acceptance rate (benchmark) = 8-10 playlist adds from 15 pitches
     - Expected streams: 2,000-5,000
   • **Total Spotify impact: 10,000-13,000 streams**
   • Rationale: Primary platform for algorithmic growth with high hook repetition (0.80)

2. **TikTok (30% = $180):**
   • Hook repetition score of 0.80 = HIGH viral potential (prioritize)
   • Influencer collaborations: $100 (3-5 micro-influencers, 10K-50K followers)
   • Content creation: $80
   • Expected reach: 50K-200K organic views
   • Indirect Spotify streams: 2,000-5,000

**Week 1 Metrics (Day 6 Checkpoint - CRITICAL):**
- Target: 400-600 streams, 200+ saves (40%+ save rate)
- Rationale: Verified campaigns show 40%+ save rate by Day 6 triggers algorithm on Day 7
- Front-load ad spend: $150/day Days 1-2, then $85/day Days 3-6

YOUR STRATEGIES MUST CITE SPECIFIC BENCHMARK NUMBERS LIKE THIS.
"""

# System prompt incorporating benchmarks and examples
SYSTEM_PROMPT = f"""You are an expert music marketing advisor specializing in independent artists. 
Your role is to create actionable, budget-conscious marketing strategies based on verified industry data.

{INDUSTRY_BENCHMARKS}

{EXAMPLE_STRATEGY_OUTPUT}

You provide:
1. Platform prioritization with CITED benchmark data
2. Calculated stream projections using verified CPS data
3. Week-by-week action plans referencing algorithmic trigger points
4. Realistic expectations using proven ROI benchmarks

CRITICAL RULES:
- ALWAYS cite specific benchmark numbers (e.g., "Based on {{genre}} CPS of $X...")
- ALWAYS calculate expected streams (budget ÷ CPS, then apply organic multiplier)
- ALWAYS reference Day 6 checkpoint for save rate targets (40%+ triggers algorithm)
- ALWAYS show your math (CPC × clicks = cost, clicks × conversion = streams)
- NEVER give generic advice without backing it with benchmark data
- Format monetary amounts as $ values, not just percentages

Consider:
- Artist's current social presence and career stage
- Genre-specific tactics and proven CPS benchmarks
- Budget constraints and expected stream ranges based on verified data
- Predicted success probability
- Algorithmic trigger thresholds (Day 6 checkpoint, save rates, stream velocity)
"""

STRATEGY_PROMPT_TEMPLATE = """
Generate a detailed marketing strategy for an independent artist with the following profile:

**TRACK PREDICTION:**
- Probability of reaching 1,000 streams in 90 days: {prediction_probability}%
- Confidence level: {confidence_level}

**ARTIST PROFILE:**
- Genre: {genre}
- Current Instagram followers: {instagram_followers}
- Current Spotify monthly listeners: {spotify_listeners}
- YouTube subscribers: {youtube_subscribers}
- Has existing fanbase: {has_fanbase}
- Artist career stage: {career_stage}

**TRACK CHARACTERISTICS:**
- Energy level: {energy}/10
- Danceability: {danceability}/10
- Tempo: {tempo} BPM

**LYRIC ANALYSIS** (from Model 2 - NLP):
- Sentiment: {sentiment} (positive/negative/neutral)
- Lexical Diversity: {lexical_diversity}/1.0 (vocabulary richness)
- Hook Repetition: {hook_repetition}/1.0 (catchiness potential)
- Semantic Coherence: {semantic_coherence}/1.0 (lyrical consistency)
- Contains Profanity: {profanity_detected}

**Marketing Implications from Lyrics:**
- {sentiment_marketing_note}
- {lexical_diversity_note}
- {hook_repetition_note}

**BUDGET:**
- Total marketing budget: ${budget}
- Budget flexibility: {budget_flexibility}

---

**MARKETING STRATEGY REQUIREMENTS:**

1. **Investment Recommendation**
   - Should the artist invest in marketing this track? (Yes/No/Maybe)
   - Rationale based on prediction probability and budget
   - **REQUIRED: Calculate expected streams using genre-specific benchmark CPS**
   - Format: "Based on {genre} benchmark CPS of $X, your ${budget} budget should generate approximately Y-Z streams over 4 weeks (budget ÷ CPS × organic multiplier)"
   - Show your math clearly

2. **Platform Prioritization** (Rank 1-4 with budget allocation)
   
   **For EACH platform you recommend, provide:**
   - Percentage of budget + dollar amount
   - Specific tactics (from the options below)
   - Expected results with benchmark citations
   - Timeline for implementation
   
   **Available platforms and tactics:**
   - **Spotify:** Meta ads (cite CPS benchmarks), playlist pitching (Groover/SubmitHub - 55% acceptance rate), pre-save campaigns ($0.47 per pre-save)
   - **Instagram:** Reels, Stories, paid promotions (use for audience building, not primary stream driver)
   - **TikTok:** Organic content, creator partnerships, viral challenges (prioritize if hook_repetition > 0.7)
   - **YouTube:** Music video, lyric video, YouTube Shorts (typically 5-10% of budget unless artist has strong presence)
   
   **REQUIRED FORMAT for each platform:**
   
   "[Rank]. [Platform Name] ([X%] = $[amount]):
      • Tactic 1: $[amount] - [specific expected outcome with benchmark data]
      • Tactic 2: $[amount] - [specific expected outcome with benchmark data]
      • Total expected impact: [quantified result with calculation]
      • Rationale: [Why this platform/budget allocation based on artist profile]"
   
   **Example of proper citation:**
   "1. Spotify (50% = $300):
      • Meta Ads: $180 targeting US/Canada/Brazil geo-mix
        - Expected CPC: $0.27-$0.32 (based on benchmark data)
        - Expected clicks: ~600 ($180 ÷ $0.30)
        - Landing page conversion: 60% = 360 conversions
        - With 8.3x organic multiplier: ~3,000 streams
        - Based on {genre} CPS of $0.022: total ~8,200 streams
      • Playlist Pitching: $120 via Groover/SubmitHub
        - 55% acceptance rate (benchmark) = 8-10 playlist adds from 15 pitches
        - Conservative estimate: 2,000-5,000 additional streams
      • Total Spotify streams: 10,000-13,000 over 4 weeks
      • Rationale: Primary revenue platform, algorithmic potential with 40%+ save rate"
   
   Continue this format for remaining platforms (Instagram, TikTok, YouTube)

3. **4-Week Action Plan**
   Reference algorithmic trigger points:
   
   **Week 1 (Days 1-7): Build toward Day 6 checkpoint**
   - Days 1-2: Front-load Meta Ads ($100-150/day) to build engagement signals
   - Days 3-6: Moderate spend ($50-85/day), focus on driving saves
   - Day 6 TARGET: 40%+ save rate (CRITICAL - this triggers algorithm on Day 7)
   - Calculate saves needed: [X streams × 40% = Y saves]
   - Other specific actions: [platform-specific activities]
   
   **Week 2 (Days 8-14): Algorithmic testing phase**
   - Day 7: Watch for Release Radar expansion (algorithm testing begins)
   - Maintain ad spend at $30/day to reinforce positive signals
   - Monitor: Save rate, streams per listener (target 1.5-2.5+), playlist adds
   - Other specific actions: [platform-specific activities]
   
   **Week 3 (Days 15-21): Scale or pivot**
   - If algorithm picked up (Radio/Discover Weekly appearing): Scale ad spend
   - If not: Analyze root cause (save rate? creative quality? targeting?)
   - Decision point: Continue, pivot, or cut losses
   - Other specific actions: [platform-specific activities]
   
   **Week 4 (Days 22-28): Evaluate and sustain**
   - Compare actual CPS vs. projected CPS
   - Calculate ROI: Total streams ÷ total spend
   - Document learnings for next release
   - Other specific actions: [platform-specific activities]

4. **Budget Breakdown**
   Provide detailed line-item budget with expected outcomes:
   
   **Platform 1:** $X ([X%])
   - Tactic A: $X (expected outcome based on benchmarks with calculation)
   - Tactic B: $X (expected outcome based on benchmarks with calculation)
   
   **Platform 2:** $X ([X%])
   - [Continue same format]
   
   **Reserve fund:** $X (10-15% for pivots/optimization)

5. **Key Success Metrics**
   **CRITICAL: Reference algorithmic trigger thresholds from benchmark data**
   
   **Day 6 Checkpoint (Pre-Algorithm):**
   - Target saves: [X] (calculate: projected streams × 40% = Y saves needed)
   - Rationale: Verified campaigns show 40%+ save rate by Day 6 triggers algorithm testing on Day 7
   
   **Week 1 Target:**
   - Streams: [X range based on front-loaded spend]
   - Saves: [X] (40%+ save rate = CRITICAL)
   - Playlist adds: [X]
   - Streams per listener: 1.5-2.0 (indicates replay value)
   
   **Week 2 Target (Algorithm Testing Phase):**
   - Streams: [X range] (watch for acceleration from Release Radar expansion)
   - Algorithmic sources: Monitor for Radio, Release Radar appearance
   - Streams per listener: Should maintain or increase (2.0+)
   
   **Week 3 Target:**
   - Streams: [X range]
   - Look for: Discover Weekly adds (threshold: 10K streams + 10% save rate in 28 days per benchmark)
   
   **Week 4 Target:**
   - Total streams: [X] (including organic multiplier)
   - Algorithmic traffic percentage: 30-50% (benchmark for successful campaigns)
   - Final CPS: Compare against {genre} benchmark of $[X]

6. **Risk Mitigation**
   - What could go wrong? [Specific scenarios based on {prediction_probability}% confidence]
   - **If not hitting Day 6 checkpoint (40%+ save rate):**
     - Immediate actions: [specific pivots]
     - Budget reallocation: [where to move money from/to]
     - Creative adjustment: [what to change in ads/content]
   - **If algorithmic pickup doesn't happen by Week 3:**
     - Root cause analysis: Save rate too low? Skip rate too high? Wrong targeting?
     - Decision point: Cut campaign or continue at reduced $10-20/day maintenance
   - **If outperforming projections by Week 2:**
     - Scale strategy: Increase daily ad spend to $50-100/day
     - Geographic expansion: Add Germany/UK to targeting
     - Additional playlist pitching: Allocate reserve fund

**CRITICAL REMINDERS:** 
- Every dollar amount MUST have a calculated expected outcome
- Every stream projection MUST reference the verified benchmark CPS data
- Every timeline MUST reference the Day 6 checkpoint and algorithmic triggers
- Show your calculations: Don't just say "expect 10K streams" - show how you got there
- Compare final projections to the {prediction_probability}% confidence level for reasonability check
"""

def create_marketing_prompt(
    prediction_probability: float,
    budget: int,
    genre: str = "Indie Pop",
    instagram_followers: int = 500,
    spotify_listeners: int = 100,
    youtube_subscribers: int = 0,
    has_fanbase: bool = False,
    energy: float = 7.0,
    danceability: float = 6.5,
    tempo: float = 120.0,
    # New parameters from Model 2 (Stephanie's NLP analysis)
    sentiment: str = "positive",
    lexical_diversity: float = 0.5,
    hook_repetition: float = 0.5,
    semantic_coherence: float = 0.5,
    profanity_detected: bool = False,
    career_stage: str = "emerging"
):
    """
    Create a formatted prompt for the LLM
    
    Args:
        prediction_probability: Model 1's prediction (0-100%)
        budget: Marketing budget in USD
        sentiment: Lyric sentiment from Model 2 (positive/negative/neutral)
        lexical_diversity: Vocabulary richness (0-1)
        hook_repetition: Catchiness score (0-1)
        semantic_coherence: Lyrical consistency (0-1)
        Other params: Artist and track characteristics
    
    Returns:
        Formatted prompt string
    """
    
    # Determine confidence level
    if prediction_probability >= 85:
        confidence_level = "Very High"
    elif prediction_probability >= 70:
        confidence_level = "High"
    elif prediction_probability >= 50:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    # Determine budget flexibility
    if budget >= 3000:
        budget_flexibility = "High - can test multiple channels"
    elif budget >= 1000:
        budget_flexibility = "Moderate - focus on 2-3 channels"
    else:
        budget_flexibility = "Low - must prioritize single best channel"
    
    # Generate marketing insights from NLP features
    if sentiment == "positive":
        sentiment_marketing_note = "Positive sentiment → Good for mainstream playlists, wider appeal"
    elif sentiment == "negative":
        sentiment_marketing_note = "Negative/dark sentiment → Target alternative/emo playlists, niche audiences"
    else:
        sentiment_marketing_note = "Neutral sentiment → Versatile, can target multiple playlist types"
    
    if lexical_diversity >= 0.7:
        lexical_diversity_note = "High lyrical complexity → Appeal to lyrics-focused blogs, Genius annotations"
    elif lexical_diversity >= 0.4:
        lexical_diversity_note = "Moderate lyrical complexity → Balance between accessibility and depth"
    else:
        lexical_diversity_note = "Simple, repetitive lyrics → Good for TikTok, catchy radio potential"
    
    if hook_repetition >= 0.7:
        hook_repetition_note = "Strong hook repetition → HIGH TikTok/viral potential, prioritize short-form video"
    elif hook_repetition >= 0.4:
        hook_repetition_note = "Moderate hook strength → Standard playlist approach"
    else:
        hook_repetition_note = "Weak hook → Focus on production quality, mood-based playlists"
    
    return STRATEGY_PROMPT_TEMPLATE.format(
        prediction_probability=prediction_probability,
        confidence_level=confidence_level,
        genre=genre,
        instagram_followers=instagram_followers,
        spotify_listeners=spotify_listeners,
        youtube_subscribers=youtube_subscribers,
        has_fanbase="Yes" if has_fanbase else "No",
        career_stage=career_stage,
        energy=energy,
        danceability=danceability,
        tempo=tempo,
        sentiment=sentiment,
        lexical_diversity=lexical_diversity,
        hook_repetition=hook_repetition,
        semantic_coherence=semantic_coherence,
        profanity_detected="Yes" if profanity_detected else "No",
        sentiment_marketing_note=sentiment_marketing_note,
        lexical_diversity_note=lexical_diversity_note,
        hook_repetition_note=hook_repetition_note,
        budget=budget,
        budget_flexibility=budget_flexibility
    )