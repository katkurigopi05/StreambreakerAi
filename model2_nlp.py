"""
Model 2 — NLP Lyric Analyzer
StreamBreaker AI — Stephanie's lyric analysis model.

Analyzes song lyrics to extract NLP features:
  - Sentiment (positive/negative/neutral)
  - Lexical diversity (vocabulary richness)
  - Hook repetition (catchiness score)
  - Semantic coherence (thematic consistency)
  - Profanity detection
"""

import re
import math
from collections import Counter
from textblob import TextBlob


# Common profanity list (subset for detection)
PROFANITY_WORDS = {
    "fuck", "shit", "damn", "ass", "bitch", "hell", "dick", "cock",
    "pussy", "bastard", "crap", "piss", "slut", "whore", "nigga",
    "nigger", "fag", "faggot", "motherfucker", "bullshit",
}


class LyricAnalyzer:
    """
    Analyzes song lyrics and returns NLP features for the pipeline.
    """

    def analyze(self, lyrics: str) -> dict:
        """
        Analyze lyrics text and return NLP features.

        Args:
            lyrics: Raw lyrics text (can include section headers like [Chorus])

        Returns:
            dict with sentiment, lexical_diversity, hook_repetition,
            semantic_coherence, profanity_detected
        """
        if not lyrics or not lyrics.strip():
            return self._empty_result()

        # Clean lyrics (remove section headers like [Chorus], [Verse 1])
        cleaned = self._clean_lyrics(lyrics)
        words = self._tokenize(cleaned)

        if len(words) < 5:
            return self._empty_result()

        sentiment = self._analyze_sentiment(cleaned)
        lexical_diversity = self._compute_lexical_diversity(words)
        hook_repetition = self._compute_hook_repetition(lyrics, cleaned)
        semantic_coherence = self._compute_semantic_coherence(lyrics)
        profanity_detected = self._detect_profanity(words)

        return {
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "lexical_diversity": round(lexical_diversity, 4),
            "hook_repetition": round(hook_repetition, 4),
            "semantic_coherence": round(semantic_coherence, 4),
            "profanity_detected": profanity_detected,
        }

    def _clean_lyrics(self, lyrics: str) -> str:
        """Remove section headers and extra whitespace."""
        # Remove [Chorus], [Verse 1], etc.
        cleaned = re.sub(r"\[.*?\]", "", lyrics)
        # Remove extra whitespace and newlines
        cleaned = re.sub(r"\n+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _tokenize(self, text: str) -> list:
        """Tokenize text into lowercase words, supporting any language."""
        # [^\W\d_]+ matches any unicode word character (excluding numbers and underscores)
        return re.findall(r"[^\W\d_]+", text.lower())

    def _analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment using TextBlob.
        Returns label (positive/negative/neutral) and score (-1 to 1).
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": round(polarity, 4)}

    def _compute_lexical_diversity(self, words: list) -> float:
        """
        Lexical diversity = unique words / total words.
        0 = very repetitive, 1 = all unique words.
        """
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _compute_hook_repetition(self, raw_lyrics: str, cleaned: str) -> float:
        """
        Measure hook/catchiness by detecting repeated phrases (2-5 word n-grams).
        Higher score = more repetition = catchier.
        """
        words = self._tokenize(cleaned)
        if len(words) < 10:
            return 0.0

        # Extract lines for line-level repetition
        lines = [
            line.strip().lower()
            for line in raw_lyrics.split("\n")
            if line.strip() and not line.strip().startswith("[")
        ]

        if not lines:
            return 0.0

        # Count repeated lines (exact match)
        line_counts = Counter(lines)
        repeated_lines = sum(c for c in line_counts.values() if c > 1)
        total_lines = len(lines)
        line_repeat_ratio = repeated_lines / total_lines if total_lines > 0 else 0

        # Count repeated n-grams (2-4 words)
        ngram_repeat_scores = []
        for n in [2, 3, 4]:
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                continue
            ngram_counts = Counter(ngrams)
            repeated = sum(c for c in ngram_counts.values() if c > 1)
            ngram_repeat_scores.append(repeated / len(ngrams) if ngrams else 0)

        avg_ngram_repeat = (
            sum(ngram_repeat_scores) / len(ngram_repeat_scores)
            if ngram_repeat_scores else 0
        )

        # Combine: 60% line repetition, 40% n-gram repetition
        score = 0.6 * line_repeat_ratio + 0.4 * avg_ngram_repeat

        # Normalize to 0-1 range (cap at 1)
        return min(score * 1.5, 1.0)

    def _compute_semantic_coherence(self, raw_lyrics: str) -> float:
        """
        Measure thematic consistency by comparing vocabulary overlap between
        sections (verses, choruses). Higher = more coherent theme.
        """
        # Split into sections
        sections = re.split(r"\[.*?\]", raw_lyrics)
        sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]

        if len(sections) < 2:
            return 0.7  # Default for single-section lyrics

        # Compute word sets for each section
        section_words = [set(self._tokenize(s)) for s in sections]

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(section_words)):
            for j in range(i + 1, len(section_words)):
                intersection = len(section_words[i] & section_words[j])
                union = len(section_words[i] | section_words[j])
                if union > 0:
                    similarities.append(intersection / union)

        if not similarities:
            return 0.7

        return sum(similarities) / len(similarities)

    def _detect_profanity(self, words: list) -> bool:
        """Check if any profanity words are present."""
        word_set = set(words)
        return bool(word_set & PROFANITY_WORDS)

    def _empty_result(self) -> dict:
        """Return default values when lyrics are empty/unavailable."""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "lexical_diversity": 0.5,
            "hook_repetition": 0.5,
            "semantic_coherence": 0.5,
            "profanity_detected": False,
        }


# ---------------------------------------------------------------------------
# CLI — test with sample lyrics when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STREAMBREAKER AI — MODEL 2 NLP TEST")
    print("=" * 60)

    # Sample indie pop lyrics for testing
    sample_lyrics = """
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

[Bridge]
And when the morning comes around
We'll still be lost and never found
Just two dreamers in the dark
Lighting up each other's spark

[Chorus]
Take me higher, take me higher
Set my heart on fire, fire
We're burning brighter through the rain
Take me higher once again
"""

    analyzer = LyricAnalyzer()
    result = analyzer.analyze(sample_lyrics)

    print(f"\n🎤 NLP Analysis Results:")
    print(f"   Sentiment:          {result['sentiment']} ({result['sentiment_score']})")
    print(f"   Lexical Diversity:  {result['lexical_diversity']}")
    print(f"   Hook Repetition:    {result['hook_repetition']}")
    print(f"   Semantic Coherence: {result['semantic_coherence']}")
    print(f"   Profanity Detected: {result['profanity_detected']}")
    print()

    # Test with edgy lyrics
    edgy_lyrics = """
[Verse 1]
Damn this broken world we live in
Nothing left that's worth forgiving
Fuck the pain that keeps on giving
I'm just barely even living

[Chorus]
Burn it down, burn it down
Watch the ashes hit the ground
Burn it down, burn it down
Nothing left to be found
"""

    print("--- Edgy lyrics test ---")
    result2 = analyzer.analyze(edgy_lyrics)
    print(f"   Sentiment:          {result2['sentiment']} ({result2['sentiment_score']})")
    print(f"   Lexical Diversity:  {result2['lexical_diversity']}")
    print(f"   Hook Repetition:    {result2['hook_repetition']}")
    print(f"   Profanity Detected: {result2['profanity_detected']}")
