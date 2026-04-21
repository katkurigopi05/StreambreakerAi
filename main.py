"""
Model 3 — Marketing Strategy Generator (LLM)
StreamBreaker AI — Miguel's marketing strategy generator.

Supports multiple LLM backends:
  1. Ollama (local, free) — default
  2. Groq (fast, free tier)
  3. OpenAI (paid)
"""

import json
import os
import requests
from prompts import SYSTEM_PROMPT, create_marketing_prompt


class MarketingStrategyGenerator:
    """
    Generates marketing strategies using an LLM.
    Supports Ollama (local), Groq, and OpenAI backends.
    """

    def __init__(self, backend="ollama", model=None, api_key=None,
                 temperature=0.7, ollama_url="http://localhost:11434"):
        """
        Initialize the generator.

        Args:
            backend: "ollama", "groq", or "openai"
            model: Model name (auto-selected if None)
            api_key: API key for Groq/OpenAI (can also use env vars)
            temperature: Creativity level (0.0-1.0)
            ollama_url: Ollama API base URL (only for ollama backend)
        """
        self.backend = backend.lower()
        self.temperature = temperature
        self.ollama_url = ollama_url

        # Auto-detect backend from environment
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key and not self.backend:
                self.backend = "groq" if os.getenv("GROQ_API_KEY") else "openai"

        self.api_key = api_key

        # Set model defaults per backend
        if model:
            self.model = model
        elif self.backend == "groq":
            self.model = "llama-3.3-70b-versatile"
        elif self.backend == "openai":
            self.model = "gpt-3.5-turbo"
        else:
            self.model = "qwen3:30b"

        # Set API endpoints
        if self.backend == "groq":
            self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        elif self.backend == "openai":
            self.api_endpoint = "https://api.openai.com/v1/chat/completions"
        else:
            self.api_endpoint = f"{ollama_url}/api/chat"

    def generate_strategy(
        self,
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
        sentiment: str = "positive",
        lexical_diversity: float = 0.5,
        hook_repetition: float = 0.5,
        semantic_coherence: float = 0.5,
        profanity_detected: bool = False,
        career_stage: str = "emerging"
    ) -> dict:
        """
        Generate a marketing strategy based on track prediction and artist profile.
        """
        user_prompt = create_marketing_prompt(
            prediction_probability=prediction_probability,
            budget=budget,
            genre=genre,
            instagram_followers=instagram_followers,
            spotify_listeners=spotify_listeners,
            youtube_subscribers=youtube_subscribers,
            has_fanbase=has_fanbase,
            energy=energy,
            danceability=danceability,
            tempo=tempo,
            sentiment=sentiment,
            lexical_diversity=lexical_diversity,
            hook_repetition=hook_repetition,
            semantic_coherence=semantic_coherence,
            profanity_detected=profanity_detected,
            career_stage=career_stage
        )

        if self.backend == "ollama":
            return self._call_ollama(user_prompt)
        else:
            return self._call_openai_compatible(user_prompt)

    def _call_ollama(self, user_prompt: str) -> dict:
        """Call Ollama local API."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 2000,
                }
            }

            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=600,  # 10 min — qwen3:30b can be slow
            )
            response.raise_for_status()
            result = response.json()
            strategy_text = result["message"]["content"]

            # Strip thinking tags if present (qwen3 uses <think> blocks)
            import re
            strategy_text = re.sub(r'<think>.*?</think>', '', strategy_text, flags=re.DOTALL).strip()

            total_tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)

            return {
                "success": True,
                "strategy": strategy_text,
                "metadata": {
                    "model": self.model,
                    "tokens_used": total_tokens,
                    "cost_estimate": 0.00,
                }
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Make sure Ollama is running.",
                "strategy": None
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Ollama timed out (10 min). Try a smaller model or use an API key.",
                "strategy": None
            }
        except Exception as e:
            return {"success": False, "error": str(e), "strategy": None}

    def _call_openai_compatible(self, user_prompt: str) -> dict:
        """Call OpenAI-compatible API (works for OpenAI and Groq)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": 2000,
            }

            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            strategy_text = result["choices"][0]["message"]["content"]
            total_tokens = result.get("usage", {}).get("total_tokens", 0)

            # Estimate cost
            cost = self._estimate_cost(total_tokens)

            return {
                "success": True,
                "strategy": strategy_text,
                "metadata": {
                    "model": self.model,
                    "tokens_used": total_tokens,
                    "cost_estimate": cost,
                }
            }
        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                error_body = str(e)
            return {"success": False, "error": f"API error: {error_body}", "strategy": None}
        except Exception as e:
            return {"success": False, "error": str(e), "strategy": None}

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate API cost."""
        if self.backend == "groq":
            return 0.00  # Groq free tier
        elif "gpt-4" in self.model:
            return (tokens / 1000) * 0.03
        elif "gpt-3.5" in self.model:
            return (tokens / 1000) * 0.002
        return 0.00

    def generate_strategy_json(self, prediction_probability: float,
                                budget: int, **kwargs) -> dict:
        """Generate structured JSON output for web app."""
        result = self.generate_strategy(
            prediction_probability=prediction_probability,
            budget=budget, **kwargs
        )
        if not result["success"]:
            return result

        strategy_text = result["strategy"]
        text_lower = strategy_text.lower()[:500]

        recommendation = "maybe"
        if "yes" in text_lower or "invest" in text_lower:
            recommendation = "invest"
        elif "no" in text_lower or "skip" in text_lower:
            recommendation = "skip"

        if prediction_probability >= 85:
            confidence = "very_high"
        elif prediction_probability >= 70:
            confidence = "high"
        elif prediction_probability >= 50:
            confidence = "moderate"
        else:
            confidence = "low"

        return {
            "success": True,
            "recommendation": recommendation,
            "confidence": confidence,
            "prediction_probability": prediction_probability,
            "budget": budget,
            "strategy_text": strategy_text,
            "platforms": self._extract_platforms(strategy_text),
            "budget_allocation": self._estimate_budget_allocation(budget, strategy_text),
            "metadata": result["metadata"],
        }

    def _extract_platforms(self, strategy_text: str) -> list:
        platforms = []
        text_lower = strategy_text.lower()
        for p in ["spotify", "instagram", "tiktok", "youtube"]:
            if p in text_lower:
                platforms.append(p)
        return platforms

    def _estimate_budget_allocation(self, total_budget: int,
                                     strategy_text: str) -> dict:
        platforms = self._extract_platforms(strategy_text)
        if not platforms:
            return {}
        allocation = {}
        per_platform = int(total_budget * 0.9 / len(platforms))
        for p in platforms:
            allocation[p] = per_platform
        allocation["reserve"] = total_budget - sum(v for k, v in allocation.items() if k != "reserve")
        return allocation

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running."""
        if self.backend != "ollama":
            return True  # API backends don't need Ollama
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = self.model.split(":")[0]
                return any(model_base in m for m in models)
            return False
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎵 StreamBreaker AI — Model 3 Test")
    print("=" * 60)

    # Auto-detect backend
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        print("Using Groq API (fast)")
        gen = MarketingStrategyGenerator(backend="groq", api_key=groq_key)
    elif openai_key:
        print("Using OpenAI API")
        gen = MarketingStrategyGenerator(backend="openai", api_key=openai_key)
    else:
        print("Using Ollama (local)")
        gen = MarketingStrategyGenerator(backend="ollama")
        if not gen.check_ollama_status():
            print(f"⚠️  Ollama not detected. Run: ollama pull {gen.model}")
            exit(1)

    print(f"Model: {gen.model}")
    print("\nGenerating strategy...\n")

    result = gen.generate_strategy(
        prediction_probability=75.0,
        budget=1500,
        genre="Indie Pop",
        sentiment="positive",
        hook_repetition=0.85,
    )

    if result["success"]:
        print(result["strategy"][:500] + "...\n")
        print(f"✅ Tokens: {result['metadata']['tokens_used']}")
    else:
        print(f"❌ Error: {result['error']}")
