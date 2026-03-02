"""
Marketing Strategy Generator using OpenAI GPT
Miguel Davila - StreamBreaker AI Project
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, create_marketing_prompt

# Load environment variables
load_dotenv()

class MarketingStrategyGenerator:
    """
    Generates marketing strategies using GPT-3.5-turbo or GPT-4
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        """
        Initialize the generator
        
        Args:
            model: OpenAI model to use (gpt-3.5-turbo or gpt-4)
            temperature: Creativity level (0.0-1.0)
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        
    def generate_strategy(
        self,
        prediction_probability: float,
        budget: int,
        genre: str = "Indie Pop",
        instagram_followers: int = 500,
        spotify_listeners: int = 100,
        has_fanbase: bool = False,
        energy: float = 7.0,
        has_lyrics: bool = True,
        hook_strength: float = 6.0,
        danceability: float = 6.5,
        career_stage: str = "emerging"
    ) -> dict:
        """
        Generate a marketing strategy based on track prediction and artist profile
        
        Returns:
            dict with 'strategy' (text) and 'metadata' (usage stats)
        """
        
        # Create the prompt
        user_prompt = create_marketing_prompt(
            prediction_probability=prediction_probability,
            budget=budget,
            genre=genre,
            instagram_followers=instagram_followers,
            spotify_listeners=spotify_listeners,
            has_fanbase=has_fanbase,
            energy=energy,
            has_lyrics=has_lyrics,
            hook_strength=hook_strength,
            danceability=danceability,
            career_stage=career_stage
        )
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            strategy_text = response.choices[0].message.content
            
            return {
                "success": True,
                "strategy": strategy_text,
                "metadata": {
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens,
                    "cost_estimate": self._estimate_cost(response.usage.total_tokens)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "strategy": None
            }
    
    def _estimate_cost(self, tokens: int) -> float:
        """
        Estimate API cost based on tokens used
        GPT-3.5-turbo: ~$0.002 per 1K tokens
        GPT-4: ~$0.03 per 1K tokens
        """
        if "gpt-4" in self.model:
            return (tokens / 1000) * 0.03
        else:
            return (tokens / 1000) * 0.002


def main():
    """
    Example usage - test the generator
    """
    print("🎵 StreamBreaker AI - Marketing Strategy Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = MarketingStrategyGenerator(model="gpt-3.5-turbo")
    
    # Example scenario: High prediction, moderate budget
    print("\n📊 SCENARIO: High-potential indie pop track, $1500 budget\n")
    
    result = generator.generate_strategy(
        prediction_probability=87.5,  # High confidence from Model 1
        budget=1500,
        genre="Indie Pop",
        instagram_followers=1200,
        spotify_listeners=350,
        has_fanbase=True,
        energy=7.5,
        has_lyrics=True,
        hook_strength=8.0,
        danceability=7.0,
        career_stage="emerging"
    )
    
    if result["success"]:
        print(result["strategy"])
        print("\n" + "=" * 60)
        print(f"✅ Tokens used: {result['metadata']['tokens_used']}")
        print(f"💰 Estimated cost: ${result['metadata']['cost_estimate']:.4f}")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()
