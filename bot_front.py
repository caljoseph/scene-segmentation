"""
Simplified LLM-based scene boundary classifier using OpenAI API.
Drop-in replacement for neural network classifiers.
"""

from openai import OpenAI, AsyncOpenAI
import os
import asyncio


class SceneLLM:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize LLM classifier for scene boundary detection.

        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
        """
        self.model = model
        self.api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.sample_responses = []  # Store sample responses for verification

        # Load API credentials
        self._load_credentials()
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

        # System prompt - based on Zehe et al. annotation guidelines
        self.system_prompt = """You are an expert at identifying scene boundaries in narrative text.

A scene boundary occurs when there is a change in:
- TIME: Time jumps, flashbacks, or shifts in narrative timeframe
- SPACE: Location changes (same building = no boundary, different location = boundary)
- CHARACTERS: Different character focus or character enters/leaves
- ACTION: Shift in event sequence or narrative focus

EXAMPLES:

YES: "Und es war kalt. Wir waren am 19. Juni des Jahres 1883 in New York losgesegelt, und wenn ich nicht irgendwo auf dem Atlantik die Übersicht verloren hatte, dann mussten wir jetzt den 24. Juli schreiben. Hochsommer, dachte ich."

NO: "Jetzt lag das Schiff ohne Fahrt auf der Stelle. Die großen, an vielen Stellen geflickten Segel hingen schlaff von den Rahen, und an Masten und Tauwerk sammelte sich Feuchtigkeit und lief in kleinen glitzernden Bahnen zu Boden. Es war still, eine unheimliche, an den Nerven zerrende Stille, die mit dem Nebel über das Meer herangekrochen war und den schnittigen Viermastersegler einhüllte."

YES: "Das Leben findet er schön, denn es ist für ihn voller Abenteuer. Und Abenteuer machen seiner Meinung nach das Leben süß. Als er wieder einen Schluck vom roten Wein nimmt, da hört er im Nebenzimmer laute Stimmen."

NO: "Als er wieder einen Schluck vom roten Wein nimmt, da hört er im Nebenzimmer laute Stimmen. Es ist sogar ein Gebrüll. Und dann kreischt eine Mädchenstimme um Hilfe."

YES: "Kirby Mahoun ist ein Mann von jener Sorte, die stets schnell und gründlich handelt. Und so tritt er die Tür ein und gleitet in das Zimmer. Er kann noch sehen, wie ein Mann sich aus dem Fenster schwingt und sich offenbar hinunterfallen lässt wie ein geschmeidiger Wildkater."

NO: "Der Killer wollte sie haben. Er kam durch das Fenster, schlich an diesem wunderschönen Bett vorbei zum Kleiderständer da in der Ecke. Aber ich war aufgewacht und sprang ihn an, wollte … Aaaah, ich verschwende nur meinen Atem."

Determine if there is a scene boundary WITHIN the text window provided.

Respond ONLY: "YES" or "NO"."""

    def _load_credentials(self):
        """Load OpenAI API credentials from api_key.py"""
        if os.path.exists("api_key.py"):
            import api_key
            self.api_key = api_key.api_key
            if hasattr(api_key, 'org_id'):
                self.org_id = api_key.org_id
        else:
            raise FileNotFoundError("api_key.py not found. Please create it with your OpenAI API key.")

    def classify_window(self, window_text):
        """
        Classify whether there's a scene boundary in the middle of the text window.

        Args:
            window_text: String containing the window of sentences to analyze

        Returns:
            bool: True if scene boundary detected, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Text window:\n\n{window_text}\n\nIs there a scene boundary in this text?"}
                ],
                reasoning_effort="low",
                timeout=30  # 30 second timeout
            )

            # Track usage
            self.api_calls += 1
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

            # Parse response
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer

        except Exception as e:
            print(f"\n⚠️  LLM API Error: {type(e).__name__}: {e}")
            print(f"Model attempted: {self.model}")
            import traceback
            traceback.print_exc()
            return False

    async def _classify_window_async(self, window_text):
        """Async version of classify_window for parallel processing"""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Text window:\n\n{window_text}\n\nIs there a scene boundary in this text?"}
                ],
                reasoning_effort="low",
                timeout=30
            )

            # Track usage
            self.api_calls += 1
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

            # Parse response
            answer = response.choices[0].message.content.strip().upper()
            result = "YES" in answer

            # Save a few samples for verification (max 5)
            if len(self.sample_responses) < 5:
                self.sample_responses.append({
                    'text': window_text[:200] + "..." if len(window_text) > 200 else window_text,
                    'response': answer,
                    'classification': result
                })

            return result

        except Exception as e:
            print(f"\n⚠️  LLM API Error: {type(e).__name__}: {e}")
            return False

    def classify_batch(self, window_texts, batch_size=20):
        """
        Classify multiple windows in parallel batches.

        Args:
            window_texts: List of text windows to classify
            batch_size: Number of concurrent API calls (default: 20)

        Returns:
            List of bool: True if scene boundary detected, False otherwise
        """
        async def process_batch(texts):
            tasks = [self._classify_window_async(text) for text in texts]
            return await asyncio.gather(*tasks)

        # Process in batches to avoid overwhelming the API
        results = []
        for i in range(0, len(window_texts), batch_size):
            batch = window_texts[i:i + batch_size]
            batch_results = asyncio.run(process_batch(batch))
            results.extend(batch_results)
            print(f"  Progress: {min(i + batch_size, len(window_texts))}/{len(window_texts)} windows classified...", end='\r')

        return results

    def get_usage_stats(self):
        """Return usage statistics"""
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }

    def print_sample_responses(self):
        """Print sample LLM responses for verification"""
        if not self.sample_responses:
            print("\nNo sample responses available.")
            return

        print("\n" + "="*80)
        print("SAMPLE LLM RESPONSES (for verification):")
        print("="*80)
        for i, sample in enumerate(self.sample_responses, 1):
            print(f"\n📝 Sample {i}:")
            print(f"Text: {sample['text']}")
            print(f"LLM Response: '{sample['response']}'")
            print(f"Classification: {'✅ BOUNDARY' if sample['classification'] else '❌ NO BOUNDARY'}")
            print("-" * 80)


if __name__ == "__main__":
    # Test the classifier
    llm = SceneLLM()

    # Test case: clear scene transition
    test_window = """The battle raged on through the night, swords clashing and men shouting.
    Finally, as dawn broke, the enemy retreated into the forest.
    We had won, but at a terrible cost.
    Three days later, I found myself in the quiet halls of the castle, reporting to the king.
    The throne room was silent except for the crackling fireplace.
    His Majesty listened intently as I recounted the events of that fateful night."""

    result = llm.classify_window(test_window)
    print(f"Scene boundary detected: {result}")
    print(f"Usage: {llm.get_usage_stats()}")
