"""
Required by the spec: compare at least two strategies/prompts/model
configurations on dataset/sample_claims.csv, and state which one was
chosen for the final output.csv run.
"""

from dataclasses import dataclass

from core import data_loader, pipeline
from evaluation import metrics


@dataclass
class Strategy:
    name: str
    description: str
    provider: str
    model: str


STRATEGIES: list[Strategy] = [
    Strategy(
        name="groq_per_image",
        description="Groq llama-4-scout, one call per image",
        provider="groq",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    Strategy(
        name="openrouter_free",
        description="OpenRouter nex-n2-pro:free, one call per image",
        provider="openrouter",
        model="nex-agi/nex-n2-pro:free",
    ),
]


def run_all(records, evidence_requirements, user_history):
    """Run each strategy and return {strategy_name: results}."""
    import os

    results = {}
    for strategy in STRATEGIES:
        print(f"\nRunning strategy: {strategy.name}")
        # Temporarily override provider
        old_provider = os.environ.get("VLM_PROVIDER")
        old_model = os.getenv("VLM_MODEL")
        os.environ["VLM_PROVIDER"] = strategy.provider
        os.environ["VLM_MODEL"] = strategy.model

        try:
            predictions = pipeline.run_batch(records, evidence_requirements, user_history)
            expected = [r.expected or {} for r in records]
            score = metrics.score(predictions, expected)
            results[strategy.name] = score
        except Exception as e:
            results[strategy.name] = {"error": str(e)}
        finally:
            # Restore original provider
            if old_provider:
                os.environ["VLM_PROVIDER"] = old_provider
            if old_model:
                os.environ["VLM_MODEL"] = old_model

    return results
