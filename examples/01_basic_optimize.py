"""Quickstart — optimize a single prompt and print the result.

Run from the project root:
    python examples/01_basic_optimize.py
"""

from contextops import optimize, count_tokens
from contextops.models import Prompt


def main() -> None:
    # Your prompt as it currently looks — section order is whatever you wrote.
    p = Prompt(
        query="What's the weather in Berlin?",
        history=[
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
        documents="Berlin weather API docs: GET /weather?city=...",
        system="You are a helpful weather assistant. Answer concisely.",
        tools='[{"name": "get_weather", "parameters": {"city": "string"}}]',
        role="weather-agent",
        model="gpt-4o",
    )

    print(f"Original tokens: {count_tokens(p.query + p.system, p.model)}")
    result = optimize(p)

    print("\nOriginal section order:")
    for sec, content in result.original_sections:
        print(f"  {sec}: {content[:60]}{'...' if len(content) > 60 else ''}")

    print("\nOptimized section order (cache-friendly):")
    for sec, content in result.optimized_sections:
        print(f"  {sec}: {content[:60]}{'...' if len(content) > 60 else ''}")

    print(f"\nTokens: {result.original_tokens} → {result.optimized_tokens}")
    print(f"Estimated cache hit rate: {result.estimated_cache_hit_rate:.1%}")
    print(f"Estimated cost savings per 1k calls: ${result.estimated_cost_savings_usd:.4f}")

    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  • {note}")


if __name__ == "__main__":
    main()