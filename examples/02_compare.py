"""Side-by-side compare of two prompts."""

from contextops.eval import compare
from contextops.models import Prompt


def main() -> None:
    # "Bad" order — variable content first, stable content last.
    bad = Prompt(
        query="Summarize this article.",
        documents="Long article text here...",
        history=[{"role": "user", "content": "hi"}],
        system="You are a helpful summarizer.",
        tools="[]",
        model="claude-sonnet-4.6",
    )

    # "Good" order — stable content first.
    good = Prompt(
        system="You are a helpful summarizer.",
        tools="[]",
        query="Summarize this article.",
        documents="Long article text here...",
        history=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4.6",
    )

    report = compare(baseline=bad, optimized=good)

    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()