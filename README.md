# ContextOps

[![CI](https://github.com/QuickLeopard/contextops/actions/workflows/ci.yml/badge.svg)](https://github.com/QuickLeopard/contextops/actions)
[![PyPI](https://img.shields.io/pypi/v/contextops-tool.svg)](https://pypi.org/project/contextops-tool/)
[![Python](https://img.shields.io/pypi/pyversions/contextops-tool.svg)](https://pypi.org/project/contextops-tool/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Cache-aware prompt structure optimizer + LLM-as-judge eval + local cost/usage logger.**

Stop paying for the same tokens twice. ContextOps reorders your prompt
sections so stable content (system prompt, tools) sits at the top — and
variable content (query, history) sits at the bottom — maximizing cache
hit rate on Anthropic / OpenAI / DeepSeek / any provider that does
prefix caching.

No cloud, no SaaS, no SDK lock-in. Just `pip install contextops-tool` and go.

---

## ⚡ Quickstart

```bash
pip install contextops-tool
```

```python
from contextops import optimize, Prompt

p = Prompt(
    query="What's the weather in Berlin?",
    history=[{"role": "user", "content": "Hi!"}],
    documents="Berlin weather API docs...",
    system="You are a helpful weather assistant.",
    tools='[{"name": "get_weather"}]',
    model="gpt-4o",
)

result = optimize(p)
print(result.diff())                            # history → documents → ... → query
print(f"Cache hit: {result.estimated_cache_hit_rate:.1%}")
print(f"Saves ~${result.estimated_cost_savings_usd:.4f} per 1k calls")
```

Output:

```
Section order: history → documents → ... → system → ... → query
Cache hit: 71.0%
Saves ~$0.1006 per 1k calls
```

That's it. Same prompt, same tokens, ~70% cache hit rate instead of ~5%.

---

## 🤔 Why?

LLM providers (Anthropic, OpenAI, DeepSeek, Google) cache the **prefix**
of your prompt. If the prefix is stable across calls, you pay 10% of the
cached-token price instead of the full price.

The trick: keep the prefix stable by **putting variable content (query,
history) at the end**.

ContextOps knows the canonical ordering by stability:

```
system → tools → role → context → documents → history → query
  ↑ stable                                                ↑ variable
```

Estimated impact on a typical workload:

| Setup                          | Cache hit rate | Effective $/1M input |
|--------------------------------|----------------|----------------------|
| Random order                   | ~5%            | $X (full price)      |
| ContextOps optimized           | ~78%           | ~$0.3·X (10% on cached prefix) |

---

## 🧰 What's in the box

| Feature | Description |
|---|---|
| **Cache-aware reordering** | Moves stable sections to the top, variable to the bottom. Same total tokens, much higher cache hit rate. |
| **Token counting** | tiktoken-based, model-aware (`gpt-4o`, `claude-*`, `qwen*`, fallback to `cl100k_base`). |
| **Cost estimation** | Per-model pricing baked in; estimates $/1k calls before vs after reorder. |
| **LLM-as-judge eval** | Built-in metrics: `faithfulness`, `relevance`, `completeness`, `conciseness`. |
| **A/B testing** | Run two prompts over a golden dataset, get structural + quality deltas. |
| **Local SQLite logger** | Every LLM call goes to `~/.contextops/calls.db`. Zero cloud. |
| **Dataset loaders** | `.json`, `.jsonl`, `.csv` golden QA datasets. |
| **Rich CLI** | `optimize / stats / recent / compare / eval / reset` with tables and progress bars. |
| **LiteLLM auto-log (opt)** | One line to auto-log every litellm call. `pip install "contextops[integrations]"` |
| **Bench harness** | 1000+ prompts through Ollama, LM Studio, OpenRouter, or direct APIs (Anthropic / OpenAI / Gemini / OpenCode-ZEN). |

---

## 📦 Install

```bash
# Core (optimizer + logger + eval + CLI)
pip install contextops-tool

# With LiteLLM auto-callback for real LLM logging
pip install "contextops-tool[integrations]"

# With dev tooling (pytest, ruff, mypy)
pip install "contextops-tool[dev]"

# Everything
pip install "contextops-tool[all]"
```

From source:

```bash
git clone https://github.com/QuickLeopard/contextops.git
cd contextops
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,integrations]"
pytest                                          # 53 tests
python -m contextops_bench smoke               # offline smoke
```

Requires **Python 3.10+**.

### macOS / Linux gotcha: `python` not found

If you see `zsh: command not found: python`, use `python3` instead. Or install
Python 3.12 via Homebrew and add it to PATH:

```bash
brew install python@3.12
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"
python --version   # should print 3.12.x
```

If you want a fully automated setup, run the bootstrap script after cloning:

```bash
git clone https://github.com/QuickLeopard/contextops.git
cd contextops
./scripts/bootstrap.sh   # installs Python, creates venv, runs tests + smoke
```

### No `python -m venv`?

Some Linux distros ship Python without `venv`:

```bash
# Debian / Ubuntu
sudo apt install python3.12-venv

# Fedora
sudo dnf install python3.12-venv

# Or use virtualenv instead
pip install virtualenv
virtualenv .venv
```

---

## 📖 Usage

### 1. Optimize a prompt (Python)

```python
from contextops import optimize, Prompt

p = Prompt(
    query="What's the weather in Berlin?",
    documents="API docs...",
    system="You are a helpful weather assistant.",
    tools="[]",
    model="gpt-4o",
)

result = optimize(p)
print(result.diff())                  # before → after
print(result.optimized_sections)      # [(Section, content), ...]
```

### 2. Compare two prompts (Python)

```python
from contextops.eval import compare

report = compare(baseline=bad_prompt, optimized=good_prompt)
print(report["delta"])
# {"tokens": 0, "cache_hit_rate": 0.65, "cost_savings_per_1k_usd": 4.21}
```

### 3. A/B eval with LLM-as-judge

```python
from contextops import evaluate_ab, load_dataset, Prompt, LiteLLMJudge

dataset = load_dataset("evals/sample_dataset.jsonl")

baseline = Prompt(system="...", query="", documents="{ctx}", model="gpt-4o-mini")
optimized = Prompt(system="...", documents="{ctx}", query="", model="gpt-4o-mini")

def my_llm(prompt_str: str) -> str:
    return call_my_llm(prompt_str)

report = evaluate_ab(
    baseline, optimized,
    run_fn=my_llm,
    dataset=dataset,
    metrics=["faithfulness", "relevance", "completeness"],
    judge=LiteLLMJudge(),
    on_render=lambda p, item: p.system.replace("{ctx}", item.context),
)

print(report["structural"])   # tokens / cache / cost deltas
print(report["quality"])      # per-metric judge deltas
```

### 4. CLI

```bash
# Optimize a prompt inline
contextops optimize \
    --system "You are a weather assistant." \
    --query "What's the weather in Berlin?" \
    --documents "API docs..." \
    --model gpt-4o

# Load a prompt from a JSON file
contextops optimize --from-json my_prompt.json

# Side-by-side comparison
contextops compare baseline.json optimized.json

# A/B eval with offline echo judge
contextops eval \
    --baseline evals/baseline_prompt.json \
    --optimized evals/optimized_prompt.json \
    --dataset evals/sample_dataset.jsonl \
    --metrics relevance,completeness,faithfulness \
    --echo --run-fn echo \
    --output report.json

# Real LLM-as-judge
pip install "contextops[integrations]"
contextops eval \
    --baseline evals/baseline_prompt.json \
    --dataset evals/sample_dataset.jsonl \
    --judge-model gpt-4o-mini \
    --metrics relevance,completeness,faithfulness,conciseness

# Local call stats
contextops stats
contextops recent --limit 50

# Reset the local database
contextops reset
```

### 5. Auto-log every LiteLLM call

```python
from contextops.integrations import install_callback
install_callback()

import litellm
litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
# → automatically logged to ~/.contextops/calls.db
```

---

## 🆚 Comparison

| Tool | What it does | Where ContextOps is different |
|---|---|---|
| **DSPy** | Auto-rewrites prompt *text* using a dataset | We **reorder sections** — no dataset, no model rewrite |
| **RAGAS / DeepEval** | Evaluate answer quality via LLM-judge | We measure **structure + cost**, complementary not competing |
| **Langfuse** | Cloud LLM observability | We stay **local-first**: SQLite, no signup |
| **prompt-cache / token-optimizer** | Cache responses, compress tokens | We focus on **provider cache** (Anthropic / OpenAI), not response cache |
| **vaibkumr/prompt-optimizer** | Compress text (LLMLingua-style) | We **reorder**, never change tokens or text |

---

## 🧪 Bench harness

1000+ prompts through Ollama, LM Studio, OpenRouter, or direct APIs:

```bash
# Smoke (10 prompts, <30s, no LLM, for CI)
python -m contextops_bench smoke

# Local (100 prompts via Ollama)
python -m contextops_bench local --provider ollama --model llama3.1:8b --n 100

# Cloud via OpenRouter, multi-model, parallel
export OPENROUTER_API_KEY=sk-or-v1-...
python -m contextops_bench cloud --provider openrouter \
    --model openai/gpt-4o-mini,anthropic/claude-3.5-haiku,meta-llama/llama-3.1-8b-instruct \
    --n 1000 --parallel 4

# Direct APIs (bypasses OpenRouter translation — definitive cache signal).
# Pick the provider that matches your target's cache mechanics:
export ZEN_API_KEY=...            # or ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
python -m contextops_bench cloud --provider direct_anthropic \
    --model claude-sonnet-4-6 --preset-agent realistic --n 30   # explicit cache_control markers
python -m contextops_bench cloud --provider direct_openai \
    --model gpt-4o-mini --preset-agent realistic --n 30          # automatic caching, 50% off
python -m contextops_bench cloud --provider direct_google \
    --model google/gemini-2.5-flash --preset-agent realistic --n 30  # implicit caching, 10% off
```

**About `--preset-agent`:** the bench needs a stable system prompt + tool schema + role across all calls for cache hits to be non-zero. `--preset-agent realistic` pins all three. On cache-bearing providers (OpenRouter + the four `direct_*`), the bench auto-applies `realistic` if you don't pass a preset — with a loud warning explaining why. Use `--preset-agent none` to opt out and use randomized prompts.

Each run writes:

- `bench/results/<label>.csv` — every observation (prompt_id, model, tokens, cache hit, cost, latency, error, section order)
- `bench/results/<label>.summary.json` — aggregated stats with optimized vs baseline deltas

**Troubleshooting cache reads showing 0?** Read [`docs/POSTMORTEM_realistic_cache.md`](docs/POSTMORTEM_realistic_cache.md) — it covers the realistic-preset cache key regression, why OpenRouter drops `cache_control` markers during translation, and why EchoClient (used in unit tests) hides the bug.

See [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) for the formal pass criteria.

---

## 🗺️ Roadmap

- ✅ **v0.1** — reorder, token count, SQLite logger, CLI
- ✅ **v0.2** — LLM-as-judge eval + A/B testing + dataset loaders
- ✅ **v0.3** — realistic-preset cache-key regression fix + direct providers (Anthropic / Zen / OpenAI / Gemini) + CI bench regression gate + safety-net auto-default on cache-bearing providers. See [`docs/POSTMORTEM_realistic_cache.md`](docs/POSTMORTEM_realistic_cache.md).
- 🔜 **v0.4** — RAG curator (multi-signal retrieval + strict threshold)
- 🔜 **v1.0** — Access-aware context + audit trail (on-prem / enterprise)

---

## 📚 Documentation

- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — formal pass/fail criteria
- [`CHANGELOG.md`](CHANGELOG.md) — version history
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute
- [`SECURITY.md`](SECURITY.md) — how to report vulnerabilities
- [`evals/`](evals/) — sample datasets and prompts

---

## 🤝 Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for workflow, conventions, and release process.

Good first contributions:

- New `metric` in `contextops/judge.py` (e.g. `safety`, `format_compliance`)
- New `provider` in `contextops_bench/clients.py` (e.g. `vllm`, `tgi`)
- Better pricing tables for non-USD regions
- Translations of `docs/` and `README.md`

---

## 📜 License

[MIT](LICENSE).

---

## ✨ Credits

Built with:

- [pydantic](https://docs.pydantic.dev/) for prompt modeling
- [tiktoken](https://github.com/openai/tiktoken) for token counting
- [click](https://click.palletsprojects.com/) + [rich](https://rich.readthedocs.io/) for the CLI
- [litellm](https://github.com/BerriAI/litellm) for the optional auto-callback