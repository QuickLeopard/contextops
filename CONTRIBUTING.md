# Contributing to ContextOps

Thanks for your interest in ContextOps! We welcome contributions of all kinds:
bug fixes, new features, docs, examples, benchmarks, and discussions.

## Quick start

```bash
git clone https://github.com/QuickLeopard/contextops.git
cd contextops
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 35 unit tests, <2s
python -m contextops_bench smoke   # 10 prompts, <1s
```

## Project layout

```
contextops/
├── contextops/         # the library (importable as `contextops`)
├── contextops_bench/   # the benchmark runner (script package)
├── docs/               # ACCEPTANCE.md and design docs
├── evals/              # sample datasets + prompts for demos
├── examples/           # runnable Python examples
└── tests/              # unit tests
```

## Conventions

### Code style

- Python 3.10+ syntax (`X | Y` unions, `match` statements when they help).
- `ruff` for linting (run `ruff check .`).
- `mypy --strict` for type-checking the public API (in CI).
- All public functions have docstrings; all CLI commands have `--help` text.

### Tests

Every change MUST come with tests:

- New optimizer logic → add to `tests/test_optimizer.py`.
- New eval / judge feature → add to `tests/test_v02_eval.py`.
- New bench feature → add to `tests/test_bench_unit.py`.
- Backwards-incompatible change → also add a regression test under
  the appropriate file with a name like `test_v01_still_works`.

Run the full suite before submitting a PR:

```bash
pytest
python -m contextops_bench smoke
```

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` docs only
- `test:` test additions or fixes
- `refactor:` no behaviour change
- `bench:` changes to the bench harness only
- `chore:` tooling, CI, deps

### Acceptance criteria

Before opening a PR that targets a release, run through
[`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md). Every `MUST` criterion
touched by your change MUST still pass.

## Adding a new metric

1. Add the metric definition to `_METRICS` in `contextops/judge.py`.
2. Add a unit test in `tests/test_v02_eval.py` exercising both the happy path and a judge that returns non-JSON.
3. Document it in `README.md` under the metrics table.
4. Optionally add it to the default metrics list in `evaluate_ab()`.

## Adding a new bench provider

1. Implement a class with `PROVIDER` string attribute, `complete()` and optionally `list_models()` methods in `contextops_bench/clients.py`.
2. Add the provider name to the `choices=` list in `contextops_bench/__main__.py:_add_common_args`.
3. Add a unit test in `tests/test_bench_unit.py` (at minimum: factory returns correct class, unknown name raises `ValueError`).

## Release process

1. Bump version in `pyproject.toml`.
2. Add a section to `CHANGELOG.md`.
3. Run full acceptance suite (`pytest`, smoke bench, manual OpenRouter test if you have a key).
4. Tag: `git tag -a v0.X.Y -m "v0.X.Y"`.
5. Build & publish:
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```
6. Push the tag: `git push origin v0.X.Y`.

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Be kind, be patient, assume good faith.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.