#!/usr/bin/env bash
# publish_to_github.sh — initialize git repo, commit, push to GitHub, publish to PyPI.
#
# Prerequisites:
#   - You created an empty GitHub repo at https://github.com/<owner>/contextops
#   - You have git, the gh CLI (or git push access), and twine installed
#   - You have a PyPI API token (set PYPI_TOKEN)
#
# Usage:
#   GITHUB_OWNER=your-username ./scripts/publish_to_github.sh

set -euo pipefail

# --- Pick a Python interpreter ---------------------------------------------
# macOS / Linux often only have `python3`, not `python`. Find something that works.
PYTHON=""
for candidate in python python3 python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    # Must be 3.10+ (we use `X | Y` union syntax)
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌ No Python 3.10+ found."
  echo "   Install via one of:"
  echo "     brew install python@3.12        # macOS (Homebrew)"
  echo "     sudo apt install python3.12     # Debian/Ubuntu"
  echo "     pyenv install 3.12 && pyenv global 3.12   # any Unix"
  exit 1
fi

echo "==> Using Python: $($PYTHON --version)"

OWNER="${GITHUB_OWNER:-your-username}"
REPO="contextops"
BRANCH="${BRANCH:-main}"

cd "$(dirname "$0")/.."

# --- Helper: run pip through the chosen interpreter ------------------------
run_pip() {
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip "$@"
  else
    echo "❌ pip not available for $PYTHON"
    echo "   Try: $PYTHON -m ensurepip --upgrade"
    echo "   Or:  brew install python@3.12 && brew link python@3.12"
    exit 1
  fi
}

# --- Helper: ensure pip + venv exist --------------------------------------
ensure_pip() {
  if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "==> Bootstrapping pip via ensurepip"
    "$PYTHON" -m ensurepip --upgrade || {
      echo "❌ ensurepip failed. Install pip manually:"
      echo "   brew install python@3.12     # macOS"
      echo "   sudo apt install python3-pip python3-venv  # Debian/Ubuntu"
      exit 1
    }
  fi
}

ensure_pip

# --- Git init + commit -----------------------------------------------------
echo "==> Initializing git repo"
if [ ! -d .git ]; then
  git init
  git checkout -b "$BRANCH" 2>/dev/null || git checkout -b main
fi

echo "==> Adding files"
git add .

echo "==> Committing"
git commit -m "feat: initial public release v0.2.0

- Cache-aware prompt reordering (system/tools first, query/history last)
- Token counting via tiktoken (cl100k_base fallback)
- Cost + cache hit rate estimation
- LLM-as-judge eval with 4 built-in metrics
- A/B testing with structural + quality deltas
- Local SQLite logger at ~/.contextops/calls.db
- Rich CLI: optimize / stats / recent / compare / eval / reset
- LiteLLM auto-callback (opt-in via [integrations])
- Bench harness: Ollama / LM Studio / OpenRouter
- 35 unit tests + smoke bench + 1000-prompt stress bench
- Full docs: ACCEPTANCE.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md" || echo "    (nothing to commit)"

# --- Push to GitHub --------------------------------------------------------
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "==> Creating GitHub repo via gh CLI"
  if gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
    echo "    Repo ${OWNER}/${REPO} already exists, pushing instead"
    git remote remove origin 2>/dev/null || true
    gh repo set-default "${OWNER}/${REPO}" 2>/dev/null || true
    git remote add origin "https://github.com/${OWNER}/${REPO}.git" 2>/dev/null || true
    git push -u origin "$BRANCH"
  else
    gh repo create "${OWNER}/${REPO}" --public --source=. --remote=origin --push \
      --description "Cache-aware prompt structure optimizer + LLM-as-judge eval + local cost logger"
  fi
else
  REMOTE="git@github.com:${OWNER}/${REPO}.git"
  echo "==> Setting remote $REMOTE and pushing (gh CLI not authenticated, using plain git)"
  git remote remove origin 2>/dev/null || true
  git remote add origin "$REMOTE"
  git push -u origin "$BRANCH" || {
    echo "❌ Push failed. Either:"
    echo "   - Install gh CLI: brew install gh && gh auth login"
    echo "   - Or add your SSH key: ssh-add ~/.ssh/id_ed25519"
    exit 1
  }
fi

# --- Build sdist + wheel ---------------------------------------------------
echo "==> Installing build tooling"
run_pip install --upgrade build twine

echo "==> Building sdist + wheel"
rm -rf dist build
"$PYTHON" -m build

# --- Upload to PyPI --------------------------------------------------------
echo "==> Uploading to PyPI"
if [ -n "${PYPI_TOKEN:-}" ]; then
  TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" "$PYTHON" -m twine upload dist/*
else
  "$PYTHON" -m twine upload dist/*
fi

echo ""
echo "✅ Done!"
echo "   Repo:  https://github.com/${OWNER}/${REPO}"
echo "   PyPI:  https://pypi.org/project/contextops-tool/"
echo "   Install: pip install contextops-tool"
echo ""
echo "Next steps:"
echo "  1. Visit https://github.com/${OWNER}/${REPO} → enable Issues + Discussions"
echo "  2. Add repo topics: llm, prompt-optimization, cache, observability, eval, ai"
echo "  3. Enable GitHub Actions (CI is in .github/workflows/ci.yml)"
echo "  4. Add PyPI + coverage badges to README"