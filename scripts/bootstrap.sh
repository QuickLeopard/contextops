#!/usr/bin/env bash
# bootstrap.sh — установить всё нужное для разработки/публикации ContextOps.
# Запусти один раз после распаковки архива.

set -e

echo "==> Detecting OS"
case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux)  OS=linux ;;
  *)      echo "❌ Unsupported OS: $(uname -s)"; exit 1 ;;
esac
echo "    OS: $OS"

# --- Python 3.10+ ---------------------------------------------------------
PY=""
for candidate in python3 python python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "==> No Python 3.10+ found. Installing..."
  if [ "$OS" = "macos" ]; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "❌ Homebrew not installed. Install from https://brew.sh first."
      exit 1
    fi
    brew install python@3.12
    PY=python3.12
  else
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3-pip
    PY=python3.12
  fi
fi

echo "==> Python: $($PY --version)"

# --- pip -------------------------------------------------------------------
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip"
  "$PY" -m ensurepip --upgrade || {
    if [ "$OS" = "macos" ]; then
      brew install python@3.12
    else
      sudo apt install -y python3-pip python3.12-venv
    fi
  }
fi

# --- venv ------------------------------------------------------------------
echo "==> Creating virtualenv at .venv"
"$PY" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

echo ""
echo "==> Running tests"
pytest -q

echo ""
echo "==> Running smoke bench"
python -m contextops_bench smoke

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "To activate later:    source .venv/bin/activate"
echo "To run tests:         pytest"
echo "To publish:           ./scripts/publish_to_github.sh"