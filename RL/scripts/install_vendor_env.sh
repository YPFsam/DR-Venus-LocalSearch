#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RL_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}
VLLM_SPEC=${VLLM_SPEC:-vllm}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-true}
UV_BIN=${UV_BIN:-}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [ "$(uname -s)" != "Linux" ]; then
    fail "This installer must run on the Linux GPU machine."
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN not found."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. Install the NVIDIA driver first."

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python 3.10 or newer is required.")
print(f"Python: {sys.version.split()[0]}")
PY

echo "NVIDIA devices:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ -z "$UV_BIN" ]; then
    UV_BIN="$(command -v uv || true)"
fi
if [ -z "$UV_BIN" ]; then
    echo "Installing uv for the current user..."
    "$PYTHON_BIN" -m pip install --user --upgrade uv
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="$(command -v uv || true)"
fi
[ -n "$UV_BIN" ] || fail "uv was installed but is not on PATH. Add \$HOME/.local/bin to PATH and retry."

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    "$UV_BIN" venv --python "$PYTHON_BIN" --seed "$VENV_DIR"
else
    echo "Using existing virtual environment: $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
UV_PIP=("$UV_BIN" pip install --python "$PYTHON")

echo "Installing build helpers..."
"${UV_PIP[@]}" --upgrade setuptools wheel packaging ninja

echo "Installing vLLM GPU stack: $VLLM_SPEC"
"${UV_PIP[@]}" "$VLLM_SPEC" --torch-backend=auto

if [ "$INSTALL_FLASH_ATTN" = "true" ]; then
    echo "Installing flash-attn. This requires a CUDA toolkit compatible with the installed PyTorch build."
    "${UV_PIP[@]}" flash-attn --no-build-isolation
else
    echo "Skipping flash-attn because INSTALL_FLASH_ATTN=$INSTALL_FLASH_ATTN"
fi

echo "Installing DR-Venus RL dependencies..."
"${UV_PIP[@]}" -r requirements.txt
"${UV_PIP[@]}" -e .

"$PYTHON" - <<'PY'
import importlib.util
import torch
import vllm

required = ["datasets", "flask", "qwen_agent", "ray", "tantivy", "wandb"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"ERROR: missing Python modules after installation: {', '.join(missing)}")

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA. Check the NVIDIA driver and installed wheel.")
PY

cat <<EOF

Environment installation completed.

Activate it with:
  source $VENV_DIR/bin/activate

Prepare model, data, and the local index with:
  bash scripts/vendor_train.sh ready
EOF
