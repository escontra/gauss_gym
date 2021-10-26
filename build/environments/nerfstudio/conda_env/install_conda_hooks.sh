#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is not set. Activate your conda env first."
  echo "Example: conda activate ns"
  exit 1
fi

HOOK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Hook root: $HOOK_ROOT"
SRC_ACT="$HOOK_ROOT/activate.d/env_vars.sh"
SRC_DEACT="$HOOK_ROOT/deactivate.d/env_vars.sh"
DST_ACT="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
DST_DEACT="$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"

mkdir -p "$(dirname "$DST_ACT")" "$(dirname "$DST_DEACT")"

ln -sf "$SRC_ACT"   "$DST_ACT"
ln -sf "$SRC_DEACT" "$DST_DEACT"

echo "Installed conda hooks:"
echo "  $DST_ACT -> $SRC_ACT"
echo "  $DST_DEACT -> $SRC_DEACT"

# Source the activation script to apply environment variables immediately
# source "$SRC_ACT"
# echo "Applied environment variables from $SRC_ACT"
