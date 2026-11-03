#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-surgiplot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required to run this installer."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python=3.11 pip
fi

echo "Installing Surgiplot with the optional PyVista 3D scene backend..."
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}[scene3d]"

echo ""
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run GUI with:  surgiplot-gui"
