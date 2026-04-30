#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-surgiplot}"

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
conda run -n "${ENV_NAME}" pip install -e ".[scene3d]"

echo ""
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run GUI with:  surgiplot_gui"
