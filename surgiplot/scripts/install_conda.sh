#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="surgiplot_master"
ENV_FILE="env/surgiplot_master.yml"

# Create env if missing, otherwise update
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

echo ""
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run GUI with:  surgiplot_gui"
