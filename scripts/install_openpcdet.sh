#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPCDET_ROOT="${CVCI_OPENPCDET_ROOT:-$REPO_ROOT/third_party/OpenPCDet}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$OPENPCDET_ROOT/pcdet" ]]; then
  echo "OpenPCDet source not found: $OPENPCDET_ROOT" >&2
  exit 2
fi

cd "$OPENPCDET_ROOT"
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" setup.py develop

echo "OpenPCDet installed from $OPENPCDET_ROOT"
