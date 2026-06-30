#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${CVCI_LIDAR_DETECTOR_MODEL:-$REPO_ROOT/weights/pointpillar_kitti.pth}"
FILE_ID="${OPENPCDET_POINTPILLAR_FILE_ID:-1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm}"
URL="https://drive.google.com/file/d/${FILE_ID}/view?usp=sharing"

mkdir -p "$(dirname "$OUT_PATH")"

if [[ -s "$OUT_PATH" ]]; then
  echo "Detector checkpoint already exists: $OUT_PATH"
  exit 0
fi

echo "Downloading OpenPCDet PointPillar checkpoint"
echo "source: $URL"
echo "target: $OUT_PATH"

if command -v gdown >/dev/null 2>&1; then
  gdown --fuzzy "$URL" -O "$OUT_PATH"
else
  echo "gdown is not installed. Installing it in the active Python environment."
  python -m pip install gdown
  gdown --fuzzy "$URL" -O "$OUT_PATH"
fi

if [[ ! -s "$OUT_PATH" ]]; then
  echo "Download failed or produced an empty file: $OUT_PATH" >&2
  exit 2
fi

echo "Saved detector checkpoint to $OUT_PATH"
