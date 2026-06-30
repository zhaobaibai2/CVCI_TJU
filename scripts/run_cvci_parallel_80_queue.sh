#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DT_ROOT="${DT_ROOT:-$REPO_ROOT}"
export CVCI_ROOT="${CVCI_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/CVCI_Benchmark/CVCI_BenchMark}"
export CARLA_ROOT="${CARLA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/carla}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/weights/iter_25000.pth}"
export CVCI_OPENPCDET_ROOT="${CVCI_OPENPCDET_ROOT:-$REPO_ROOT/third_party/OpenPCDet}"
export CVCI_LIDAR_DETECTOR_CONFIG="${CVCI_LIDAR_DETECTOR_CONFIG:-$CVCI_OPENPCDET_ROOT/tools/cfgs/kitti_models/pointpillar.yaml}"
export CVCI_LIDAR_DETECTOR_MODEL="${CVCI_LIDAR_DETECTOR_MODEL:-$REPO_ROOT/weights/pointpillar_kitti.pth}"
export CVCI_LIDAR_DETECTOR_ENABLED="${CVCI_LIDAR_DETECTOR_ENABLED:-1}"
export CVCI_LIDAR_DETECTOR_REQUIRED="${CVCI_LIDAR_DETECTOR_REQUIRED:-1}"
export CVCI_LIDAR_DETECTOR_ASYNC="${CVCI_LIDAR_DETECTOR_ASYNC:-0}"
export BASE_ROUTES="${BASE_ROUTES:-$CVCI_ROOT/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml}"
export RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/outputs/cvci_parallel_80}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

QUEUE_JSON="${QUEUE_JSON:-$REPO_ROOT/configs/sample_cvci_queue.json}"
GPUS="${GPUS:-0,1,2}"
BASE_PORT="${BASE_PORT:-45700}"
BASE_TM_PORT="${BASE_TM_PORT:-64700}"
CARLA_GRAPHICS_ADAPTERS="${CARLA_GRAPHICS_ADAPTERS:-0,1,2}"

if [[ ! -f "$QUEUE_JSON" ]]; then
  echo "Queue JSON not found: $QUEUE_JSON" >&2
  echo "Set QUEUE_JSON=/path/to/queue.json or create one with tools/build_cvci_80_queue.py." >&2
  exit 2
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Put iter_25000.pth under weights/ or set CKPT_PATH=/path/to/model.pth" >&2
  exit 2
fi

if [[ "$CVCI_LIDAR_DETECTOR_ENABLED" == "1" || "$CVCI_LIDAR_DETECTOR_REQUIRED" == "1" ]]; then
  if [[ ! -d "$CVCI_OPENPCDET_ROOT" ]]; then
    echo "OpenPCDet root not found: $CVCI_OPENPCDET_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "$CVCI_LIDAR_DETECTOR_CONFIG" ]]; then
    echo "LiDAR detector config not found: $CVCI_LIDAR_DETECTOR_CONFIG" >&2
    exit 2
  fi
  if [[ ! -s "$CVCI_LIDAR_DETECTOR_MODEL" ]]; then
    echo "LiDAR detector checkpoint not found: $CVCI_LIDAR_DETECTOR_MODEL" >&2
    echo "Put pointpillar_kitti.pth under weights/ or set CVCI_LIDAR_DETECTOR_MODEL=/path/to/checkpoint.pth" >&2
    exit 2
  fi
fi

mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" "$REPO_ROOT/tools/run_cvci_parallel_80_queue.py" \
  --queue "$QUEUE_JSON" \
  --gpus "$GPUS" \
  --run-root "$RUN_ROOT" \
  --dt-root "$DT_ROOT" \
  --cvci-root "$CVCI_ROOT" \
  --xml "$BASE_ROUTES" \
  --ckpt "$CKPT_PATH" \
  --carla-root "$CARLA_ROOT" \
  --python "$PYTHON_BIN" \
  --threshold "${CVCI_PASS_THRESHOLD:-80}" \
  --base-port "$BASE_PORT" \
  --base-tm-port "$BASE_TM_PORT" \
  --carla-graphics-adapters "$CARLA_GRAPHICS_ADAPTERS" \
  --allow-route-prior "${CVCI_ALLOW_ROUTE_PRIOR:-0}" \
  --reverse-vehicle-rule "${CVCI_REVERSE_VEHICLE_RULE_ENABLED:-1}" \
  --rule-version "${RULE_VERSION:-github_repro_default}" \
  --freeze-on-pass
