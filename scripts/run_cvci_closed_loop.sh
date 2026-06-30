#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DT_ROOT="${DT_ROOT:-$REPO_ROOT}"
export CVCI_ROOT="${CVCI_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/CVCI_Benchmark/CVCI_BenchMark}"
export CARLA_ROOT="${CARLA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/carla}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/weights/iter_25000.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/cvci_closed_loop}"
export BASE_ROUTES="${BASE_ROUTES:-$CVCI_ROOT/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export GPU_ID="${GPU_ID:-0}"
export CARLA_PORT="${CARLA_PORT:-45700}"
export TM_PORT="${TM_PORT:-64700}"
export CVCI_CARLA_GRAPHICS_ADAPTER="${CVCI_CARLA_GRAPHICS_ADAPTER:-0}"

export CVCI_AUXILIARY_SYSTEM_ENABLED="${CVCI_AUXILIARY_SYSTEM_ENABLED:-1}"
export CVCI_AUXILIARY_PERCEPTION_ENABLED="${CVCI_AUXILIARY_PERCEPTION_ENABLED:-1}"
export CVCI_LIDAR_ENABLED="${CVCI_LIDAR_ENABLED:-1}"
export CVCI_LIDAR_DETECTOR_ENABLED="${CVCI_LIDAR_DETECTOR_ENABLED:-1}"
export CVCI_LIDAR_DETECTOR_REQUIRED="${CVCI_LIDAR_DETECTOR_REQUIRED:-1}"
export CVCI_LIDAR_DETECTOR_ASYNC="${CVCI_LIDAR_DETECTOR_ASYNC:-0}"
export CVCI_LEGACY_DETECTION_RULES_ENABLED="${CVCI_LEGACY_DETECTION_RULES_ENABLED:-0}"
export CVCI_REVERSE_VEHICLE_RULE_ENABLED="${CVCI_REVERSE_VEHICLE_RULE_ENABLED:-1}"
export CVCI_ALLOW_ROUTE_PRIOR="${CVCI_ALLOW_ROUTE_PRIOR:-0}"
export CVCI_AUX_LOG_PERIOD="${CVCI_AUX_LOG_PERIOD:-100}"
export CVCI_OPENPCDET_ROOT="${CVCI_OPENPCDET_ROOT:-$REPO_ROOT/third_party/OpenPCDet}"
export CVCI_LIDAR_DETECTOR_CONFIG="${CVCI_LIDAR_DETECTOR_CONFIG:-$CVCI_OPENPCDET_ROOT/tools/cfgs/kitti_models/pointpillar.yaml}"
export CVCI_LIDAR_DETECTOR_MODEL="${CVCI_LIDAR_DETECTOR_MODEL:-$REPO_ROOT/weights/pointpillar_kitti.pth}"

export CARLA_SERVER="${CARLA_SERVER:-$CARLA_ROOT/CarlaUE4.sh}"
export SCENARIO_RUNNER_ROOT="$CVCI_ROOT/scenario_runner"
export LEADERBOARD_ROOT="$CVCI_ROOT/leaderboard"
export CHALLENGE_TRACK_CODENAME="${CHALLENGE_TRACK_CODENAME:-SENSORS}"
export IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
export DISABLE_BEV_SENSOR="${DISABLE_BEV_SENSOR:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export CVCI_CARLA_CUDA_VISIBLE="${CVCI_CARLA_CUDA_VISIBLE:-$GPU_ID}"

export PYTHONPATH="$CVCI_ROOT:$DT_ROOT:$DT_ROOT/adzoo:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$CVCI_ROOT/leaderboard:$CVCI_ROOT/scenario_runner:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Put iter_25000.pth under weights/ or set CKPT_PATH=/path/to/model.pth" >&2
  exit 2
fi

if [[ ! -f "$BASE_ROUTES" ]]; then
  echo "Route XML not found: $BASE_ROUTES" >&2
  echo "Set CVCI_ROOT or BASE_ROUTES to your official CVCI benchmark path." >&2
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

ln -sfn "$DT_ROOT" "$CVCI_ROOT/DriveTransformer"
ln -sfn "$DT_ROOT/adzoo" "$CVCI_ROOT/adzoo"
ln -sfn "$DT_ROOT/team_code" "$CVCI_ROOT/team_code"

TEAM_AGENT="$DT_ROOT/team_code/drivetransformer_b2d_agent.py"
TEAM_CONFIG="$DT_ROOT/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+$CKPT_PATH"
RESULT_JSON="$OUTPUT_DIR/result.json"
LOG_FILE="$OUTPUT_DIR/leaderboard.log"

cd "$CVCI_ROOT"
echo "Running CVCI closed-loop evaluation"
echo "agent=$TEAM_AGENT"
echo "config=$TEAM_CONFIG"
echo "routes=$BASE_ROUTES"
echo "output=$RESULT_JSON"

"$PYTHON_BIN" leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$BASE_ROUTES" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="$RESULT_JSON" \
  --agent="$TEAM_AGENT" \
  --agent-config="$TEAM_CONFIG" \
  --debug=0 \
  --resume=False \
  --port="$CARLA_PORT" \
  --traffic-manager-port="$TM_PORT" \
  --client-timeout=300 \
  --scenario-timeout=300 \
  --agent-timeout=120 \
  --gpu-rank=0 2>&1 | tee "$LOG_FILE"
