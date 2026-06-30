# CVCI_TJU

Code-only release for the TJU CVCI 2026 Bench2InterActDrive closed-loop submission. This repository contains the DriveTransformer-based agent, CVCI rule supervisor, auxiliary LiDAR/perception modules, route tools, and runnable shell entry points. Model weights and benchmark assets are intentionally not included.

本仓库是 TJU CVCI 2026 Bench2InterActDrive 闭环提交代码版，只包含可复现代码、规则、辅助感知、工具脚本和运行入口，不包含模型权重、CARLA、CVCI 官方 benchmark 数据或历史结果。

## What Is Included

- `team_code/drivetransformer_b2d_agent.py`: CVCI submission agent entry point.
- `team_code/cvci_auxiliary_system.py`: rule-based safety supervisor and recovery controller, enabled by default.
- `team_code/cvci_scenario_*.py`: scenario context, route/scenario classification, and CVCI scenario-specific rule support.
- `team_code/auxiliary_perception/`: LiDAR geometry, required LiDAR detector wrapper, and object tracking helpers.
- `adzoo/drivetransformer/`: DriveTransformer model/config/plugin code used by the agent.
- `mmcv/`: local MMCV-compatible source used by the original DriveTransformer environment.
- `tools/`: CVCI route parsing, queue running, score summarization, and analysis helpers.
- `scripts/run_cvci_closed_loop.sh`: one-command closed-loop evaluation entry.
- `scripts/run_cvci_parallel_80_queue.sh`: multi-GPU route queue runner for repeated CVCI route attempts.
- `third_party/OpenPCDet/`: bundled OpenPCDet detector source code used by the LiDAR detection head.

## Not Included

The repository does not include:

- `iter_25000.pth` or any other `.pth/.pt/.ckpt` model file.
- `pointpillar_kitti.pth` or any other LiDAR detector checkpoint.
- CARLA simulator binaries.
- Official CVCI/Bench2InterActDrive benchmark package.
- Evaluation result JSON files, videos, logs, datasets, or cached build products.

Place the model separately at:

```bash
weights/iter_25000.pth
```

or pass it explicitly:

```bash
export CKPT_PATH=/absolute/path/to/iter_25000.pth
```

The LiDAR detector checkpoint is also required by default:

```bash
weights/pointpillar_kitti.pth
```

Download source:

- Official OpenPCDet KITTI PointPillar model zoo entry: https://github.com/open-mmlab/OpenPCDet#kitti-3d-object-detection-baselines
- Official PointPillar checkpoint file: https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
- The official filename is `pointpillar_7728.pth`; this repository expects it at `weights/pointpillar_kitti.pth`.

Download with the helper script:

```bash
bash scripts/download_lidar_detector_checkpoint.sh
```

or pass it explicitly:

```bash
export CVCI_LIDAR_DETECTOR_MODEL=/absolute/path/to/pointpillar_kitti.pth
```

## Environment Assumptions

You need an environment compatible with the original DriveTransformer and CVCI benchmark stack:

- Linux with CUDA GPU.
- CARLA simulator required by the official CVCI package.
- Official CVCI/Bench2InterActDrive benchmark tree containing `leaderboard/` and `scenario_runner/`.
- Python environment with the dependencies required by DriveTransformer, CARLA, leaderboard, and the CVCI benchmark.

Recommended directory layout:

```text
workspace/
  CVCI_TJU/
  CVCI_Benchmark/
    CVCI_BenchMark/
      leaderboard/
      scenario_runner/
      runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml
  carla/
    CarlaUE4.sh
```

If your paths differ, set environment variables before running.

## Quick Start

```bash
cd CVCI_TJU
mkdir -p weights outputs

# Put the model here, or set CKPT_PATH to another location.
ls weights/iter_25000.pth

export CVCI_ROOT=/path/to/CVCI_Benchmark/CVCI_BenchMark
export CARLA_ROOT=/path/to/carla
export PYTHON_BIN=/path/to/python

bash scripts/run_cvci_closed_loop.sh
```

The default route XML is:

```bash
$CVCI_ROOT/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml
```

Override it with:

```bash
export BASE_ROUTES=/path/to/CVCI_BenchMark.xml
```

Results are written to:

```text
outputs/cvci_closed_loop/result.json
outputs/cvci_closed_loop/leaderboard.log
```

## Default Rule/Perception Switches

The release defaults are set for our CVCI code path:

```bash
export CVCI_AUXILIARY_SYSTEM_ENABLED=1
export CVCI_AUXILIARY_PERCEPTION_ENABLED=1
export CVCI_LIDAR_ENABLED=1
export CVCI_LIDAR_DETECTOR_ENABLED=1
export CVCI_LIDAR_DETECTOR_REQUIRED=1
export CVCI_LIDAR_DETECTOR_ASYNC=0
export CVCI_REVERSE_VEHICLE_RULE_ENABLED=1
export CVCI_LEGACY_DETECTION_RULES_ENABLED=0
export CVCI_ALLOW_ROUTE_PRIOR=0
```

`CVCI_ALLOW_ROUTE_PRIOR=0` is the default reproducible/fair setting. It avoids reading route-id priors unless you explicitly enable debugging or route-specific ablations.

The LiDAR detector is required by default. If OpenPCDet, the detector config, or the detector checkpoint is missing, the shell scripts exit before evaluation. If detector loading or inference fails inside Python, `team_code/auxiliary_perception/lidar_detector.py` raises an exception instead of silently falling back.

Optional LiDAR detector paths:

```bash
export CVCI_OPENPCDET_ROOT=/path/to/OpenPCDet
export CVCI_LIDAR_DETECTOR_CONFIG=/path/to/pointpillar.yaml
export CVCI_LIDAR_DETECTOR_MODEL=/path/to/pointpillar_kitti.pth
```

Install or rebuild the bundled detector source in the active Python environment:

```bash
bash scripts/install_openpcdet.sh
```

## Multi-GPU Queue Runner

Create or edit a queue JSON:

```json
[
  {"route_id": "3", "macro_scenario": "Lead Vehicle Occlusion with Abrupt Departure"}
]
```

Run:

```bash
export CVCI_ROOT=/path/to/CVCI_Benchmark/CVCI_BenchMark
export CARLA_ROOT=/path/to/carla
export CKPT_PATH=/path/to/iter_25000.pth
export CVCI_LIDAR_DETECTOR_MODEL=/path/to/pointpillar_kitti.pth
export QUEUE_JSON=configs/sample_cvci_queue.json
export GPUS=0,1,2
export CARLA_GRAPHICS_ADAPTERS=0,1,2

bash scripts/run_cvci_parallel_80_queue.sh
```

Queue outputs are written to:

```text
outputs/cvci_parallel_80/
```

## Chinese Usage Notes

### 目录用途

- `team_code/`: 官方评测会加载的 agent 和我们新增的 CVCI 规则/辅助感知代码。
- `team_code/auxiliary_perception/`: LiDAR 点云几何、必需检测头、目标跟踪。
- `third_party/OpenPCDet/`: 已放入仓库的检测头源码，不含检测头权重。
- `tools/`: 路线拆分、队列运行、分数统计、场景分析工具。
- `scripts/`: 推荐使用的运行入口，所有路径都可以通过环境变量覆盖。
- `weights/`: 模型占位目录，只放你自己通过其他方式上传的权重，不提交到 git。
- `outputs/`: 运行输出目录，不提交到 git。

### 单次闭环运行

```bash
cd CVCI_TJU
export CVCI_ROOT=/你的/CVCI_Benchmark/CVCI_BenchMark
export CARLA_ROOT=/你的/carla
export CKPT_PATH=/你的/iter_25000.pth
export CVCI_LIDAR_DETECTOR_MODEL=/你的/pointpillar_kitti.pth
export PYTHON_BIN=/你的/python
bash scripts/run_cvci_closed_loop.sh
```

如果权重放在仓库默认位置，可以不设置 `CKPT_PATH`:

```bash
weights/iter_25000.pth
```

检测头权重默认位置:

```bash
weights/pointpillar_kitti.pth
```

检测头权重下载地址:

- OpenPCDet 官方 KITTI PointPillar Model Zoo: https://github.com/open-mmlab/OpenPCDet#kitti-3d-object-detection-baselines
- 官方 PointPillar 权重: https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
- 官方文件名是 `pointpillar_7728.pth`，本仓库默认读取 `weights/pointpillar_kitti.pth`。

可以直接运行:

```bash
bash scripts/download_lidar_detector_checkpoint.sh
```

### 改端口/GPU/路线

```bash
export GPU_ID=0
export CARLA_PORT=45700
export TM_PORT=64700
export BASE_ROUTES=/path/to/routes.xml
bash scripts/run_cvci_closed_loop.sh
```

### 默认开启的能力

默认会开启我们当前的规则系统、LiDAR 辅助、检测头辅助和 reverse vehicle 规则。检测头是强依赖，缺 OpenPCDet、缺 yaml 或缺 `pointpillar_kitti.pth` 都会直接退出，不会静默降级。模型权重路径、检测头权重路径、CVCI 路径、CARLA 路径、端口、GPU 都不写死，均可用环境变量覆盖。

## Citation

This repository builds on DriveTransformer:

```bibtex
@inproceedings{jia2025drivetransformer,
  title={DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving},
  author={Xiaosong Jia and Junqi You and Zhiyuan Zhang and Junchi Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
