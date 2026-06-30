# Submission Checklist

- Code only: model weights and benchmark assets are excluded.
- Default model path: `weights/iter_25000.pth`.
- Override model path: `CKPT_PATH=/path/to/model.pth`.
- Default LiDAR detector checkpoint path: `weights/pointpillar_kitti.pth`.
- Override LiDAR detector checkpoint path: `CVCI_LIDAR_DETECTOR_MODEL=/path/to/pointpillar_kitti.pth`.
- Official detector checkpoint source: OpenPCDet KITTI PointPillar `pointpillar_7728.pth`.
- Official detector checkpoint URL: `https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing`.
- Helper download script: `scripts/download_lidar_detector_checkpoint.sh`.
- OpenPCDet source is bundled under `third_party/OpenPCDet/`.
- Main agent: `team_code/drivetransformer_b2d_agent.py`.
- Main config: `adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py`.
- Main script: `scripts/run_cvci_closed_loop.sh`.
- Queue script: `scripts/run_cvci_parallel_80_queue.sh`.
- Rules and auxiliary perception are enabled by default through environment variables.
- LiDAR detector is required by default; missing detector source/config/checkpoint exits before evaluation.
- `CVCI_ALLOW_ROUTE_PRIOR=0` by default for fair closed-loop reproduction.

Before running on a new machine:

```bash
export CVCI_ROOT=/path/to/CVCI_Benchmark/CVCI_BenchMark
export CARLA_ROOT=/path/to/carla
export CKPT_PATH=/path/to/iter_25000.pth
export CVCI_LIDAR_DETECTOR_MODEL=/path/to/pointpillar_kitti.pth
bash scripts/run_cvci_closed_loop.sh
```
