# CVCI Auxiliary Perception and Scenario Rule System Plan

## Current Call Chain

- Agent entry: `team_code/drivetransformer_b2d_agent.py`, class `DriveTransformerAgent`.
- Sensor declaration: `DriveTransformerAgent.sensors()` builds from `self.cameras + [IMU, GPS, SPEED]`, optional BEV, and optional `LIDAR` when `CVCI_LIDAR_ENABLED` and `CVCI_AUXILIARY_PERCEPTION_ENABLED` are enabled.
- Per-frame input: `tick()` compresses camera images, reads GPS/IMU/speed, runs `_route_planner` and `_command_planner`, and passes `lidar_points` through the tick dictionary when enabled.
- Model path: `run_step()` builds camera/lidar calibration tensors, invokes the DriveTransformer model, decodes predicted trajectory / target-speed style outputs, and obtains the nominal PID control.
- Rule hook: the current control insertion point is after model inference and before `return control`: detection context is built from model detections plus optional LiDAR geometry, then `handle_current_frame_detection_result()` and `_apply_rule_based_control()` adjust the final `carla.VehicleControl`.
- Current auxiliary files: `team_code/auxiliary_perception/pointcloud_geometry.py`, `team_code/cvci_scenario_classifier.py`, `team_code/cvci_scenario_context.py`, `team_code/cvci_scenario_rules_v4.py`, and `team_code/cvci_rule_config.py`.

## Recommended Insertion Points

1. `sensors()`: keep LiDAR behind environment/config switches so baseline runs remain comparable.
2. `tick()`: normalize raw LiDAR only as sensor data; do not query CARLA actors, semantic labels, XML route ids, or ScenarioRunner blackboard at runtime.
3. Post-model detection/context builder: fuse model detections, LiDAR geometry fallback, traffic-light/stop cues already available to the agent, and route-planner topology features.
4. `ScenarioClassifier.classify()`: default to sensor/route-observation classification. `CVCI_ALLOW_ROUTE_PRIOR` must remain false for formal runs; forced/route-prior modes are debug-only.
5. `_apply_rule_based_control()`: keep nominal DriveTransformer control intact, then apply bounded caps, braking, steering scaling, and recovery release. This is the only place that should alter the returned control.

## CenterPoint / LiDAR Detector Integration Options

| Option | Fit for current environment | Advantages | Risks | Recommendation |
|---|---|---|---|---|
| In-process Autoware CenterPoint | Weak | Reuses mature detector | ROS2/Autoware dependency and Python/CUDA ABI mismatch risk | Do not use by default |
| ONNX/TensorRT PointPillars/CenterPoint wrapper | Good | Small runtime surface, legal raw LiDAR input, can be disabled cleanly | Requires export/calibration and TensorRT engine build | Preferred production path |
| Separate detector process via ZeroMQ/Unix socket/shared memory | Good | Isolates dependency conflicts and GPU memory spikes | IPC timeout/fallback code required | Preferred if detector stack cannot import in `drivetransformer` |
| Geometry fallback only | Already present | No new heavy dependency, immediate testing, legal raw LiDAR | Not a real 3D detector/tracker | Keep as safe fallback and regression baseline |

## Compatibility Audit

- Active conda env: `drivetransformer`.
- Observed Python: `3.8.20` on the current remote host, while the target CVCI/CARLA stack is Python 3.7-compatible. New code should avoid Python 3.9+ syntax and heavy dependency changes.
- CARLA launcher in current runs: `/root/autodl-tmp/projects/carla/CarlaUE4.sh`, with CVCI closed-loop evaluator from the benchmark tree.
- GPU policy used by recent CVCI runs: `GPU_LIST=0,1,2`, `CARLA_GRAPHICS_ADAPTER_LIST=2,2,0`.
- Current LiDAR fallback depends only on NumPy and raw LiDAR points; no ROS/Autoware runtime is introduced.
- Before adding TensorRT/ONNX, verify PyTorch/CUDA/TensorRT versions inside `drivetransformer`, exported detector input shape, max latency, and VRAM on a single CARLA+agent shard.

## GPU and Latency Risks

- Closed-loop evaluation already runs CARLA plus DriveTransformer on each GPU. A detector must have a hard timeout and stale-result policy.
- Detector latency above one control tick should degrade to the latest non-stale result or the original DriveTransformer control.
- Avoid allocating large tensors in `run_step()` when the auxiliary switches are disabled.
- Record per-frame auxiliary latency, stale count, and applied action in metadata for later score debugging.

## Expected Modified Files

- `team_code/drivetransformer_b2d_agent.py`: sensor switch, LiDAR extraction, final control arbitration.
- `team_code/auxiliary_perception/*`: raw LiDAR detector/fallback, optional tracker, timeout wrapper.
- `team_code/cvci_scenario_classifier.py`: observation-based 12-family classification with route-prior disabled by default.
- `team_code/cvci_scenario_rules_v4.py` and `team_code/cvci_rule_config.py`: scenario FSM/rule policies.
- `tools/build_cvci_scenario_catalog.py`: offline CVCI scenario catalog builder.
- `configs/cvci_scenario_catalog.yaml` and `reports/cvci_scenario_catalog.md`: generated offline design artifacts, not runtime truth.

## Fallback Mechanism

- The requested auxiliary system is enabled by default for this work: `CVCI_AUXILIARY_SYSTEM_ENABLED=1`, `CVCI_AUXILIARY_PERCEPTION_ENABLED=1`, `CVCI_LIDAR_ENABLED=1`, `CVCI_SCENARIO_RULES_ENABLED=1`, and `CVCI_SAFETY_SUPERVISOR_ENABLED=1`. Baseline A/B is still available by explicitly setting these to `0`.
- If auxiliary input is missing, stale, low-confidence, malformed, or too slow, return the already-computed DriveTransformer/legacy-rule control.
- Rule states must include release/recovery paths and deadlock recovery; no scenario rule may permanently stop the vehicle after risk clears.
- Baseline A/B command: run the same CVCI route XML and checkpoint with all auxiliary switches disabled, then run the new system with switches enabled and compare `score_challenge` from `results/cvci_*.json`.

## Current Implementation Status

The runtime module `team_code/cvci_auxiliary_system.py` now implements the requested side chain with legal-observation inputs: auxiliary perception, local object tracking and velocity estimation, traffic-light/stop-sign adapter, weak lane/topology extraction, scene features, scenario recognition, FSM-style rule planning, and safety supervision. It is enabled by default for this goal while preserving explicit baseline switches. Existing DriveTransformer detection-head rules are protected so prior good closed-loop behavior is not overwritten by the new auxiliary path. Remaining work is closed-loop CVCI validation and iterative rule refinement on the poor scenario families.


## Current Implementation Update

- Runtime module added: `team_code/cvci_auxiliary_system.py`.
- Tracker module added: `team_code/auxiliary_perception/object_tracker.py`.
- Chain implemented by name: `AuxiliaryPerception -> SceneFeatureBuilder -> ScenarioRecognizer -> ScenarioRulePlanner -> SafetySupervisor`.
- The module defaults to enabled for this goal. Explicit baseline disable remains possible through environment variables.
- `CVCI_ALLOW_ROUTE_PRIOR` remains default false. The runtime module does not read CVCI XML, scenario names, actor truth, ScenarioRunner blackboard, or CARLA actor state.
- Existing DriveTransformer detection-head rules are protected: when the legacy rule path has already produced an action, the new auxiliary system preserves that control. This is intended to avoid breaking previously good closed-loop results.
- `ObjectTracker` estimates local track ids, longitudinal/lateral velocity, closing speed, TTC, ego-corridor intersection, and stable reverse-vehicle evidence from legal detector outputs only. These fields are exposed through `AuxFeatures` and debug logs as evidence before stronger control is attempted.
- Observed remote compatibility: Python 3.8.20, PyTorch 2.4.1+cu118, CARLA import available, TensorRT not installed, 3 x RTX 4090 D with 24564 MiB each.
- Current validation: `python -m py_compile team_code/cvci_auxiliary_system.py team_code/auxiliary_perception/pointcloud_geometry.py team_code/auxiliary_perception/object_tracker.py team_code/drivetransformer_b2d_agent.py` and `pytest -q tests/test_pointcloud_geometry.py tests/test_object_tracker.py tests/test_cvci_auxiliary_system.py` passed with 21 tests.
