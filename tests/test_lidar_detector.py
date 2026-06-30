import numpy as np

from team_code.auxiliary_perception.lidar_detector import LidarDetector, LidarDetectorConfig


def test_carla_to_detector_lateral_axis_is_mirrored():
    points = np.array([[10.0, -2.0, 0.5, 0.0], [8.0, 3.0, 0.2, 0.0]], dtype=np.float32)
    converted = LidarDetector.carla_to_detector_points(points)
    assert converted[0, 1] == 2.0
    assert converted[1, 1] == -3.0
    assert points[0, 1] == -2.0


def test_detector_box_is_converted_back_to_agent_frame():
    box = np.array([12.0, 1.5, 0.4, 4.0, 1.8, 1.6, 0.25], dtype=np.float32)
    converted = LidarDetector.detector_box_to_agent(box)
    assert converted["x"] == 12.0
    assert converted["y"] == -1.5
    assert converted["yaw"] == -0.25


def test_missing_checkpoint_is_reported_without_crashing(tmp_path):
    cfg = LidarDetectorConfig(
        enabled=True,
        async_enabled=False,
        root="/root/autodl-tmp/projects/lidar_detectors/OpenPCDet",
        config_path="/root/autodl-tmp/projects/lidar_detectors/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
        checkpoint_path=str(tmp_path / "missing.pth"),
    )
    detector = LidarDetector(cfg)
    result = detector.latest_or_submit(np.zeros((4, 4), dtype=np.float32), timestamp=1.0)
    assert result["available"] is False
    assert result["status"] == "missing_checkpoint"
    assert "missing_checkpoint" in result["error"]
