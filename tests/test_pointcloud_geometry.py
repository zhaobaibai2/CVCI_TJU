import numpy as np

from team_code.auxiliary_perception.pointcloud_geometry import PointCloudGeometryFallback


def test_front_blockage_detected():
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(5.0, 10.0, 100)
    pts[:, 1] = np.linspace(-0.5, 0.5, 100)
    out = PointCloudGeometryFallback().update(pts, timestamp=1.0)
    assert out["available"]
    assert out["front_blocked"]
    assert out["near_blocked"]
    assert out["front_distance"] <= 5.1
    assert out["center_density"] > 0
    assert out["center_blockage_ratio"] > 0.0


def test_side_points_not_front_blockage():
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 0] = 8.0
    pts[:, 1] = 5.0
    out = PointCloudGeometryFallback().update(pts, timestamp=1.0)
    assert out["available"]
    assert not out["front_blocked"]
    assert out["front_point_count"] == 0


def test_lateral_corridor_reports_open_side():
    pts = np.zeros((60, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(4.0, 12.0, 60)
    pts[:, 1] = np.linspace(-2.0, -0.9, 60)
    out = PointCloudGeometryFallback().update(pts, timestamp=1.0)
    assert out["left_density"] == 60
    assert out["right_density"] == 0
    assert out["open_side"] == "right"
    assert out["left_nearest_distance"] <= 4.1
    assert out["right_nearest_distance"] is None


def test_balanced_corridor_when_both_sides_blocked():
    left = np.zeros((30, 4), dtype=np.float32)
    left[:, 0] = np.linspace(5.0, 9.0, 30)
    left[:, 1] = -1.2
    right = np.zeros((30, 4), dtype=np.float32)
    right[:, 0] = np.linspace(5.0, 9.0, 30)
    right[:, 1] = 1.2
    out = PointCloudGeometryFallback().update(np.concatenate([left, right], axis=0), timestamp=1.0)
    assert out["left_density"] == 30
    assert out["right_density"] == 30
    assert out["open_side"] == "balanced"
