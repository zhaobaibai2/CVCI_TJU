import pytest

from team_code.auxiliary_perception.object_tracker import ObjectTracker


def test_tracker_outputs_velocity_ttc_and_corridor_fields():
    tracker = ObjectTracker(max_match_distance=5.0)
    first = [{"score": 0.9, "class_name": "car", "box_lidar": {"x": 12.0, "y": 0.2, "z": 0.0}}]
    second = [{"score": 0.9, "class_name": "car", "box_lidar": {"x": 10.0, "y": 0.0, "z": 0.0}}]
    tracker.update(first, 1.0)
    tracks = tracker.update(second, 2.0)
    assert len(tracks) == 1
    track = tracks[0]
    assert track["track_id"] == 1
    assert track["longitudinal_velocity"] == pytest.approx(-2.0)
    assert track["lateral_velocity"] == pytest.approx(-0.2)
    assert track["closing_speed"] == pytest.approx(2.0)
    assert track["ttc"] == pytest.approx(5.0)
    assert track["intersects_ego_corridor"] is True
    assert track["is_reversing_candidate"] is True


def test_tracker_does_not_report_ttc_for_receding_vehicle():
    tracker = ObjectTracker(max_match_distance=5.0)
    tracker.update([{"score": 0.9, "class_name": "car", "box_lidar": {"x": 8.0, "y": 0.0}}], 1.0)
    tracks = tracker.update([{"score": 0.9, "class_name": "car", "box_lidar": {"x": 10.0, "y": 0.0}}], 2.0)
    assert len(tracks) == 1
    assert tracks[0]["longitudinal_velocity"] == pytest.approx(2.0)
    assert tracks[0]["ttc"] is None
    assert tracks[0]["is_reversing_candidate"] is False


def test_tracker_expires_unmatched_tracks():
    tracker = ObjectTracker(max_match_distance=5.0, max_age=1)
    tracker.update([{"score": 0.9, "class_name": "truck", "box_lidar": {"x": 8.0, "y": 0.0}}], 1.0)
    assert len(tracker.update([], 2.0)) == 1
    assert tracker.update([], 3.0) == []
