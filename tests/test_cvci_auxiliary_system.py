import os
import sys
import types

import pytest
import yaml


class Control:
    def __init__(self, steer=0.25, throttle=0.7, brake=0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake


def load_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", types.SimpleNamespace(VehicleControl=Control))
    import importlib
    import team_code.cvci_auxiliary_system as mod

    return importlib.reload(mod)


def test_defaults_enable_auxiliary_but_not_route_prior(monkeypatch):
    for key in (
        "CVCI_AUXILIARY_SYSTEM_ENABLED",
        "CVCI_AUXILIARY_PERCEPTION_ENABLED",
        "CVCI_LIDAR_ENABLED",
        "CVCI_SCENARIO_RULES_ENABLED",
        "CVCI_SAFETY_SUPERVISOR_ENABLED",
        "CVCI_ALLOW_ROUTE_PRIOR",
    ):
        monkeypatch.delenv(key, raising=False)
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    assert cfg.enabled is True
    assert cfg.perception_enabled is True
    assert cfg.lidar_enabled is True
    assert cfg.scenario_rules_enabled is True
    assert cfg.safety_supervisor_enabled is True
    assert cfg.allow_route_prior is False
    assert cfg.lidar_open_side_progress_recovery_frames == 800


def test_pedestrian_emergency_brake(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 1,
        "timestamp": 0.1,
        "objects": [
            {"score": 0.95, "class_name": "pedestrian", "box_lidar": {"x": 3.2, "y": 0.1}},
        ],
        "map_objects": [],
    }
    ctrl = system.process(Control(), detection, {"speed": 5.0, "command_near": 4}, 0.1)
    assert ctrl.throttle == 0.0
    assert ctrl.brake >= 0.85
    assert system.last_debug["macro_scenario"] == "four_students_crossing_the_road"


def test_legacy_rule_is_preserved_for_non_emergency(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 1,
        "timestamp": 0.1,
        "objects": [
            {"score": 0.90, "class_name": "traffic_cone", "box_lidar": {"x": 12.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    raw = Control(steer=0.33, throttle=0.66, brake=0.0)
    ctrl = system.process(raw, detection, {"speed": 4.0, "command_near": 2}, 0.1, legacy_rule_action="clear_stuck_recovery")
    assert ctrl is raw
    assert system.last_debug["legacy_preserved"] is True


def test_tracker_estimates_velocity(monkeypatch):
    mod = load_module(monkeypatch)
    tracker = mod.ObjectTracker(max_match_distance=5.0)
    first = [{"score": 0.9, "class_name": "car", "box_lidar": {"x": 10.0, "y": 0.0}}]
    second = [{"score": 0.9, "class_name": "car", "box_lidar": {"x": 12.0, "y": 0.0}}]
    tracker.update(first, 1.0)
    tracks = tracker.update(second, 2.0)
    assert len(tracks) == 1
    assert tracks[0]["vx"] == pytest.approx(2.0)
    assert tracks[0]["speed"] == pytest.approx(2.0)


def test_roundabout_and_blind_spot_recognition(monkeypatch):
    mod = load_module(monkeypatch)
    recognizer = mod.ScenarioRecognizer(mod.AuxiliaryConfig())
    roundabout = mod.AuxFeatures(confidence=0.6, junction_like=True, route_curvature=4.0)
    assert recognizer.recognize(roundabout).macro_scenario == "roundabout"
    blind = mod.AuxFeatures(confidence=0.7, junction_like=True, side_risk=True, left_clear=False)
    assert recognizer.recognize(blind).macro_scenario == "blind_spot_hidden_car"


def test_roundabout_layout_does_not_use_construction_open_side_unwedge(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=0.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.2,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    for _ in range(12):
        action = planner.plan(features, estimate)

    assert action.reason != "construction_close_static_obstacle_open_side_unwedge"
    assert action.reason != "construction_open_side_reverse_unwedge"


def test_roundabout_layout_blockage_gets_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=3.05,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.02,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=3.05,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_layout_blockage_cautious_creep"
    assert action.throttle_floor == pytest.approx(0.45)
    assert action.throttle_cap == pytest.approx(0.65)
    assert action.steer_limit == pytest.approx(0.35)


def test_roundabout_memory_suppresses_later_construction_open_side_nudge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.red_final_context_frames = 20
    roundabout_features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=0,
        ego_speed=3.0,
        junction_like=True,
        route_curvature=3.2,
    )
    roundabout_estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")
    planner.plan(roundabout_features, roundabout_estimate)

    construction_features = mod.AuxFeatures(
        confidence=0.7,
        front_clear=False,
        front_obstacle_distance=6.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=0.0,
        junction_like=False,
        route_curvature=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=6.3,
        lidar_blockage_ratio=0.65,
        lidar_center_blockage_ratio=0.45,
        lidar_left_blockage_ratio=0.70,
        lidar_right_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    construction_estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.7,
        "PREPARE",
        "temporary construction after roundabout",
    )
    for _ in range(8):
        action = planner.plan(construction_features, construction_estimate)

    assert planner.roundabout_context_frames > 0
    assert action.reason != "distant_lidar_open_side_nudge"
    assert action.reason == "roundabout_static_obstacle_pre_stop"


def test_roundabout_memory_survives_long_post_yield_phase(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    roundabout_features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=0,
        ego_speed=3.0,
        junction_like=True,
        route_curvature=3.2,
    )
    roundabout_estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")
    planner.plan(roundabout_features, roundabout_estimate)

    construction_features = mod.AuxFeatures(
        confidence=0.7,
        front_clear=False,
        front_obstacle_distance=6.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=0.0,
        junction_like=False,
        route_curvature=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=6.3,
        lidar_blockage_ratio=0.65,
        lidar_center_blockage_ratio=0.45,
        lidar_left_blockage_ratio=0.70,
        lidar_right_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    construction_estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.7,
        "PREPARE",
        "temporary construction after roundabout",
    )
    for _ in range(360):
        action = planner.plan(construction_features, construction_estimate)

    assert planner.roundabout_context_frames > 0
    assert action.reason == "roundabout_static_obstacle_pre_stop"


def test_roundabout_ultra_close_static_caps_speed_before_route_deviation(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.85,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.99,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.85,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.31,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_ultra_close_static_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.42)
    assert action.steer_limit == pytest.approx(0.10)


def test_roundabout_ultra_close_static_stalled_uses_reverse_clearance(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.06,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.04,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.06,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout route66 ultra close stall")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_ultra_close_static_reverse_clearance"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_roundabout_ultra_close_static_low_speed_keeps_control(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.16,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.23,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.16,
        lidar_blockage_ratio=0.65,
        lidar_center_blockage_ratio=0.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_ultra_close_static_controlled_forward"
    assert action.throttle_cap == pytest.approx(0.42)
    assert action.steer_bias == pytest.approx(-0.10)


def test_roundabout_close_static_gets_cautious_creep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.red_final_context_frames = 20
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.4,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.02,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.4,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.20,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_close_static_cautious_creep"
    assert action.throttle_floor == pytest.approx(0.18)
    assert action.steer_limit == pytest.approx(0.22)


def test_roundabout_close_static_caps_near_speed_before_release(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.62,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=1.72,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.62,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=86,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_near_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.56)
    assert action.steer_limit == pytest.approx(0.12)
    assert action.steer_bias == pytest.approx(-0.08)


def test_roundabout_mid_close_static_gets_progress_creep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=4.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.1,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.2,
        lidar_blockage_ratio=0.75,
        lidar_center_blockage_ratio=0.55,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_close_static_progress_creep"
    assert action.throttle_floor == pytest.approx(0.24)
    assert action.throttle_cap == pytest.approx(0.40)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_limit == pytest.approx(0.58)


def test_roundabout_mid_static_progress_covers_six_meter_edge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8, front_clear=False, front_obstacle_distance=5.85,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.0, junction_like=True, route_curvature=3.2,
        lidar_available=True, lidar_stale=False, lidar_front_distance=5.85,
        lidar_blockage_ratio=0.20, lidar_center_blockage_ratio=0.0, lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_progress_creep"
    assert action.throttle_floor == pytest.approx(0.24)


def test_roundabout_mid_static_progress_escalates_to_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 36
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=4.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.2,
        lidar_blockage_ratio=0.80,
        lidar_center_blockage_ratio=0.55,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_progress_side_push"
    assert action.throttle_floor == pytest.approx(0.38)
    assert action.throttle_cap == pytest.approx(0.58)
    assert action.steer_bias == pytest.approx(-0.30)
    assert action.steer_limit == pytest.approx(0.42)


def test_roundabout_low_conf_close_blockage_keeps_forward_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.balanced_blockage_progress_frames = 12
    planner.blocked_frames = 105
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=3.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.05,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=3.3,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout close low-conf blockage")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_progress_side_push"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.40)
    assert action.steer_bias == pytest.approx(-0.30)
    assert action.steer_limit == pytest.approx(0.46)


def test_roundabout_close_static_progress_caps_speed_near_pole(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=3.88,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=1.74,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=3.88,
        lidar_blockage_ratio=0.90,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_progress_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.30)
    assert action.steer_limit == pytest.approx(0.12)


def test_roundabout_close_static_high_blocked_gets_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 58
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.02,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.2,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.20,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_close_static_high_blocked_push"
    assert action.throttle_floor == pytest.approx(0.34)
    assert action.throttle_cap == pytest.approx(0.52)
    assert action.steer_bias == pytest.approx(-0.10)


def test_roundabout_close_static_balanced_reverse_sweep_after_long_block(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 95
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.46,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.46,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")
    action = planner.plan(features, estimate)
    assert action.reason == "roundabout_close_static_balanced_reverse_sweep"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.48)
    assert action.steer_limit == pytest.approx(0.58)


def test_roundabout_far_static_obstacle_caps_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=21.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=6.5,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=21.5,
        lidar_blockage_ratio=0.25,
        lidar_center_blockage_ratio=0.15,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_static_obstacle_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.42)
    assert action.steer_limit == pytest.approx(0.08)




def test_roundabout_close_vehicle_yield_brakes_before_prius_conflict(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=0.75,
        ego_speed=3.4,
        front_clear=False,
        front_vehicle_distance=4.8,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        junction_like=True,
        route_curvature=3.8,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="roundabout",
        confidence=0.75,
        phase="APPROACH",
        reason="roundabout close prius",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_vehicle_yield_brake"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.68)
    assert action.target_speed == pytest.approx(0.0)
    assert planner.roundabout_vehicle_yield_cooldown_frames > 0


def test_roundabout_close_vehicle_yield_skips_red_context(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=0.75,
        ego_speed=3.4,
        front_clear=False,
        front_vehicle_distance=4.8,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=4.2,
        junction_like=True,
        route_curvature=3.8,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="roundabout", confidence=0.75, phase="APPROACH")

    action = planner.plan(features, estimate)

    assert action.reason != "roundabout_close_vehicle_yield_brake"


def test_roundabout_approach_triggers_scored_brake_response(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.70,
        ego_speed=4.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        junction_like=True,
        route_curvature=3.6,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="roundabout",
        confidence=0.70,
        phase="APPROACH",
        reason="curved junction topology",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_approach_scored_brake_response"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.48)
    assert planner.roundabout_approach_brake_cooldown_frames > 0


def test_roundabout_approach_brake_skips_red_context(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.70,
        ego_speed=4.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=4.0,
        junction_like=True,
        route_curvature=3.6,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="roundabout", confidence=0.70, phase="APPROACH")

    action = planner.plan(features, estimate)

    assert action.reason != "roundabout_approach_scored_brake_response"


def test_roundabout_mid_static_low_speed_releases_raw_brake(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True)
    planner = mod.ScenarioRulePlanner(cfg)
    supervisor = mod.SafetySupervisor(cfg)
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=15.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.4,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=15.0,
        lidar_blockage_ratio=0.25,
        lidar_center_blockage_ratio=0.15,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)
    ctrl = supervisor.apply(
        Control(throttle=0.0, brake=1.0, steer=-0.3),
        features,
        estimate,
        action,
    )

    assert action.reason == "roundabout_static_obstacle_speed_cap"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle >= 0.36
    assert abs(ctrl.steer) <= 0.20


def test_roundabout_far_static_low_speed_gets_progress_release(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=22.4,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=22.4,
        lidar_blockage_ratio=0.25,
        lidar_center_blockage_ratio=0.15,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_far_static_progress_release"
    assert action.throttle_floor == pytest.approx(0.45)
    assert action.brake_cap == pytest.approx(0.0)


def test_roundabout_mid_static_high_blocked_gets_progress_release(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 52
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=16.8,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.10,
        junction_like=True,
        route_curvature=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=16.8,
        lidar_blockage_ratio=0.25,
        lidar_center_blockage_ratio=0.15,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout topology")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_mid_static_blocked_progress_release"
    assert action.throttle_floor == pytest.approx(0.72)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_limit == pytest.approx(0.16)


def test_roundabout_far_static_low_speed_gets_progress_probe(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 30
    features = mod.AuxFeatures(
        confidence=0.7,
        front_clear=False,
        front_obstacle_distance=29.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=29.0,
        lidar_blockage_ratio=0.12,
        lidar_center_blockage_ratio=0.10,
        lidar_open_side="right",
        detection_object_count=40,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.7, "PREPARE", "roundabout far static obstacle")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_far_static_progress_probe"
    assert action.throttle_floor == pytest.approx(0.24)
    assert action.brake_cap == pytest.approx(0.0)


def test_roundabout_clear_lane_speed_guard_limits_post_pass_speed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=7.6,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
        route_curvature=1.2,
        junction_like=True,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.75, "NORMAL", "roundabout clear after obstacle")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_clear_lane_speed_guard"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake >= 0.38
    assert action.steer_limit <= 0.18


def test_legacy_rule_is_preserved_even_for_emergency_pedestrian(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 1,
        "timestamp": 0.1,
        "objects": [
            {"score": 0.95, "class_name": "pedestrian", "box_lidar": {"x": 3.0, "y": 0.1}},
        ],
        "map_objects": [],
    }
    raw = Control(steer=0.11, throttle=0.22, brake=0.33)
    ctrl = system.process(raw, detection, {"speed": 2.0, "command_near": 4}, 0.1, legacy_rule_action="front_pedestrian_brake")
    assert ctrl is raw
    assert system.last_debug["legacy_preserved"] is True

def test_side_risk_without_longitudinal_conflict_does_not_take_over(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(confidence=0.8, side_risk=True, front_clear=True, risk_level=2)
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_reckless_lane_cutting",
        confidence=0.6,
        phase="PREPARE",
        reason="side risk near ego lane",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "PREPARE"
    assert "side_risk_observed" in action.reason

def test_front_vehicle_near_hazard_crawls_instead_of_hard_stop(monkeypatch):
    monkeypatch.setenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.balanced_blockage_progress_frames = 12
    planner.blocked_frames = 105
    features = mod.AuxFeatures(
        confidence=1.0,
        immediate_hazard=True,
        risk_level=3,
        front_vehicle_distance=3.0,
        front_clear=False,
        ego_speed=0.2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="reverse_vehicle",
        confidence=1.0,
        phase="YIELD_OR_BRAKE",
        reason="close vehicle conflict at low ego speed",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.brake == 0.0
    assert action.throttle_cap == pytest.approx(0.28)
    assert action.steer_limit == pytest.approx(0.75)

def test_tracked_reversing_vehicle_uses_ttc_limited_brake(monkeypatch):
    monkeypatch.setenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    first = {
        "frame": 1,
        "timestamp": 1.0,
        "objects": [
            {"score": 0.95, "class_name": "car", "box_lidar": {"x": 8.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    second = {
        "frame": 2,
        "timestamp": 2.0,
        "objects": [
            {"score": 0.95, "class_name": "car", "box_lidar": {"x": 5.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    system.process(Control(throttle=0.5), first, {"speed": 0.2, "command_near": 2}, 1.0)
    ctrl = system.process(Control(throttle=0.5), second, {"speed": 0.2, "command_near": 2}, 2.0)
    assert ctrl.throttle == 0.0
    assert ctrl.brake >= 0.35
    assert system.last_debug["macro_scenario"] == "reverse_vehicle"
    assert system.last_debug["reversing_vehicle_evidence"] is True
    assert system.last_debug["front_vehicle_ttc"] == pytest.approx(5.0 / 3.0)

def test_reverse_vehicle_ttc_brake_even_when_default_observe(monkeypatch):
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        immediate_hazard=True,
        risk_level=3,
        front_clear=False,
        front_vehicle_distance=3.98,
        front_vehicle_ttc=2.32,
        front_vehicle_closing_speed=1.72,
        reversing_vehicle_evidence=True,
        ego_speed=2.65,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="reverse_vehicle",
        confidence=1.0,
        phase="YIELD_OR_BRAKE",
        reason="tracked front vehicle closing/reversing",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "reverse_vehicle_ttc_defensive_brake"
    assert action.brake == pytest.approx(0.48)
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.steer_limit == pytest.approx(0.25)


def test_reverse_vehicle_close_brake_without_reverse_macro(monkeypatch):
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        risk_level=3,
        front_vehicle_distance=4.46875,
        front_vehicle_ttc=7.15,
        front_vehicle_closing_speed=0.625,
        reversing_vehicle_evidence=True,
        ego_speed=3.31,
        red_light_active=True,
        red_stop_distance=4.46875,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_reckless_lane_cutting",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "reverse_vehicle_ttc_defensive_brake"
    assert action.brake == pytest.approx(0.48)
    assert action.throttle_cap == pytest.approx(0.0)


def test_reverse_vehicle_close_brake_persists_when_track_flickers(monkeypatch):
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    first = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        immediate_hazard=True,
        risk_level=3,
        front_vehicle_distance=3.2,
        front_vehicle_ttc=5.1,
        front_vehicle_closing_speed=0.625,
        front_obstacle_distance=2.95,
        reversing_vehicle_evidence=True,
        ego_speed=2.73,
        red_light_active=True,
        red_stop_distance=3.2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=1.0,
        phase="PREPARE",
        reason="close obstacle with reversing trace",
    )
    action = planner.plan(first, estimate)
    assert action.reason == "reverse_vehicle_ttc_defensive_brake"

    second = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        immediate_hazard=True,
        risk_level=3,
        front_vehicle_distance=3.97,
        front_obstacle_distance=1.67,
        reversing_vehicle_evidence=False,
        ego_speed=2.04,
        red_light_active=True,
        red_stop_distance=3.97,
    )
    action = planner.plan(second, estimate)
    assert action.active is True
    assert action.reason == "reverse_vehicle_ttc_defensive_brake"
    assert action.brake == pytest.approx(0.48)


def test_supervisor_forward_action_clears_raw_reverse(monkeypatch):
    mod = load_module(monkeypatch)
    supervisor = mod.SafetySupervisor(mod.AuxiliaryConfig())
    raw = Control(steer=0.0, throttle=0.1, brake=0.0)
    raw.reverse = True
    features = mod.AuxFeatures(confidence=1.0, ego_speed=0.0)
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "RECOVER", "forced route-prior macro")
    action = mod.PlannerAction(
        True,
        "RECOVER",
        throttle_cap=0.8,
        throttle_floor=0.5,
        brake_cap=0.0,
        reverse=False,
        reason="reverse_vehicle_observed_static_forward_resume",
    )

    ctrl = supervisor.apply(raw, features, estimate, action)

    assert ctrl.reverse is False
    assert ctrl.throttle >= 0.5
    assert ctrl.brake == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_observed_unwedges(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=1.55,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_floor >= 0.50
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_near_two_meter_gap_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=1.78,
        reversing_vehicle_evidence=False,
        ego_speed=-0.25,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False


def test_reverse_vehicle_route_prior_static_observed_mid_gap_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.39,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_observed_forward_resumes(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.64,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.80,
        lidar_center_blockage_ratio=0.85,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.90
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_uses_stronger_nudge(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.02,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.steer_bias >= 0.30
    assert action.steer_min_magnitude >= 0.20
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_reverse_swing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.44,
        reversing_vehicle_evidence=False,
        ego_speed=0.004,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    first_action = planner.plan(features, estimate)
    action = planner.plan(features, estimate)

    assert first_action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reason == "reverse_vehicle_observed_static_high_blockage_reverse_swing"
    assert action.reverse is True
    assert action.steer_bias <= -0.44
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_low_speed_swing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.83,
        reversing_vehicle_evidence=False,
        ego_speed=0.48,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    first_action = planner.plan(features, estimate)
    action = planner.plan(features, estimate)

    assert first_action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reason == "reverse_vehicle_observed_static_high_blockage_reverse_swing"
    assert action.reverse is True
    assert action.steer_bias <= -0.44
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_balanced_swing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.45,
        reversing_vehicle_evidence=False,
        ego_speed=0.004,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=1.0,
        lidar_open_side="balanced",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    first_action = planner.plan(features, estimate)
    action = planner.plan(features, estimate)

    assert first_action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reason == "reverse_vehicle_observed_static_high_blockage_reverse_swing"
    assert action.reverse is True
    assert abs(action.steer_bias) >= 0.36
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_two_meter_swing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.05,
        reversing_vehicle_evidence=False,
        ego_speed=0.01,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    first_action = planner.plan(features, estimate)
    action = planner.plan(features, estimate)

    assert first_action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reason == "reverse_vehicle_observed_static_high_blockage_reverse_swing"
    assert action.reverse is True
    assert action.steer_bias <= -0.44
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_high_blockage_one_eight_meter_swing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=1.80,
        reversing_vehicle_evidence=False,
        ego_speed=0.10,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    first_action = planner.plan(features, estimate)
    action = planner.plan(features, estimate)

    assert first_action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reason == "reverse_vehicle_observed_static_high_blockage_reverse_swing"
    assert action.reverse is True
    assert action.steer_bias <= -0.44
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_two_meter_gap_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.01,
        reversing_vehicle_evidence=False,
        ego_speed=-0.10,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_mid_gap_resumes_at_moderate_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=4.36,
        reversing_vehicle_evidence=False,
        ego_speed=2.83,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_mid_gap_resumes_at_higher_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.34,
        reversing_vehicle_evidence=False,
        ego_speed=3.69,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_low_blockage_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=3.22,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.075,
        lidar_center_blockage_ratio=0.075,
        lidar_left_blockage_ratio=0.05,
        lidar_right_blockage_ratio=0.10,
        lidar_open_side="balanced",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_low_blockage_forward_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.90
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_low_blockage_far_edge_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=4.86,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.14,
        lidar_center_blockage_ratio=0.14,
        lidar_left_blockage_ratio=0.10,
        lidar_right_blockage_ratio=0.12,
        lidar_open_side="balanced",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_low_blockage_forward_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.90
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_low_blockage_far_gap_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=12.57,
        reversing_vehicle_evidence=False,
        ego_speed=3.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.40,
        lidar_center_blockage_ratio=0.40,
        lidar_left_blockage_ratio=0.45,
        lidar_right_blockage_ratio=0.20,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_low_blockage_forward_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.90
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_clear_low_blockage_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        reversing_vehicle_evidence=False,
        ego_speed=3.48,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.6625,
        lidar_center_blockage_ratio=0.6625,
        lidar_left_blockage_ratio=0.8,
        lidar_right_blockage_ratio=0.1,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_route_prior_clear_low_blockage_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.60
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_clear_low_blockage_higher_speed_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        reversing_vehicle_evidence=False,
        ego_speed=4.46,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.3125,
        lidar_center_blockage_ratio=0.3125,
        lidar_left_blockage_ratio=0.42,
        lidar_right_blockage_ratio=0.18,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_route_prior_clear_low_blockage_resume"
    assert action.reverse is False
    assert action.throttle_floor >= 0.60
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_clear_unknown_open_side_resumes_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        reversing_vehicle_evidence=False,
        ego_speed=2.85,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
        lidar_left_blockage_ratio=0.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="unknown",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_route_prior_clear_low_blockage_resume"
    assert action.reverse is False
    assert action.steer_bias == pytest.approx(0.0)
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_far_gap_keeps_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=8.18,
        reversing_vehicle_evidence=False,
        ego_speed=2.36,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_far_forward_keepalive"
    assert action.reverse is False
    assert action.throttle_floor >= 0.50
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_far_gap_keeps_forward_at_higher_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=10.51,
        reversing_vehicle_evidence=False,
        ego_speed=3.22,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_far_forward_keepalive"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_forward_resumes_while_slowly_reversing(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.52,
        reversing_vehicle_evidence=False,
        ego_speed=-0.45,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_forward_resumes_while_rolling(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.76,
        reversing_vehicle_evidence=False,
        ego_speed=2.65,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_forward_resume"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_close_gap_reverses_while_rolling(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=1.35,
        reversing_vehicle_evidence=False,
        ego_speed=1.52,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_reverse_unwedge"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_route_prior_static_ultraclose_reverses_without_high_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=0.008,
        reversing_vehicle_evidence=False,
        ego_speed=0.0,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=0.35,
        lidar_center_blockage_ratio=0.20,
        lidar_left_blockage_ratio=0.45,
        lidar_right_blockage_ratio=0.05,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "PREPARE", "forced route-prior macro")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_static_ultraclose_reverse_unwedge"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)


def test_reverse_vehicle_default_observe_only(monkeypatch):
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    first = {
        "frame": 1,
        "timestamp": 1.0,
        "objects": [
            {"score": 0.95, "class_name": "car", "box_lidar": {"x": 8.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    second = {
        "frame": 2,
        "timestamp": 2.0,
        "objects": [
            {"score": 0.95, "class_name": "car", "box_lidar": {"x": 5.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    system.process(Control(throttle=0.5), first, {"speed": 0.2, "command_near": 2}, 1.0)
    raw = Control(throttle=0.5)
    ctrl = system.process(raw, second, {"speed": 0.2, "command_near": 2}, 2.0)
    assert ctrl is raw
    assert system.last_debug["macro_scenario"] == "reverse_vehicle"
    assert system.last_debug["action_active"] is False
    assert system.last_debug["reason"] == "reverse_vehicle_observed_only"

def test_agent_legacy_detection_rules_default_disabled_source():
    text = open("team_code/drivetransformer_b2d_agent.py", "r").read()
    assert "CVCI_LEGACY_DETECTION_RULES_ENABLED" in text
    assert "os.environ.get('CVCI_LEGACY_DETECTION_RULES_ENABLED', '0')" in text
    assert "if CVCI_LEGACY_DETECTION_RULES_ENABLED:" in text


def test_cvci_scenario_catalog_covers_all_families():
    with open("configs/cvci_scenario_catalog.yaml", "r") as f:
        catalog = yaml.safe_load(f)
    families = catalog["scenario_families"]
    assert catalog["metadata"]["num_families"] == 12
    assert len(families) == 12
    assert sum(len(item["routes"]) for item in families.values()) == 144
    for name, item in families.items():
        assert len(item["routes"]) == 12, name
        assert len(item["route_ids"]) == 12, name
        assert item["xml_scenario_types"], name
        assert item["python_scenario_classes"], name
        assert item["recommended_rule_state_machine"], name
    poor7 = {
        "trucks_encountered_during_construction",
        "highway_accident_vehicle",
        "four_students_crossing_the_road",
        "reverse_vehicle",
        "roundabout",
        "high_speed_reckless_lane_cutting",
        "blind_spot_hidden_car",
    }
    assert poor7.issubset(families)

def test_static_obstacle_low_risk_is_observed_without_takeover(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=7.0,
        risk_level=2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "PREPARE"
    assert "static_obstacle_observed" in action.reason

def test_static_obstacle_near_but_not_extreme_is_observed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        immediate_hazard=True,
        risk_level=3,
        front_obstacle_distance=3.2,
        front_clear=False,
        ego_speed=0.2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "PREPARE"
    assert "static_obstacle_observed" in action.reason

def test_static_obstacle_extreme_distance_is_observed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        immediate_hazard=True,
        risk_level=3,
        front_obstacle_distance=1.8,
        front_clear=False,
        ego_speed=1.5,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "PREPARE"
    assert "static_obstacle_observed" in action.reason

def test_construction_static_creep_release_after_stable_stop(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    action = None
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.7,
        risk_level=2,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.9,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    for _ in range(8):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.state == "AVOID_OR_PASS"
    assert action.reason == "construction_static_creep_release"
    assert action.throttle_floor == pytest.approx(0.16)
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_static_creep_release_covers_six_meter_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    action = None
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=5.55,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.9,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    for _ in range(8):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_static_creep_release"
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_static_side_gap_forward_push(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    action = None
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.0,
        risk_level=2,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.52,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    for _ in range(8):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_static_side_gap_forward_push"
    assert action.throttle_floor == pytest.approx(0.68)
    assert action.throttle_cap == pytest.approx(0.86)
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_static_open_side_push_after_persistent_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    action = None
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.6,
        risk_level=2,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    for _ in range(16):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_open_side_push_release"
    assert action.throttle_floor == pytest.approx(0.32)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias < 0.0


def test_construction_static_open_side_push_covers_five_meter_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=4.8,
        risk_level=2,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = None
    for _ in range(16):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_open_side_push_release"
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_static_open_side_push_reverses_after_long_close_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.12,
        risk_level=2,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = None
    for _ in range(40):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias < 0.0


def test_full_blockage_open_side_escape_reverses_after_no_progress(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.9,
        risk_level=2,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = None
    for _ in range(17):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias < 0.0


def test_full_blockage_open_side_reverse_window_is_sticky(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.7,
        risk_level=2,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    for _ in range(17):
        action = planner.plan(features, estimate)
    assert action.reason == "construction_open_side_reverse_unwedge"

    for _ in range(20):
        action = planner.plan(features, estimate)
        assert action.reason == "construction_open_side_reverse_unwedge"
        assert action.reverse is True


def test_close_open_side_reverse_continuation_preempts_memory_escape(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 20
    planner.reverse_unwedge_frames = 8
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.7,
        risk_level=2,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert planner.reverse_unwedge_frames == 7


def test_construction_close_static_open_side_continues_after_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.static_creep_frames = 12
    planner.open_side_pass_memory_frames = 4
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=4.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.10,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.0,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="persistent close static blockage with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_open_side_continue"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.24)
    assert action.steer_bias > 0.0


def test_construction_close_static_open_side_continue_covers_seven_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.static_creep_frames = 9
    planner.open_side_pass_memory_frames = 3
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=6.8,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.20,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=6.8,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="persistent seven meter static blockage with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_open_side_continue"
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_open_side_memory_escape_covers_two_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.static_creep_frames = 10
    planner.open_side_pass_memory_frames = 4
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.08,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.3,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="persistent two meter static blockage with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.30)
    assert action.steer_min_magnitude == pytest.approx(0.35)


def test_construction_close_open_side_memory_push_covers_high_speed_route_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "APPROACH"
    planner.static_creep_frames = 5
    planner.open_side_pass_memory_frames = 0
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.4,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.4,
        lidar_blockage_ratio=0.88,
        lidar_center_blockage_ratio=0.88,
        lidar_left_blockage_ratio=0.88,
        lidar_right_blockage_ratio=0.12,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="route27 close open side stall",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason in {
        "construction_close_open_side_memory_push",
        "distant_lidar_open_side_nudge",
    }
    assert action.throttle_floor is not None
    assert action.steer_bias > 0.0


def test_construction_vehicle_observation_can_continue_open_side_push(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_vehicle_distance=3.9,
        front_vehicle_ttc=5.0,
        front_obstacle_distance=3.9,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="vehicle-like static blockage with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_vehicle_open_side_push_release"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_construction_vehicle_observation_far_vehicle_pushes_five_meter_blockage(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_vehicle_distance=16.0,
        front_vehicle_ttc=5.0,
        front_obstacle_distance=5.4,
        risk_level=2,
        ego_speed=0.15,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.4,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="far vehicle plus close construction blockage",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_vehicle_open_side_push_release"
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_vehicle_observation_near_vehicle_does_not_push(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_vehicle_distance=5.0,
        front_vehicle_ttc=5.0,
        front_obstacle_distance=5.4,
        risk_level=2,
        ego_speed=0.15,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.4,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="near vehicle should not be pushed",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "construction_vehicle_open_side_push_release"


def test_high_speed_sparse_lidar_cone_approach_slows_under_low_confidence(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=18.5,
        lidar_front_distance=20.5,
        lidar_blockage_ratio=0.09,
        lidar_center_blockage_ratio=0.0,
        lidar_left_density=7,
        lidar_right_density=0,
        lidar_lateral_centroid=-1.6,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="low-confidence sparse construction geometry",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_high_speed_sparse_lidar_approach"
    assert action.brake == pytest.approx(0.58)
    assert action.throttle_cap == pytest.approx(0.0)
    assert planner.construction_cone_entry_slow_frames == 120


def test_high_speed_sparse_lidar_cone_approach_respects_red_stop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        red_stop_distance=18.0,
        red_light_active=True,
        risk_level=0,
        ego_speed=18.5,
        lidar_front_distance=20.5,
        lidar_blockage_ratio=0.09,
        lidar_center_blockage_ratio=0.0,
        lidar_left_density=7,
        lidar_lateral_centroid=-1.6,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="red stop has priority",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "construction_high_speed_sparse_lidar_approach"



def test_construction_corridor_memory_caps_low_confidence_clear_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.construction_corridor_memory_frames = 12
    features = mod.AuxFeatures(
        confidence=0.3,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=5.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.3,
        phase="NORMAL",
        reason="construction memory after sparse corridor",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_corridor_memory_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.48)
    assert planner.construction_corridor_memory_frames == 11


def test_construction_corridor_memory_recovers_without_permanent_stop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.construction_corridor_memory_frames = 5
    features = mod.AuxFeatures(
        confidence=0.3,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=0.4,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.3,
        phase="NORMAL",
        reason="construction memory can recover",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_corridor_memory_cautious_recovery"
    assert action.throttle_floor == pytest.approx(0.12)
    assert action.brake_cap == pytest.approx(0.0)

def test_high_speed_lateral_vehicle_guard_ignores_static_high_score_side_track(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=10.2,
        lidar_front_distance=28.0,
        lidar_open_side="right",
        tracked_objects=[
            {
                "class_name": "car",
                "x": 30.9,
                "y": -2.3,
                "score": 0.62,
                "observed_frames": 3,
                "closing_speed": 0.0,
                "lateral_velocity": 0.0,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="low-confidence lateral vehicle",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"

    features.tracked_objects = [
        {
            "class_name": "car",
            "x": 30.9,
            "y": -2.3,
            "score": 0.62,
            "observed_frames": 3,
            "closing_speed": 1.1,
            "lateral_velocity": 0.0,
        }
    ]
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "high_speed_lateral_vehicle_cutin_guard"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.62)
    assert action.steer_limit == pytest.approx(0.16)


def test_high_speed_lateral_vehicle_guard_ignores_low_score_static_side_track(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=6.9,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 38.3,
                "y": -2.1,
                "score": 0.32,
                "observed_frames": 4,
                "closing_speed": 0.0,
                "lateral_velocity": 0.0,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="low-score static side track",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"


def test_high_speed_lateral_vehicle_guard_accepts_lateral_motion_low_score(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=7.2,
        lidar_front_distance=28.0,
        lidar_open_side="right",
        tracked_objects=[
            {
                "class_name": "car",
                "x": 32.0,
                "y": 2.2,
                "score": 0.31,
                "observed_frames": 4,
                "closing_speed": 0.0,
                "lateral_velocity": -0.5,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="moving lateral side track",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "lateral_intersection_scored_brake_response"
    assert action.target_speed == pytest.approx(2.0)
    assert action.brake == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.68)


def test_high_speed_lateral_vehicle_guard_uses_vx_at_route135_speed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=5.3,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 41.8,
                "y": -3.3,
                "score": 0.42,
                "observed_frames": 9,
                "closing_speed": 0.0,
                "vx": 12.4,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="route135 lateral car",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_scored_brake_response"
    assert action.target_speed == pytest.approx(2.0)
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.68)

    features.tracked_objects = []
    features.ego_speed = 2.2
    action = planner.plan(features, estimate)
    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"


def test_route135_single_frame_openpcdet_lateral_car_triggers_guard(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=6.4,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 36.9,
                "y": -2.1,
                "score": 0.40,
                "observed_frames": 1,
                "closing_speed": 0.0,
                "vx": 0.0,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="route135 openpcdet lateral car",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_scored_brake_response"
    assert action.target_speed == pytest.approx(2.0)
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.68)


def test_route135_lateral_memory_keeps_rolling_after_initial_detection(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=6.4,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 36.9,
                "y": -2.1,
                "score": 0.40,
                "observed_frames": 1,
                "closing_speed": 0.0,
                "vx": 0.0,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    first = planner.plan(features, estimate)
    assert first.reason == "lateral_intersection_scored_brake_response"

    features.ego_speed = 2.4
    features.tracked_objects = []
    second = planner.plan(features, estimate)

    assert second.active is True
    assert second.reason == "lateral_intersection_keep_rolling"
    assert second.target_speed == pytest.approx(9.0)
    assert second.throttle_floor == pytest.approx(0.88)
    assert second.brake_cap == pytest.approx(0.0)


def test_route135_lateral_release_memory_covers_high_raw_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.lateral_intersection_release_frames = 800
    planner.high_speed_lateral_guard_frames = 0
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=9.8,
        lidar_front_distance=18.8,
        lidar_blockage_ratio=0.7,
        lidar_open_side="right",
        tracked_objects=[],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_keep_rolling"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.throttle_floor is None
    assert action.brake_cap == pytest.approx(0.0)


def test_route135_lateral_release_memory_soft_trims_overspeed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.lateral_intersection_release_frames = 800
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=10.8,
        tracked_objects=[],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_keep_rolling"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.throttle_floor is None
    assert action.brake == pytest.approx(0.10)
    assert action.brake_cap == pytest.approx(0.16)


def test_route135_lateral_release_memory_overrides_disabled_raw_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.lateral_intersection_release_frames = 800
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=8.8,
        tracked_objects=[],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_keep_rolling"
    assert action.throttle_floor == pytest.approx(0.88)
    assert action.brake_cap == pytest.approx(0.0)


def test_route135_lateral_release_memory_overrides_far_static_raw_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.lateral_intersection_release_frames = 800
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=17.0,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=9.8,
        lidar_available=True,
        lidar_front_distance=17.0,
        tracked_objects=[],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "lateral_intersection_far_static_memory_override"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake_cap == pytest.approx(0.0)


def test_route135_scored_brake_response_only_fires_once(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=6.2,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 31.2,
                "y": -2.0,
                "score": 0.42,
                "observed_frames": 3,
                "closing_speed": 0.9,
                "vx": 0.0,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    first = planner.plan(features, estimate)
    second = planner.plan(features, estimate)

    assert first.reason == "lateral_intersection_scored_brake_response"
    assert first.brake == pytest.approx(0.58)
    assert first.brake_cap == pytest.approx(0.68)
    assert second.reason == "lateral_intersection_scored_brake_response"
    assert second.throttle_cap == pytest.approx(0.0)
    assert second.brake == pytest.approx(0.58)
    assert second.brake_cap == pytest.approx(0.68)


def test_route135_lateral_memory_scores_brake_at_near_guard_edge(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.lateral_intersection_release_frames = 100
    planner.high_speed_lateral_guard_frames = 10
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=4.2,
        tracked_objects=[
            {
                "class_name": "car",
                "x": 34.2,
                "y": -1.9,
                "score": 0.35,
                "observed_frames": 6,
            }
        ],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.4, phase="NORMAL")

    action = planner.plan(features, estimate)

    assert action.reason == "lateral_intersection_scored_brake_response"
    assert action.brake == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.68)


def test_lateral_guard_memory_enables_stronger_mid_red_release(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    system.red_final_clamp_hold_frames = 90

    class MidRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.6},
                "lidar_geometry": None,
            }

    system.perception = MidRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.1},
        0.1,
    )

    assert 0.78 <= ctrl.throttle <= 0.95
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] in {"active_red_far_prolonged_creep_release", "students_long_active_red_final_release"}


def test_lateral_guard_memory_strengthens_mid_red_brake_response(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    tracked = [{"class_name": "car", "x": 32.0, "y": -2.0}]

    class MidRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 5.2},
                "lidar_geometry": None,
            }

    system.perception = MidRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 4.4},
        0.1,
    )

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.90
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"


def test_lateral_guard_memory_releases_late_red_until_close_conflict(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    tracked = [{"class_name": "car", "x": 55.4, "y": -2.4}]

    class LateRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 11.2},
                "lidar_geometry": None,
            }

    system.perception = LateRedPerception()
    ctrl = system.process(
        Control(throttle=0.4, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 5.9},
        0.1,
    )

    assert ctrl.throttle >= 0.58
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_lateral_guard_memory_releases_far_crossing_red_false_positive(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    tracked = [
        {"class_name": "car", "x": 30.8, "y": -20.4},
        {"class_name": "car", "x": 50.5, "y": -20.7},
    ]

    class FarCrossingRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 11.4},
                "lidar_geometry": None,
            }

    system.perception = FarCrossingRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.55, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 5.6},
        0.1,
    )

    assert ctrl.throttle >= 0.58
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_lateral_guard_memory_delays_mid_red_brake_until_track_closer(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    tracked = [{"class_name": "car", "x": 44.0, "y": -2.0}]

    class FarLateralRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 5.2},
                "lidar_geometry": None,
            }

    system.perception = FarLateralRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 3.2},
        0.1,
    )

    assert 0.45 <= ctrl.throttle <= 0.70
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_lateral_guard_memory_releases_far_red_false_hold_with_side_track(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    system.red_final_clamp_hold_frames = 40
    tracked = [{"class_name": "car", "x": 52.0, "y": -2.2}]

    class FarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 11.8},
                "lidar_geometry": None,
            }

    system.perception = FarRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.35, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.8},
        0.1,
    )

    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_lateral_guard_memory_releases_close_red_false_hold_with_side_track(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.rule_planner.lateral_intersection_release_frames = 100
    tracked = [{"class_name": "car", "x": 52.0, "y": -1.4}]

    class CloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": tracked,
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.8},
                "lidar_geometry": None,
            }

    system.perception = CloseRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.2, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.1},
        0.1,
    )

    assert ctrl.throttle >= 0.86
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_high_speed_lateral_vehicle_guard_drops_stale_memory_without_lidar_context(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.high_speed_lateral_guard_frames = 20
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=5.95,
        lidar_front_distance=None,
        lidar_open_side="unknown",
        tracked_objects=[],
        detection_object_count=0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="post-red recovery without current lateral evidence",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"


def test_high_speed_lateral_vehicle_guard_drops_stale_memory_with_far_lidar_context(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.high_speed_lateral_guard_frames = 20
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=7.3,
        lidar_front_distance=23.2,
        lidar_open_side="right",
        tracked_objects=[],
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="far sparse lidar should not sustain stale cut-in guard",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"


def test_high_speed_lateral_vehicle_guard_respects_pedestrian(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        front_pedestrian_distance=7.0,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        ego_speed=10.2,
        tracked_objects=[{"class_name": "car", "x": 30.9, "y": -2.3, "observed_frames": 3}],
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="pedestrian has priority",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "high_speed_lateral_vehicle_cutin_guard"


def test_balanced_construction_blockage_creeps_from_six_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=6.8,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.05,
        lidar_front_distance=6.8,
        lidar_blockage_ratio=0.03,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="balanced cone blockage at six meters",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "balanced_construction_blockage_straight_creep"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.22)


def test_balanced_construction_blockage_progress_push_after_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=7.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.04,
        lidar_front_distance=7.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.04,
        lidar_center_blockage_ratio=0.03,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="balanced cone blockage no progress",
    )

    action = None
    for _ in range(25):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_progress_push"
    assert action.throttle_floor == pytest.approx(0.65)
    assert action.steer_limit == pytest.approx(0.20)

    features.front_obstacle_distance = 5.7
    features.lidar_front_distance = 5.7
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_progress_push"
    assert action.throttle_floor == pytest.approx(0.65)
    assert action.brake_cap == pytest.approx(0.0)


def test_balanced_construction_blockage_escape_sweeps_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=6.4,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.03,
        lidar_front_distance=6.4,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.04,
        lidar_center_blockage_ratio=0.03,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="balanced cone blockage very long stall",
    )

    action = None
    for _ in range(105):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.48)
    assert action.steer_limit == pytest.approx(0.60)
    assert abs(action.steer_bias) == pytest.approx(0.40)


def test_balanced_construction_blockage_covers_nine_point_six_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=9.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.0,
        lidar_front_distance=9.6,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.04,
        lidar_center_blockage_ratio=0.03,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="balanced cone blockage at nine point six meters",
    )

    action = None
    for _ in range(25):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_progress_push"
    assert action.throttle_floor == pytest.approx(0.65)
    assert action.brake_cap == pytest.approx(0.0)


def test_balanced_blockage_memory_preempts_clear_road_reverse_on_lidar_gap(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 84
    planner.balanced_blockage_progress_frames = 12
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.02,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="balanced blockage lidar gap",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "balanced_construction_blockage_progress_push"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.65)
    assert planner.balanced_blockage_progress_frames == 11


def test_low_conf_center_blockage_escalates_progress_push_after_short_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=6.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.06,
        lidar_front_distance=6.6,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.90,
        lidar_center_blockage_ratio=0.92,
        lidar_left_blockage_ratio=0.45,
        lidar_right_blockage_ratio=0.45,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="low confidence balanced center blockage",
    )

    action = None
    for _ in range(10):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.reason == "low_conf_center_blockage_progress_push"
    assert action.target_speed == pytest.approx(3.2)
    assert action.throttle_cap == pytest.approx(1.0)
    assert action.throttle_floor == pytest.approx(0.65)
    assert planner.balanced_blockage_progress_frames == 80


def test_low_conf_center_blockage_escape_sweeps_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.55,
        front_clear=False,
        front_obstacle_distance=5.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.0,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.95,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-0.2,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.55,
        phase="PREPARE",
        reason="low confidence center blockage long stall",
    )

    action = None
    for _ in range(105):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reason == "low_conf_center_blockage_reverse_escape_sweep"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.48)
    assert action.steer_limit == pytest.approx(0.62)


def test_low_conf_center_blockage_reverses_after_close_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.balanced_blockage_progress_frames = 12
    planner.blocked_frames = 105
    features = mod.AuxFeatures(
        confidence=0.55,
        front_clear=False,
        front_obstacle_distance=4.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.0,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.95,
        lidar_open_side="right",
        lidar_lateral_centroid=-0.2,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.55,
        phase="PREPARE",
        reason="low confidence close center blockage long stall",
    )

    action = None
    for _ in range(105):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reverse is True
    assert action.reason == "low_conf_center_blockage_reverse_unwedge"
    assert action.throttle_floor == pytest.approx(0.30)
    assert action.steer_bias < 0.0


def test_low_conf_center_blockage_progress_memory_persists_at_moving_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.balanced_blockage_progress_frames = 12
    planner.blocked_frames = 0
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=6.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.65,
        lidar_front_distance=6.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.90,
        lidar_center_blockage_ratio=0.92,
        lidar_left_blockage_ratio=0.45,
        lidar_right_blockage_ratio=0.45,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="PREPARE",
        reason="low confidence balanced center blockage moving",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "low_conf_center_blockage_progress_push"
    assert action.throttle_floor == pytest.approx(0.65)
    assert planner.balanced_blockage_progress_frames == 11


def test_construction_far_progress_recovery_reverses_when_stalled(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.state = "RECOVER"
    planner.progress_recovery_frames = 20
    planner.last_open_side = "right"
    planner.blocked_frames = 5
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=15.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=15.2,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.70,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="RECOVER",
        reason="stalled far construction recovery",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_far_progress_stalled_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.46)
    assert action.steer_bias < 0.0


def test_static_open_side_push_covers_five_meter_full_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=5.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.06,
        lidar_front_distance=5.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="PREPARE",
        reason="five meter full blockage open side",
    )

    action = None
    for _ in range(16):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.reason in {
        "construction_static_open_side_push_release",
        "low_conf_center_blockage_progress_push",
        "construction_full_blockage_open_side_escape",
    }
    assert action.throttle_floor >= 0.32


def test_static_open_side_push_covers_moving_six_meter_full_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=5.8,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=0.64,
        lidar_front_distance=5.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="moving five meter full blockage open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason in {
        "construction_full_blockage_open_side_escape",
        "construction_static_open_side_push_release",
    }
    assert action.throttle_floor >= 0.40
    assert action.brake_cap == pytest.approx(0.0)
    assert abs(action.steer_bias) >= 0.34


def test_construction_close_obstacle_slowdown_at_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.6,
        risk_level=3,
        ego_speed=6.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_close_obstacle_open_side_slowdown"
    assert action.target_speed == pytest.approx(0.25)
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.88)
    assert action.steer_bias == pytest.approx(0.06)


def test_high_speed_temporary_construction_close_obstacle_slowdown(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=3.05,
        risk_level=2,
        ego_speed=3.6,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.3,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="temporary construction close obstacle",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_close_obstacle_open_side_slowdown"


def test_high_speed_temporary_construction_close_obstacle_brakes(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=2.4,
        risk_level=3,
        ego_speed=13.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.8,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.8,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="temporary construction close obstacle",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_high_speed_close_obstacle_brake"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.85)


def test_sparse_construction_cone_entry_slows_before_close_obstacle(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=5.9,
        risk_level=1,
        ego_speed=6.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.025,
        lidar_left_blockage_ratio=0.05,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-1.8,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="sparse construction cone entry",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_sparse_cone_entry_slowdown"
    assert action.target_speed == pytest.approx(0.45)
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.58)


def test_sparse_construction_cone_entry_memory_requires_construction_estimate(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="sparse construction cone entry",
    )
    first = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=5.9,
        risk_level=1,
        ego_speed=6.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.025,
        lidar_left_blockage_ratio=0.05,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-1.8,
        detection_object_count=100,
    )
    action = planner.plan(first, estimate)
    assert action.reason == "construction_sparse_cone_entry_slowdown"

    dropout = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        risk_level=0,
        ego_speed=5.5,
        detection_object_count=100,
    )
    action = planner.plan(dropout, mod.ScenarioEstimate("unknown", 0.4, "NORMAL", "lidar dropout"))
    assert action.reason != "construction_sparse_cone_entry_memory_slowdown"

    action = planner.plan(dropout, estimate)
    assert action.active is True
    assert action.reason == "construction_sparse_cone_entry_memory_slowdown"
    assert action.brake == pytest.approx(0.55)
    assert action.throttle_cap == pytest.approx(0.0)


def test_low_confidence_construction_does_not_open_side_recover(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=10.0,
        risk_level=1,
        ego_speed=3.4,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="low confidence corridor blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is False

    planner.state = "AVOID_OR_PASS"
    planner.post_pass_frames = 12
    features.front_obstacle_distance = 8.0
    features.ego_speed = 0.4
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "low_conf_center_blockage_straight_creep"
    assert action.steer_bias is None


def test_low_confidence_far_construction_blockage_gets_progress_release(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.62,
        front_clear=False,
        front_obstacle_distance=20.0,
        risk_level=1,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.62,
        phase="PREPARE",
        reason="far low confidence corridor blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "low_conf_far_center_blockage_progress_release"
    assert action.target_speed == pytest.approx(3.2)
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.0)




def test_route51_full_blockage_long_hold_reverse_covers_three_meter_edge(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.construction_full_blockage_escape_frames = 60
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=2,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="route51 full blockage edge",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_long_hold_reverse"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.42)
    assert action.steer_min_magnitude == pytest.approx(0.30)


def test_balanced_construction_blockage_reverses_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 54
    features = mod.AuxFeatures(
        confidence=0.62,
        front_clear=False,
        front_obstacle_distance=6.0,
        risk_level=1,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.30,
        lidar_center_blockage_ratio=0.10,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-0.2,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.62,
        phase="PREPARE",
        reason="balanced construction blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.48)


def test_observable_risk_recovery_does_not_creep_on_red_stop(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.observable_risk_creep_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=2,
        ego_speed=0.6,
        red_light_active=True,
        red_stop_distance=8.5,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="observable red stop hint",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"


def test_low_conf_clear_road_recovery_tolerates_lidar_center_noise(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        front_clear=True,
        risk_level=0,
        ego_speed=0.0,
        lidar_front_distance=5.0,
        lidar_center_blockage_ratio=0.35,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="low confidence clear road",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "clear_road_cautious_creep_recovery"


def test_close_construction_slowdown_ignores_high_speed_late_trigger(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=3.0,
        risk_level=3,
        ego_speed=13.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.reason != "construction_close_obstacle_open_side_slowdown"


def test_very_close_construction_open_side_escape(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.2,
        risk_level=3,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="very close construction blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_very_close_open_side_escape"
    assert action.steer_bias == pytest.approx(0.46)
    assert action.steer_min_magnitude == pytest.approx(0.38)
    assert action.throttle_floor == pytest.approx(0.34)


def test_very_close_construction_open_side_escape_covers_one_point_seven_meters(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.7,
        risk_level=3,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="very close construction blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_very_close_open_side_escape"
    assert action.steer_bias == pytest.approx(0.70)
    assert action.throttle_floor == pytest.approx(0.42)


def test_very_close_construction_open_side_stays_forward_after_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.5,
        risk_level=3,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="very close construction blockage",
    )
    action = None
    for _ in range(12):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_very_close_open_side_escape"
    assert action.reverse is False
    assert action.steer_bias == pytest.approx(0.70)
    assert action.throttle_floor == pytest.approx(0.42)
def test_very_close_construction_escape_handles_ultra_close_open_side(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=0.8,
        risk_level=3,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="too close construction blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_very_close_open_side_escape"
    assert action.steer_bias == pytest.approx(0.55)
    assert action.throttle_floor == pytest.approx(0.55)
    assert action.brake_cap == pytest.approx(0.0)


def test_ultra_close_construction_open_side_reverses_without_distant_mode(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=0.7,
        risk_level=3,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="ultra close construction blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias == pytest.approx(-0.35)


def test_full_blockage_construction_uses_open_side_escape(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=5.9,
        risk_level=1,
        ego_speed=0.55,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.0,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="center blockage with open side",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.58)
    assert action.steer_min_magnitude == pytest.approx(0.58)
    assert planner.last_open_side == "right"


def test_close_full_blockage_construction_keeps_open_side_escape(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=3.4,
        risk_level=1,
        ego_speed=0.25,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-0.9,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="close center blockage with open side",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.32)
    assert action.steer_limit == pytest.approx(0.55)
    assert action.target_speed == pytest.approx(1.8)
    assert action.throttle_cap == pytest.approx(0.58)
    assert action.throttle_floor == pytest.approx(0.36)


def test_full_blockage_open_side_escape_memory_tolerates_distance_flicker(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=3.8,
        risk_level=1,
        ego_speed=0.35,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="flickering center blockage distance",
    )
    action = planner.plan(features, estimate)
    assert action.reason == "construction_full_blockage_open_side_escape"

    features.front_obstacle_distance = 17.5
    features.ego_speed = 0.32
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.70)
    assert action.target_speed == pytest.approx(2.4)
    assert action.throttle_cap == pytest.approx(0.72)
    assert action.throttle_floor == pytest.approx(0.50)


def test_full_blockage_open_side_memory_overrides_vehicle_misclassification(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.open_side_pass_memory_frames = 12
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_vehicle_distance=2.4,
        front_obstacle_distance=2.4,
        risk_level=2,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="construction blockage misclassified as vehicle",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.25)
    assert action.throttle_floor == pytest.approx(0.36)
    assert action.throttle_cap == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.0)


def test_static_construction_creep_uses_open_side_escape_when_side_is_clear(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.75,
        risk_level=2,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="static construction blockage with open side",
    )
    action = None
    for _ in range(8):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.25)
    assert action.throttle_floor == pytest.approx(0.36)
    assert action.throttle_cap == pytest.approx(0.58)


def test_reverse_vehicle_observed_only_uses_reverse_open_side_probe(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_vehicle_distance=4.0,
        front_obstacle_distance=1.9,
        reversing_vehicle_evidence=False,
        risk_level=2,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="reverse_vehicle",
        confidence=0.8,
        phase="YIELD_OR_BRAKE",
        reason="close vehicle conflict at low ego speed",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "reverse_vehicle_open_side_cautious_probe"
    assert action.reason != "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.22)
    assert action.brake_cap == pytest.approx(0.0)


def test_distant_near_memory_escape_uses_stronger_steer(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.95,
        risk_level=1,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="near memory escape",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias == pytest.approx(-0.35)
    assert action.throttle_floor == pytest.approx(0.30)


def test_distant_near_memory_escape_reverses_after_persistent_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.02,
        risk_level=1,
        ego_speed=0.005,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="near memory escape",
    )
    action = None
    for _ in range(45):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias == pytest.approx(-0.35)
    assert action.throttle_floor == pytest.approx(0.30)


def test_distant_near_full_blockage_reverses_before_forward_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.82,
        risk_level=1,
        ego_speed=0.58,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="near full blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_distant_near_full_blockage_keeps_reverse_until_clear(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 8
    planner.reverse_unwedge_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=4.00,
        risk_level=1,
        ego_speed=1.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="near full blockage reverse continuation",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_distant_very_close_open_side_escape_reverses_after_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.75,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="very close static blockage",
    )
    action = None
    for _ in range(10):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias == pytest.approx(-0.35)


def test_ultra_close_open_side_accepts_lower_center_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=0.9,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.62,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="ultra close static blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_very_close_open_side_escape"
    assert action.brake_cap == pytest.approx(0.0)


def test_ultra_close_open_side_immediately_reverses_inside_collision_margin(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=0.4,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.62,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="ultra close static blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_ultra_close_static_fallback_reverses_instead_of_releasing(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.open_side_pass_memory_frames = 6
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=0.8,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="ultra close static fallback",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_near_static_reverse_recovery_requires_open_side_memory(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.75,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="near static fallback without memory",
    )
    action = planner.plan(features, estimate)
    assert action.reason != "construction_open_side_reverse_unwedge"
    assert action.reverse is False


def test_near_static_reverse_recovery_continues_open_side_memory(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.open_side_pass_memory_frames = 6
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=1.75,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="near static fallback with memory",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_near_static_reverse_recovery_covers_three_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.open_side_pass_memory_frames = 6
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.82,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="three meter static fallback with memory",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_near_static_reverse_recovery_rearms_at_four_meters(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.open_side_pass_memory_frames = 6
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=4.05,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="AVOID_OR_PASS",
        reason="four meter static fallback with memory",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.reverse is True


def test_reverse_action_sets_vehicle_control_reverse(monkeypatch):
    mod = load_module(monkeypatch)
    supervisor = mod.SafetySupervisor(mod.AuxiliaryConfig())
    raw = Control(steer=0.0, throttle=0.0, brake=0.0)
    features = mod.AuxFeatures(ego_speed=0.0)
    estimate = mod.ScenarioEstimate()
    action = mod.PlannerAction(
        True,
        "AVOID_OR_PASS",
        throttle_cap=0.4,
        throttle_floor=0.3,
        brake_cap=0.0,
        steer_bias=-0.35,
        steer_limit=0.55,
        reverse=True,
        reason="construction_open_side_reverse_unwedge",
    )
    ctrl = supervisor.apply(raw, features, estimate, action)
    assert ctrl.reverse is True
    assert ctrl.throttle == pytest.approx(0.3)
    assert ctrl.brake == pytest.approx(0.0)


def test_low_conf_center_blockage_gets_straight_creep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=10.2,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="low confidence corridor blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "low_conf_center_blockage_straight_creep"
    assert action.steer_bias is None
    assert action.steer_limit == pytest.approx(0.25)


def test_low_conf_center_blockage_progress_push_after_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=6.0,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="low confidence corridor blockage",
    )
    action = None
    for _ in range(25):
        action = planner.plan(features, estimate)
    assert action is not None
    assert action.active is True
    assert action.reason == "low_conf_center_blockage_progress_push"
    assert action.throttle_floor == pytest.approx(0.65)
    assert action.steer_limit == pytest.approx(0.20)


def test_active_red_stop_decelerates_before_line(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=2,
        ego_speed=3.2,
        red_light_active=True,
        red_stop_distance=3.8,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.8, phase="APPROACH", reason="red stop")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "active_red_stop_deceleration"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake >= 0.65


def test_center_full_blockage_does_not_open_side_nudge_after_confidence_rises(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=5.0,
        risk_level=1,
        ego_speed=0.55,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "construction_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.58)


def test_center_full_close_obstacle_does_not_construction_slowdown(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.8,
        risk_level=3,
        ego_speed=3.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.reason != "construction_close_obstacle_open_side_slowdown"

def test_red_center_blockage_releases_after_hold(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.4,
        red_light_active=True,
        red_stop_distance=2.9,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    action = None
    for _ in range(19):
        action = planner.plan(features, estimate)
        assert action.reason == "static_obstacle_observed_without_immediate_conflict"
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "red_center_blockage_straight_creep"
    assert action.target_speed == pytest.approx(0.9)
    assert action.steer_bias == pytest.approx(0.0)


def test_red_center_blockage_escalates_after_prolonged_hold(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.55,
        red_light_active=True,
        red_stop_distance=1.8,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    action = None
    for _ in range(60):
        action = planner.plan(features, estimate)
    features.red_light_active = False
    features.red_stop_distance = None
    for _ in range(30):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "red_center_blockage_straight_creep"
    assert action.target_speed == pytest.approx(4.0)
    assert action.throttle_cap == pytest.approx(1.0)
    assert action.throttle_floor == pytest.approx(0.75)
    assert action.brake == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(0.52)
    assert action.steer_min_magnitude == pytest.approx(0.60)


def test_red_center_balanced_uses_lateral_centroid_after_prolonged_hold(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.55,
        red_light_active=True,
        red_stop_distance=1.8,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-0.28,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    action = None
    for _ in range(60):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "red_center_blockage_straight_creep"
    assert action.steer_bias == pytest.approx(0.52)
    assert action.steer_min_magnitude == pytest.approx(0.60)


def test_red_center_blockage_tolerates_brief_red_dropout(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.94,
        red_light_active=True,
        red_stop_distance=2.9,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    for _ in range(10):
        planner.plan(features, estimate)
    features.red_light_active = False
    features.red_stop_distance = None
    for _ in range(5):
        action = planner.plan(features, estimate)
    assert action.reason == "static_obstacle_observed_without_immediate_conflict"
    features.red_light_active = True
    features.red_stop_distance = 3.0
    for _ in range(10):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "red_center_blockage_straight_creep"


def test_close_static_obstacle_defensive_brake_prevents_fast_collision(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=2.4,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=5.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.4,
        lidar_blockage_ratio=0.30,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="close static obstacle at speed",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_obstacle_defensive_brake"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.72)


def test_close_static_balanced_obstacle_creeps_instead_of_indefinite_brake(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=2.7,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.7,
        lidar_blockage_ratio=0.30,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="close static obstacle stalled",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_balanced_creep"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.22)


def test_close_static_balanced_high_blocked_gets_reverse_sweep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 125
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=2.9,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.9,
        lidar_blockage_ratio=0.30,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="close static obstacle high blocked",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_balanced_reverse_sweep"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.36)
    assert action.steer_min_magnitude == pytest.approx(0.30)


def test_ultra_close_cone_uses_reverse_clearance_before_open_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=1.8,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.8,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="ultra close cone",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_cone_reverse_clearance"
    assert action.reverse is True
    assert action.steer_bias < 0.0
    assert action.throttle_cap == pytest.approx(0.28)


def test_close_cone_tight_creep_keeps_gentle_open_side_steer(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 12
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=1.85,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.85,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="close cone",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_cone_tight_creep"
    assert action.reverse is False
    assert action.steer_bias == pytest.approx(0.10)
    assert action.steer_min_magnitude is None
    assert action.throttle_cap == pytest.approx(0.24)


def test_high_blocked_close_static_uses_stronger_open_side_creep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 120
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=1.65,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.65,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.05,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="close static obstacle repeatedly blocked",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_high_blocked_open_side_creep"
    assert action.reverse is False
    assert action.steer_bias == pytest.approx(0.22)
    assert action.steer_min_magnitude == pytest.approx(0.18)
    assert action.throttle_cap == pytest.approx(0.34)


def test_unknown_close_static_high_blocked_gets_reverse_sweep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 120
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=1.72,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.72,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.05,
        lidar_open_side="unknown",
        lidar_lateral_centroid=-0.12,
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="unknown close static",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_unknown_reverse_sweep"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.38)


def test_roundabout_ultra_close_static_uses_forward_unwedge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=0.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.5,
        lidar_blockage_ratio=0.8,
        lidar_center_blockage_ratio=0.2,
        lidar_open_side="unknown",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="roundabout",
        confidence=0.75,
        phase="PREPARE",
        reason="ultra close static obstacle",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_ultra_close_forward_unwedge"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.throttle_cap == pytest.approx(0.72)
    assert action.steer_limit == pytest.approx(0.12)


def test_roundabout_ultra_close_static_low_speed_skips_construction_brake(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=0.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=1.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.6,
        lidar_blockage_ratio=0.8,
        lidar_center_blockage_ratio=0.2,
        lidar_open_side="unknown",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="roundabout",
        confidence=0.75,
        phase="PREPARE",
        reason="ultra close static obstacle at low speed in roundabout",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_ultra_close_forward_unwedge"
    assert action.reverse is False
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.throttle_cap == pytest.approx(0.72)


def test_roundabout_far_side_blockage_overrides_raw_brake(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 40
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=7.36,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=7.36,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "roundabout_far_side_blockage_forward_push"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.throttle_cap == pytest.approx(0.82)
    assert action.steer_bias < 0.0
    assert action.steer_min_magnitude == pytest.approx(0.16)


def test_construction_far_side_blockage_pushes_before_immediate_hazard(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.red_final_context_frames = 20
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=6.52,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=6.52,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_far_side_blockage_forward_push"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.throttle_cap == pytest.approx(0.82)
    assert action.steer_bias > 0.0


def test_ultra_close_static_high_blocked_uses_escape_reverse_sweep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 120
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=0.95,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.95,
        lidar_blockage_ratio=0.8,
        lidar_center_blockage_ratio=0.2,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="ultra close static obstacle",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "ultra_close_static_escape_reverse_sweep"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.40)
    assert action.steer_min_magnitude == pytest.approx(0.30)
    assert action.steer_bias < 0.0


def test_construction_ultra_close_static_long_hold_uses_forward_clearance(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 239
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=0.82,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.82,
        lidar_blockage_ratio=0.55,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="RECOVER",
        reason="route51 ultra close static long hold",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_ultra_close_static_long_hold_forward_clearance"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.62)
    assert action.steer_bias > 0.0
    assert action.steer_min_magnitude == pytest.approx(0.22)


def test_construction_ultra_close_static_long_hold_alternates_reverse_clearance(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 264
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=0.81,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=-0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.81,
        lidar_blockage_ratio=0.60,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="RECOVER",
        reason="route51 ultra close static long hold",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_ultra_close_static_long_hold_reverse_clearance"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.60)
    assert action.steer_bias > 0.0
    assert action.steer_min_magnitude == pytest.approx(0.42)


def test_high_blocked_one_meter_static_obstacle_stays_in_escape(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.blocked_frames = 60
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=1.35,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.02,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.35,
        lidar_blockage_ratio=0.8,
        lidar_center_blockage_ratio=0.2,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="blocked near static obstacle",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "ultra_close_static_escape_reverse_sweep"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.40)
    assert action.steer_min_magnitude == pytest.approx(0.30)
    assert abs(action.steer_bias) >= 0.24


def test_construction_mid_static_obstacle_speed_cap_prevents_raw_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=8.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=5.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=8.6,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="mid static construction obstacle",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_mid_static_collision_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.58)
    assert action.steer_limit == pytest.approx(0.16)


def test_construction_center_blockage_corridor_caps_low_speed_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.red_final_clamp_gap_frames = 96
    features = mod.AuxFeatures(
        confidence=0.60,
        front_clear=False,
        front_obstacle_distance=16.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=1.5,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=16.5,
        lidar_blockage_ratio=0.64,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.60,
        phase="PREPARE",
        reason="center blocked construction corridor",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_center_blockage_corridor_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.22)
    assert action.steer_limit == pytest.approx(0.12)


def test_construction_center_blockage_corridor_brakes_six_meter_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.red_final_clamp_gap_frames = 141
    features = mod.AuxFeatures(
        confidence=0.75,
        front_clear=False,
        front_obstacle_distance=6.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=2.9,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=6.3,
        lidar_blockage_ratio=0.43,
        lidar_center_blockage_ratio=0.97,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.75,
        phase="PREPARE",
        reason="near center blocked construction corridor",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_center_blockage_corridor_speed_cap"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.42)
    assert action.steer_limit == pytest.approx(0.12)


def test_static_obstacle_approach_brake_before_close_collision(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=5.0,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=3.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.0,
        lidar_blockage_ratio=0.30,
        lidar_center_blockage_ratio=0.40,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="static obstacle approach too fast",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_obstacle_approach_brake"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.36)


def test_red_center_blockage_requires_center_full(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=2.4,
        red_light_active=True,
        red_stop_distance=1.6,
        risk_level=2,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.4,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    for _ in range(70):
        action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "static_obstacle_observed_without_immediate_conflict"


def test_red_center_open_side_bypasses_active_red_deceleration(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=3.5,
        red_light_active=True,
        red_stop_distance=1.8,
        risk_level=2,
        ego_speed=0.50,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="PREPARE",
        reason="center blockage near red stop",
    )
    action = planner.plan(features, estimate)
    assert action.reason == "static_obstacle_observed_without_immediate_conflict"
    assert action.reason != "active_red_stop_deceleration"
    assert planner.red_stop_hold_frames == 1


def test_balanced_construction_blockage_enters_straight_creep(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=8.05,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.35,
        lidar_center_blockage_ratio=0.82,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="balanced corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "balanced_construction_blockage_straight_creep"
    assert action.steer_bias == pytest.approx(0.0)
    assert action.brake_cap == pytest.approx(0.0)


def test_balanced_construction_blockage_covers_five_meter_edge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=5.6,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.35,
        lidar_center_blockage_ratio=0.82,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="balanced corridor blockage from detector/lidar",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "balanced_construction_blockage_straight_creep"


def test_static_obstacle_open_side_creep_replaces_inactive_baseline_handoff(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=4.1,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.1,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.1,
        lidar_blockage_ratio=0.40,
        lidar_center_blockage_ratio=0.50,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="static obstacle with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_obstacle_open_side_creep"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.30)


def test_static_obstacle_open_side_creep_tolerates_sparse_detector(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.55,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=4.2,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.1,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.2,
        lidar_blockage_ratio=0.20,
        lidar_center_blockage_ratio=0.25,
        lidar_open_side="right",
        detection_object_count=10,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.55,
        phase="PREPARE",
        reason="sparse detector static obstacle with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_obstacle_open_side_creep"
    assert action.brake_cap == pytest.approx(0.0)


def test_close_static_obstacle_open_side_unwedge_replaces_braking_baseline(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=3.0,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=3.0,
        lidar_blockage_ratio=0.40,
        lidar_center_blockage_ratio=0.50,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="close static obstacle with open side",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_close_static_obstacle_open_side_unwedge"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor == pytest.approx(0.26)
    assert action.throttle_cap == pytest.approx(0.46)
    assert action.steer_bias == pytest.approx(0.28)
    assert action.steer_limit == pytest.approx(0.45)

    for _ in range(5):
        action = planner.plan(features, estimate)
    assert action.reason == "construction_close_static_obstacle_open_side_unwedge"
    assert action.throttle_floor == pytest.approx(0.34)


def test_static_obstacle_balanced_creep_replaces_inactive_baseline_handoff(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=4.1,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.55,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.1,
        lidar_blockage_ratio=0.40,
        lidar_center_blockage_ratio=0.50,
        lidar_open_side="balanced",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="balanced static obstacle",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_static_obstacle_balanced_creep"
    assert action.steer_bias == pytest.approx(0.0)
    assert action.brake_cap == pytest.approx(0.0)


def test_far_static_obstacle_no_progress_recovery_replaces_braking_baseline(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=15.4,
        red_light_active=False,
        red_stop_distance=None,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=15.4,
        lidar_blockage_ratio=0.08,
        lidar_center_blockage_ratio=0.12,
        lidar_open_side="right",
        detection_object_count=40,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="PREPARE",
        reason="far static obstacle without direct conflict",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "static_obstacle_far_no_progress_recovery"
    assert action.throttle_floor == pytest.approx(0.42)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(0.12)

    for _ in range(7):
        action = planner.plan(features, estimate)
    assert action.reason == "static_obstacle_far_no_progress_recovery"
    assert action.throttle_floor == pytest.approx(0.58)


def test_construction_static_balanced_creep_escalates_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        risk_level=2,
        ego_speed=0.02,
        front_obstacle_distance=4.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.6,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.45,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-0.35,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="high_speed_temporary_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="route12 static balanced stall",
    )

    action = None
    for _ in range(35):
        action = planner.plan(features, estimate)
    assert action.reason == "construction_static_balanced_progress_push"
    assert action.throttle_floor == pytest.approx(0.56)

    for _ in range(50):
        action = planner.plan(features, estimate)
    assert action.reason == "construction_static_balanced_reverse_sweep"
    assert action.reverse is True
    assert action.steer_bias < 0.0


def test_balanced_construction_blockage_respects_red_stop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.65,
        front_clear=False,
        front_obstacle_distance=8.05,
        red_light_active=True,
        red_stop_distance=12.0,
        risk_level=1,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.35,
        lidar_center_blockage_ratio=0.82,
        lidar_open_side="balanced",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.65,
        phase="PREPARE",
        reason="balanced corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "static_obstacle_observed_without_immediate_conflict"


def test_red_stop_static_obstacle_blocks_open_side_nudge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=7.3,
        red_light_active=True,
        red_stop_distance=12.0,
        risk_level=1,
        ego_speed=0.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.6,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = None
    for _ in range(10):
        action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "static_obstacle_observed_without_immediate_conflict"
    assert planner.open_side_pass_memory_frames == 0
    assert planner.progress_recovery_frames == 0

def test_construction_near_vehicle_with_safe_ttc_is_observed(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        immediate_hazard=True,
        risk_level=3,
        front_clear=False,
        front_vehicle_distance=3.05,
        front_vehicle_ttc=11.7,
        front_vehicle_closing_speed=0.26,
        front_obstacle_distance=3.69,
        red_stop_distance=3.05,
        ego_speed=3.06,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=101,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=1.0,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "PREPARE"
    assert action.reason == "construction_vehicle_observed_without_confirmed_collision"

def test_construction_static_creep_release_requires_static_only(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=3.7,
        front_vehicle_distance=3.5,
        risk_level=3,
        ego_speed=0.01,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.9,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.reason != "construction_static_creep_release"

def test_construction_static_creep_release_caps_brake(monkeypatch):
    mod = load_module(monkeypatch)
    supervisor = mod.SafetySupervisor(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(ego_speed=0.0)
    estimate = mod.ScenarioEstimate(macro_scenario="trucks_encountered_during_construction", confidence=0.9)
    action = mod.PlannerAction(
        active=True,
        state="AVOID_OR_PASS",
        throttle_cap=0.24,
        throttle_floor=0.16,
        brake_cap=0.0,
        steer_limit=0.65,
        reason="construction_static_creep_release",
    )
    ctrl = supervisor.apply(Control(steer=0.8, throttle=0.0, brake=1.0), features, estimate, action)
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.16)
    assert ctrl.steer == pytest.approx(0.65)

def test_distant_center_lidar_blockage_enters_creep_recovery(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 1,
        "timestamp": 1.0,
        "objects": [],
        "map_objects": [],
    }
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": [
            [13.8, -0.2, 0.0, 0.0],
            [13.9, 0.1, 0.0, 0.0],
        ] * 45,
    }
    ctrl = None
    for i in range(4):
        raw = Control(steer=0.1, throttle=0.0, brake=1.0)
        ctrl = system.process(raw, detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["front_obstacle_distance"] == pytest.approx(13.8)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_blockage_creep_release"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.45)

def test_distant_lidar_creep_default_observe_only(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": [[13.8, -0.2, 0.0, 0.0], [13.9, 0.1, 0.0, 0.0]] * 45,
    }
    raw = Control(steer=0.1, throttle=0.0, brake=1.0)
    for i in range(8):
        ctrl = system.process(raw, detection, tick_data, 1.0 + i * 0.1)
    assert ctrl is raw
    assert system.last_debug["front_obstacle_distance"] == pytest.approx(13.8)
    assert system.last_debug["action_active"] is False


def test_distant_lidar_creep_continues_after_initial_movement(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=13.5,
        risk_level=1,
        ego_speed=0.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.8,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("unknown", 0.6, "NORMAL", "distant blockage")
    planner.static_creep_frames = 4
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "distant_lidar_blockage_creep_release"
    assert action.throttle_floor == pytest.approx(0.45)
    assert action.brake_cap == pytest.approx(0.0)


def test_observable_full_blockage_open_side_escape_covers_near_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=6.95,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("unknown", 0.6, "NORMAL", "observable static blockage")
    planner.static_creep_frames = 3
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "observable_full_blockage_open_side_escape"
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(0.28)


def test_observable_full_blockage_open_side_escape_continues_after_motion(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.open_side_pass_memory_frames = 6
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=5.5,
        risk_level=1,
        ego_speed=0.85,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("unknown", 0.6, "NORMAL", "observable static blockage")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "observable_full_blockage_open_side_escape"
    assert action.throttle_floor == pytest.approx(0.58)


def test_observable_full_blockage_open_side_escape_covers_three_meter_band(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.static_creep_frames = 2
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=3.9,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("unknown", 0.6, "NORMAL", "observable static blockage")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "observable_full_blockage_open_side_escape"
    assert action.brake_cap == pytest.approx(0.0)


def test_observable_full_blockage_open_side_escape_overrides_stalled_low_conf_suppression(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.static_creep_frames = 2
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=7.62,
        risk_level=1,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.6,
        "PREPARE",
        "low confidence full blockage",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "observable_full_blockage_open_side_escape"
    assert action.steer_bias == pytest.approx(0.28)


def test_distant_lidar_open_side_nudge_experiment(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[13.8, -1.2, 0.0, 0.0]] * 45
            + [[13.9, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.16)
    assert ctrl.steer == pytest.approx(0.14)


def test_distant_lidar_open_side_nudge_continues_near_six_meters(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[6.3, -1.2, 0.0, 0.0]] * 45
            + [[6.35, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.steer == pytest.approx(0.42)


def test_distant_lidar_open_side_nudge_uses_stronger_close_pass(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[5.3, -1.2, 0.0, 0.0]] * 45
            + [[5.35, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.steer == pytest.approx(0.42)


def test_distant_lidar_open_side_nudge_continues_inside_three_meters(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[2.5, -1.2, 0.0, 0.0]] * 45
            + [[2.55, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=-0.2, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.48)
    assert ctrl.steer == pytest.approx(0.55)


def test_distant_lidar_open_side_nudge_boosts_two_point_five_meter_stall(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ESCAPE_FRAMES", "24")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[2.5, -1.2, 0.0, 0.0]] * 45
            + [[2.52, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(8):
        ctrl = system.process(Control(steer=-0.2, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.48)
    assert ctrl.steer == pytest.approx(0.55)


def test_distant_lidar_open_side_nudge_escalates_when_close_and_stalled(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ESCAPE_FRAMES", "8")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[1.9, -1.2, 0.0, 0.0]] * 45
            + [[1.95, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(8):
        ctrl = system.process(Control(steer=-0.2, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_escape"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.30)
    assert ctrl.steer == pytest.approx(0.70)


def test_distant_lidar_open_side_nudge_escape_covers_two_point_four_meters(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ESCAPE_FRAMES", "8")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[2.4, -1.2, 0.0, 0.0]] * 45
            + [[2.45, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(8):
        ctrl = system.process(Control(steer=-0.2, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_escape"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.48)
    assert ctrl.steer == pytest.approx(0.70)


def test_distant_lidar_open_side_nudge_respects_one_meter_lower_bound(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[0.8, -1.2, 0.0, 0.0]] * 45
            + [[0.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(8):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["reason"] != "distant_lidar_open_side_nudge"


def test_distant_lidar_open_side_nudge_accepts_lower_center_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[3.2, -1.2, 0.0, 0.0]] * 56
            + [[3.25, 0.1, 0.0, 0.0]] * 14
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, tick_data, 1.0 + i * 0.1)
    assert system.last_debug["lidar_center_blockage_ratio"] == pytest.approx(0.4)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.steer == pytest.approx(0.42)


def test_distant_lidar_open_side_nudge_hysteresis_keeps_open_side_pass(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    entry_tick = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[4.0, -1.2, 0.0, 0.0]] * 45
            + [[4.05, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, entry_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.steer == pytest.approx(0.42)

    low_center_tick = {
        "speed": 0.2,
        "command_near": 2,
        "lidar_points": (
            [[2.6, -1.2, 0.0, 0.0]] * 70
            + [[2.65, 0.1, 0.0, 0.0]] * 9
        ),
    }
    ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, low_center_tick, 2.0)
    assert system.last_debug["lidar_center_blockage_ratio"] < 0.35
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.steer == pytest.approx(0.55)


def test_open_side_nudge_continuation_overrides_legacy_brake_without_rewaiting(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    entry_tick = {
        "speed": 0.0,
        "command_near": 2,
        "lidar_points": (
            [[4.0, -1.2, 0.0, 0.0]] * 45
            + [[4.05, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, entry_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"

    continuation_tick = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[2.4, -1.2, 0.0, 0.0]] * 45
            + [[2.45, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = system.process(
        Control(steer=-0.2, throttle=0.0, brake=0.65),
        detection,
        continuation_tick,
        2.0,
        legacy_rule_action="front_obstacle_brake",
    )
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["legacy_preserved"] is False
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.48)


def test_open_side_pass_memory_continues_after_observed_jitter(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "PREPARE"
    planner.last_open_side = "right"
    planner.open_side_pass_memory_frames = 5
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=2.7,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        risk_level=2,
        ego_speed=0.02,
        lidar_blockage_ratio=0.50,
        lidar_center_blockage_ratio=0.10,
        lidar_open_side="unknown",
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "distant_lidar_open_side_nudge"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(0.55)
    assert planner.open_side_pass_memory_frames > 0


def test_open_side_nudge_post_pass_recovery_after_obstacle_moves_far(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    entry_tick = {
        "speed": 0.2,
        "command_near": 2,
        "lidar_points": (
            [[4.1, -1.2, 0.0, 0.0]] * 45
            + [[4.15, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, entry_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"

    far_tick = {
        "speed": 0.25,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, far_tick, 2.0)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["fsm_state"] == "RECOVER"
    assert system.last_debug["reason"] == "distant_lidar_open_side_post_pass_recovery"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.28)
    assert ctrl.steer == pytest.approx(0.10)


def test_open_side_post_pass_recovery_does_not_override_front_vehicle(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "AVOID_OR_PASS"
    planner.post_pass_frames = 10
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=15.0,
        front_vehicle_distance=8.0,
        lidar_center_blockage_ratio=0.1,
        ego_speed=0.2,
        risk_level=1,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.9,
        phase="PREPARE",
        reason="corridor blockage from detector/lidar",
    )
    action = planner.plan(features, estimate)
    assert action.reason != "distant_lidar_open_side_post_pass_recovery"


def test_open_side_post_pass_recovery_overrides_legacy_clear_crawl(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    entry_tick = {
        "speed": 0.2,
        "command_near": 2,
        "lidar_points": (
            [[4.1, -1.2, 0.0, 0.0]] * 45
            + [[4.15, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, entry_tick, 1.0 + i * 0.1)
    far_tick = {
        "speed": 0.25,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    ctrl = system.process(
        Control(steer=0.0, throttle=0.0, brake=1.0),
        detection,
        far_tick,
        2.0,
        legacy_rule_action="clear_crawl_release",
    )
    assert system.last_debug["reason"] == "distant_lidar_open_side_post_pass_recovery"
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["legacy_preserved"] is False
    assert system.last_debug["action_active"] is True
    assert ctrl.throttle == pytest.approx(0.28)
    assert ctrl.brake == pytest.approx(0.0)


def test_close_obstacle_memory_keeps_nudge_when_lidar_jumps_far(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    close_tick = {
        "speed": 0.4,
        "command_near": 2,
        "lidar_points": (
            [[3.8, -1.2, 0.0, 0.0]] * 45
            + [[3.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, close_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"

    far_tick = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, far_tick, 2.0)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_close_memory_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.18)
    assert ctrl.steer == pytest.approx(0.55)


def test_close_obstacle_memory_accepts_twenty_meter_jump(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    close_tick = {
        "speed": 0.35,
        "command_near": 2,
        "lidar_points": (
            [[1.2, -1.2, 0.0, 0.0]] * 45
            + [[1.25, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(24):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, close_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] in ("distant_lidar_open_side_nudge", "distant_lidar_open_side_escape")

    jump_tick = {
        "speed": 0.01,
        "command_near": 2,
        "lidar_points": (
            [[20.7, -1.4, 0.0, 0.0]] * 60
            + [[20.8, 0.2, 0.0, 0.0]] * 6
        ),
    }
    ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, jump_tick, 4.0)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_close_memory_nudge"
    assert ctrl.brake == pytest.approx(0.0)


def test_close_obstacle_memory_nudge_overrides_legacy_clear_crawl(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    close_tick = {
        "speed": 0.4,
        "command_near": 2,
        "lidar_points": (
            [[3.8, -1.2, 0.0, 0.0]] * 45
            + [[3.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, close_tick, 1.0 + i * 0.1)
    far_tick = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    ctrl = system.process(
        Control(steer=0.0, throttle=0.0, brake=1.0),
        detection,
        far_tick,
        2.0,
        legacy_rule_action="clear_crawl_release",
    )
    assert system.last_debug["reason"] == "distant_lidar_open_side_close_memory_nudge"
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["legacy_preserved"] is False
    assert ctrl.throttle == pytest.approx(0.18)


def test_progress_recovery_follows_close_memory_after_far_lidar_return(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    close_tick = {
        "speed": 0.35,
        "command_near": 2,
        "lidar_points": (
            [[3.8, -1.2, 0.0, 0.0]] * 45
            + [[3.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, close_tick, 1.0 + i * 0.1)
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"

    far_tick = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, far_tick, 2.0)
    ctrl = system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, far_tick, 2.1)
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_progress_recovery"
    assert system.last_debug["fsm_state"] == "RECOVER"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.45)


def test_progress_recovery_overrides_legacy_clear_stuck(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    close_tick = {
        "speed": 0.35,
        "command_near": 2,
        "lidar_points": (
            [[3.8, -1.2, 0.0, 0.0]] * 45
            + [[3.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    for i in range(6):
        system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, close_tick, 1.0 + i * 0.1)

    far_tick = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[15.5, -1.4, 0.0, 0.0]] * 60
            + [[15.6, 0.2, 0.0, 0.0]] * 6
        ),
    }
    system.process(Control(steer=0.0, throttle=0.0, brake=1.0), detection, far_tick, 2.0)
    ctrl = system.process(
        Control(steer=0.0, throttle=0.0, brake=1.0),
        detection,
        far_tick,
        2.1,
        legacy_rule_action="clear_stuck_recovery",
    )
    assert system.last_debug["reason"] == "distant_lidar_open_side_progress_recovery"
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["legacy_preserved"] is False
    assert system.last_debug["action_active"] is True
    assert ctrl.throttle == pytest.approx(0.45)
    assert ctrl.brake == pytest.approx(0.0)


def test_progress_recovery_continues_after_lidar_front_clears(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.progress_recovery_frames = 10
    features = mod.AuxFeatures(
        confidence=0.2,
        risk_level=0,
        front_clear=True,
        ego_speed=0.2,
        lidar_front_distance=None,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.2, phase="NORMAL", reason="no active scenario")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.state == "RECOVER"
    assert action.reason == "distant_lidar_open_side_progress_recovery"
    assert action.throttle_cap == pytest.approx(0.68)
    assert action.throttle_floor == pytest.approx(0.45)
    assert action.brake_cap == pytest.approx(0.0)


def test_progress_recovery_continues_on_clear_road_before_normal_reset(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.progress_recovery_frames = 10
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=0,
        front_clear=True,
        ego_speed=0.4,
        lidar_front_distance=None,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.9,
        phase="NORMAL",
        reason="no active scenario",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.state == "RECOVER"
    assert action.reason == "distant_lidar_open_side_progress_recovery"
    assert planner.state == "RECOVER"


def test_clear_road_progress_recovery_allows_low_cruise_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.progress_recovery_frames = 10
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=0,
        front_clear=True,
        ego_speed=2.0,
        lidar_front_distance=None,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.9, phase="NORMAL")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "distant_lidar_open_side_progress_recovery"
    assert action.target_speed == pytest.approx(2.6)


def test_low_confidence_progress_recovery_allows_low_cruise_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.progress_recovery_frames = 10
    features = mod.AuxFeatures(
        confidence=0.2,
        risk_level=0,
        front_clear=True,
        ego_speed=2.0,
        lidar_front_distance=None,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.2, phase="NORMAL")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "distant_lidar_open_side_progress_recovery"
    assert planner.state == "RECOVER"


def test_open_side_nudge_can_override_legacy_static_obstacle_brake_when_enabled(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[2.5, -1.2, 0.0, 0.0]] * 45
            + [[2.55, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(
            Control(steer=-0.2, throttle=0.0, brake=0.65),
            detection,
            tick_data,
            1.0 + i * 0.1,
            legacy_rule_action="front_obstacle_brake",
        )
    assert system.last_debug["legacy_preserved"] is False
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle == pytest.approx(0.48)
    assert ctrl.steer == pytest.approx(0.55)


def test_open_side_nudge_can_override_legacy_clear_stuck_when_enabled(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[4.8, -1.2, 0.0, 0.0]] * 45
            + [[4.85, 0.1, 0.0, 0.0]] * 25
        ),
    }
    ctrl = None
    for i in range(6):
        ctrl = system.process(
            Control(steer=0.0, throttle=0.4, brake=0.0),
            detection,
            tick_data,
            1.0 + i * 0.1,
            legacy_rule_action="clear_stuck_recovery",
        )
    assert system.last_debug["legacy_preserved"] is False
    assert system.last_debug["legacy_override_allowed"] is True
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "distant_lidar_open_side_nudge"
    assert ctrl.throttle == pytest.approx(0.36)
    assert ctrl.steer == pytest.approx(0.42)


def test_legacy_static_obstacle_brake_is_preserved_when_open_side_nudge_disabled(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    monkeypatch.delenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {"frame": 1, "timestamp": 1.0, "objects": [], "map_objects": []}
    tick_data = {
        "speed": 0.05,
        "command_near": 2,
        "lidar_points": (
            [[2.5, -1.2, 0.0, 0.0]] * 45
            + [[2.55, 0.1, 0.0, 0.0]] * 25
        ),
    }
    raw = Control(steer=-0.2, throttle=0.0, brake=0.65)
    for i in range(6):
        ctrl = system.process(
            raw,
            detection,
            tick_data,
            1.0 + i * 0.1,
            legacy_rule_action="front_obstacle_brake",
        )
    assert ctrl is raw
    assert system.last_debug["legacy_preserved"] is True
    assert system.last_debug["legacy_override_allowed"] is False
    assert system.last_debug["action_active"] is False


def test_debug_reports_nearest_detector_and_tracked_vehicles(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_DETECTOR_ENABLED", "0")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 1,
        "timestamp": 1.0,
        "objects": [
            {"score": 0.9, "class_name": "car", "source": "openpcdet", "box_lidar": {"x": 14.0, "y": 1.0}},
            {"score": 0.8, "class_name": "car", "source": "openpcdet", "box_lidar": {"x": 6.0, "y": 4.0}},
        ],
        "map_objects": [],
    }
    system.process(Control(), detection, {"speed": 0.0, "command_near": 2}, 1.0)
    debug = system.last_debug
    assert debug["nearest_detector_vehicles"][0]["x"] == pytest.approx(14.0)
    assert debug["nearest_detector_vehicles"][0]["source"] == "openpcdet"
    assert debug["nearest_tracked_vehicles"][0]["observed_frames"] == 1

def test_traffic_light_hint_is_observed_without_takeover(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=5.0,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.state == "APPROACH"
    assert "observable_risk" in action.reason


def test_stable_red_light_hint_does_not_release(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=2.7,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    for _ in range(12):
        action = planner.plan(features, estimate)
        assert action.active is False
        assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"


def test_unstable_red_stop_hint_gets_bounded_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_gap_frames = 14
    planner.red_stop_hold_frames = 20
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=2.7,
        red_light_active=False,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "unstable_red_stop_cautious_creep_recovery"
    assert action.target_speed == pytest.approx(2.0)
    assert action.throttle_cap == pytest.approx(0.55)
    assert action.throttle_floor == pytest.approx(0.32)
    assert planner.red_stop_release_frames == 45

    features.ego_speed = 0.8
    features.red_stop_distance = None
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "unstable_red_stop_cautious_creep_recovery"
    assert action.throttle_floor == pytest.approx(0.32)
    assert planner.red_stop_release_frames == 44


def test_recent_red_stop_blocks_unstable_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_gap_frames = 20
    planner.red_stop_hold_frames = 8
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=2.7,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"


def test_active_red_without_stopline_decelerates_before_low_confidence(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.2,
        red_stop_distance=None,
        red_light_active=True,
        risk_level=1,
        front_clear=True,
        ego_speed=4.4,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.2,
        phase="APPROACH",
        reason="red light without stable stopline",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "active_red_without_stopline_deceleration"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.55)


def test_active_red_without_stopline_blocks_clear_recovery(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.observable_risk_creep_frames = 12
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=None,
        red_light_active=True,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        ego_speed=3.4,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="NORMAL",
        reason="clear road but red light active",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "active_red_without_stopline_deceleration"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.35)


def test_active_red_stop_stall_does_not_release_until_red_clears(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 44
    planner.blocked_frames = 35
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=3.6,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_front_distance=None,
        lidar_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="clear stop-line stall",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"
    assert planner.red_stop_release_frames == 0

    features.red_light_active = False
    features.red_stop_distance = None
    planner.red_stop_gap_frames = 29
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "unstable_red_stop_cautious_creep_recovery"
    assert planner.red_stop_release_frames == 45


def test_red_stop_dropout_clear_stall_starts_release_before_low_confidence_return(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 36
    planner.red_stop_gap_frames = 47
    features = mod.AuxFeatures(
        confidence=0.4,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=0,
        front_clear=True,
        ego_speed=0.0,
        lidar_front_distance=None,
        lidar_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="red stop dropout",
    )
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "unstable_red_stop_cautious_creep_recovery"
    assert planner.red_stop_release_frames == 45


def test_prolonged_active_red_stop_remains_stopped(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 180
    planner.blocked_frames = 90
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=3.8,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"
    assert planner.red_stop_release_frames == 0


def test_active_red_stop_with_lidar_center_noise_remains_stopped(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 180
    planner.blocked_frames = 90
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=2.3,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_front_distance=6.5,
        lidar_center_blockage_ratio=0.35,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "observable_risk_without_confirmed_longitudinal_conflict"


def test_near_stop_line_without_red_gets_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=2.8,
        red_light_active=False,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="near stop line without red",
    )
    action = planner.plan(features, estimate)
    assert action.active is False
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "near_stop_line_cautious_creep_recovery"
    assert action.target_speed == pytest.approx(2.6)
    features.ego_speed = 1.2
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "near_stop_line_cautious_creep_recovery"


def test_recent_red_stop_blocks_clear_road_recovery(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 80
    planner.red_stop_gap_frames = 0
    planner.blocked_frames = 4
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=0,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.8, phase="NORMAL", reason="clear")
    action = planner.plan(features, estimate)
    assert action.active is False
    assert action.reason == "clear"
    assert planner.red_stop_gap_frames == 1


def test_clear_road_recovery_after_stable_red_gap(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.red_stop_hold_frames = 80
    planner.red_stop_gap_frames = 120
    planner.blocked_frames = 4
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=True,
        risk_level=0,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(macro_scenario="unknown", confidence=0.8, phase="NORMAL", reason="clear")
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "clear_road_cautious_creep_recovery"


def test_distant_red_stop_deadlock_gets_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        red_stop_distance=8.0,
        red_light_active=True,
        risk_level=2,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.8,
        phase="APPROACH",
        reason="traffic light or stop sign",
    )
    action = None
    for _ in range(4):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.state == "RECOVER"
    assert action.target_speed == pytest.approx(2.6)
    assert action.throttle_cap == pytest.approx(0.68)
    assert action.throttle_floor == pytest.approx(0.45)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.reason == "observable_risk_cautious_creep_recovery"
    features.red_stop_distance = None
    features.risk_level = 0
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "observable_risk_cautious_creep_recovery"
    low_confidence = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.1,
        phase="NORMAL",
        reason="no active scenario",
    )
    action = planner.plan(features, low_confidence)
    assert action.active is True
    assert action.reason in {"observable_risk_cautious_creep_recovery", "clear_road_cautious_creep_recovery"}


def test_clear_road_deadlock_gets_cautious_creep(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="no active scenario",
    )
    action = None
    for _ in range(2):
        action = planner.plan(features, estimate)
    assert action.active is True
    assert action.state == "RECOVER"
    assert action.reason == "clear_road_cautious_creep_recovery"
    assert action.target_speed == pytest.approx(2.6)
    assert action.throttle_cap == pytest.approx(0.68)
    assert action.throttle_floor == pytest.approx(0.45)

    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features.ego_speed = 0.5
    action = planner.plan(features, estimate)
    assert action.active is False
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "clear_road_cautious_creep_recovery"

    features.ego_speed = 0.8
    action = planner.plan(features, estimate)
    assert action.active is True
    assert action.reason == "clear_road_cautious_creep_recovery"


def test_clear_road_no_progress_gets_stronger_recovery(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 20
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="clear road but no progress",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "clear_road_no_progress_stronger_recovery"
    assert action.target_speed == pytest.approx(3.2)
    assert action.throttle_cap == pytest.approx(0.85)
    assert action.throttle_floor == pytest.approx(0.62)
    assert action.reverse is False


def test_clear_road_recovery_escalates_when_still_stalled(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_front_distance=16.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="clear road but still stalled",
    )

    action = None
    for _ in range(21):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "clear_road_no_progress_stronger_recovery"
    assert action.throttle_cap == pytest.approx(0.85)
    assert action.throttle_floor == pytest.approx(0.62)


def test_clear_road_no_progress_gets_forward_unwedge_before_reverse(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 45
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="clear road but no progress",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "clear_road_no_progress_forward_unwedge"
    assert action.reverse is False
    assert action.target_speed == pytest.approx(4.5)
    assert action.throttle_cap == pytest.approx(1.0)
    assert action.throttle_floor == pytest.approx(0.86)


def test_cone_post_pass_memory_limits_clear_road_forward_unwedge(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 45
    planner.open_side_pass_memory_frames = 12
    planner.last_open_side = "right"
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="clear road after cone pass",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_cone_post_pass_cautious_recovery"
    assert action.reverse is False
    assert action.target_speed == pytest.approx(2.0)
    assert action.throttle_cap == pytest.approx(0.46)
    assert action.throttle_floor == pytest.approx(0.26)
    assert action.steer_bias > 0.0


def test_clear_road_no_progress_gets_reverse_unwedge(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 90
    features = mod.AuxFeatures(
        confidence=0.4,
        risk_level=0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_center_blockage_ratio=0.0,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.4,
        phase="NORMAL",
        reason="clear road but no progress",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "clear_road_no_progress_reverse_unwedge"
    assert action.reverse is True
    assert action.throttle_cap == pytest.approx(0.34)
    assert action.throttle_floor == pytest.approx(0.24)


def test_clear_road_no_progress_roundabout_keeps_forward_recovery(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 95

    action = planner._clear_road_no_progress_action(
        "clear_road_cautious_creep_recovery",
        allow_reverse=False,
    )

    assert action.active is True
    assert action.reason == "roundabout_clear_road_forward_recovery"
    assert action.reverse is False
    assert action.target_speed == pytest.approx(4.8)
    assert action.throttle_cap == pytest.approx(1.0)
    assert action.throttle_floor == pytest.approx(0.88)
    assert abs(action.steer_bias) == pytest.approx(0.16)
    assert action.steer_min_magnitude == pytest.approx(0.14)


def test_construction_far_full_blockage_open_side_gets_recovery(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=8.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=8.1,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="static obstacle with open side",
    )

    action = None
    for _ in range(3):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.state == "AVOID_OR_PASS"
    assert action.reason == "construction_far_full_blockage_open_side_recovery"
    assert action.target_speed == pytest.approx(2.4)
    assert action.throttle_cap == pytest.approx(0.78)
    assert action.throttle_floor == pytest.approx(0.72)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0
    assert action.steer_min_magnitude == pytest.approx(0.36)


def test_construction_far_full_blockage_covers_ten_meter_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=10.3,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=10.3,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="static obstacle with open side",
    )

    action = None
    for _ in range(3):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_far_full_blockage_open_side_recovery"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_construction_far_full_blockage_covers_twelve_meter_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=12.4,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=12.4,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="far static obstacle with open side",
    )

    action = None
    for _ in range(3):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "construction_far_full_blockage_open_side_recovery"
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_far_full_blockage_speed_guard_limits_steer(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    supervisor = mod.SafetySupervisor(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=9.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=2.7,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=9.5,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="static obstacle with open side",
    )

    action = planner.plan(features, estimate)
    control = supervisor.apply(Control(steer=1.0, throttle=1.0, brake=0.0), features, estimate, action)

    assert action.active is True
    assert action.reason == "construction_far_full_blockage_open_side_speed_guard"
    assert action.brake_cap == pytest.approx(0.0)
    assert control.steer == pytest.approx(0.45)
    assert control.throttle >= 0.12 - 1e-6
    assert control.brake == pytest.approx(0.0)


def test_construction_far_full_blockage_respects_red_stop(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=8.2,
        red_stop_distance=5.0,
        red_light_active=True,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=8.1,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="static obstacle with red stop",
    )

    for _ in range(4):
        action = planner.plan(features, estimate)

    assert action.reason != "construction_far_full_blockage_open_side_recovery"


def test_reverse_unwedge_survives_far_lidar_jump(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.last_open_side = "right"
    planner.reverse_unwedge_frames = 8
    planner.progress_recovery_frames = 20
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=14.1,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=14.1,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="far lidar jump during reverse unwedge",
    )

    action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reverse is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.steer_bias < 0.0
    assert planner.reverse_unwedge_frames == 7


def test_observable_very_close_open_side_creep_without_nudge_flag(monkeypatch):
    monkeypatch.delenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", raising=False)
    monkeypatch.delenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=1.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.6,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="APPROACH",
        reason="observable close blockage",
    )

    action = None
    for _ in range(4):
        action = planner.plan(features, estimate)

    assert action.active is True
    assert action.reason == "observable_very_close_open_side_creep"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_observable_very_close_open_side_creep_reverses_after_stall(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=1.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.6,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="APPROACH",
        reason="observable close blockage",
    )

    action = None
    for _ in range(10):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reverse is True
    assert action.reason == "construction_open_side_reverse_unwedge"


def test_observable_very_close_open_side_creep_reverses_at_two_point_three_meters(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=2.25,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.25,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="APPROACH",
        reason="observable close blockage",
    )

    action = None
    for _ in range(10):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.active is True
    assert action.reverse is True
    assert action.reason == "construction_open_side_reverse_unwedge"
    assert action.steer_bias < 0.0


def test_observable_very_close_open_side_creep_respects_red_stop(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.6,
        front_clear=False,
        front_obstacle_distance=1.6,
        red_stop_distance=3.0,
        red_light_active=True,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.6,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.6,
        phase="APPROACH",
        reason="observable close blockage with red stop",
    )

    for _ in range(4):
        action = planner.plan(features, estimate)

    assert action.reason != "observable_very_close_open_side_creep"


def test_observable_close_open_side_speed_guard_limits_steer(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    supervisor = mod.SafetySupervisor(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.7,
        front_clear=False,
        front_obstacle_distance=4.0,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=1.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.0,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="unknown",
        confidence=0.7,
        phase="APPROACH",
        reason="observable close open side",
    )

    action = planner.plan(features, estimate)
    control = supervisor.apply(Control(steer=1.0, throttle=1.0, brake=0.0), features, estimate, action)

    assert action.active is True
    assert action.reason == "observable_close_open_side_speed_guard"
    assert action.brake_cap is None
    assert control.steer == pytest.approx(0.42)
    assert control.throttle == pytest.approx(0.0)
    assert control.brake > 0.0


def test_reverse_unwedge_far_lidar_jump_respects_red_stop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "RECOVER"
    planner.last_open_side = "right"
    planner.reverse_unwedge_frames = 8
    planner.progress_recovery_frames = 20
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_obstacle_distance=14.1,
        red_stop_distance=4.0,
        red_light_active=True,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=14.1,
        lidar_blockage_ratio=0.95,
        lidar_center_blockage_ratio=0.95,
        lidar_left_blockage_ratio=0.95,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="trucks_encountered_during_construction",
        confidence=0.8,
        phase="APPROACH",
        reason="far lidar jump with red stop",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "construction_open_side_reverse_unwedge"


def test_debug_reports_intervention_counters(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    detection = {
        "frame": 3,
        "timestamp": 0.3,
        "objects": [
            {"score": 0.95, "class_name": "pedestrian", "box_lidar": {"x": 3.0, "y": 0.0}},
        ],
        "map_objects": [],
    }
    ctrl = system.process(Control(throttle=0.6), detection, {"speed": 4.0, "command_near": 4}, 0.3)
    assert ctrl.brake >= 0.85
    assert system.last_debug["action_active"] is True
    assert system.last_debug["intervention_count"] == 1
    assert system.last_debug["emergency_count"] == 1
    assert system.last_debug["ego_speed"] == pytest.approx(4.0)


def test_process_clamps_active_red_without_stopline_even_when_rule_inactive(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    class FakePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True},
                "lidar_geometry": None,
            }

    system.perception = FakePerception()
    raw = Control(throttle=0.9, brake=0.0, steer=0.1)
    ctrl = system.process(
        raw,
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 4.2},
        0.1,
    )
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.55
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"

    raw = Control(throttle=0.7, brake=0.0, steer=0.1)
    ctrl = system.process(
        raw,
        {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []},
        {"speed": 0.2},
        0.2,
    )
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.20


def test_active_red_final_clamp_overrides_far_front_vehicle(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class RedWithFarVehiclePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [
                    {
                        "class_name": "car",
                        "x": 20.0,
                        "y": 0.0,
                        "distance": 20.0,
                    }
                ],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 12.0},
                "lidar_geometry": None,
            }

    system.perception = RedWithFarVehiclePerception()
    ctrl = system.process(
        Control(throttle=1.0, brake=0.0, steer=0.8),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 2.0},
        0.1,
    )

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.35
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"


def test_active_red_final_clamp_yields_to_near_two_wheeler_track(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class RedWithTwoWheelerPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [
                    {"class_name": "bicycle", "x": 10.0, "y": 0.7, "observed_frames": 3}
                ],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.3},
                "lidar_geometry": None,
            }

    system.perception = RedWithTwoWheelerPerception()
    ctrl = system.process(
        Control(throttle=0.8, brake=0.0, steer=0.02),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.2},
        0.1,
    )

    assert ctrl.throttle == pytest.approx(0.8)
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] != "active_red_without_stopline_final_clamp"


def test_active_red_far_side_blockage_creeps_in_roundabout_context(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 6

    class FarSideBlockageRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [
                    {"class_name": "others", "score": 0.9, "box_lidar": {"x": 7.4, "y": 0.0}},
                ],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 13.0},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 7.4,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_density": 30,
                    "right_density": 0,
                    "center_density": 0,
                    "open_side": "right",
                    "lateral_centroid": -1.5,
                },
            }

    system.perception = FarSideBlockageRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=1.0, steer=-0.2),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer > 0.0
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"




def test_construction_suppressed_long_gap_deadlock_gets_forward_pulse(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_gap_frames = 265

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 20.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.7, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.02},
        0.1,
    )

    assert ctrl.throttle >= 0.96
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) >= 0.22
    assert getattr(ctrl, "reverse", False) is False

def test_students_close_red_flicker_releases_after_short_safe_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 20

    class StudentsCloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.6},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = StudentsCloseRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.8, steer=-0.05),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.1},
        0.1,
    )

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.45
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"

def test_construction_very_close_red_deadlock_releases_when_clear(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 55

    class ConstructionVeryCloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.2},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ConstructionVeryCloseRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.8, steer=-0.05),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.46 <= ctrl.throttle <= 0.62
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.12
    assert getattr(ctrl, "reverse", False) is False
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

def test_process_clamps_active_red_near_stopline_even_when_rule_inactive(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class FakePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.0},
                "lidar_geometry": None,
            }

    system.perception = FakePerception()
    raw = Control(throttle=1.0, brake=0.0, steer=0.02)
    ctrl = system.process(
        raw,
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.1},
        0.1,
    )
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.20
    assert system.last_debug["action_active"] is True
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"

    class MidDistanceRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 12.5},
                "lidar_geometry": None,
            }

    system.perception = MidDistanceRedPerception()
    ctrl = system.process(
        Control(throttle=1.0, brake=0.0, steer=0.02),
        {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []},
        {"speed": 0.3},
        0.2,
    )
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.20
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"


def test_process_creeps_after_prolonged_far_red_false_hold(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class FarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.3},
                "lidar_geometry": None,
            }

    system.perception = FarRedPerception()
    ctrl = None
    for i in range(80):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle >= 0.16
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"
    assert system.last_debug["red_final_clamp_hold_frames"] >= 80

    for i in range(80, 180):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl.throttle >= 0.55
    assert ctrl.throttle <= 0.75
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["red_final_clamp_hold_frames"] >= 180

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 181, "timestamp": 18.1, "objects": [], "map_objects": []},
        {"speed": 0.15},
        18.1,
    )
    assert ctrl.throttle >= 0.55
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 304, "timestamp": 30.4, "objects": [], "map_objects": []},
        {"speed": 1.6},
        30.4,
    )
    assert ctrl.throttle >= 0.55
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    previous_hold = system.last_debug["red_final_clamp_hold_frames"]
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 182, "timestamp": 18.2, "objects": [], "map_objects": []},
        {"speed": 0.29},
        18.2,
    )
    assert ctrl.throttle >= 0.55
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"
    assert system.last_debug["red_final_clamp_hold_frames"] > previous_hold

    for i in range(183, 302):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.throttle <= 0.90
    assert ctrl.brake == pytest.approx(0.0)

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 302, "timestamp": 30.2, "objects": [], "map_objects": []},
        {"speed": 0.6},
        30.2,
    )
    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 303, "timestamp": 30.3, "objects": [], "map_objects": []},
        {"speed": 0.95},
        30.3,
    )
    assert ctrl.throttle >= 0.65
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_far_clear_red_false_hold_gets_route_speed_release(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 80

    class FarClearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 11.4},
                "lidar_geometry": {
                    "front_distance": None,
                    "blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_density": 0,
                    "right_density": 0,
                    "center_density": 0,
                    "open_side": "unknown",
                    "lateral_centroid": 0.0,
                },
            }

    system.perception = FarClearRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.02},
        0.1,
    )

    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_creeps_after_prolonged_close_red_false_hold(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.7},
                "lidar_geometry": None,
            }

    system.perception = CloseRedPerception()
    ctrl = None
    for i in range(140):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.50 <= ctrl.throttle <= 0.70
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"
    assert system.last_debug["red_final_clamp_hold_frames"] >= 140


def test_three_meter_red_blocked_releases_early_for_scenario_progress(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 80
    system.rule_planner.blocked_frames = 80

    class ThreeMeterRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.3},
                "lidar_geometry": None,
            }

    system.perception = ThreeMeterRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.35 <= ctrl.throttle <= 0.50
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []},
        {"speed": 0.6},
        0.2,
    )

    assert 0.35 <= ctrl.throttle <= 0.50
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_close_red_release_continues_after_creep_speed_builds(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.4},
                "lidar_geometry": None,
            }

    system.perception = CloseRedPerception()
    ctrl = None
    for i in range(180):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.50 <= ctrl.throttle <= 0.70
    assert ctrl.brake == pytest.approx(0.0)

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 181, "timestamp": 18.1, "objects": [], "map_objects": []},
        {"speed": 0.6},
        18.1,
    )
    assert 0.50 <= ctrl.throttle <= 0.70
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_red_release_memory_bridges_five_meter_stopline_jitter(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    distances = [5.56, 5.31]

    class JitterRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            distance = distances.pop(0) if distances else 5.31
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": distance},
                "lidar_geometry": None,
            }

    system.perception = JitterRedPerception()
    system.red_final_clamp_hold_frames = 84
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )
    assert ctrl.throttle >= 0.16
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.2,
    )
    assert 0.24 <= ctrl.throttle <= 0.36
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_close_red_long_deadlock_gets_stronger_release(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 100
    system.cut_in_clear_recovery_frames = 0
    system.cut_in_post_unwedge_commit_frames = 0
    system.rule_planner.blocked_frames = 100

    class CloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.05},
                "lidar_geometry": None,
            }

    system.perception = CloseRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=1.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.35 <= ctrl.throttle <= 0.50
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_post_red_long_deadlock_release_overrides_raw_brake(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 220
    system.red_final_clamp_gap_frames = 5
    system.rule_planner.blocked_frames = 120

    class ClearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": None,
            }

    system.perception = ClearRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=1.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert ctrl.throttle >= 0.65
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_uses_stronger_close_red_release_after_long_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.8},
                "lidar_geometry": None,
            }

    system.perception = CloseRedPerception()
    ctrl = None
    for i in range(259):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.50 <= ctrl.throttle <= 0.70
    assert ctrl.brake == pytest.approx(0.0)

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 260, "timestamp": 26.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        26.0,
    )
    assert 0.68 <= ctrl.throttle <= 0.90
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 261, "timestamp": 26.1, "objects": [], "map_objects": []},
        {"speed": 0.52},
        26.1,
    )
    assert 0.68 <= ctrl.throttle <= 0.90
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    for i in range(262, 301):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.80 <= ctrl.throttle <= 1.00
    assert ctrl.brake == pytest.approx(0.0)

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 301, "timestamp": 30.1, "objects": [], "map_objects": []},
        {"speed": 1.16},
        30.1,
    )
    assert 0.80 <= ctrl.throttle <= 1.00
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_creeps_after_prolonged_near_red_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class NearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.6},
                "lidar_geometry": None,
            }

    system.perception = NearRedPerception()
    ctrl = None
    for i in range(159):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 160, "timestamp": 16.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        16.0,
    )
    assert 0.12 <= ctrl.throttle <= 0.20
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_near_stopline_red_deadlock_gets_probe_release(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class NearStoplineRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.18},
                "lidar_geometry": {
                    "front_distance": 8.5,
                    "center_blockage_ratio": 0.0,
                    "blockage_ratio": 0.0,
                },
            }

    system.perception = NearStoplineRedPerception()
    ctrl = None
    for i in range(59):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 60, "timestamp": 6.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        6.0,
    )
    assert 0.22 <= ctrl.throttle <= 0.35
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"
    assert system.last_debug["red_final_near_line_hold_frames"] >= 60


def test_near_stopline_red_deadlock_uses_total_hold_when_distance_jitters(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 220
    system.rule_planner.blocked_frames = 40

    class NearStoplineRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.18},
                "lidar_geometry": {
                    "front_distance": 8.5,
                    "center_blockage_ratio": 0.0,
                    "blockage_ratio": 0.0,
                },
            }

    system.perception = NearStoplineRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.22 <= ctrl.throttle <= 0.35
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_creeps_after_three_meter_red_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class ThreeMeterRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.0},
                "lidar_geometry": None,
            }

    system.perception = ThreeMeterRedPerception()
    ctrl = None
    for i in range(99):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 100, "timestamp": 10.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        10.0,
    )
    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_creeps_after_two_point_five_meter_red_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class TwoPointFiveMeterRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.45},
                "lidar_geometry": None,
            }

    system.perception = TwoPointFiveMeterRedPerception()
    ctrl = None
    for i in range(159):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 160, "timestamp": 16.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        16.0,
    )
    assert 0.12 <= ctrl.throttle <= 0.20
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_uses_stronger_two_meter_red_release_after_long_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class TwoMeterRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.05},
                "lidar_geometry": None,
            }

    system.perception = TwoMeterRedPerception()
    ctrl = None
    for i in range(139):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 140, "timestamp": 14.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        14.0,
    )
    assert 0.16 <= ctrl.throttle <= 0.28
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    for i in range(141, 260):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 260, "timestamp": 26.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        26.0,
    )
    assert 0.24 <= ctrl.throttle <= 0.38
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_process_uses_stronger_near_red_release_after_long_deadlock(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class NearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.9},
                "lidar_geometry": None,
            }

    system.perception = NearRedPerception()
    ctrl = None
    for i in range(259):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 260, "timestamp": 26.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        26.0,
    )
    assert 0.55 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 261, "timestamp": 26.1, "objects": [], "map_objects": []},
        {"speed": 0.42},
        26.1,
    )
    assert 0.42 <= ctrl.throttle <= 0.60
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_far_red_release_reverses_when_close_obstacle_blocks_open_side(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 100
    system.red_final_clamp_last_distance = 11.0

    class EmptyPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {}

    class CloseObstacleFeatureBuilder:
        def build(self, observation, tick_data):
            return mod.AuxFeatures(
                confidence=0.8,
                front_clear=False,
                front_obstacle_distance=2.0,
                front_vehicle_distance=None,
                front_pedestrian_distance=None,
                red_stop_distance=11.0,
                red_light_active=True,
                ego_speed=0.0,
                lidar_available=True,
                lidar_stale=False,
                lidar_front_distance=2.0,
                lidar_blockage_ratio=1.0,
                lidar_center_blockage_ratio=1.0,
                lidar_left_blockage_ratio=1.0,
                lidar_right_blockage_ratio=0.0,
                lidar_open_side="right",
                detection_object_count=100,
            )

        def debug_nearest_objects(self, observation):
            return {"nearest_detector_vehicles": [], "nearest_tracked_vehicles": []}

    class UnknownRecognizer:
        def recognize(self, features):
            return mod.ScenarioEstimate("unknown", 0.8, "PREPARE", "far red close obstacle")

    system.perception = EmptyPerception()
    system.feature_builder = CloseObstacleFeatureBuilder()
    system.recognizer = UnknownRecognizer()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 0, "timestamp": 0.0},
        {"speed": 0.0},
        0.0,
    )
    assert ctrl.reverse is True
    assert ctrl.throttle == pytest.approx(0.30)
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer == pytest.approx(-0.35)


def test_far_red_close_obstacle_creeps_before_long_reverse_hold(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 6
    system.red_final_clamp_last_distance = 11.0

    class EmptyPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {}

    class CloseObstacleFeatureBuilder:
        def build(self, observation, tick_data):
            return mod.AuxFeatures(
                confidence=0.8,
                front_clear=False,
                front_obstacle_distance=3.9,
                front_vehicle_distance=None,
                front_pedestrian_distance=None,
                red_stop_distance=11.0,
                red_light_active=True,
                ego_speed=0.1,
                lidar_available=True,
                lidar_stale=False,
                lidar_front_distance=3.9,
                lidar_open_side="right",
                detection_object_count=100,
            )

        def debug_nearest_objects(self, observation):
            return {"nearest_detector_vehicles": [], "nearest_tracked_vehicles": []}

    class UnknownRecognizer:
        def recognize(self, features):
            return mod.ScenarioEstimate("unknown", 0.8, "PREPARE", "far red close obstacle")

    system.perception = EmptyPerception()
    system.feature_builder = CloseObstacleFeatureBuilder()
    system.recognizer = UnknownRecognizer()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 0, "timestamp": 0.0},
        {"speed": 0.1},
        0.0,
    )
    assert getattr(ctrl, "reverse", False) is False
    assert 0.28 <= ctrl.throttle <= 0.42
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer == pytest.approx(0.42)


def test_far_red_close_obstacle_creep_ignores_distant_front_vehicle(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 30
    system.red_final_clamp_last_distance = 11.0

    class EmptyPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {}

    class CloseObstacleFeatureBuilder:
        def build(self, observation, tick_data):
            return mod.AuxFeatures(
                confidence=0.8,
                front_clear=False,
                front_obstacle_distance=3.5,
                front_vehicle_distance=14.0,
                front_pedestrian_distance=None,
                red_stop_distance=11.0,
                red_light_active=True,
                ego_speed=0.0,
                lidar_available=True,
                lidar_stale=False,
                lidar_front_distance=3.5,
                lidar_open_side="right",
                detection_object_count=100,
            )

        def debug_nearest_objects(self, observation):
            return {"nearest_detector_vehicles": [], "nearest_tracked_vehicles": []}

    class UnknownRecognizer:
        def recognize(self, features):
            return mod.ScenarioEstimate("unknown", 0.8, "PREPARE", "far red close obstacle")

    system.perception = EmptyPerception()
    system.feature_builder = CloseObstacleFeatureBuilder()
    system.recognizer = UnknownRecognizer()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 0, "timestamp": 0.0},
        {"speed": 0.0},
        0.0,
    )
    assert getattr(ctrl, "reverse", False) is False
    assert 0.28 <= ctrl.throttle <= 0.42
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer == pytest.approx(0.42)


def test_post_red_clear_stuck_recovery_releases_baseline_brake(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 0
    system.red_final_clamp_gap_frames = 25

    class EmptyPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {}

    class ClearFeatureBuilder:
        def build(self, observation, tick_data):
            return mod.AuxFeatures(
                confidence=0.4,
                front_clear=True,
                front_vehicle_distance=None,
                front_pedestrian_distance=None,
                front_obstacle_distance=None,
                red_stop_distance=None,
                red_light_active=False,
                ego_speed=0.0,
            )

        def debug_nearest_objects(self, observation):
            return {}

    class UnknownRecognizer:
        def recognize(self, features):
            return mod.ScenarioEstimate("unknown", 0.4, "NORMAL", "rules_disabled_or_low_confidence")

    system.perception = EmptyPerception()
    system.feature_builder = ClearFeatureBuilder()
    system.recognizer = UnknownRecognizer()
    ctrl = system.process(
        Control(throttle=0.0, brake=1.0, steer=0.0),
        {"frame": 0, "timestamp": 0.0},
        {"speed": 0.0},
        0.0,
    )

    assert ctrl.throttle >= 0.45
    assert ctrl.brake == pytest.approx(0.0)


def test_process_creeps_after_very_near_red_deadlock_only_after_long_hold(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class VeryNearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 1.5},
                "lidar_geometry": None,
            }

    system.perception = VeryNearRedPerception()
    ctrl = None
    for i in range(299):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake > 0.0

    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 300, "timestamp": 30.0, "objects": [], "map_objects": []},
        {"speed": 0.0},
        30.0,
    )
    assert 0.05 <= ctrl.throttle <= 0.08
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_far_red_hold_survives_short_detection_gaps(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    state = {"active": True}

    class IntermittentRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            traffic = {"active": True, "distance": 12.5} if state["active"] else {"active": False}
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": traffic,
                "lidar_geometry": None,
            }

    system.perception = IntermittentRedPerception()
    for i in range(60):
        system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    state["active"] = False
    for i in range(60, 70):
        system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert system.red_final_clamp_hold_frames >= 60
    state["active"] = True
    ctrl = None
    for i in range(70, 91):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert ctrl.throttle >= 0.16
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_construction_suppressed_post_red_extreme_gap_prefers_sustained_reverse(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_gap_frames = 1199
    system.rule_planner.blocked_frames = 520

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False},
                "lidar_geometry": {
                    "front_distance": None,
                    "blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.1},
        0.1,
    )

    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.reverse is True
    assert 0.72 <= ctrl.throttle <= 0.86
    assert abs(ctrl.steer) >= 0.58
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_construction_suppressed_post_red_mid_gap_reverses_when_blocked(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_gap_frames = 529
    system.rule_planner.blocked_frames = 440

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False},
                "lidar_geometry": {
                    "front_distance": None,
                    "blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.reverse is True
    assert 0.58 <= ctrl.throttle <= 0.76
    assert abs(ctrl.steer) >= 0.44
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_construction_suppressed_post_red_long_gap_alternates_reverse(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_gap_frames = 629
    system.rule_planner.blocked_frames = 0

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False},
                "lidar_geometry": {
                    "front_distance": None,
                    "blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.reverse is True
    assert 0.88 <= ctrl.throttle <= 1.0
    assert abs(ctrl.steer) >= 0.46
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_construction_suppressed_post_red_long_gap_commits_forward_when_moving(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_gap_frames = 629
    system.rule_planner.blocked_frames = 0

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False},
                "lidar_geometry": {
                    "front_distance": None,
                    "blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.0, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.8},
        0.1,
    )

    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.reverse is False
    assert ctrl.throttle >= 0.96
    assert abs(ctrl.steer) >= 0.18
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_near_red_release_continues_through_short_detection_gap(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    state = {"active": True}

    class IntermittentNearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            traffic = {"active": True, "distance": 2.9} if state["active"] else {"active": False}
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": traffic,
                "lidar_geometry": None,
            }

    system.perception = IntermittentNearRedPerception()
    ctrl = None
    for i in range(270):
        ctrl = system.process(
            Control(throttle=0.0, brake=0.0, steer=0.0),
            {"frame": i, "timestamp": i * 0.1, "objects": [], "map_objects": []},
            {"speed": 0.0},
            i * 0.1,
        )
    assert ctrl is not None
    assert 0.42 <= ctrl.throttle <= 0.60
    assert ctrl.brake == pytest.approx(0.0)

    state["active"] = False
    ctrl = system.process(
        Control(throttle=0.0, brake=1.0, steer=0.0),
        {"frame": 271, "timestamp": 27.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        27.1,
    )
    assert ctrl.throttle >= 0.65
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"
    assert system.last_debug["red_final_clamp_gap_frames"] == 1


def test_debug_reports_lidar_corridor_evidence(monkeypatch):
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    pts = []
    for i in range(60):
        pts.append([6.0 + 0.02 * i, -1.2, 0.0, 0.0])
    ctrl = system.process(
        Control(throttle=0.2),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 1.0, "command_near": 2, "lidar_points": pts},
        0.1,
    )
    assert ctrl.throttle == pytest.approx(0.2)
    assert system.last_debug["lidar_left_density"] == 60
    assert system.last_debug["lidar_right_density"] == 0
    assert system.last_debug["lidar_open_side"] == "right"
    assert system.last_debug["lidar_left_blockage_ratio"] > 0.0

def test_recovery_uses_bounded_release_control(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.state = "PREPARE"
    planner.clear_frames = 2
    features = mod.AuxFeatures(confidence=0.8, risk_level=0, front_clear=True)
    estimate = mod.ScenarioEstimate(confidence=0.8, phase="NORMAL", reason="no active scenario")
    action = planner.plan(features, estimate)
    assert action.state == "RECOVER"
    assert action.active is True
    assert action.throttle_cap == pytest.approx(0.45)
    assert action.steer_limit == pytest.approx(0.55)
    assert "risk cleared" in action.reason


def test_construction_very_close_stall_keeps_forward_escape_instead_of_reverse_loop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=1.52,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        junction_like=False,
        route_curvature=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=1.52,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.16,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        1.0,
        "PREPARE",
        "route63 close one-sided construction blockage",
    )

    actions = [planner.plan(features, estimate) for _ in range(18)]

    assert all(not action.reverse for action in actions)
    assert actions[-1].reason == "construction_very_close_open_side_escape"
    assert actions[-1].throttle_floor >= 0.34


def test_construction_ultra_close_still_allows_limited_reverse_unwedge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=0.55,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        junction_like=False,
        route_curvature=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=0.55,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        1.0,
        "PREPARE",
        "ultra-close contact unwedge",
    )

    action = planner.plan(features, estimate)

    assert action.reverse is True
    assert action.reason == "construction_open_side_reverse_unwedge"



def test_students_very_close_red_deadlock_releases_when_clear(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 55

    class StudentsVeryCloseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.0},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = StudentsVeryCloseRedPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.8, steer=-0.05),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.45
    assert getattr(ctrl, "reverse", False) is False
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"


def test_construction_extreme_gap_uses_forward_release_not_reverse(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.config.suppress_lateral_intersection_rules = True
    system.red_final_clamp_gap_frames = 720

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 20.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.7, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": -0.35},
        0.1,
    )

    assert ctrl.throttle >= 0.92
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.18
    assert getattr(ctrl, "reverse", False) is False



def test_construction_mid_gap_uses_short_reverse_unwedge(monkeypatch):
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.config.suppress_lateral_intersection_rules = True
    system.red_final_clamp_gap_frames = 360
    system.rule_planner.blocked_frames = 220

    class ClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 20.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearPerception()
    ctrl = system.process(
        Control(throttle=0.0, brake=0.7, steer=0.0),
        {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []},
        {"speed": 0.0},
        0.1,
    )

    assert 0.48 <= ctrl.throttle <= 0.64
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) >= 0.30
    assert getattr(ctrl, "reverse", False) is True



def test_construction_sparse_cone_low_speed_releases_at_five_meter_edge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=5.35,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.08,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.35,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.26,
        lidar_left_blockage_ratio=0.9,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.1,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 sparse cone low speed edge",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "construction_sparse_cone_low_speed_open_side_release"
    assert action.throttle_floor == pytest.approx(0.50)
    assert action.steer_bias > 0.0
    assert action.brake_cap == pytest.approx(0.0)


def test_construction_sparse_cone_long_hold_releases_instead_of_permanent_brake(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=4.95,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.95,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=0.18,
        lidar_left_blockage_ratio=0.75,
        lidar_right_blockage_ratio=0.15,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.05,
        detection_object_count=80,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 sparse cone long hold",
    )

    action = None
    for _ in range(31):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.reason == "construction_sparse_cone_long_hold_forward_release"
    assert action.throttle_floor >= 0.36
    assert action.brake_cap == pytest.approx(0.0)
    assert action.reverse is False



def test_construction_full_blockage_long_hold_reverses_near_full_center_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=4.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.2,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.1,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 full blockage long hold",
    )

    action = None
    for _ in range(20):
        action = planner.plan(features, estimate)

    assert action is not None
    assert action.reason == "construction_full_blockage_open_side_long_hold_reverse"
    assert action.throttle_floor >= 0.46
    assert action.reverse is True
    assert action.steer_min_magnitude >= 0.34


def test_construction_full_blockage_continues_forward_when_already_moving(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    planner.construction_full_blockage_escape_frames = 18
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=4.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.42,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.5,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.1,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 moving full blockage",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "construction_full_blockage_open_side_long_hold_push"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.50)


def test_construction_full_blockage_long_hold_pushes_after_reverse_clearance(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    planner.construction_full_blockage_escape_frames = 18
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=5.6,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.20,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=5.6,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.1,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 full blockage reverse clearance",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "construction_full_blockage_open_side_long_hold_push"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.50)
    assert action.steer_min_magnitude >= 0.28


def test_construction_full_blockage_long_hold_can_reverse_unwedge(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.setenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=True,
        )
    )
    planner.construction_full_blockage_escape_frames = 59
    features = mod.AuxFeatures(
        confidence=0.9,
        risk_level=1,
        front_clear=False,
        front_obstacle_distance=4.2,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.2,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.1,
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        0.9,
        "PREPARE",
        "route51 full blockage reverse unwedge",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "construction_full_blockage_open_side_long_hold_reverse"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)



def test_dynamic_forced_macros_suppress_construction_escape_actions(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInLikeStaticPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 4.0,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.92,
                    "center_blockage_ratio": 0.95,
                    "left_blockage_ratio": 0.45,
                    "right_blockage_ratio": 0.45,
                    "open_side": "balanced",
                    "lateral_centroid": -0.2,
                },
            }

    def construction_like_plan(features, estimate):
        return mod.PlannerAction(
            True,
            "RECOVER",
            throttle_cap=0.62,
            throttle_floor=0.46,
            brake_cap=0.0,
            steer_limit=0.68,
            reverse=True,
            reason="low_conf_center_blockage_reverse_escape_sweep",
        )

    system.perception = CutInLikeStaticPerception()
    system.rule_planner.plan = construction_like_plan
    raw = Control(throttle=0.7, brake=0.0, steer=0.05)
    ctrl = system.process(raw, {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl is raw
    assert system.last_debug.get("macro_scenario") == "high_speed_reckless_lane_cutting", system.last_debug
    assert system.last_debug["reason"] == "forced_macro_suppressed_construction_rule"
    assert system.last_debug["action_active"] is False


def test_blind_spot_suppresses_construction_creep_action(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class BlindSpotStaticPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 1.7,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.95,
                    "center_blockage_ratio": 0.95,
                    "left_blockage_ratio": 0.90,
                    "right_blockage_ratio": 0.05,
                    "open_side": "right",
                },
            }

    def construction_creep_plan(features, estimate):
        return mod.PlannerAction(
            True,
            "AVOID_OR_PASS",
            target_speed=1.1,
            throttle_cap=0.34,
            throttle_floor=0.22,
            brake_cap=0.0,
            steer_limit=0.38,
            reason="construction_close_static_high_blocked_open_side_creep",
        )

    system.perception = BlindSpotStaticPerception()
    system.rule_planner.plan = construction_creep_plan
    raw = Control(throttle=0.5, brake=0.0, steer=0.02)
    ctrl = system.process(raw, {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl is raw
    assert system.last_debug["macro_scenario"] == "blind_spot_hidden_car"
    assert system.last_debug["reason"] == "forced_macro_suppressed_construction_rule"
    assert system.last_debug["action_active"] is False



def test_blind_spot_far_static_speed_keepalive_overrides_model_brake(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=0.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=16.9,
        red_light_active=False,
        red_stop_distance=None,
        lidar_front_distance=18.0,
        lidar_center_blockage_ratio=0.0,
        route_curvature=0.1,
        tracked_objects=[],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced route-prior macro scenario",
    )

    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.16), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "blind_spot_far_static_speed_keepalive", system.last_debug
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle >= 0.82
    assert abs(ctrl.steer) <= 0.12


def test_blind_spot_clear_speed_keepalive_overrides_clear_braking(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class ClearBlindSpotPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {"route_curvature": 0.1, "junction_like": False},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 18.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = ClearBlindSpotPerception()
    raw = Control(throttle=0.0, brake=0.9, steer=0.04)
    ctrl = system.process(raw, {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 5.5}, 0.1)

    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.18
    assert system.last_debug["reason"] == "blind_spot_clear_speed_keepalive"



def test_cut_in_forward_commit_uses_straight_recovery_not_open_side(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInBlockedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {"route_curvature": 0.2, "junction_like": False},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 6.2,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.65,
                    "center_blockage_ratio": 0.90,
                    "left_blockage_ratio": 0.40,
                    "right_blockage_ratio": 0.40,
                    "open_side": "balanced",
                },
            }

    def suppressed_construction_plan(features, estimate):
        return mod.PlannerAction(False, "RECOVER", reason="forced_macro_suppressed_construction_rule")

    system.perception = CutInBlockedPerception()
    system.rule_planner.plan = suppressed_construction_plan
    raw = Control(throttle=0.0, brake=1.0, steer=0.45)
    ctrl = system.process(raw, {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.26
    assert getattr(ctrl, "reverse", False) is False
    assert system.last_debug["reason"] == "cut_in_forward_commit_no_open_side"


def test_students_active_red_deadlock_release_persists_after_first_motion(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 110

    class StudentsStaleRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.3},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = StudentsStaleRedPerception()
    first = system.process(Control(throttle=0.0, brake=0.8, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    second = system.process(Control(throttle=0.0, brake=0.8, steer=0.2), {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []}, {"speed": 0.45}, 0.2)

    assert first.throttle >= 0.88
    assert second.throttle >= 0.88
    assert second.brake == pytest.approx(0.0)
    assert abs(second.steer) <= 0.04
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_students_near_red_deadlock_release_before_block(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 46

    class StudentsNearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.3},
                "lidar_geometry": {"available": True, "front_distance": 12.0, "center_blockage_ratio": 0.0},
            }

    system.perception = StudentsNearRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05
    assert system.last_debug["reason"] in {"active_red_far_prolonged_creep_release", "students_long_active_red_final_release"}


def test_cut_in_open_side_bypass_forward_commits_after_stuck_frames(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 10

    class CutInRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 1.9,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert 0.56 <= ctrl.throttle <= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert 0.10 <= ctrl.steer <= 0.34
    assert system.last_debug["reason"] == "cut_in_ultra_close_reverse_escape"



def test_cut_in_close_open_side_overrides_suppressed_construction(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInCloseRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 2.38,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 0.95,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    def suppressed_construction_plan(features, estimate):
        return mod.PlannerAction(False, "RECOVER", reason="forced_macro_suppressed_construction_rule")

    system.perception = CutInCloseRightOpenPerception()
    system.rule_planner.plan = suppressed_construction_plan
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.55}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert 0.92 <= ctrl.throttle <= 1.0
    assert ctrl.brake == pytest.approx(0.0)
    assert 0.10 <= ctrl.steer <= 0.34
    assert system.last_debug["reason"] == "cut_in_controlled_open_side_bypass"


def test_cut_in_upper_mid_gap_stuck_short_reverses(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 4

    class CutInUpperMidGapRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "front_distance": 3.34, "front_blocked": True, "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0, "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInUpperMidGapRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_cut_in_mid_gap_stuck_short_reverses_before_forward_commit(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 24

    class CutInMidGapRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 2.75,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInMidGapRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert 0.56 <= ctrl.throttle <= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert -0.34 <= ctrl.steer <= -0.10
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_cut_in_lower_mid_gap_stuck_short_reverses(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 4

    class CutInLowerMidGapRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 2.46,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInLowerMidGapRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert ctrl.throttle >= 0.68
    assert ctrl.brake == pytest.approx(0.0)
    assert -0.34 <= ctrl.steer <= -0.18
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_cut_in_uses_controlled_open_side_when_right_is_clear(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 2.9,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.06}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert 0.92 <= ctrl.throttle <= 1.0
    assert ctrl.brake == pytest.approx(0.0)
    assert 0.10 <= ctrl.steer <= 0.34
    assert system.last_debug["reason"] == "cut_in_controlled_open_side_bypass"


def test_cut_in_three_meter_stall_reverses_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 8
    class CutInThreeMeterStallPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                    "traffic_light_state": {"active": False, "distance": None},
                    "lidar_geometry": {"available": True, "front_distance": 3.08, "front_blocked": True,
                                       "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                                       "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right"}}
    system.perception = CutInThreeMeterStallPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert getattr(ctrl, "reverse", False) is True
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_cut_in_post_reverse_stall_switches_to_opposite_sweep(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_sustain_frames = 12
    system.cut_in_open_side_stuck_frames = 18

    class CutInPostReverseStallPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True, "front_distance": 3.26, "front_blocked": True,
                    "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right",
                },
            }

    system.perception = CutInPostReverseStallPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.05}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert ctrl.throttle >= 0.94
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.34
    assert system.last_debug["reason"] == "cut_in_post_reverse_backout"


def test_cut_in_post_reverse_sustains_open_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_sustain_frames = 12
    system.cut_in_open_side_sustain_side = "right"

    class CutInPostReverseRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True, "front_distance": 2.89, "front_blocked": True,
                    "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right",
                },
            }

    system.perception = CutInPostReverseRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert ctrl.throttle >= 0.92
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer <= -0.30
    assert system.last_debug["reason"] == "cut_in_post_reverse_sustained_open_side_push"


def test_cut_in_high_mid_open_side_commit_does_not_wiggle_or_reverse(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 10

    class CutInHighMidRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True, "front_distance": 3.55, "front_blocked": True,
                    "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right",
                },
            }

    system.perception = CutInHighMidRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert ctrl.throttle >= 0.96
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer == pytest.approx(0.34)
    assert system.last_debug["reason"] == "cut_in_controlled_open_side_bypass"


def test_cut_in_high_mid_stuck_reverses_after_long_no_progress(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 24

    class CutInHighMidStuckPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True, "front_distance": 3.42, "front_blocked": True,
                    "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right",
                },
            }

    system.perception = CutInHighMidStuckPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_students_near_red_deadlock_releases_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig(); cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 8
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=True, front_vehicle_distance=None, front_pedestrian_distance=None,
        front_obstacle_distance=None, red_light_active=True, red_stop_distance=2.2, ego_speed=2.2,
        lidar_available=True, lidar_stale=False, lidar_front_distance=None, lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "APPROACH", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=0.85, steer=0.02), {}, {}, 0.0)
    assert ctrl.throttle >= 0.74
    assert ctrl.brake == pytest.approx(0.0)


def test_cut_in_open_side_allows_far_front_vehicle(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInFarVehicleRightOpenPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [{"class_name": "car", "box_lidar": {"x": 23.0, "y": 0.0}}],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True, "front_distance": 3.35, "front_blocked": True,
                    "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right",
                },
            }

    system.perception = CutInFarVehicleRightOpenPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert system.last_debug["reason"] in {"cut_in_controlled_open_side_bypass", "cut_in_open_side_short_reverse_unstuck"}
    assert ctrl.brake == pytest.approx(0.0)



def test_cut_in_post_unwedge_right_open_keeps_right_bias_without_full_brake(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_post_unwedge_commit_frames = 10

    class CutInAfterBackoffPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 3.4,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInAfterBackoffPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 3.6}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert ctrl.brake == pytest.approx(0.0)
    assert -0.34 <= ctrl.steer <= -0.10
    assert system.last_debug["reason"] == "cut_in_post_unwedge_open_side_commit"


def test_cut_in_post_unwedge_forces_forward_commit(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_post_unwedge_commit_frames = 10

    class CutInAfterBackoffPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 5.0,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 1.0,
                    "center_blockage_ratio": 1.0,
                    "left_blockage_ratio": 1.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "right",
                },
            }

    system.perception = CutInAfterBackoffPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=-0.4), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert 0.58 <= ctrl.throttle <= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert -0.34 <= ctrl.steer <= -0.10
    assert system.last_debug["reason"] == "cut_in_post_unwedge_open_side_commit"


def test_cut_in_close_stuck_uses_straight_reverse_unwedge(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInCloseStuckPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 10.5},
                "lidar_geometry": {
                    "available": True,
                    "front_distance": 2.8,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.4,
                    "center_blockage_ratio": 0.4,
                    "left_blockage_ratio": 0.2,
                    "right_blockage_ratio": 0.2,
                    "open_side": "balanced",
                },
            }

    system.perception = CutInCloseStuckPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.4), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert 0.44 <= ctrl.throttle <= 0.58
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.03
    assert system.last_debug["reason"] == "cut_in_straight_reverse_unwedge"


def test_students_far_red_deadlock_release_after_second_false_stop(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 42

    class StudentsFarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 8.6},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = StudentsFarRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.4, steer=-0.3), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_cut_in_far_red_false_positive_keeps_straight_close_crawl(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 0
    system.students_red_deadlock_release_frames = 12

    class CutInFarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {"route_curvature": 0.2, "junction_like": False},
                "traffic_light_state": {"active": True, "distance": 10.8},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 3.5,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.55,
                    "center_blockage_ratio": 0.60,
                    "left_blockage_ratio": 0.30,
                    "right_blockage_ratio": 0.30,
                    "open_side": "balanced",
                },
            }

    system.perception = CutInFarRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.42), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.2}, 0.1)

    assert ctrl.throttle >= 0.90
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.10
    assert system.last_debug["reason"] == "cut_in_far_red_false_positive_close_crawl"





def test_cut_in_far_red_false_positive_uses_open_side_crawl(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 120

    class CutInFarRedOpenSidePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {"route_curvature": 0.2, "junction_like": False},
                "traffic_light_state": {"active": True, "distance": 10.8},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 3.0, "front_blocked": True, "corridor_blockage_ratio": 0.55, "center_blockage_ratio": 0.60, "left_blockage_ratio": 0.50, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInFarRedOpenSidePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.2}, 0.1)

    assert 0.54 <= ctrl.throttle <= 0.68
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer > 0.0
    assert abs(ctrl.steer) <= 0.28
    assert system.last_debug["reason"] == "cut_in_far_red_false_positive_open_side_crawl"

def test_students_active_red_deadlock_creeps_only_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 110

    class StudentsStaleRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.3},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = StudentsStaleRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.8, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert 0.88 <= ctrl.throttle <= 1.0
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"





def test_ebike_ped_cross_red_deadlock_release_has_independent_reason(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ebike_and_pedestrian_cross")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 40

    class EbikeRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.6},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 12.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = EbikeRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08
    assert system.last_debug["reason"] == "ebike_ped_cross_red_deadlock_release"


def test_ghost_probe_red_deadlock_release_has_independent_reason(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 34

    class GhostProbeRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.6},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 14.0,
                    "front_blocked": False,
                    "corridor_blockage_ratio": 0.0,
                    "center_blockage_ratio": 0.0,
                    "left_blockage_ratio": 0.0,
                    "right_blockage_ratio": 0.0,
                    "open_side": "unknown",
                },
            }

    system.perception = GhostProbeRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=-0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08
    assert system.last_debug["reason"] == "ghost_probe_far_red_false_hold_creep"


def test_ebike_ped_cross_far_red_deadlock_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ebike_and_pedestrian_cross")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 60

    class EbikeFarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 13.4},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = EbikeFarRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=-0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ebike_ped_cross_red_deadlock_release"


def test_ghost_probe_near_red_deadlock_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 28

    class GhostNearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.8},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 12.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.1), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle >= 0.84
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_midline_false_hold_release"


def test_ghost_probe_stopline_like_front_obstacle_releases(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 30

    class GhostStoplineObstaclePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [{"class_name": "unknown", "x": 3.55, "y": 0.0, "confidence": 0.5}],
                "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.5},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 3.55, "front_blocked": True, "corridor_blockage_ratio": 0.2, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostStoplineObstaclePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.1), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle >= 0.84
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_red_deadlock_release"




def test_ghost_probe_far_red_false_hold_creeps_after_short_confirmed_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 4

    class GhostFarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.5},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 25.7, "front_blocked": False, "corridor_blockage_ratio": 0.20, "center_blockage_ratio": 0.25, "left_blockage_ratio": 0.25, "right_blockage_ratio": 0.25, "open_side": "balanced"},
            }

    system.perception = GhostFarRedPerception()
    ctrl = system.process(Control(throttle=0.8, brake=0.0, steer=0.12), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert system.last_debug["reason"] == "ghost_probe_far_red_false_hold_creep"
    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08


def test_ghost_probe_far_red_false_hold_releases_midline_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 80

    class GhostNearRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.2},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearRedPerception()
    ctrl = system.process(Control(throttle=0.8, brake=0.0, steer=0.12), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert system.last_debug["reason"] == "ghost_probe_midline_false_hold_release"
    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)

def test_ghost_probe_near_line_active_red_holds(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 30

    class GhostNearLinePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 1.20},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 10.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearLinePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.45
    assert system.last_debug["reason"] == "ghost_probe_active_red_hold"


def test_ghost_probe_near_line_finish_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class GhostNearLineClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.72},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 14.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearLineClearPerception()
    ctrl = system.process(Control(throttle=0.45, brake=0.0, steer=0.15), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.02}, 0.1)
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.45
    assert abs(ctrl.steer) <= 0.08
    assert system.last_debug["reason"] == "ghost_probe_active_red_hold"


def test_ebike_forced_macro_skips_generic_red_final_clamp(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ebike_and_pedestrian_cross")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 5

    class EbikeRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": True, "distance": 11.0}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = EbikeRedPerception()
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 3.0}, 0.1)
    assert system.last_debug["reason"] != "active_red_without_stopline_final_clamp"
    assert ctrl.brake < 0.2


def test_ghost_forced_macro_skips_generic_red_final_clamp(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 5

    class GhostRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": True, "distance": 9.0}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = GhostRedPerception()
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 3.0}, 0.1)
    assert system.last_debug["reason"] == "ghost_probe_active_red_hold"
    assert ctrl.brake >= 0.82


def test_ghost_probe_skips_early_active_red_stop_deceleration(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class GhostNearRedMovingPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.8},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 12.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearRedMovingPerception()
    ctrl = system.process(Control(throttle=0.65, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 1.4}, 0.1)
    assert system.last_debug["reason"] == "ghost_probe_active_red_hold"
    assert ctrl.brake >= 0.82


def test_ebike_skips_early_active_red_stop_deceleration(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ebike_and_pedestrian_cross")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class EbikeNearRedMovingPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.2},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 12.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = EbikeNearRedMovingPerception()
    ctrl = system.process(Control(throttle=0.65, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 1.4}, 0.1)
    assert system.last_debug["reason"] != "active_red_stop_deceleration"
    assert ctrl.brake < 0.45



def test_highway_accident_observed_hazard_brake_response_without_forced_prior(monkeypatch):
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.02, throttle=0.8, brake=0.0)
    features = mod.AuxFeatures(
        confidence=0.75,
        ego_speed=8.5,
        front_clear=False,
        front_vehicle_distance=32.0,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="highway_accident_vehicle",
        confidence=0.75,
        phase="PREPARE",
        reason="front vehicle at high speed",
    )

    ctrl = system.process(raw, {}, {}, 0.0)

    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.80


def test_highway_accident_observed_hazard_brake_skips_red_context(monkeypatch):
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.02, throttle=0.8, brake=0.0)
    features = mod.AuxFeatures(
        confidence=0.75,
        ego_speed=8.5,
        front_clear=False,
        front_vehicle_distance=32.0,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=4.0,
        tracked_objects=[],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="highway_accident_vehicle",
        confidence=0.75,
        phase="PREPARE",
        reason="front vehicle at high speed red",
    )

    ctrl = system.process(raw, {}, {}, 0.0)

    assert system.last_debug["reason"] != "highspeed_accident_observed_hazard_brake_response", system.last_debug


def test_highway_accident_skips_early_active_red_stop_deceleration(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayNearRedMovingPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 2.5},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 25.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = HighwayNearRedMovingPerception()
    ctrl = system.process(Control(throttle=0.65, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 1.4}, 0.1)
    assert system.last_debug["reason"] != "active_red_stop_deceleration"
    assert ctrl.brake < 0.45


def test_highway_accident_no_longer_preserves_speed_before_scored_activation(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayPretriggerPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 34.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "balanced"},
            }

    system.perception = HighwayPretriggerPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "pos": [158.0, 210.2]}, {"speed": 5.0}, 0.1)

    assert ctrl.brake >= 0.85
    assert system.last_debug["reason"] != "highspeed_accident_pretrigger_speed_preserve"


def test_highway_accident_brake_probe_overrides_late_red_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.highspeed_brake_response_frames = 20

    class HighwayLateReleasePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = HighwayLateReleasePerception()
    ctrl = system.process(Control(throttle=0.72, brake=0.0, steer=0.02), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert system.highspeed_brake_response_frames >= 0
    assert system.last_debug["macro_scenario"] == "highway_accident_vehicle"


def test_highway_accident_lidar_window_starts_short_brake_probe(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayLidarHazardPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {
                    "available": True,
                    "stale": False,
                    "front_distance": 25.0,
                    "front_blocked": True,
                    "corridor_blockage_ratio": 0.35,
                    "center_blockage_ratio": 0.35,
                    "left_blockage_ratio": 0.2,
                    "right_blockage_ratio": 0.2,
                    "open_side": "balanced",
                },
            }

    system.perception = HighwayLidarHazardPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "pos": [170.0, 205.5]}, {"speed": 4.5}, 0.1)

    assert ctrl.brake >= 0.85
    assert ctrl.throttle == pytest.approx(0.0)
    assert 8 <= system.highspeed_brake_response_frames <= 18
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"


def test_highway_accident_forced_route_brake_probe_is_short_and_strong(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": False, "distance": None}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = HighwayClearPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "pos": [170.0, 205.5]}, {"speed": 5.0}, 0.1)
    assert ctrl.brake >= 0.85
    assert ctrl.throttle == pytest.approx(0.0)
    assert 8 <= system.highspeed_brake_response_frames <= 18
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"


def test_highway_accident_uses_local_route_start_when_xml_progress_is_negative(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayLocalPosPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": False, "distance": None}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 30.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = HighwayLocalPosPerception()
    first = {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "ego": {"pos": [28.93, 212.22]}}
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), first, {"speed": 12.0}, 0.1)
    assert system.highspeed_route_local_start_pos == pytest.approx((28.93, 212.22))
    assert ctrl.brake < 0.85

    in_window = {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": [], "ego": {"pos": [66.2, 197.6]}}
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), in_window, {"speed": 5.0}, 0.2)
    assert ctrl.brake >= 0.85
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"


def test_highway_accident_reads_ego_pos_from_model_detection(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayModelDetectionPosPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": False, "distance": None}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = HighwayModelDetectionPosPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "ego": {"pos": [166.8, 206.7]}}, {"speed": 4.2}, 0.1)
    assert ctrl.brake >= 0.85
    assert ctrl.throttle == pytest.approx(0.0)
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"


def test_highway_accident_route36_reference_window_triggers_brake(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayRoute36Perception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": False, "distance": None}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = HighwayRoute36Perception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": [], "pos": [166.8, 206.7]}, {"speed": 4.2}, 0.1)
    assert ctrl.brake >= 0.85
    assert ctrl.throttle == pytest.approx(0.0)
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"




def test_ghost_probe_midline_false_hold_releases_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 70

    class GhostProbeMidlineClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.0},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeMidlineClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_midline_false_hold_release"


def test_ghost_probe_stopline_false_hold_releases_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 50

    class GhostProbeStoplineClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 1.1},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeStoplineClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.88
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_stopline_false_hold_release"


def test_ghost_probe_near_line_false_hold_releases_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 120

    class GhostProbeNearLineClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 5.1},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeNearLineClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert 0.82 <= ctrl.throttle <= 0.95
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_near_line_false_hold_release"


def test_ghost_probe_final_cross_release_does_not_extend_commit_loop(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 66
    system.ghost_probe_active_red_hold_frames = 10
    system.ghost_probe_line_commit_frames = 0
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=2.1,
        ego_speed=0.15,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("ghost_probe", 1.0, "NORMAL", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=0.5, steer=0.08), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "ghost_probe_final_cross_release", system.last_debug
    assert ctrl.throttle >= 0.96
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.04
    assert system.ghost_probe_line_commit_frames <= 12


def test_ghost_probe_line_commit_survives_rolling_red_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_line_commit_frames = 12

    class GhostProbeLineCommitPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 1.9},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeLineCommitPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 1.2}, 0.1)

    assert ctrl.throttle >= 0.86
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_line_commit_release"
    assert system.last_debug["ghost_probe_line_commit_frames"] == 11


def test_ghost_probe_far_red_false_hold_keeps_creeping_while_rolling(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 120

    class GhostProbeFarRedClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.1},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeFarRedClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.68}, 0.1)

    assert 0.58 <= ctrl.throttle <= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08
    assert system.last_debug["reason"] == "ghost_probe_far_red_false_hold_creep"


def test_ghost_probe_far_red_obstacle_open_side_release(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.ghost_probe_active_red_hold_frames = 32
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=8.6,
        red_light_active=True,
        red_stop_distance=9.4,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=8.6,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="ghost_probe",
        confidence=1.0,
        phase="NORMAL",
        reason="forced test",
    )
    ctrl = system.process(Control(throttle=0.0, brake=0.4, steer=0.0), {}, {}, 0.1)
    assert system.last_debug["reason"] == "ghost_probe_far_red_obstacle_open_side_release"
    assert ctrl.throttle >= 0.46
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer < 0.0


def test_ghost_probe_close_static_red_hold_bypass_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    cfg.allow_route_prior = True
    cfg.forced_macro_scenario = "ghost_probe"
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 180
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.44,
        red_light_active=False,
        red_stop_distance=None,
        ego_speed=0.004,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.44,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=0.8,
        lidar_right_blockage_ratio=0.8,
        lidar_open_side="balanced",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="ghost_probe",
        confidence=1.0,
        phase="APPROACH",
        reason="forced test",
    )

    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.0), {}, {"speed": 0.004}, 0.1)

    assert system.last_debug["reason"] == "ghost_probe_close_static_red_hold_bypass"
    assert 0.42 <= ctrl.throttle <= 0.62
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) >= 0.14


def test_ghost_probe_close_static_red_hold_reverse_sweep_after_long_stall(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    cfg.allow_route_prior = True
    cfg.forced_macro_scenario = "ghost_probe"
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 260
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.60,
        red_light_active=True,
        red_stop_distance=10.8,
        ego_speed=0.001,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.60,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="ghost_probe",
        confidence=1.0,
        phase="APPROACH",
        reason="forced test",
    )

    ctrl = system.process(Control(throttle=0.0, brake=0.45, steer=0.0), {}, {"speed": 0.001}, 0.1)

    assert system.last_debug["reason"] == "ghost_probe_close_static_red_hold_reverse_sweep"
    assert getattr(ctrl, "reverse", False) is True
    assert 0.40 <= ctrl.throttle <= 0.58
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer < 0.0



def test_ghost_probe_far_red_false_hold_gently_caps_high_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 12

    class GhostProbeFarRedFastPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.8},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeFarRedFastPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 6.3}, 0.1)

    assert ctrl.throttle == pytest.approx(0.0)
    assert 0.18 <= ctrl.brake < 0.82
    assert system.last_debug["reason"] == "ghost_probe_far_red_false_hold_creep"


def test_cut_in_close_gap_prefers_open_side_forward_commit(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInVeryClosePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 2.82, "front_blocked": True, "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0, "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInVeryClosePerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert getattr(ctrl, "reverse", False) is False
    assert 0.92 <= ctrl.throttle <= 1.0
    assert ctrl.brake == pytest.approx(0.0)
    assert -0.34 <= ctrl.steer <= -0.10
    assert system.last_debug["reason"] == "cut_in_controlled_open_side_bypass"




def test_highway_accident_red_release_never_reverses(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 220
    system.red_final_clamp_gap_frames = 6
    system.red_final_clamp_last_distance = 6.8
    system.rule_planner.blocked_frames = 140

    class HighwayClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 20.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = HighwayClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": -1.2}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_far_prolonged_creep_release"


def test_highway_accident_done_state_rearms_near_hazard_brake_probe(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.highspeed_brake_response_done = True

    class HighwayNearHazardPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 4.1, "front_blocked": True, "corridor_blockage_ratio": 0.4, "center_blockage_ratio": 0.4, "left_blockage_ratio": 0.2, "right_blockage_ratio": 0.2, "open_side": "unknown"},
            }

    system.perception = HighwayNearHazardPerception()
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 1.7}, 0.1)

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.85
    assert system.last_debug["reason"] == "highspeed_accident_brake_response_probe"


def test_highway_accident_mid_hazard_creep_bypass_after_brake(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.highspeed_brake_response_done = True

    class HighwayMidHazardPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 7.5, "front_blocked": True, "corridor_blockage_ratio": 0.5, "center_blockage_ratio": 0.5, "left_blockage_ratio": 0.2, "right_blockage_ratio": 0.2, "open_side": "balanced"},
            }

    system.perception = HighwayMidHazardPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle >= 0.92
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) >= 0.14
    assert system.last_debug["reason"] == "highspeed_accident_mid_hazard_creep_bypass"


def test_highway_accident_close_hazard_creep_bypass_after_brake(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.highspeed_brake_response_done = True

    class HighwayCloseHazardPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [{"class_name": "vehicle", "distance": 3.3, "x": 3.3, "y": 0.0}],
                "map_objects": [],
                "tracked_objects": [],
                "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 3.3},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 3.3, "front_blocked": True, "corridor_blockage_ratio": 0.6, "center_blockage_ratio": 0.6, "left_blockage_ratio": 0.2, "right_blockage_ratio": 0.2, "open_side": "unknown"},
            }

    system.perception = HighwayCloseHazardPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.02}, 0.1)
    assert ctrl.throttle >= 0.52
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) >= 0.18
    assert system.last_debug["reason"] == "highspeed_accident_close_hazard_creep_bypass"


def test_highway_accident_suppresses_reverse_vehicle_ttc_defensive_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=2.0,
        front_vehicle_distance=3.5,
        front_pedestrian_distance=None,
        reversing_vehicle_evidence=True,
        front_clear=False,
        risk_level=3,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=3.5,
        lidar_blockage_ratio=0.8,
        lidar_center_blockage_ratio=0.8,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("highway_accident_vehicle", 1.0, "APPROACH", "forced")
    action = planner.plan(features, estimate)
    assert action.reason != "reverse_vehicle_ttc_defensive_brake"


def test_reverse_vehicle_still_uses_ttc_defensive_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=2.0,
        front_vehicle_distance=3.5,
        front_pedestrian_distance=None,
        reversing_vehicle_evidence=True,
        front_clear=False,
        risk_level=3,
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "APPROACH", "forced")
    action = planner.plan(features, estimate)
    assert action.reason == "reverse_vehicle_ttc_defensive_brake"


def test_cut_in_three_meter_gap_prefers_open_side_forward(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInThreeMeterPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 3.0, "front_blocked": True, "corridor_blockage_ratio": 0.35, "center_blockage_ratio": 0.35, "left_blockage_ratio": 0.45, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInThreeMeterPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.05}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert system.last_debug["reason"] != "cut_in_close_reverse_gap_reset"


def test_cut_in_ultra_close_gap_still_reverses(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CutInUltraClosePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 1.8, "front_blocked": True, "corridor_blockage_ratio": 0.45, "center_blockage_ratio": 0.45, "left_blockage_ratio": 0.55, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInUltraClosePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.05}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert system.last_debug["reason"] == "cut_in_ultra_close_reverse_escape"


def test_cut_in_clear_recovery_releases_clear_false_red_after_cut_in(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_clear_recovery_frames = 5

    class CutInRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.5},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 22.0, "front_blocked": False, "corridor_blockage_ratio": 0.1, "center_blockage_ratio": 0.1, "left_blockage_ratio": 0.1, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInRedPerception()
    ctrl = system.process(Control(throttle=0.7, brake=0.0, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.2}, 0.1)

    assert system.cut_in_false_red_release_frames > 0
    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.04
    assert system.last_debug["reason"] == "cut_in_false_red_clear_path_release"


def test_cut_in_far_red_uses_hold_not_generic_creep_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 100

    class CutInFarRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 9.5},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 22.0, "front_blocked": False, "corridor_blockage_ratio": 0.1, "center_blockage_ratio": 0.1, "left_blockage_ratio": 0.1, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInFarRedPerception()
    ctrl = system.process(Control(throttle=0.7, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert ctrl.throttle == pytest.approx(0.0)
    assert system.last_debug["reason"] == "active_red_without_stopline_final_clamp"

def test_highway_accident_does_not_brake_before_route_activation_window(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "highway_accident_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class HighwayTooEarlyHazardPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {"objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {}, "traffic_light_state": {"active": False, "distance": None}, "lidar_geometry": {"available": True, "stale": False, "front_distance": 30.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"}}

    system.perception = HighwayTooEarlyHazardPerception()
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 5.0, "pos": [150.0, 213.3]}, 0.1)
    assert system.last_debug["reason"] != "highspeed_accident_brake_response_probe"
    assert ctrl.brake < 0.85



def test_cut_in_mid_gap_uses_longer_open_side_reverse_escape(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_open_side_stuck_frames = 4

    class CutInMidGapPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 3.6, "front_blocked": True, "corridor_blockage_ratio": 1.0, "center_blockage_ratio": 1.0, "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInMidGapPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert getattr(ctrl, "reverse", False) is True
    assert system.cut_in_open_side_reverse_frames >= 17
    assert system.cut_in_post_unwedge_commit_frames >= 96
    assert ctrl.throttle >= 0.58
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer <= -0.24
    assert system.last_debug["reason"] == "cut_in_open_side_short_reverse_unstuck"


def test_ghost_probe_far_red_persistent_creep_survives_interrupted_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 0
    system.ghost_probe_far_red_release_frames = 2

    class GhostProbeFarRedInterruptedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 10.2},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 29.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeFarRedInterruptedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.12), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.95
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08
    assert system.ghost_probe_far_red_release_frames >= 3
    assert system.last_debug["reason"] in {"ghost_probe_far_red_false_hold_creep", "ghost_probe_far_red_persistent_creep"}



def test_cut_in_wide_gap_push_overrides_post_unwedge_commit(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_post_unwedge_commit_frames = 50

    class CutInWideGapPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 4.58, "front_blocked": True, "corridor_blockage_ratio": 0.8, "center_blockage_ratio": 0.8, "left_blockage_ratio": 1.0, "right_blockage_ratio": 0.0, "open_side": "right"},
            }

    system.perception = CutInWideGapPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.05}, 0.1)

    assert getattr(ctrl, "reverse", False) is False
    assert ctrl.throttle >= 0.98
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer <= -0.36
    assert system.last_debug["reason"] == "cut_in_wide_gap_open_side_push"



def test_ghost_probe_far_red_clear_creep_not_overridden_by_guard(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_far_red_release_frames = 3

    class GhostProbeFarRedFastClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 8.3},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeFarRedFastClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 4.4}, 0.1)

    assert ctrl.throttle >= 0.95
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] in {"ghost_probe_far_red_false_hold_creep", "ghost_probe_far_red_persistent_creep"}



def test_ghost_probe_upper_near_line_releases_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 8

    class GhostProbeUpperNearLinePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 7.75},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostProbeUpperNearLinePerception()
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_near_line_false_hold_release"


def test_ghost_probe_stopline_releases_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 7

    class GhostStoplinePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.42},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 9.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostStoplinePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.02}, 0.1)

    assert ctrl.throttle >= 0.88
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_stopline_false_hold_release"


def test_cut_in_wide_gap_push_memory_blocks_post_commit(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.cut_in_wide_gap_push_frames = 5
    system.cut_in_wide_gap_push_side = "right"
    system.cut_in_post_unwedge_commit_frames = 20

    class CutInWideMemoryPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 4.7, "front_blocked": True, "corridor_blockage_ratio": 0.50, "center_blockage_ratio": 0.50, "left_blockage_ratio": 0.45, "right_blockage_ratio": 0.15, "open_side": "unknown"},
            }

    system.perception = CutInWideMemoryPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert system.last_debug["reason"] == "cut_in_wide_gap_open_side_push_memory"
    assert ctrl.throttle >= 0.98
    assert ctrl.steer <= -0.20


def test_cut_in_close_false_red_releases_after_short_hold_without_memory(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 5

    class CutInCloseFalseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 1.95},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 24.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = CutInCloseFalseRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.01}, 0.1)

    assert ctrl.throttle >= 0.88
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.04
    assert system.last_debug["reason"] == "cut_in_false_red_clear_path_release"


def test_cut_in_very_close_false_red_releases_after_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 40

    class CutInVeryCloseFalseRedPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.65},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = CutInVeryCloseFalseRedPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=-0.30), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)

    assert ctrl.throttle >= 0.88
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.04
    assert system.last_debug["reason"] == "cut_in_false_red_clear_path_release"


def test_ghost_probe_near_line_releases_with_far_obstacle(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.ghost_probe_active_red_hold_frames = 8

    class GhostNearLineFarObstaclePerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [{"class_name": "others", "x": 23.0, "y": 0.1, "distance": 23.0}],
                "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 7.4},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": 23.0, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = GhostNearLineFarObstaclePerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.02}, 0.1)

    assert ctrl.throttle >= 0.82
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "ghost_probe_near_line_false_hold_release"



def test_construction_dense_close_hold_releases_to_side_gap(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.blocked_frames = 7
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        risk_level=2,
        ego_speed=0.0,
        front_obstacle_distance=3.0,
        lidar_front_distance=3.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_blockage_ratio=0.96,
        lidar_center_blockage_ratio=0.52,
        lidar_left_blockage_ratio=0.92,
        lidar_right_blockage_ratio=0.12,
        lidar_open_side="right",
        detection_object_count=100,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
    )
    estimate = mod.ScenarioEstimate("trucks_encountered_during_construction", 1.0, "PREPARE", "test")
    action = None
    for _ in range(8):
        action = planner.plan(features, estimate)
    assert action.reason == "construction_static_side_gap_hold_release"
    assert action.throttle_floor >= 0.54
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_cut_in_route_rejoin_stabilizes_after_obstacle_clear(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.cut_in_route_rejoin_frames = 5
    system.cut_in_route_rejoin_side = "right"
    raw = Control(steer=0.69, throttle=1.0, brake=0.0)
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        ego_speed=8.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=18.0,
        lidar_center_blockage_ratio=0.0,
        lidar_left_blockage_ratio=0.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="unknown",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="high_speed_reckless_lane_cutting",
        confidence=0.9,
        phase="RECOVER",
        reason="forced test",
    )
    control = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("macro_scenario") == "high_speed_reckless_lane_cutting", system.last_debug
    assert system.last_debug["reason"] == "cut_in_route_rejoin_speed_stabilize"
    assert control.throttle == pytest.approx(0.0)
    assert control.brake >= 0.34
    assert abs(control.steer) <= 0.06
    assert system.cut_in_route_rejoin_frames == 4


def test_construction_sparse_cone_low_speed_side_gap_release_without_suppression(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    monkeypatch.delenv("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(
        mod.AuxiliaryConfig(
            lidar_open_side_nudge_enabled=True,
            suppress_lateral_intersection_rules=False,
        )
    )
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        risk_level=2,
        ego_speed=0.02,
        front_obstacle_distance=4.55,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.55,
        lidar_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_center_blockage_ratio=0.18,
        lidar_open_side="right",
        lidar_lateral_centroid=-1.68,
        detection_object_count=102,
    )
    estimate = mod.ScenarioEstimate(
        "trucks_encountered_during_construction",
        1.0,
        "PREPARE",
        "route48 sparse cone hold",
    )
    action = None
    for _ in range(12):
        action = planner.plan(features, estimate)
    assert action is not None
    assert action.reason == "construction_sparse_cone_low_speed_side_gap_release"
    assert action.throttle_floor == pytest.approx(0.42)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_cut_in_route_rejoin_limits_raw_steer_without_side_bias(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.cut_in_route_rejoin_frames = 3
    system.cut_in_route_rejoin_side = "left"
    raw = Control(steer=0.50, throttle=1.0, brake=0.0)
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        ego_speed=2.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=20.0,
        lidar_center_blockage_ratio=0.0,
        lidar_open_side="unknown",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="high_speed_reckless_lane_cutting",
        confidence=0.9,
        phase="RECOVER",
        reason="forced test",
    )
    control = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("reason") == "cut_in_route_rejoin_speed_stabilize", system.last_debug
    assert control.steer == pytest.approx(-0.045)
    assert control.throttle <= 0.68
    assert control.throttle >= 0.44



def test_cut_in_post_rejoin_static_obstacle_bypass(monkeypatch):
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "high_speed_reckless_lane_cutting")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.cut_in_route_rejoin_side = "right"
    raw = Control(steer=0.0, throttle=0.0, brake=1.0)
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.25,
        red_light_active=False,
        red_stop_distance=None,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.25,
        lidar_center_blockage_ratio=0.08,
        lidar_left_blockage_ratio=0.20,
        lidar_right_blockage_ratio=0.20,
        lidar_open_side="balanced",
        lidar_lateral_centroid=-0.02,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="high_speed_reckless_lane_cutting",
        confidence=0.9,
        phase="RECOVER",
        reason="forced test",
    )

    control = system.process(raw, {}, {}, 0.0)

    assert system.last_debug.get("reason") == "cut_in_post_rejoin_static_obstacle_bypass", system.last_debug
    assert control.brake == pytest.approx(0.0)
    assert control.throttle >= 0.36
    assert abs(control.steer) >= 0.12


def test_roundabout_ultra_close_reverse_unwedge_after_long_block(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    planner.roundabout_context_frames = 10
    planner.blocked_frames = 520
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        ego_speed=0.02,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=0.32,
        red_light_active=False,
        red_stop_distance=None,
        lidar_available=True,
        lidar_front_distance=0.32,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
    )
    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "RECOVER", "test"))
    assert action.reason == "roundabout_ultra_close_reverse_unwedge"
    assert action.reverse is True
    assert action.steer_bias > 0.0


def test_students_no_stopline_red_deadlock_release_at_rolling_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig(); cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 4
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=True, front_vehicle_distance=None, front_pedestrian_distance=None,
        front_obstacle_distance=None, red_light_active=True, red_stop_distance=None, ego_speed=4.4,
        lidar_available=True, lidar_stale=False, lidar_front_distance=None, lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "APPROACH", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=0.85, steer=-0.10), {}, {}, 0.0)
    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert system.last_debug["reason"] == "students_no_stopline_red_deadlock_release"


def test_students_no_stopline_red_deadlock_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig(); cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 9
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=True, front_vehicle_distance=None, front_pedestrian_distance=None,
        front_obstacle_distance=None, red_light_active=True, red_stop_distance=None, ego_speed=2.4,
        lidar_available=True, lidar_stale=False, lidar_front_distance=None, lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "APPROACH", "forced test")

    ctrl = system.process(Control(throttle=0.0, brake=0.85, steer=-0.10), {}, {}, 0.0)

    assert ctrl.throttle >= 0.72
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.steer == pytest.approx(-0.06)
    assert system.last_debug["reason"] == "students_no_stopline_red_deadlock_release"


def test_students_far_red_deadlock_release_covers_twelve_meter_false_stop(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 34
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=12.4,
        ego_speed=0.04,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "NORMAL", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=0.5, steer=0.09), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "active_red_far_prolonged_creep_release", system.last_debug
    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05


def test_students_stopline_red_deadlock_release_after_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 14

    class StudentsStoplineClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 0.49},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = StudentsStoplineClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=-0.02), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.0}, 0.1)
    assert ctrl.throttle >= 0.74
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05

def test_blind_spot_mid_red_false_release_after_clamp(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 10

    class BlindSpotMidRedClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.6},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = BlindSpotMidRedClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=0.04), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 2.2}, 0.1)

    assert ctrl.throttle >= 0.92
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08



def test_blind_spot_junction_side_risk_triggers_rule_planner_scored_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.45,
        ego_speed=5.8,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        junction_like=True,
        side_risk=True,
        route_curvature=0.2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=0.45,
        phase="PREPARE",
        reason="junction side risk",
    )

    action = planner.plan(features, estimate)

    assert action.reason == "blind_spot_junction_scored_brake_response"
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.brake == pytest.approx(0.62)
    assert planner.lateral_intersection_release_frames >= 900


def test_blind_spot_junction_scored_brake_requires_side_risk(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.45,
        ego_speed=5.8,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        junction_like=True,
        side_risk=False,
        route_curvature=0.2,
    )
    estimate = mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=0.45,
        phase="PREPARE",
        reason="junction without side risk",
    )

    action = planner.plan(features, estimate)

    assert action.reason != "blind_spot_junction_scored_brake_response"


def test_blind_spot_side_vehicle_prebrake_from_tracked_objects(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.02, throttle=1.0, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=6.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[{"class_name": "car", "x": 30.0, "y": -2.2, "vx": 0.0}],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced test",
    )
    ctrl = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("reason") == "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.90
    assert abs(ctrl.steer) <= 0.12


def test_blind_spot_forced_route_prior_prebrakes_far_side_vehicle(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=3.8,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[
            {"class_name": "car", "x": 42.0, "y": -2.2, "observed_frames": 4, "score": 0.55},
        ],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced route-prior macro scenario",
    )

    ctrl = system.process(Control(throttle=0.8, brake=0.0), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.92


def test_blind_spot_forced_route_prior_side_vehicle_overrides_cooldown(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.blind_spot_prebrake_cooldown_frames = 80
    system.blind_spot_prebrake_frames = 0
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=4.45,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[
            {"class_name": "car", "x": 35.6, "y": -1.1, "observed_frames": 4, "score": 0.41},
        ],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced route-prior macro scenario",
    )

    ctrl = system.process(Control(throttle=1.0, brake=0.0), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.92


def test_blind_spot_side_vehicle_prebrake_uses_cooldown(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.01, throttle=0.8, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=6.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[{"class_name": "car", "x": 30.0, "y": 2.0, "vx": 0.0}],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced test",
    )

    ctrl = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("reason") == "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.brake >= 0.76
    assert system.blind_spot_prebrake_cooldown_frames == 120

    system.blind_spot_prebrake_frames = 0
    ctrl = system.process(raw, {}, {}, 0.1)
    assert system.last_debug.get("reason") != "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.brake == pytest.approx(0.0)


def test_blind_spot_clear_approach_prebrake_covers_forced_route_prior_window(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.03, throttle=0.72, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=5.6,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        tracked_objects=[{"class_name": "car", "x": 40.6, "y": -1.3, "vx": 0.0}],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="NORMAL",
        reason="forced route-prior macro scenario",
    )

    ctrl = system.process(raw, {}, {}, 0.0)

    assert system.last_debug.get("reason") == "blind_spot_side_vehicle_prebrake", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.90
    assert abs(ctrl.steer) <= 0.12


def _run_blind_spot_route_prior_trigger_case(monkeypatch, *, ego_pos, red_light_active=False):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.16, throttle=0.82, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=9.8,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=red_light_active,
        red_stop_distance=8.0 if red_light_active else None,
        tracked_objects=[],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced route-prior macro scenario",
    )

    return system, system.process(raw, {}, {"pos": ego_pos}, 0.0)


def test_blind_spot_late_red_brake_response(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "blind_spot_hidden_car")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 8
    raw = Control(steer=0.06, throttle=0.85, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=3.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=9.6,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
        tracked_objects=[],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="blind_spot_hidden_car",
        confidence=1.0,
        phase="PREPARE",
        reason="forced route-prior macro scenario",
    )

    ctrl = system.process(raw, {}, {}, 0.0)

    assert system.last_debug.get("reason") == "blind_spot_late_red_brake_response", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.82
    assert abs(ctrl.steer) <= 0.08


def test_blind_spot_route_prior_trigger_zone_brakes_inside_window(monkeypatch):
    system, ctrl = _run_blind_spot_route_prior_trigger_case(monkeypatch, ego_pos=[-2.0, 216.0])

    assert system.last_debug.get("reason") == "blind_spot_route_prior_trigger_zone_brake", system.last_debug
    assert system.last_debug.get("blind_spot_route_prior_ego_pos") == [-2.0, 216.0]
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.62
    assert abs(ctrl.steer) <= 0.08


def test_blind_spot_route_prior_trigger_zone_skips_outside_window(monkeypatch):
    system, ctrl = _run_blind_spot_route_prior_trigger_case(monkeypatch, ego_pos=[24.0, 205.0])

    assert system.last_debug.get("reason") != "blind_spot_route_prior_trigger_zone_brake", system.last_debug
    assert ctrl.brake == pytest.approx(0.0)

def test_blind_spot_route_prior_trigger_zone_skips_early_window(monkeypatch):
    system, ctrl = _run_blind_spot_route_prior_trigger_case(monkeypatch, ego_pos=[-2.0, 205.0])

    assert system.last_debug.get("reason") != "blind_spot_route_prior_trigger_zone_brake", system.last_debug
    assert ctrl.brake == pytest.approx(0.0)


def test_blind_spot_route_prior_trigger_zone_skips_red_light(monkeypatch):
    system, ctrl = _run_blind_spot_route_prior_trigger_case(
        monkeypatch,
        ego_pos=[-2.0, 216.0],
        red_light_active=True,
    )

    assert system.last_debug.get("reason") != "blind_spot_route_prior_trigger_zone_brake", system.last_debug
    assert ctrl.brake < 0.62


def test_students_clear_route_speed_guard_caps_post_risk_speed(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.22, throttle=1.0, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=7.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=24.0,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="four_students_crossing_the_road",
        confidence=1.0,
        phase="NORMAL",
        reason="forced test",
    )
    system.rule_planner.plan = lambda built_features, estimate: mod.PlannerAction(False, "NORMAL", reason="clear")
    ctrl = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("reason") == "students_clear_route_speed_guard", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.22
    assert abs(ctrl.steer) <= 0.12


def test_students_long_active_red_final_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 120
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=0.45,
        ego_speed=0.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "APPROACH", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.2), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "students_long_active_red_final_release", system.last_debug
    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08


def test_ghost_probe_long_active_red_final_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "ghost_probe")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.ghost_probe_active_red_hold_frames = 45
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=3.6,
        ego_speed=0.0,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("ghost_probe", 1.0, "APPROACH", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.12), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "ghost_probe_long_active_red_final_release", system.last_debug
    assert ctrl.throttle >= 0.74
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05


def test_roundabout_mid_far_static_open_side_progress(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8, front_clear=False, front_obstacle_distance=15.5, front_vehicle_distance=None,
        front_pedestrian_distance=None, red_stop_distance=None, red_light_active=False, ego_speed=0.02,
        junction_like=True, route_curvature=3.2, lidar_available=True, lidar_stale=False,
        lidar_front_distance=15.5, lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0, lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout far static")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_mid_far_static_open_side_progress"
    assert action.throttle_floor == pytest.approx(0.68)
    assert action.steer_bias == pytest.approx(-0.22)
    assert action.reverse is False


def test_roundabout_mid_far_static_open_side_progress_skips_at_cruise_speed(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.8, front_clear=False, front_obstacle_distance=16.8, front_vehicle_distance=None,
        front_pedestrian_distance=None, red_stop_distance=None, red_light_active=False, ego_speed=3.8,
        junction_like=True, route_curvature=3.2, lidar_available=True, lidar_stale=False,
        lidar_front_distance=16.8, lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=0.9,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0, lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 0.8, "APPROACH", "roundabout far static")

    action = planner.plan(features, estimate)

    assert action.reason != "roundabout_mid_far_static_open_side_progress"


def test_roundabout_route_prior_no_stopline_close_obstacle_reverses(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.rule_planner.blocked_frames = 14
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.66,
        red_light_active=True,
        red_stop_distance=None,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=2.66,
        lidar_center_blockage_ratio=0.45,
        lidar_open_side="balanced",
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("roundabout", 1.0, "NORMAL", "forced test")

    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=0.48), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "roundabout_route_prior_no_stopline_close_obstacle_reverse", system.last_debug
    assert getattr(ctrl, "reverse", False) is True
    assert 0.52 <= ctrl.throttle <= 0.68
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.42


def test_roundabout_route_prior_no_stopline_red_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=None,
        ego_speed=4.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("roundabout", 1.0, "NORMAL", "forced test")

    ctrl = system.process(Control(throttle=0.9, brake=0.0, steer=-0.2), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "roundabout_route_prior_no_stopline_red_release", system.last_debug
    assert 0.56 <= ctrl.throttle <= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.12


def test_roundabout_active_mid_red_release_after_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=4.3,
        ego_speed=0.05,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("roundabout", 1.0, "NORMAL", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=0.5, steer=0.12), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "roundabout_active_mid_red_release", system.last_debug
    assert 0.34 <= ctrl.throttle <= 0.48
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05


def test_roundabout_active_mid_red_yields_when_tracked_scene_is_busy(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 24
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=True,
        red_stop_distance=4.2,
        ego_speed=0.04,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=None,
        lidar_center_blockage_ratio=0.0,
        tracked_objects=[{"id": i, "class_name": "vehicle"} for i in range(6)],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {"tracked_objects": features.tracked_objects}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("roundabout", 1.0, "NORMAL", "forced test")
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.12), {}, {}, 0.0)
    assert system.last_debug.get("reason") == "roundabout_active_mid_red_yield_hold", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert 0.22 <= ctrl.brake <= 0.38
    assert abs(ctrl.steer) <= 0.04


def test_roundabout_active_mid_red_yield_times_out_to_release(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 24
    system.roundabout_mid_red_yield_frames = 18
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=True, front_vehicle_distance=None, front_pedestrian_distance=None,
        front_obstacle_distance=None, red_light_active=True, red_stop_distance=4.4, ego_speed=0.02,
        lidar_available=True, lidar_stale=False, lidar_front_distance=None, lidar_center_blockage_ratio=0.0,
        tracked_objects=[{"id": i, "class_name": "vehicle"} for i in range(7)],
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {"tracked_objects": features.tracked_objects}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("roundabout", 1.0, "NORMAL", "forced test")

    ctrl = system.process(Control(throttle=0.0, brake=0.5, steer=0.10), {}, {}, 0.0)

    assert system.last_debug.get("reason") == "roundabout_active_mid_red_yield_timeout_release", system.last_debug
    assert 0.28 <= ctrl.throttle <= 0.40
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.04



def test_roundabout_red_gap_release_keeps_small_steer(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 220
    system.red_final_clamp_gap_frames = 4
    system.red_final_clamp_last_distance = 6.2

    class RoundaboutRedGapPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "stale": False, "front_distance": None, "front_blocked": False, "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0, "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = RoundaboutRedGapPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=-0.22), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 0.1}, 0.1)

    assert ctrl.throttle <= 0.46
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.08


def test_roundabout_three_meter_stall_reverses_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.blocked_frames = 7
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=3.00,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.01, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=3.00,
        lidar_blockage_ratio=0.80, lidar_center_blockage_ratio=0.80,
        lidar_left_blockage_ratio=0.8, lidar_right_blockage_ratio=0.0,
        lidar_left_density=20, lidar_right_density=0, lidar_center_density=20,
        lidar_open_side="right", detection_object_count=100,
    )
    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout three meter stall"))
    assert action.reason == "roundabout_very_close_reverse_clearance"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.72)


def test_roundabout_very_close_forward_stall_reverses_before_long_hold(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.blocked_frames = 119
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.60,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.01, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.60,
        lidar_blockage_ratio=0.94, lidar_center_blockage_ratio=0.94,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=42, lidar_right_density=0, lidar_center_density=33,
        lidar_open_side="right", detection_object_count=100,
    )
    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout forward stall"))
    assert action.reason == "roundabout_very_close_reverse_clearance"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.72)


def test_roundabout_very_close_forward_stall_uses_short_reverse_clearance(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.blocked_frames = 139
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.58,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.01, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.58,
        lidar_blockage_ratio=0.94, lidar_center_blockage_ratio=0.94,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=42, lidar_right_density=0, lidar_center_density=33,
        lidar_open_side="right", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout forward stall")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_very_close_reverse_clearance"
    assert action.reverse is True
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias > 0.0


def test_roundabout_global_long_loop_close_obstacle_reverses_before_final_commit(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.roundabout_long_loop_frames = 320
    planner.blocked_frames = 118
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.46,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.01, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.46,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=50, lidar_right_density=0, lidar_center_density=125,
        lidar_open_side="right", detection_object_count=100,
    )

    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "route63 long loop"))

    assert action.reason == "roundabout_global_close_obstacle_reverse_clearance"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.72)
    assert action.steer_bias > 0.0


def test_roundabout_global_long_loop_close_obstacle_commits_when_already_reversing(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.roundabout_long_loop_frames = 420
    planner.blocked_frames = 43
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.55,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=-1.18, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.55,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=66, lidar_right_density=0, lidar_center_density=350,
        lidar_open_side="right", detection_object_count=100,
    )

    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "route63 reversing near close obstacle"))

    assert action.reason == "roundabout_global_close_obstacle_post_reverse_commit"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.82)
    assert action.steer_bias < 0.0


def test_roundabout_global_long_loop_close_obstacle_reverses_again_when_post_commit_stalls(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.roundabout_long_loop_frames = 420
    planner.blocked_frames = 43
    planner.roundabout_post_reverse_forward_frames = 12
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.95,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.003, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.95,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=66, lidar_right_density=0, lidar_center_density=350,
        lidar_open_side="right", detection_object_count=100,
    )

    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "route63 stalled post reverse"))

    assert action.reason == "roundabout_global_close_obstacle_post_reverse_stall_backout"
    assert action.reverse is True
    assert action.throttle_floor == pytest.approx(0.78)
    assert action.steer_bias > 0.0


def test_roundabout_global_long_loop_close_obstacle_commits_after_reverse(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.roundabout_long_loop_frames = 320
    planner.blocked_frames = 118
    planner.roundabout_post_reverse_forward_frames = 12
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.46,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.02, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.46,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_left_density=50, lidar_right_density=0, lidar_center_density=125,
        lidar_open_side="right", detection_object_count=100,
    )

    action = planner.plan(features, mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "route63 post reverse"))

    assert action.reason == "roundabout_global_close_obstacle_post_reverse_commit"
    assert action.reverse is False
    assert action.throttle_floor == pytest.approx(0.82)
    assert action.steer_bias < 0.0


def test_roundabout_very_close_balanced_preempts_near_speed_cap(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.62,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=1.02, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.62,
        lidar_blockage_ratio=0.39, lidar_center_blockage_ratio=0.72,
        lidar_left_blockage_ratio=0.17, lidar_right_blockage_ratio=0.0,
        lidar_left_density=6, lidar_right_density=0, lidar_center_density=25,
        lidar_open_side="balanced", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout balanced speed cap edge")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_very_close_open_side_forward_push"
    assert action.throttle_floor == pytest.approx(0.58)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(-0.30)


def test_roundabout_very_close_balanced_uses_memory_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.66,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.0, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.66,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=0.20,
        lidar_left_blockage_ratio=0.5, lidar_right_blockage_ratio=0.5,
        lidar_open_side="balanced", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout balanced close")
    action = planner.plan(features, estimate)
    assert action.reason == "roundabout_very_close_open_side_forward_push"
    assert action.throttle_floor == pytest.approx(0.50)
    assert action.steer_bias == pytest.approx(-0.26)


def test_students_far_red_deadlock_releases_after_short_hold(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig(); cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    system.red_final_clamp_hold_frames = 8
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=True, front_vehicle_distance=None, front_pedestrian_distance=None,
        front_obstacle_distance=None, red_light_active=True, red_stop_distance=9.1, ego_speed=0.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=None, lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate("four_students_crossing_the_road", 1.0, "APPROACH", "forced test")
    ctrl = system.process(Control(throttle=0.0, brake=1.0, steer=0.02), {}, {}, 0.0)
    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)


def test_roundabout_very_close_open_side_forward_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=2.55,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.02, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=2.55,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_open_side="right", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout very close")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_very_close_open_side_forward_push"
    assert action.throttle_floor == pytest.approx(0.50)
    assert action.steer_bias == pytest.approx(-0.26)


def test_roundabout_low_conf_center_blockage_uses_open_side_push(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=4.15,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.62, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=4.15,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_open_side="right", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout low conf center")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_low_conf_open_side_forward_push"
    assert action.throttle_floor == pytest.approx(0.52)
    assert action.steer_bias == pytest.approx(-0.22)


def test_roundabout_close_static_open_side_pushes_before_long_wait(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    planner.blocked_frames = 8
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=5.65,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.75, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=5.65,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=0.52,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_open_side="right", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout close side gap")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_close_static_progress_side_push"
    assert action.throttle_floor == pytest.approx(0.38)
    assert action.steer_bias == pytest.approx(-0.30)


def test_roundabout_near_side_blockage_pushes_instead_of_pre_stop(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0, front_clear=False, front_obstacle_distance=7.35,
        front_vehicle_distance=None, front_pedestrian_distance=None, red_stop_distance=None,
        red_light_active=False, ego_speed=0.95, junction_like=True, route_curvature=3.0,
        lidar_available=True, lidar_stale=False, lidar_front_distance=7.35,
        lidar_blockage_ratio=1.0, lidar_center_blockage_ratio=0.60,
        lidar_left_blockage_ratio=1.0, lidar_right_blockage_ratio=0.0,
        lidar_open_side="right", detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout near side blockage")

    action = planner.plan(features, estimate)

    assert action.reason == "roundabout_near_side_blockage_forward_push"
    assert action.throttle_floor == pytest.approx(0.54)
    assert action.brake_cap == pytest.approx(0.0)
    assert action.steer_bias == pytest.approx(-0.20)


def test_roundabout_static_obstacle_pre_stop_limits_steer_near_obstacle(monkeypatch):
    monkeypatch.setenv("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "1")
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    planner.roundabout_context_frames = 20
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_obstacle_distance=9.5,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        risk_level=1,
        ego_speed=2.8,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=9.5,
        lidar_center_blockage_ratio=1.0,
        lidar_open_side="right",
        detection_object_count=100,
    )
    estimate = mod.ScenarioEstimate("roundabout", 1.0, "APPROACH", "roundabout static route guard")
    action = planner.plan(features, estimate)
    assert action.reason == "roundabout_static_obstacle_pre_stop"
    assert action.brake >= 0.46
    assert action.throttle_cap == pytest.approx(0.0)
    assert action.steer_limit == pytest.approx(0.06)


def test_roundabout_clear_passthrough_speed_cap_overrides_raw_throttle(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "roundabout")
    mod = load_module(monkeypatch)
    cfg = mod.AuxiliaryConfig()
    cfg.max_aux_latency_ms = 100000.0
    system = mod.CVCIAuxiliarySystem(cfg)
    raw = Control(steer=0.08, throttle=1.0, brake=0.0)
    features = mod.AuxFeatures(
        confidence=1.0,
        ego_speed=5.2,
        front_clear=True,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=None,
        red_light_active=False,
        red_stop_distance=None,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=28.0,
        lidar_center_blockage_ratio=0.0,
    )
    system.perception.update = lambda model_detection, tick_data, timestamp: {}
    system.feature_builder.build = lambda observation, tick_data: features
    system.recognizer.recognize = lambda built_features: mod.ScenarioEstimate(
        macro_scenario="roundabout",
        confidence=1.0,
        phase="NORMAL",
        reason="forced test",
    )
    system.rule_planner.plan = lambda built_features, estimate: mod.PlannerAction(False, "NORMAL", reason="clear")
    ctrl = system.process(raw, {}, {}, 0.0)
    assert system.last_debug.get("reason") == "roundabout_clear_passthrough_speed_cap", system.last_debug
    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.58
    assert abs(ctrl.steer) <= 0.10



def test_crazy_bike_rule_holds_brake_for_decelerate_response(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    monkeypatch.setenv("CVCI_CRAZY_BIKE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class CrazyBikeRiskPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [{"class_name": "bicycle", "score": 0.06, "box_lidar": {"x": 4.2, "y": 0.2}}],
                "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "front_distance": None, "front_blocked": False,
                                   "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0,
                                   "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = CrazyBikeRiskPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.2), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 4.4}, 0.1)

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.70
    assert abs(ctrl.steer) <= 0.05
    assert system.last_debug["reason"] == "crazy_bike_decelerate_response_hold"


def test_crazy_bike_rule_does_not_brake_before_bike_window(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    monkeypatch.setenv("CVCI_CRAZY_BIKE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()

    class RedOnlyPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": True, "distance": 4.0},
                "lidar_geometry": {"available": True, "front_distance": None, "front_blocked": False,
                                   "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0,
                                   "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = RedOnlyPerception()
    ctrl = system.process(Control(throttle=0.6, brake=0.0, steer=0.0), {"frame": 1, "timestamp": 0.1, "objects": [], "map_objects": []}, {"speed": 4.0}, 0.1)

    assert system.last_debug["reason"] != "crazy_bike_decelerate_response_hold"
    assert system.crazy_bike_decelerate_frames == 0
    assert not system.crazy_bike_decelerate_done


def test_crazy_bike_rule_accelerates_after_decelerate_response(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    monkeypatch.setenv("CVCI_CRAZY_BIKE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.crazy_bike_decelerate_done = True
    system.crazy_bike_resume_frames = 24

    class CrazyBikeClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "front_distance": 18.0, "front_blocked": False,
                                   "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0,
                                   "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = CrazyBikeClearPerception()
    ctrl = system.process(Control(throttle=0.0, brake=0.8, steer=0.0), {"frame": 2, "timestamp": 0.2, "objects": [], "map_objects": []}, {"speed": 2.0}, 0.2)

    assert ctrl.throttle >= 0.78
    assert ctrl.brake == pytest.approx(0.0)
    assert abs(ctrl.steer) <= 0.05
    assert system.last_debug["reason"] == "crazy_bike_resume_route_accelerate"



def test_crazy_bike_resume_speed_guard_prevents_route_deviation(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "four_students_crossing_the_road")
    monkeypatch.setenv("CVCI_CRAZY_BIKE_RULE_ENABLED", "1")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.crazy_bike_decelerate_done = True
    system.crazy_bike_resume_frames = 24

    class CrazyBikeClearPerception:
        def update(self, model_detection, tick_data, timestamp):
            return {
                "objects": [], "map_objects": [], "tracked_objects": [], "road_topology": {},
                "traffic_light_state": {"active": False, "distance": None},
                "lidar_geometry": {"available": True, "front_distance": 20.0, "front_blocked": False,
                                   "corridor_blockage_ratio": 0.0, "center_blockage_ratio": 0.0,
                                   "left_blockage_ratio": 0.0, "right_blockage_ratio": 0.0, "open_side": "unknown"},
            }

    system.perception = CrazyBikeClearPerception()
    ctrl = system.process(Control(throttle=1.0, brake=0.0, steer=0.2), {"frame": 3, "timestamp": 0.3, "objects": [], "map_objects": []}, {"speed": 8.0}, 0.3)

    assert ctrl.throttle == pytest.approx(0.0)
    assert ctrl.brake >= 0.22
    assert abs(ctrl.steer) <= 0.05
    assert system.last_debug["reason"] == "crazy_bike_resume_speed_guard"



def test_reverse_vehicle_far_vehicle_keepalive_releases_safe_stall(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    monkeypatch.delenv("CVCI_REVERSE_VEHICLE_RULE_ENABLED", raising=False)
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=9.5,
        front_vehicle_ttc=18.0,
        front_pedestrian_distance=None,
        front_obstacle_distance=10.05,
        reversing_vehicle_evidence=True,
        ego_speed=-0.003,
        red_light_active=False,
        red_stop_distance=None,
        lidar_blockage_ratio=1.0,
        lidar_center_blockage_ratio=1.0,
        lidar_left_blockage_ratio=1.0,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 1.0, "APPROACH", "route112 far vehicle stall")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_far_vehicle_keepalive"
    assert action.brake_cap == pytest.approx(0.0)
    assert action.throttle_floor >= 0.50
    assert action.reverse is False


def test_reverse_vehicle_observed_buffer_brake(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig())
    features = mod.AuxFeatures(
        confidence=0.8,
        front_clear=False,
        front_vehicle_distance=8.0,
        front_vehicle_closing_speed=0.2,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=5.0,
        lidar_center_blockage_ratio=0.40,
    )
    estimate = mod.ScenarioEstimate("reverse_vehicle", 0.8, "YIELD_OR_BRAKE", "observed front vehicle conflict")

    action = planner.plan(features, estimate)

    assert action.reason == "reverse_vehicle_observed_buffer_brake"
    assert action.throttle_cap == 0.0
    assert action.brake == pytest.approx(0.62)
    assert planner.reverse_vehicle_brake_frames == 45


def test_construction_vehicle_open_side_push_is_conservative(monkeypatch):
    mod = load_module(monkeypatch)
    planner = mod.ScenarioRulePlanner(mod.AuxiliaryConfig(lidar_open_side_nudge_enabled=True))
    features = mod.AuxFeatures(
        confidence=0.9,
        front_clear=False,
        front_obstacle_distance=4.8,
        front_vehicle_distance=8.5,
        front_vehicle_ttc=4.0,
        front_pedestrian_distance=None,
        red_stop_distance=None,
        red_light_active=False,
        ego_speed=0.2,
        lidar_available=True,
        lidar_stale=False,
        lidar_front_distance=4.8,
        lidar_blockage_ratio=0.92,
        lidar_center_blockage_ratio=0.90,
        lidar_left_blockage_ratio=0.92,
        lidar_right_blockage_ratio=0.08,
        lidar_open_side="right",
    )
    estimate = mod.ScenarioEstimate("trucks_encountered_during_construction", 0.9, "PREPARE", "construction vehicle/open side")

    action = planner.plan(features, estimate)

    assert action.reason == "construction_vehicle_open_side_push_release"
    assert action.target_speed == pytest.approx(1.2)
    assert action.throttle_cap == pytest.approx(0.34)
    assert action.steer_limit == pytest.approx(0.42)
    assert action.steer_bias == pytest.approx(0.58)



def test_reverse_vehicle_close_red_release_allows_low_blockage_static_obstacle(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 8
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=3.84,
        ego_speed=0.0,
        red_light_active=True,
        red_stop_distance=2.38,
        lidar_blockage_ratio=0.11,
        lidar_center_blockage_ratio=0.10,
        lidar_left_blockage_ratio=0.12,
        lidar_right_blockage_ratio=0.08,
        lidar_open_side="balanced",
    )
    monkeypatch.setattr(system.feature_builder, "build", lambda observation, tick_data: features)

    ctrl = system.process(Control(throttle=0.0, brake=0.2), {"objects": [], "map_objects": []}, {"speed": 0.0}, 1.0)

    assert system.last_debug["reason"] == "reverse_vehicle_route_prior_close_red_release"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle >= 0.52
    assert ctrl.reverse is False


def test_reverse_vehicle_near_line_static_red_release_accepts_center_blockage(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 64
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=3.08,
        ego_speed=0.016,
        red_light_active=True,
        red_stop_distance=2.12,
        lidar_blockage_ratio=0.475,
        lidar_center_blockage_ratio=0.94,
        lidar_left_blockage_ratio=0.14,
        lidar_right_blockage_ratio=0.0,
        lidar_open_side="balanced",
    )
    monkeypatch.setattr(system.feature_builder, "build", lambda observation, tick_data: features)

    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=0.0), {"objects": [], "map_objects": []}, {"speed": 0.016}, 1.0)

    assert system.last_debug["reason"] == "reverse_vehicle_route_prior_near_line_static_red_release"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle >= 0.82
    assert ctrl.reverse is False


def test_reverse_vehicle_near_line_static_red_release_overrides_clamp(monkeypatch):
    monkeypatch.setenv("CVCI_ALLOW_ROUTE_PRIOR", "1")
    monkeypatch.setenv("CVCI_FORCE_MACRO_SCENARIO", "reverse_vehicle")
    mod = load_module(monkeypatch)
    system = mod.CVCIAuxiliarySystem()
    system.red_final_clamp_hold_frames = 4
    features = mod.AuxFeatures(
        confidence=1.0,
        front_clear=False,
        front_vehicle_distance=None,
        front_pedestrian_distance=None,
        front_obstacle_distance=2.47,
        ego_speed=0.0,
        red_light_active=True,
        red_stop_distance=2.09,
        lidar_blockage_ratio=0.6,
        lidar_center_blockage_ratio=0.6,
        lidar_left_blockage_ratio=0.6,
        lidar_right_blockage_ratio=0.6,
        lidar_open_side="balanced",
    )
    monkeypatch.setattr(system.feature_builder, "build", lambda observation, tick_data: features)

    ctrl = system.process(Control(throttle=0.0, brake=0.2, steer=0.0), {"objects": [], "map_objects": []}, {"speed": 0.0}, 1.0)

    assert system.last_debug["reason"] == "reverse_vehicle_route_prior_near_line_static_red_release"
    assert ctrl.brake == pytest.approx(0.0)
    assert ctrl.throttle >= 0.82
    assert abs(ctrl.steer) <= 0.02
    assert ctrl.reverse is False
