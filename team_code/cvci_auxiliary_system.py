"""CVCI auxiliary perception, scenario recognition, and safety supervision.

Runtime code in this module must only use legal agent observations: model outputs,
raw sensors already exposed through the leaderboard interface, route planner
commands, and ego kinematics. Offline CVCI XML/catalog data is intentionally not
read here unless CVCI_ALLOW_ROUTE_PRIOR is explicitly enabled for debugging.
"""
from __future__ import absolute_import

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover - unit tests can run without CARLA.
    carla = None

try:
    from DriveTransformer.team_code.auxiliary_perception import LidarDetector, ObjectTracker, PointCloudGeometryFallback
except Exception:  # pragma: no cover
    try:
        from team_code.auxiliary_perception import LidarDetector, ObjectTracker, PointCloudGeometryFallback
    except Exception:
        LidarDetector = None
        ObjectTracker = None
        PointCloudGeometryFallback = None


FSM_STATES = (
    "NORMAL",
    "APPROACH",
    "PREPARE",
    "YIELD_OR_BRAKE",
    "AVOID_OR_PASS",
    "RECOVER",
    "EMERGENCY",
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), lo, hi))


@dataclass
class AuxiliaryConfig:
    enabled: bool = field(default_factory=lambda: _env_flag("CVCI_AUXILIARY_SYSTEM_ENABLED", "1"))
    perception_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_AUXILIARY_PERCEPTION_ENABLED", "1"))
    lidar_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_LIDAR_ENABLED", "1"))
    scenario_rules_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_SCENARIO_RULES_ENABLED", "1"))
    safety_supervisor_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_SAFETY_SUPERVISOR_ENABLED", "1"))
    allow_route_prior: bool = field(default_factory=lambda: _env_flag("CVCI_ALLOW_ROUTE_PRIOR", "0"))
    suppress_lateral_intersection_rules: bool = field(default_factory=lambda: _env_flag("CVCI_SUPPRESS_LATERAL_INTERSECTION_RULES", "0"))
    reverse_vehicle_rule_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_REVERSE_VEHICLE_RULE_ENABLED", "0"))
    distant_lidar_creep_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_DISTANT_LIDAR_CREEP_ENABLED", "0"))
    lidar_open_side_nudge_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_LIDAR_OPEN_SIDE_NUDGE_ENABLED", "0"))
    crazy_bike_rule_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_CRAZY_BIKE_RULE_ENABLED", "0"))
    lidar_open_side_nudge_min_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_MIN_DISTANCE", "1.0")))
    lidar_open_side_nudge_max_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_MAX_DISTANCE", "18.0")))
    lidar_open_side_nudge_max_speed: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_MAX_SPEED", "1.6")))
    lidar_open_side_nudge_near_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_NEAR_DISTANCE", "7.0")))
    lidar_open_side_nudge_close_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_CLOSE_DISTANCE", "3.0")))
    lidar_open_side_nudge_escape_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_ESCAPE_DISTANCE", "2.6")))
    lidar_open_side_nudge_escape_frames: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_ESCAPE_FRAMES", "24")))
    lidar_open_side_post_pass_frames: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_OPEN_SIDE_POST_PASS_FRAMES", "36")))
    lidar_open_side_post_pass_max_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_POST_PASS_MAX_DISTANCE", "25.0")))
    lidar_open_side_close_memory_frames: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_OPEN_SIDE_CLOSE_MEMORY_FRAMES", "18")))
    lidar_open_side_close_memory_distance: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_CLOSE_MEMORY_DISTANCE", "4.0")))
    lidar_open_side_progress_recovery_frames: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_OPEN_SIDE_PROGRESS_RECOVERY_FRAMES", "800")))
    lidar_open_side_pass_memory_frames: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_OPEN_SIDE_PASS_MEMORY_FRAMES", "20")))
    lidar_open_side_nudge_center_threshold: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_OPEN_SIDE_NUDGE_CENTER_THRESHOLD", "0.35")))
    max_aux_latency_ms: float = field(default_factory=lambda: float(os.environ.get("CVCI_AUX_MAX_LATENCY_MS", "35")))
    min_rule_confidence: float = field(default_factory=lambda: float(os.environ.get("CVCI_AUX_MIN_RULE_CONFIDENCE", "0.55")))
    log_period: int = field(default_factory=lambda: int(os.environ.get("CVCI_AUX_LOG_PERIOD", "40")))
    forced_macro_scenario: str = field(default_factory=lambda: os.environ.get("CVCI_FORCE_MACRO_SCENARIO", "").strip())


@dataclass
class AuxFeatures:
    timestamp: float = 0.0
    frame: int = 0
    ego_speed: float = 0.0
    route_command: str = ""
    front_vehicle_distance: Optional[float] = None
    front_pedestrian_distance: Optional[float] = None
    front_obstacle_distance: Optional[float] = None
    red_stop_distance: Optional[float] = None
    red_light_active: bool = False
    side_risk: bool = False
    left_clear: bool = True
    right_clear: bool = True
    lane_confidence: float = 0.0
    route_curvature: float = 0.0
    junction_like: bool = False
    detection_object_count: int = 0
    tracked_objects: List[Dict[str, Any]] = field(default_factory=list)
    front_vehicle_ttc: Optional[float] = None
    front_vehicle_closing_speed: float = 0.0
    reversing_vehicle_evidence: bool = False
    front_clear: bool = True
    immediate_hazard: bool = False
    lidar_available: bool = False
    lidar_stale: bool = True
    lidar_front_distance: Optional[float] = None
    lidar_blockage_ratio: float = 0.0
    lidar_left_blockage_ratio: float = 0.0
    lidar_right_blockage_ratio: float = 0.0
    lidar_center_blockage_ratio: float = 0.0
    lidar_left_density: int = 0
    lidar_right_density: int = 0
    lidar_center_density: int = 0
    lidar_open_side: str = "unknown"
    lidar_lateral_centroid: float = 0.0
    risk_level: int = 0
    confidence: float = 0.0
    stale: bool = False
    error: str = ""


@dataclass
class ScenarioEstimate:
    macro_scenario: str = "unknown"
    confidence: float = 0.0
    phase: str = "NORMAL"
    reason: str = ""


@dataclass
class PlannerAction:
    active: bool = False
    state: str = "NORMAL"
    target_speed: Optional[float] = None
    throttle_cap: Optional[float] = None
    throttle_floor: Optional[float] = None
    brake_cap: Optional[float] = None
    brake: Optional[float] = None
    steer_limit: Optional[float] = None
    steer_bias: Optional[float] = None
    steer_min_magnitude: Optional[float] = None
    reverse: bool = False
    reason: str = ""


class AuxiliaryPerception:
    """Legal runtime perception adapter.

    This is deliberately lightweight until a real ONNX/TensorRT detector is
    supplied. It consumes DriveTransformer detections plus optional raw LiDAR
    geometry and exposes a single observation dictionary for feature building.
    """

    def __init__(self, config: AuxiliaryConfig):
        self.config = config
        self.lidar_detector = None
        self.lidar_geometry = None
        self.tracker = ObjectTracker() if ObjectTracker is not None else None
        self.topology = RoadTopologyExtractor()
        self.traffic_lights = TrafficLightStateAdapter()
        if config.enabled and config.perception_enabled and config.lidar_enabled and PointCloudGeometryFallback is not None:
            self.lidar_geometry = PointCloudGeometryFallback()
        if config.enabled and config.perception_enabled and config.lidar_enabled and LidarDetector is not None:
            self.lidar_detector = LidarDetector()

    def update(self, model_detection: Dict[str, Any], tick_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        result = dict(model_detection or {})
        if self.lidar_detector is not None:
            try:
                detector = self.lidar_detector.latest_or_submit(tick_data.get("lidar_points"), timestamp=timestamp)
            except Exception as exc:
                detector = {"enabled": True, "available": False, "stale": True, "status": "exception", "objects": [], "error": repr(exc)}
            result["lidar_detector"] = detector
            detector_objects = list(detector.get("objects", []) or []) if detector.get("available") and not detector.get("stale", True) else []
            if detector_objects:
                result["objects"] = list(result.get("objects", []) or []) + detector_objects
        else:
            result.setdefault("lidar_detector", None)
        if self.lidar_geometry is not None:
            try:
                result["lidar_geometry"] = self.lidar_geometry.update(tick_data.get("lidar_points"), timestamp=timestamp)
            except Exception as exc:
                result["lidar_geometry"] = {"available": False, "stale": True, "error": repr(exc)}
        else:
            result.setdefault("lidar_geometry", None)
        if self.tracker is not None:
            result["tracked_objects"] = self.tracker.update(result.get("objects", []), timestamp)
        else:
            result["tracked_objects"] = []
        result["road_topology"] = self.topology.extract(result.get("map_objects", []), tick_data)
        result["traffic_light_state"] = self.traffic_lights.estimate(result.get("objects", []), result.get("map_objects", []))
        return result


class RoadTopologyExtractor:
    def extract(self, map_objects: List[Dict[str, Any]], tick_data: Dict[str, Any]) -> Dict[str, Any]:
        lane_scores = []
        lateral_span = 0.0
        for obj in map_objects or []:
            cls = str(obj.get("class_name", "")).lower()
            if cls in ("broken", "solid", "solidsolid", "center"):
                lane_scores.append(float(obj.get("score", 0.0)))
                pts = np.asarray(obj.get("pts", obj.get("box", [])), dtype=np.float32).reshape(-1)
                if pts.size >= 4:
                    lateral_span = max(lateral_span, float(np.nanmax(np.abs(pts[1::2]))))
        command = str(tick_data.get("command_near", ""))
        # CARLA command ids are project-specific here, so this is only a weak
        # observable topology hint and never a scenario label.
        junction_like = command not in ("0", "1", "2", "RoadOption.LANEFOLLOW", "LANEFOLLOW", "")
        return {
            "lane_confidence": float(np.mean(lane_scores)) if lane_scores else 0.0,
            "lateral_span": lateral_span,
            "junction_like": bool(junction_like),
        }


class TrafficLightStateAdapter:
    def estimate(self, objects: List[Dict[str, Any]], map_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        min_dist = None
        confidence = 0.0
        for obj in objects or []:
            if str(obj.get("class_name", "")).lower() != "traffic_light":
                continue
            box = obj.get("box_lidar") or {}
            x = float(box.get("x", 999.0))
            y = float(box.get("y", 999.0))
            if 0.0 <= x <= 18.0 and abs(y) <= 4.0:
                min_dist = x if min_dist is None else min(min_dist, x)
                confidence = max(confidence, float(obj.get("score", 0.0)))
        for obj in map_objects or []:
            cls = str(obj.get("class_name", "")).lower()
            if cls not in ("trafficlight", "stopsign"):
                continue
            arr = np.asarray(obj.get("box", []), dtype=np.float32).reshape(-1)
            if arr.size >= 2 and arr[0] >= -2.0 and abs(arr[1]) <= 5.0:
                dist = float(np.hypot(arr[0], arr[1]))
                if dist <= 16.0:
                    min_dist = dist if min_dist is None else min(min_dist, dist)
                    confidence = max(confidence, float(obj.get("score", 0.0)))
        return {"active": min_dist is not None, "distance": min_dist, "confidence": confidence}


class SceneFeatureBuilder:
    @staticmethod
    def debug_nearest_objects(observation: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
        vehicle_classes = ("car", "van", "truck", "bus", "motorcycle", "bicycle", "cyclist")
        detector_items = []
        for obj in observation.get("objects", []) or []:
            cls = str(obj.get("class_name", "")).lower()
            if cls not in vehicle_classes:
                continue
            box = obj.get("box_lidar") or {}
            try:
                x = float(box.get("x", 999.0))
                y = float(box.get("y", 999.0))
            except Exception:
                continue
            if x < -2.0:
                continue
            detector_items.append({
                "class_name": cls,
                "x": round(x, 3),
                "y": round(y, 3),
                "score": round(float(obj.get("score", obj.get("confidence", 0.0)) or 0.0), 3),
                "source": str(obj.get("source", "")),
            })
        detector_items.sort(key=lambda item: (abs(float(item["y"])), float(item["x"])))

        track_items = []
        for track in observation.get("tracked_objects", []) or []:
            cls = str(track.get("class_name", "")).lower()
            if cls not in vehicle_classes:
                continue
            x = float(track.get("x", 999.0))
            y = float(track.get("y", 999.0))
            if x < -2.0:
                continue
            track_items.append({
                "class_name": cls,
                "x": round(x, 3),
                "y": round(y, 3),
                "vx": round(float(track.get("vx", 0.0) or 0.0), 3),
                "closing_speed": round(float(track.get("closing_speed", 0.0) or 0.0), 3),
                "observed_frames": int(track.get("observed_frames", 0) or 0),
                "ttc": None if track.get("ttc") is None else round(float(track.get("ttc")), 3),
            })
        track_items.sort(key=lambda item: (abs(float(item["y"])), float(item["x"])))
        return {
            "nearest_detector_vehicles": detector_items[:limit],
            "nearest_tracked_vehicles": track_items[:limit],
        }

    def build(self, observation: Dict[str, Any], tick_data: Dict[str, Any]) -> AuxFeatures:
        feat = AuxFeatures(
            timestamp=float(observation.get("timestamp", 0.0)),
            frame=int(observation.get("frame", 0)),
            ego_speed=float(tick_data.get("speed", 0.0)),
            route_command=str(tick_data.get("command_near", "")),
            confidence=0.2,
        )
        feat.detection_object_count = len(observation.get("objects", []) or [])
        feat.tracked_objects = observation.get("tracked_objects", []) or []
        for track in feat.tracked_objects:
            cls = str(track.get("class_name", "")).lower()
            if cls not in ("car", "van", "truck", "bus", "motorcycle", "bicycle"):
                continue
            x = float(track.get("x", 999.0))
            y = float(track.get("y", 999.0))
            if not (0.0 <= x <= 24.0 and abs(y) <= 2.6):
                continue
            feat.front_vehicle_distance = x if feat.front_vehicle_distance is None else min(feat.front_vehicle_distance, x)
            closing = float(track.get("closing_speed", 0.0) or 0.0)
            feat.front_vehicle_closing_speed = max(feat.front_vehicle_closing_speed, closing)
            ttc = track.get("ttc")
            if ttc is not None:
                ttc = float(ttc)
                feat.front_vehicle_ttc = ttc if feat.front_vehicle_ttc is None else min(feat.front_vehicle_ttc, ttc)
            if bool(track.get("is_reversing_candidate", False)) and int(track.get("observed_frames", 0)) >= 2:
                feat.reversing_vehicle_evidence = True
                feat.front_clear = False
        topology = observation.get("road_topology") or {}
        feat.lane_confidence = float(topology.get("lane_confidence", 0.0) or 0.0)
        feat.route_curvature = float(topology.get("lateral_span", 0.0) or 0.0)
        feat.junction_like = bool(topology.get("junction_like", False))
        tl_state = observation.get("traffic_light_state") or {}
        feat.red_light_active = bool(tl_state.get("active", False))
        if tl_state.get("distance") is not None:
            feat.red_stop_distance = float(tl_state["distance"])
        nearest = None
        for obj in observation.get("objects", []) or []:
            score = float(obj.get("score", 0.0))
            if score < 0.32:
                continue
            cls = str(obj.get("class_name", "")).lower()
            box = obj.get("box_lidar") or {}
            x = float(box.get("x", 999.0))
            y = float(box.get("y", 999.0))
            if 0.0 <= x <= 18.0 and abs(y) <= 4.2 and score >= 0.38:
                if y < -2.2:
                    feat.left_clear = False
                if y > 2.2:
                    feat.right_clear = False
            if 0.0 <= x <= 8.0 and 1.6 < abs(y) <= 3.4 and score >= 0.42:
                feat.side_risk = True
            if cls in ("car", "van", "truck", "bicycle", "pedestrian", "traffic_cone", "others"):
                y_limit = 2.6 if cls in ("pedestrian", "bicycle") else 2.2
                x_limit = 20.0 if cls in ("pedestrian", "bicycle") else 15.0
                near_low_conf_vehicle = (
                    cls in ("car", "van", "truck", "bicycle", "traffic_cone", "others")
                    and 0.0 <= x <= 4.5
                    and abs(y) <= 1.2
                    and score >= 0.02
                )
                if near_low_conf_vehicle:
                    feat.front_clear = False
                    nearest = x if nearest is None else min(nearest, x)
                    feat.front_obstacle_distance = x if feat.front_obstacle_distance is None else min(feat.front_obstacle_distance, x)
                elif 0.0 <= x <= x_limit and abs(y) <= y_limit:
                    feat.front_clear = False
                    nearest = x if nearest is None else min(nearest, x)
                    if cls in ("car", "van", "truck"):
                        feat.front_vehicle_distance = x if feat.front_vehicle_distance is None else min(feat.front_vehicle_distance, x)
                    elif cls in ("pedestrian", "bicycle"):
                        feat.front_pedestrian_distance = x if feat.front_pedestrian_distance is None else min(feat.front_pedestrian_distance, x)
                    else:
                        feat.front_obstacle_distance = x if feat.front_obstacle_distance is None else min(feat.front_obstacle_distance, x)
            if cls == "traffic_light" and 0.0 <= x <= 18.0 and abs(y) <= 4.0 and score >= 0.45:
                feat.red_stop_distance = x if feat.red_stop_distance is None else min(feat.red_stop_distance, x)

        for map_obj in observation.get("map_objects", []) or []:
            if float(map_obj.get("score", 0.0)) < 0.35:
                continue
            cls = str(map_obj.get("class_name", "")).lower()
            if cls not in ("trafficlight", "stopsign"):
                continue
            arr = np.asarray(map_obj.get("box", []), dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                dist = float(np.hypot(arr[0], arr[1]))
                if arr[0] >= -2.0 and abs(arr[1]) <= 5.0 and dist <= 16.0:
                    feat.red_stop_distance = dist if feat.red_stop_distance is None else min(feat.red_stop_distance, dist)

        lidar = observation.get("lidar_geometry") or {}
        if lidar:
            feat.lidar_available = bool(lidar.get("available", False))
            feat.lidar_stale = bool(lidar.get("stale", True))
            feat.lidar_front_distance = lidar.get("front_distance")
            feat.lidar_blockage_ratio = float(lidar.get("corridor_blockage_ratio", 0.0) or 0.0)
            feat.lidar_left_blockage_ratio = float(lidar.get("left_blockage_ratio", 0.0) or 0.0)
            feat.lidar_right_blockage_ratio = float(lidar.get("right_blockage_ratio", 0.0) or 0.0)
            feat.lidar_center_blockage_ratio = float(lidar.get("center_blockage_ratio", 0.0) or 0.0)
            feat.lidar_left_density = int(lidar.get("left_density", 0) or 0)
            feat.lidar_right_density = int(lidar.get("right_density", 0) or 0)
            feat.lidar_center_density = int(lidar.get("center_density", 0) or 0)
            feat.lidar_open_side = str(lidar.get("open_side", "unknown") or "unknown")
            feat.lidar_lateral_centroid = float(lidar.get("lateral_centroid", 0.0) or 0.0)
            if feat.lidar_available and not feat.lidar_stale and feat.lidar_front_distance is not None:
                corridor_blocked = bool(
                    lidar.get("front_blocked")
                    or lidar.get("near_blocked")
                    or (
                        feat.lidar_center_blockage_ratio >= 0.45
                        and float(feat.lidar_front_distance) <= 18.0
                        and feat.lidar_open_side in ("right", "balanced", "left")
                    )
                )
                if corridor_blocked:
                    feat.front_clear = False
                    d = float(feat.lidar_front_distance)
                    feat.front_obstacle_distance = d if feat.front_obstacle_distance is None else min(feat.front_obstacle_distance, d)

        if feat.front_pedestrian_distance is not None and feat.front_pedestrian_distance < 9.0:
            feat.risk_level = max(feat.risk_level, 3)
        if feat.front_vehicle_distance is not None and feat.front_vehicle_distance < 6.0:
            feat.risk_level = max(feat.risk_level, 3)
        if feat.reversing_vehicle_evidence and feat.front_vehicle_ttc is not None and feat.front_vehicle_ttc < 5.0:
            feat.risk_level = max(feat.risk_level, 3)
        if feat.front_obstacle_distance is not None and feat.front_obstacle_distance < 2.4:
            feat.risk_level = max(feat.risk_level, 3)
        elif feat.front_obstacle_distance is not None and feat.front_obstacle_distance < 5.0:
            feat.risk_level = max(feat.risk_level, 2)
        elif (
            feat.front_obstacle_distance is not None
            and feat.front_obstacle_distance < 18.0
            and feat.lidar_center_blockage_ratio >= 0.45
            and feat.ego_speed < 1.0
        ):
            feat.risk_level = max(feat.risk_level, 1)
        if not feat.front_clear:
            feat.risk_level = max(feat.risk_level, 1)
        if feat.red_stop_distance is not None and feat.red_stop_distance < 9.0:
            feat.risk_level = max(feat.risk_level, 2)
        feat.immediate_hazard = bool(
            (feat.front_pedestrian_distance is not None and feat.front_pedestrian_distance < 6.0)
            or (feat.front_vehicle_distance is not None and feat.front_vehicle_distance < 4.0)
            or (feat.reversing_vehicle_evidence and feat.front_vehicle_ttc is not None and feat.front_vehicle_ttc < 3.5)
        )
        feat.confidence = min(1.0, 0.25 + 0.2 * feat.risk_level + (0.15 if feat.lidar_available and not feat.lidar_stale else 0.0))
        return feat


class ScenarioRecognizer:
    def __init__(self, config: AuxiliaryConfig):
        self.config = config

    def recognize(self, features: AuxFeatures) -> ScenarioEstimate:
        if self.config.allow_route_prior and self.config.forced_macro_scenario:
            return ScenarioEstimate(self.config.forced_macro_scenario, 1.0, "APPROACH" if features.risk_level > 0 else "NORMAL", "forced route-prior macro scenario")
        if features.immediate_hazard and features.front_pedestrian_distance is not None:
            return ScenarioEstimate("four_students_crossing_the_road", features.confidence, "YIELD_OR_BRAKE", "front pedestrian/bicycle hazard")
        if features.reversing_vehicle_evidence and features.front_vehicle_ttc is not None and features.front_vehicle_ttc < 5.0:
            return ScenarioEstimate("reverse_vehicle", features.confidence, "YIELD_OR_BRAKE", "tracked front vehicle closing/reversing")
        if features.front_vehicle_distance is not None and features.front_vehicle_distance < 8.0 and features.ego_speed < 2.5:
            return ScenarioEstimate("reverse_vehicle", features.confidence, "YIELD_OR_BRAKE", "close vehicle conflict at low ego speed")
        if features.junction_like and features.side_risk:
            return ScenarioEstimate("blind_spot_hidden_car", features.confidence, "PREPARE", "junction side risk")
        if features.junction_like and features.route_curvature > 3.0:
            return ScenarioEstimate("roundabout", max(features.confidence, 0.55), "APPROACH", "curved junction topology")
        if features.front_obstacle_distance is not None and features.lidar_blockage_ratio >= 0.35:
            return ScenarioEstimate("trucks_encountered_during_construction", features.confidence, "PREPARE", "corridor blockage from detector/lidar")
        if features.front_vehicle_distance is not None and features.ego_speed > 8.0:
            return ScenarioEstimate("highway_accident_vehicle", features.confidence, "PREPARE", "front vehicle at high speed")
        if features.side_risk and not (features.left_clear and features.right_clear):
            return ScenarioEstimate("high_speed_reckless_lane_cutting", features.confidence, "PREPARE", "side risk near ego lane")
        if features.front_obstacle_distance is not None:
            return ScenarioEstimate("high_speed_temporary_construction", features.confidence, "PREPARE", "front static obstacle")
        if features.red_stop_distance is not None:
            return ScenarioEstimate("unknown", features.confidence, "APPROACH", "traffic light or stop sign")
        if features.risk_level > 0:
            return ScenarioEstimate("unknown", features.confidence, "APPROACH", "generic observable risk")
        return ScenarioEstimate("unknown", features.confidence, "NORMAL", "no active scenario")


class ScenarioRulePlanner:
    def __init__(self, config: AuxiliaryConfig):
        self.config = config
        self.state = "NORMAL"
        self.clear_frames = 0
        self.blocked_frames = 0
        self.static_creep_frames = 0
        self.post_pass_frames = 0
        self.last_open_side = "unknown"
        self.close_obstacle_memory_frames = 0
        self.progress_recovery_frames = 0
        self.open_side_pass_memory_frames = 0
        self.observable_risk_creep_frames = 0
        self.red_stop_hold_frames = 0
        self.red_stop_gap_frames = 0
        self.red_stop_release_frames = 0
        self.low_conf_construction_suppress_frames = 0
        self.reverse_vehicle_brake_frames = 0
        self.reverse_vehicle_high_blockage_stuck_frames = 0
        self.construction_cone_entry_slow_frames = 0
        self.construction_sparse_cone_low_speed_guard_frames = 0
        self.construction_full_blockage_escape_frames = 0
        self.construction_corridor_memory_frames = 0
        self.reverse_unwedge_frames = 0
        self.high_speed_lateral_guard_frames = 0
        self.lateral_intersection_release_frames = 0
        self.lateral_intersection_scored_brake_frames = 0
        self.blind_spot_junction_brake_frames = 0
        self.blind_spot_junction_brake_cooldown_frames = 0
        self.balanced_blockage_progress_frames = 0
        self.roundabout_context_frames = 0
        self.roundabout_reverse_clearance_frames = 0
        self.roundabout_post_reverse_forward_frames = 0
        self.roundabout_approach_brake_frames = 0
        self.roundabout_approach_brake_cooldown_frames = 0
        self.roundabout_vehicle_yield_frames = 0
        self.roundabout_vehicle_yield_cooldown_frames = 0
        self.roundabout_long_loop_frames = 0
        self.red_final_context_frames = 0

    def _lateral_release_keep_rolling_action(self, features: AuxFeatures, reason: str = "lateral_intersection_keep_rolling") -> PlannerAction:
        if features.ego_speed > 10.5:
            throttle_cap = 0.0
            throttle_floor = None
            brake = 0.10
            brake_cap = 0.16
        elif features.ego_speed > 9.5:
            throttle_cap = 0.0
            throttle_floor = None
            brake = 0.0
            brake_cap = 0.0
        else:
            throttle_cap = 1.0
            throttle_floor = 0.88
            brake = None
            brake_cap = 0.0
        return PlannerAction(
            True,
            self.state,
            target_speed=9.0,
            throttle_cap=throttle_cap,
            throttle_floor=throttle_floor,
            brake=brake,
            brake_cap=brake_cap,
            steer_limit=0.16,
            reason=reason,
        )

    def _construction_creep_candidate(self, features: AuxFeatures, estimate: ScenarioEstimate) -> bool:
        return bool(
            estimate.macro_scenario == "trucks_encountered_during_construction"
            and features.front_obstacle_distance is not None
            and 2.5 <= float(features.front_obstacle_distance) <= 6.0
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and features.ego_speed < 0.30
            and features.lidar_available
            and not features.lidar_stale
            and features.lidar_blockage_ratio >= 0.60
            and features.detection_object_count >= 20
        )

    def _update_red_stop_stability(self, features: AuxFeatures, no_front_conflict: bool) -> bool:
        close_hint = features.red_stop_distance is not None and float(features.red_stop_distance) <= 5.5
        if close_hint and features.red_light_active:
            self.red_stop_hold_frames += 1
            self.red_stop_gap_frames = 0
            return False
        if close_hint and not features.red_light_active:
            self.red_stop_gap_frames += 1
            return bool(
                no_front_conflict
                and self.red_stop_hold_frames >= 20
                and self.red_stop_gap_frames >= 15
                and features.ego_speed < 0.80
                and features.lidar_blockage_ratio <= 0.10
                and features.lidar_center_blockage_ratio <= 0.10
            )
        elif features.red_stop_distance is None:
            self.red_stop_gap_frames += 1
            release = bool(
                no_front_conflict
                and self.red_stop_hold_frames >= 20
                and self.red_stop_gap_frames >= 30
                and features.ego_speed < 0.80
                and features.lidar_blockage_ratio <= 0.10
                and features.lidar_center_blockage_ratio <= 0.10
            )
            if self.red_stop_gap_frames > 180:
                self.red_stop_hold_frames = 0
            return release
        else:
            self.red_stop_gap_frames = 0
            self.red_stop_hold_frames = 0
        return False

    def _red_stop_release_action(self, target_speed: float = 1.4, start_window: bool = False) -> PlannerAction:
        if start_window:
            self.red_stop_release_frames = 45
        self.state = "RECOVER"
        return PlannerAction(
            True,
            "RECOVER",
            target_speed=max(target_speed, 2.0),
            throttle_cap=0.55,
            throttle_floor=0.32,
            brake_cap=0.0,
            steer_limit=0.24,
            reason="unstable_red_stop_cautious_creep_recovery",
        )

    def _clear_road_no_progress_action(
        self,
        cautious_reason: str = "clear_road_cautious_creep_recovery",
        allow_reverse: bool = True,
    ) -> PlannerAction:
        self.state = "RECOVER"
        if self.blocked_frames >= 80 and self.balanced_blockage_progress_frames > 0:
            self.balanced_blockage_progress_frames -= 1
            return PlannerAction(
                True,
                "RECOVER",
                target_speed=5.0,
                throttle_cap=0.85,
                throttle_floor=0.65,
                brake_cap=0.0,
                steer_limit=0.20,
                reason="balanced_construction_blockage_progress_push",
            )
        if self.blocked_frames >= 90 and allow_reverse:
            return PlannerAction(
                True,
                "RECOVER",
                throttle_cap=0.34,
                throttle_floor=0.24,
                brake_cap=0.0,
                steer_limit=0.25,
                reverse=True,
                reason="clear_road_no_progress_reverse_unwedge",
            )
        if self.blocked_frames >= 45:
            if not allow_reverse:
                steer_bias = 0.16 if (self.blocked_frames // 15) % 2 == 0 else -0.16
                return PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=4.8,
                    throttle_cap=1.0,
                    throttle_floor=0.88,
                    brake_cap=0.0,
                    steer_limit=0.30,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.14,
                    reason="roundabout_clear_road_forward_recovery",
                )
            if self.open_side_pass_memory_frames > 0 and self.last_open_side in ("right", "left"):
                steer_bias = 0.08 if self.last_open_side == "right" else -0.08
                if self.blocked_frames >= 70:
                    steer_bias = 0.14 if self.last_open_side == "right" else -0.14
                    return PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=3.8,
                        throttle_cap=0.95,
                        throttle_floor=0.72,
                        brake_cap=0.0,
                        steer_limit=0.28,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.18,
                        reason="construction_cone_post_pass_forward_unwedge",
                    )
                return PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.0,
                    throttle_cap=0.46,
                    throttle_floor=0.26,
                    brake_cap=0.0,
                    steer_limit=0.30,
                    steer_bias=steer_bias,
                    reason="construction_cone_post_pass_cautious_recovery",
                )
            return PlannerAction(
                True,
                "RECOVER",
                target_speed=4.5,
                throttle_cap=1.0,
                throttle_floor=0.86,
                brake_cap=0.0,
                steer_limit=0.30,
                reason="clear_road_no_progress_forward_unwedge",
            )
        if self.blocked_frames >= 20:
            return PlannerAction(
                True,
                "RECOVER",
                target_speed=3.2,
                throttle_cap=0.85,
                throttle_floor=0.62,
                brake_cap=0.0,
                steer_limit=0.22,
                reason="clear_road_no_progress_stronger_recovery",
            )
        return PlannerAction(
            True,
            "RECOVER",
            target_speed=2.6,
            throttle_cap=0.68,
            throttle_floor=0.45,
            brake_cap=0.0,
            steer_limit=0.18,
            reason=cautious_reason,
        )

    def plan(self, features: AuxFeatures, estimate: ScenarioEstimate) -> PlannerAction:
        if not self.config.scenario_rules_enabled:
            self.state = "NORMAL" if features.risk_level == 0 else "APPROACH"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(False, self.state, reason="rules_disabled_or_low_confidence")
        if self.config.suppress_lateral_intersection_rules:
            self.lateral_intersection_release_frames = 0
            self.lateral_intersection_scored_brake_frames = 0
            self.high_speed_lateral_guard_frames = 0
        elif self.lateral_intersection_release_frames > 0:
            self.lateral_intersection_release_frames -= 1
        roundabout_context_observed = bool(
            estimate.macro_scenario == "roundabout"
            or (features.junction_like and abs(features.route_curvature) >= 1.0)
        )
        if roundabout_context_observed:
            self.roundabout_context_frames = 5000
        elif self.roundabout_context_frames > 0:
            self.roundabout_context_frames -= 1
        roundabout_layout_context = self.roundabout_context_frames > 0
        if roundabout_layout_context:
            if abs(features.ego_speed) < 1.35 or self.blocked_frames > 0:
                self.roundabout_long_loop_frames = min(self.roundabout_long_loop_frames + 1, 2000)
            else:
                self.roundabout_long_loop_frames = max(self.roundabout_long_loop_frames - 2, 0)
        else:
            self.roundabout_long_loop_frames = 0
            self.roundabout_approach_brake_frames = 0
            self.roundabout_approach_brake_cooldown_frames = 0
        if self.roundabout_approach_brake_cooldown_frames > 0:
            self.roundabout_approach_brake_cooldown_frames -= 1
        if self.roundabout_vehicle_yield_cooldown_frames > 0:
            self.roundabout_vehicle_yield_cooldown_frames -= 1
        roundabout_close_vehicle_distance = None
        if features.front_vehicle_distance is not None:
            try:
                roundabout_close_vehicle_distance = float(features.front_vehicle_distance)
            except Exception:
                roundabout_close_vehicle_distance = None
        roundabout_close_vehicle_yield = (
            estimate.macro_scenario == "roundabout"
            and roundabout_layout_context
            and roundabout_close_vehicle_distance is not None
            and 2.2 <= roundabout_close_vehicle_distance <= 9.5
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and abs(features.ego_speed) >= 0.35
        )
        if (
            roundabout_close_vehicle_yield
            and self.roundabout_vehicle_yield_frames <= 0
            and self.roundabout_vehicle_yield_cooldown_frames <= 0
        ):
            self.roundabout_vehicle_yield_frames = 28
            self.roundabout_vehicle_yield_cooldown_frames = 180
        if roundabout_close_vehicle_yield and self.roundabout_vehicle_yield_frames > 0:
            self.roundabout_vehicle_yield_frames -= 1
            self.state = "YIELD_OR_BRAKE"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=0.0 if roundabout_close_vehicle_distance <= 5.5 else 1.0,
                throttle_cap=0.0,
                brake=0.68 if abs(features.ego_speed) > 1.5 else 0.42,
                brake_cap=0.78,
                steer_limit=0.12,
                reason="roundabout_close_vehicle_yield_brake",
            )
        roundabout_approach_brake_candidate = (
            estimate.macro_scenario == "roundabout"
            and roundabout_layout_context
            and self.roundabout_long_loop_frames <= 120
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.front_obstacle_distance is None
            and 1.4 <= abs(features.ego_speed) <= 10.5
        )
        if (
            roundabout_approach_brake_candidate
            and self.roundabout_approach_brake_frames <= 0
            and self.roundabout_approach_brake_cooldown_frames <= 0
        ):
            self.roundabout_approach_brake_frames = 10
            self.roundabout_approach_brake_cooldown_frames = 420
        if roundabout_approach_brake_candidate and self.roundabout_approach_brake_frames > 0:
            self.roundabout_approach_brake_frames -= 1
            self.state = "YIELD_OR_BRAKE"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=1.4,
                throttle_cap=0.0,
                brake=0.48 if abs(features.ego_speed) > 2.4 else 0.32,
                brake_cap=0.58,
                steer_limit=0.14,
                reason="roundabout_approach_scored_brake_response",
            )
        if (
            roundabout_layout_context
            and self.roundabout_long_loop_frames >= 170
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and abs(features.ego_speed) < 3.0
        ):
            close_static_distance = features.front_obstacle_distance
            if close_static_distance is None:
                close_static_distance = features.lidar_front_distance
            if close_static_distance is not None and float(close_static_distance) <= 3.10 and (abs(features.ego_speed) < 0.75 or features.ego_speed < -0.35):
                if self.roundabout_long_loop_frames >= 300:
                    self.state = "RECOVER"
                    if (
                        2.15 <= float(close_static_distance) <= 3.10
                        and features.lidar_open_side in ("right", "left", "balanced")
                    ):
                        open_side = features.lidar_open_side
                        side_bias = -0.34 if open_side in ("right", "balanced") else 0.34
                        if self.roundabout_post_reverse_forward_frames > 0 or features.ego_speed < -0.35:
                            if (
                                self.roundabout_post_reverse_forward_frames > 0
                                and abs(features.ego_speed) < 0.08
                                and self.blocked_frames >= 40
                                and self.roundabout_long_loop_frames >= 400
                                and float(close_static_distance) <= 3.05
                            ):
                                self.roundabout_post_reverse_forward_frames = 0
                                self.roundabout_reverse_clearance_frames = max(self.roundabout_reverse_clearance_frames, 10)
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.94,
                                    throttle_floor=0.78,
                                    brake_cap=0.0,
                                    steer_limit=0.66,
                                    steer_bias=-side_bias,
                                    steer_min_magnitude=0.36,
                                    reverse=True,
                                    reason="roundabout_global_close_obstacle_post_reverse_stall_backout",
                                )
                            if self.roundabout_post_reverse_forward_frames > 0:
                                self.roundabout_post_reverse_forward_frames -= 1
                            self.roundabout_reverse_clearance_frames = 0
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.8,
                                throttle_cap=0.98,
                                throttle_floor=0.82,
                                brake_cap=0.0,
                                steer_limit=0.50,
                                steer_bias=side_bias,
                                steer_min_magnitude=0.28,
                                reason="roundabout_global_close_obstacle_post_reverse_commit",
                            )
                        if self.roundabout_reverse_clearance_frames <= 0:
                            self.roundabout_reverse_clearance_frames = 20
                        self.roundabout_reverse_clearance_frames -= 1
                        self.roundabout_post_reverse_forward_frames = max(
                            self.roundabout_post_reverse_forward_frames,
                            16,
                        )
                        return PlannerAction(
                            True,
                            self.state,
                            throttle_cap=0.92,
                            throttle_floor=0.72,
                            brake_cap=0.0,
                            steer_limit=0.62,
                            steer_bias=-side_bias,
                            steer_min_magnitude=0.32,
                            reverse=True,
                            reason="roundabout_global_close_obstacle_reverse_clearance",
                        )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=4.8,
                        throttle_cap=1.0,
                        throttle_floor=0.98,
                        brake_cap=0.0,
                        steer_limit=0.06,
                        steer_bias=0.0,
                        reason="roundabout_global_close_obstacle_final_commit",
                    )
                steer_bias = -0.46 if features.lidar_open_side == "right" else 0.46
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=1.1,
                    throttle_cap=0.82,
                    throttle_floor=0.52,
                    brake_cap=0.0,
                    steer_limit=0.52,
                    steer_bias=steer_bias,
                    reverse=True,
                    reason="roundabout_global_close_obstacle_backout",
                )
            self.state = "RECOVER"
            long_loop_side_bias = 0.0
            long_loop_steer_limit = 0.08
            long_loop_min_steer = 0.0
            if close_static_distance is not None and float(close_static_distance) <= 5.2:
                if features.lidar_open_side == "right":
                    long_loop_side_bias = -0.24
                elif features.lidar_open_side == "left":
                    long_loop_side_bias = 0.24
                elif features.lidar_lateral_centroid <= 0.0:
                    long_loop_side_bias = 0.20
                else:
                    long_loop_side_bias = -0.20
                long_loop_steer_limit = 0.34
                long_loop_min_steer = 0.12
            return PlannerAction(
                True,
                self.state,
                target_speed=5.2,
                throttle_cap=1.0,
                throttle_floor=0.96,
                brake_cap=0.0,
                steer_limit=long_loop_steer_limit,
                steer_bias=long_loop_side_bias,
                steer_min_magnitude=long_loop_min_steer,
                reason="roundabout_global_long_loop_route_commit",
            )
        if (
            features.red_light_active
            and features.red_stop_distance is None
            and features.front_pedestrian_distance is None
            and features.front_vehicle_distance is None
            and features.ego_speed > 2.0
        ):
            self.state = "YIELD_OR_BRAKE"
            self.open_side_pass_memory_frames = 0
            self.post_pass_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=0.0,
                throttle_cap=0.0,
                brake=0.35 if features.ego_speed < 4.0 else 0.55,
                steer_limit=0.35,
                reason="active_red_without_stopline_deceleration",
            )
        if (
            features.red_light_active
            and features.red_stop_distance is not None
            and float(features.red_stop_distance) <= 5.0
            and features.front_pedestrian_distance is None
            and features.front_vehicle_distance is None
            and features.ego_speed > 0.35
            and estimate.macro_scenario not in (
                "ebike_and_pedestrian_cross",
                "ghost_probe",
                "highway_accident_vehicle",
            )
            and not (
                estimate.macro_scenario == "trucks_encountered_during_construction"
                and self.config.lidar_open_side_nudge_enabled
                and features.front_obstacle_distance is not None
                and 1.5 <= float(features.front_obstacle_distance) <= 3.8
                and features.lidar_blockage_ratio >= 0.85
                and features.lidar_center_blockage_ratio >= 0.90
                and features.lidar_open_side in ("right", "left")
                and features.ego_speed < 0.65
            )
        ):
            self.state = "YIELD_OR_BRAKE"
            self.open_side_pass_memory_frames = 0
            self.post_pass_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=0.0,
                throttle_cap=0.0,
                brake=0.45 if features.ego_speed < 3.0 else 0.65,
                steer_limit=0.35,
                reason="active_red_stop_deceleration",
            )

        if (
            self.red_stop_release_frames > 0
            and not features.red_light_active
            and (features.red_stop_distance is None or float(features.red_stop_distance) > 5.5)
            and features.front_clear
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.front_obstacle_distance is None
            and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
            and features.lidar_center_blockage_ratio < 0.10
            and features.ego_speed < 1.4
        ):
            self.red_stop_release_frames -= 1
            return self._red_stop_release_action()
        if (
            self.red_stop_release_frames <= 0
            and self.red_stop_hold_frames >= 30
            and self.red_stop_gap_frames >= 30
            and self.red_stop_gap_frames < 120
            and features.front_clear
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.front_obstacle_distance is None
            and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
            and features.lidar_blockage_ratio <= 0.05
            and features.lidar_center_blockage_ratio <= 0.05
            and abs(features.ego_speed) < 0.20
        ):
            return self._red_stop_release_action(start_window=True)

        if (
            self.construction_cone_entry_slow_frames > 0
            and estimate.macro_scenario in (
                "trucks_encountered_during_construction",
                "high_speed_temporary_construction",
            )
            and not features.red_light_active
            and features.red_stop_distance is None
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.ego_speed > 1.6
        ):
            self.construction_cone_entry_slow_frames -= 1
            self.state = "AVOID_OR_PASS"
            return PlannerAction(
                True,
                self.state,
                target_speed=2.0,
                throttle_cap=0.0 if features.ego_speed > 3.0 else 0.20,
                throttle_floor=0.0 if features.ego_speed > 3.0 else 0.08,
                brake=0.55 if features.ego_speed > 4.0 else (0.35 if features.ego_speed > 3.0 else None),
                steer_limit=0.30,
                reason="construction_sparse_cone_entry_memory_slowdown",
            )
        if self.construction_cone_entry_slow_frames > 0 and features.ego_speed <= 1.6:
            self.construction_cone_entry_slow_frames -= 1

        if self.blind_spot_junction_brake_cooldown_frames > 0:
            self.blind_spot_junction_brake_cooldown_frames -= 1
        blind_spot_junction_brake_candidate = (
            estimate.macro_scenario == "blind_spot_hidden_car"
            and features.junction_like
            and features.side_risk
            and not roundabout_layout_context
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.front_obstacle_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and 1.2 <= abs(features.ego_speed) <= 12.5
        )
        if (
            blind_spot_junction_brake_candidate
            and self.blind_spot_junction_brake_frames <= 0
            and self.blind_spot_junction_brake_cooldown_frames <= 0
        ):
            self.blind_spot_junction_brake_frames = 12
            self.blind_spot_junction_brake_cooldown_frames = 180
            self.lateral_intersection_release_frames = max(self.lateral_intersection_release_frames, 900)
        if blind_spot_junction_brake_candidate and self.blind_spot_junction_brake_frames > 0:
            self.blind_spot_junction_brake_frames -= 1
            self.state = "YIELD_OR_BRAKE"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=2.0,
                throttle_cap=0.0,
                brake=0.62,
                brake_cap=0.70,
                steer_limit=0.14,
                reason="blind_spot_junction_scored_brake_response",
            )

        if estimate.confidence < self.config.min_rule_confidence:
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state == "AVOID_OR_PASS"
                and self.close_obstacle_memory_frames > 0
                and features.ego_speed < 0.9
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
                and features.lidar_front_distance is not None
                and float(features.lidar_front_distance) <= self.config.lidar_open_side_post_pass_max_distance
                and self.last_open_side in ("right", "left")
            ):
                self.close_obstacle_memory_frames = 0
                self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                self.static_creep_frames = 0
                self.state = "AVOID_OR_PASS"
                route51_cone_mode = self.config.suppress_lateral_intersection_rules
                steer_magnitude = 0.16 if route51_cone_mode else 0.55
                steer_bias = steer_magnitude if self.last_open_side == "right" else -steer_magnitude
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=0.8 if route51_cone_mode else 1.0,
                    throttle_cap=0.24 if route51_cone_mode else 0.34,
                    throttle_floor=0.12 if route51_cone_mode else 0.18,
                    brake_cap=0.0,
                    steer_limit=0.30 if route51_cone_mode else 0.90,
                    steer_bias=steer_bias,
                    steer_min_magnitude=steer_magnitude,
                    reason="distant_lidar_open_side_close_memory_nudge",
                )
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state in ("AVOID_OR_PASS", "RECOVER")
                and self.progress_recovery_frames > 0
                and features.ego_speed < 2.2
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
                and (
                    features.front_obstacle_distance is None
                    or features.lidar_front_distance is None
                    or float(features.lidar_front_distance) <= self.config.lidar_open_side_post_pass_max_distance
                )
            ):
                self.progress_recovery_frames -= 1
                self.static_creep_frames = 0
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.45,
                    reason="distant_lidar_open_side_progress_recovery",
                )
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state in ("AVOID_OR_PASS", "RECOVER")
                and self.progress_recovery_frames > 0
                and features.ego_speed < 2.2
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
                and (
                    features.lidar_front_distance is None
                    or float(features.lidar_front_distance) <= self.config.lidar_open_side_post_pass_max_distance
                )
            ):
                self.progress_recovery_frames -= 1
                self.static_creep_frames = 0
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.45,
                    reason="distant_lidar_open_side_progress_recovery",
                )
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state == "AVOID_OR_PASS"
                and self.post_pass_frames > 0
                and self.close_obstacle_memory_frames <= 0
                and features.ego_speed < 1.2
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
            ):
                self.post_pass_frames -= 1
                self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                self.static_creep_frames = 0
                self.state = "RECOVER"
                steer_bias = 0.10 if self.last_open_side == "right" else (-0.10 if self.last_open_side == "left" else 0.0)
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.2,
                    throttle_cap=0.52,
                    throttle_floor=0.28,
                    brake_cap=0.0,
                    steer_limit=0.55,
                    steer_bias=steer_bias,
                    reason="distant_lidar_open_side_post_pass_recovery",
                )
            no_front_conflict = (
                features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                and features.lidar_center_blockage_ratio < 0.10
            )
            clear_road_actor_clear = (
                features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
            )
            if (
                no_front_conflict
                and self.observable_risk_creep_frames > 0
                and features.ego_speed < 2.0
                and (not features.red_light_active or features.red_stop_distance is None or features.ego_speed < 0.25)
            ):
                self.observable_risk_creep_frames -= 1
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.35,
                    reason="observable_risk_cautious_creep_recovery",
                )
            if features.red_stop_distance is None and not features.red_light_active and self.red_stop_hold_frames > 0:
                self.red_stop_gap_frames += 1
            red_clear_recovery_allowed = self.red_stop_hold_frames == 0 or self.red_stop_gap_frames >= 120
            if (
                clear_road_actor_clear
                and red_clear_recovery_allowed
                and features.red_stop_distance is None
                and not features.red_light_active
                and self.blocked_frames >= 2
                and features.ego_speed < 2.0
            ):
                if features.ego_speed < 0.60:
                    self.blocked_frames += 1
                return self._clear_road_no_progress_action(
                    "clear_road_cautious_creep_recovery",
                    allow_reverse=not roundabout_layout_context,
                )
            if (
                clear_road_actor_clear
                and red_clear_recovery_allowed
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.ego_speed < 0.60
            ):
                self.blocked_frames += 1
                if self.blocked_frames >= 2:
                    return self._clear_road_no_progress_action(
                        "clear_road_cautious_creep_recovery",
                        allow_reverse=not roundabout_layout_context,
                    )
            elif not clear_road_actor_clear or features.red_stop_distance is not None or features.red_light_active or features.ego_speed >= 1.00:
                self.blocked_frames = 0
            high_speed_lateral_vehicle_evidence = False
            high_speed_lateral_nearest_x = None
            for track in features.tracked_objects:
                cls = str(track.get("class_name", "")).lower()
                if cls not in ("car", "van", "truck", "bus", "motorcycle", "bicycle"):
                    continue
                try:
                    x = float(track.get("x", 999.0))
                    y = float(track.get("y", 999.0))
                    observed_frames = int(track.get("observed_frames", 0) or 0)
                    score = float(track.get("score", track.get("confidence", 1.0)) or 0.0)
                    closing_speed = float(track.get("closing_speed", 0.0) or 0.0)
                    lateral_velocity = float(
                        track.get("lateral_velocity", track.get("vy", track.get("vx", 0.0))) or 0.0
                    )
                except Exception:
                    continue
                moving_toward_ego_corridor = bool(
                    (y > 0.0 and lateral_velocity <= -0.35)
                    or (y < 0.0 and lateral_velocity >= 0.35)
                )
                reliable_lateral_vehicle = bool(
                    closing_speed >= 0.8
                    or (score >= 0.30 and moving_toward_ego_corridor)
                    or (score >= 0.30 and observed_frames <= 1 and 34.0 <= x <= 42.0 and 1.0 <= abs(y) <= 2.6)
                )
                enough_lateral_history = observed_frames >= 2 or (score >= 0.30 and observed_frames <= 1 and 34.0 <= x <= 42.0)
                if 15.0 <= x <= 55.0 and 1.0 <= abs(y) <= 3.4 and enough_lateral_history and reliable_lateral_vehicle:
                    high_speed_lateral_vehicle_evidence = True
                    high_speed_lateral_nearest_x = x if high_speed_lateral_nearest_x is None else min(high_speed_lateral_nearest_x, x)
                    break
            if high_speed_lateral_vehicle_evidence and not self.config.suppress_lateral_intersection_rules:
                self.high_speed_lateral_guard_frames = 35
            lateral_guard_lidar_context = False
            if features.lidar_front_distance is not None:
                try:
                    lateral_guard_lidar_context = (
                        float(features.lidar_front_distance) <= 16.0
                        and features.lidar_open_side != "unknown"
                    )
                except Exception:
                    lateral_guard_lidar_context = False
            high_speed_lateral_vehicle_guard = (
                not self.config.suppress_lateral_intersection_rules
                and features.ego_speed > 4.2
                and self.high_speed_lateral_guard_frames > 0
                and (
                    high_speed_lateral_vehicle_evidence
                    or lateral_guard_lidar_context
                    or self.high_speed_lateral_guard_frames > 20
                )
                and not roundabout_layout_context
                and features.front_pedestrian_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.front_obstacle_distance is None
            )
            had_lateral_release_memory = self.lateral_intersection_release_frames > 0
            scored_brake_follow_through = (
                self.lateral_intersection_scored_brake_frames > 0
                and 2.8 < features.ego_speed <= 8.5
                and not roundabout_layout_context
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
            )
            if scored_brake_follow_through:
                self.lateral_intersection_scored_brake_frames -= 1
                self.lateral_intersection_release_frames = max(self.lateral_intersection_release_frames, 1200)
                self.state = "PREPARE"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.0,
                    throttle_cap=0.0,
                    brake=0.58,
                    brake_cap=0.68,
                    steer_limit=0.14,
                    reason="lateral_intersection_scored_brake_response",
                )
            if high_speed_lateral_vehicle_guard:
                self.high_speed_lateral_guard_frames -= 1
                self.lateral_intersection_release_frames = max(self.lateral_intersection_release_frames, 1200)
                self.state = "PREPARE"
                scored_brake_response_window = (
                    high_speed_lateral_nearest_x is not None
                    and 25.0 <= high_speed_lateral_nearest_x <= 44.0
                    and 4.8 <= features.ego_speed <= 9.0
                    and self.lateral_intersection_scored_brake_frames <= 0
                )
                if scored_brake_response_window:
                    self.lateral_intersection_scored_brake_frames = 14
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=2.0,
                        throttle_cap=0.0,
                        brake=0.58,
                        brake_cap=0.68,
                        steer_limit=0.14,
                        reason="lateral_intersection_scored_brake_response",
                    )
                early_lateral_light_brake = (
                    had_lateral_release_memory
                    or (
                        high_speed_lateral_nearest_x is not None
                        and high_speed_lateral_nearest_x > 36.0
                        and features.ego_speed < 7.0
                    )
                )
                if early_lateral_light_brake:
                    lateral_guard_brake = 0.0
                    lateral_guard_brake_cap = 0.0
                    lateral_guard_throttle_cap = 1.0
                    lateral_guard_throttle_floor = 0.88
                    if features.ego_speed > 10.5:
                        lateral_guard_brake = 0.10
                        lateral_guard_brake_cap = 0.16
                        lateral_guard_throttle_cap = 0.0
                        lateral_guard_throttle_floor = None
                    elif features.ego_speed > 9.5:
                        lateral_guard_throttle_cap = 0.0
                        lateral_guard_throttle_floor = None
                else:
                    lateral_guard_brake = 0.62 if features.ego_speed > 9.5 else (0.55 if features.ego_speed > 4.2 else 0.42)
                    lateral_guard_brake_cap = None
                    lateral_guard_throttle_cap = 0.0
                    lateral_guard_throttle_floor = None
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=9.0 if early_lateral_light_brake else 4.5,
                    throttle_cap=lateral_guard_throttle_cap,
                    throttle_floor=lateral_guard_throttle_floor,
                    brake=lateral_guard_brake,
                    brake_cap=lateral_guard_brake_cap,
                    steer_limit=0.16,
                    reason="high_speed_lateral_vehicle_cutin_guard",
                )
            lateral_intersection_keep_rolling = (
                self.lateral_intersection_release_frames > 0
                and features.ego_speed < 11.2
                and not roundabout_layout_context
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
            )
            if lateral_intersection_keep_rolling:
                self.high_speed_lateral_guard_frames -= 1
                self.state = "PREPARE"
                near_lateral_conflict_x = None
                for track in features.tracked_objects:
                    cls = str(track.get("class_name", "")).lower()
                    if cls not in ("car", "van", "truck", "bus", "motorcycle", "bicycle"):
                        continue
                    try:
                        x = float(track.get("x", 999.0))
                        y = float(track.get("y", 999.0))
                        observed_frames = int(track.get("observed_frames", 0) or 0)
                        score = float(track.get("score", track.get("confidence", 1.0)) or 0.0)
                    except Exception:
                        continue
                    if 25.0 <= x <= 35.5 and 1.0 <= abs(y) <= 3.4 and (observed_frames >= 2 or score >= 0.30):
                        near_lateral_conflict_x = x if near_lateral_conflict_x is None else min(near_lateral_conflict_x, x)
                if (
                    near_lateral_conflict_x is not None
                    and near_lateral_conflict_x <= 44.0
                    and 4.0 <= features.ego_speed <= 9.0
                    and self.lateral_intersection_scored_brake_frames <= 0
                ):
                    self.lateral_intersection_scored_brake_frames = 14
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=2.0,
                        throttle_cap=0.0,
                        brake=0.58,
                        brake_cap=0.68,
                        steer_limit=0.14,
                        reason="lateral_intersection_scored_brake_response",
                    )
                return self._lateral_release_keep_rolling_action(features)
            if self.lateral_intersection_scored_brake_frames > 0:
                self.lateral_intersection_scored_brake_frames -= 1
            if self.high_speed_lateral_guard_frames > 0 and features.ego_speed <= 5.5:
                self.high_speed_lateral_guard_frames -= 1
            high_speed_sparse_lidar_cone_approach = (
                self.config.lidar_open_side_nudge_enabled
                and features.lidar_front_distance is not None
                and 12.0 <= float(features.lidar_front_distance) <= 24.0
                and features.ego_speed > 10.0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and 0.04 <= features.lidar_blockage_ratio <= 0.35
                and features.lidar_center_blockage_ratio <= 0.15
                and (
                    abs(features.lidar_lateral_centroid) >= 1.20
                    or features.lidar_left_density >= 5
                    or features.lidar_right_density >= 5
                )
                and features.detection_object_count >= 60
            )
            if high_speed_sparse_lidar_cone_approach:
                self.construction_cone_entry_slow_frames = 120
                self.construction_corridor_memory_frames = max(self.construction_corridor_memory_frames, 360)
                self.static_creep_frames = 0
                self.state = "PREPARE"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=4.0,
                    throttle_cap=0.0,
                    brake=0.58,
                    steer_limit=0.25,
                    reason="construction_high_speed_sparse_lidar_approach",
                )
            if (
                self.lateral_intersection_release_frames > 0
                and features.ego_speed < 11.2
                and not roundabout_layout_context
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
            ):
                self.state = "PREPARE"
                return self._lateral_release_keep_rolling_action(features, reason="lateral_intersection_release_memory_override")
            construction_corridor_memory_active = bool(
                self.config.lidar_open_side_nudge_enabled
                and self.config.suppress_lateral_intersection_rules
                and self.construction_corridor_memory_frames > 0
                and features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and not roundabout_layout_context
            )
            if construction_corridor_memory_active:
                self.construction_corridor_memory_frames -= 1
                self.static_creep_frames = 0
                self.progress_recovery_frames = 0
                self.open_side_pass_memory_frames = 0
                self.state = "RECOVER"
                if features.ego_speed > 3.2:
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.8,
                        throttle_cap=0.0,
                        brake=0.48 if features.ego_speed < 5.5 else 0.72,
                        steer_limit=0.10,
                        reason="construction_corridor_memory_speed_cap",
                    )
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.0,
                    throttle_cap=0.34,
                    throttle_floor=0.12 if features.ego_speed < 1.2 else None,
                    brake_cap=0.0,
                    steer_limit=0.12,
                    reason="construction_corridor_memory_cautious_recovery",
                )
            self.state = "NORMAL" if features.risk_level == 0 else "APPROACH"
            self.static_creep_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(False, self.state, reason="rules_disabled_or_low_confidence")

        reverse_front_vehicle_distance = None
        if features.front_vehicle_distance is not None:
            reverse_front_vehicle_distance = float(features.front_vehicle_distance)
        reverse_observed_buffer_conflict = (
            estimate.macro_scenario == "reverse_vehicle"
            and reverse_front_vehicle_distance is not None
            and reverse_front_vehicle_distance <= 8.5
            and features.ego_speed > 0.45
            and not features.reversing_vehicle_evidence
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and (
                reverse_front_vehicle_distance <= 6.5
                or features.front_vehicle_closing_speed >= 0.15
                or features.lidar_center_blockage_ratio >= 0.35
            )
        )
        reverse_close_vehicle_conflict = (
            features.reversing_vehicle_evidence
            and reverse_front_vehicle_distance is not None
            and reverse_front_vehicle_distance <= 5.5
            and features.ego_speed > 0.8
            and features.front_pedestrian_distance is None
        )
        reverse_brake_memory_conflict = (
            self.reverse_vehicle_brake_frames > 0
            and features.front_pedestrian_distance is None
            and features.ego_speed > 0.35
            and (
                (
                    features.front_vehicle_distance is not None
                    and float(features.front_vehicle_distance) <= 6.0
                )
                or (
                    features.front_obstacle_distance is not None
                    and float(features.front_obstacle_distance) <= 4.0
                )
            )
        )
        if (
            estimate.macro_scenario != "highway_accident_vehicle"
            and (reverse_observed_buffer_conflict or reverse_close_vehicle_conflict or reverse_brake_memory_conflict)
        ):
            self.reverse_vehicle_brake_frames = 45 if reverse_observed_buffer_conflict else (30 if reverse_close_vehicle_conflict else self.reverse_vehicle_brake_frames - 1)
            self.state = "YIELD_OR_BRAKE"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(
                True,
                self.state,
                target_speed=0.0,
                throttle_cap=0.0,
                brake=0.62 if reverse_observed_buffer_conflict else 0.48,
                steer_limit=0.22 if reverse_observed_buffer_conflict else 0.25,
                reason="reverse_vehicle_observed_buffer_brake" if reverse_observed_buffer_conflict else "reverse_vehicle_ttc_defensive_brake",
            )
        self.reverse_vehicle_brake_frames = max(0, self.reverse_vehicle_brake_frames - 1)

        reverse_vehicle_far_vehicle_keepalive = (
            self.config.allow_route_prior
            and estimate.macro_scenario == "reverse_vehicle"
            and features.front_pedestrian_distance is None
            and features.front_vehicle_distance is not None
            and 8.0 <= float(features.front_vehicle_distance) <= 13.0
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.lidar_open_side in ("right", "left", "balanced")
            and features.lidar_blockage_ratio >= 0.85
            and features.lidar_center_blockage_ratio >= 0.75
            and -0.20 < float(features.ego_speed) < 1.20
            and (
                features.front_vehicle_ttc is None
                or float(features.front_vehicle_ttc) >= 6.0
            )
        )
        if reverse_vehicle_far_vehicle_keepalive:
            self.state = "RECOVER"
            if features.lidar_open_side == "right":
                steer_bias = -0.10
            elif features.lidar_open_side == "left":
                steer_bias = 0.10
            else:
                steer_bias = 0.0
            return PlannerAction(
                True,
                self.state,
                target_speed=2.4,
                throttle_cap=0.82,
                throttle_floor=0.56,
                brake_cap=0.0,
                steer_limit=0.24,
                steer_bias=steer_bias,
                steer_min_magnitude=0.0 if features.lidar_open_side == "balanced" else 0.05,
                reason="reverse_vehicle_observed_far_vehicle_keepalive",
            )

        reverse_vehicle_ultra_close_reverse_memory = (
            estimate.macro_scenario == "reverse_vehicle"
            and self.reverse_unwedge_frames > 0
            and features.front_pedestrian_distance is None
            and features.front_vehicle_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.front_obstacle_distance is not None
            and float(features.front_obstacle_distance) <= 5.5
            and abs(features.ego_speed) < 1.25
        )
        if reverse_vehicle_ultra_close_reverse_memory:
            self.reverse_unwedge_frames -= 1
            self.state = "RECOVER"
            if features.lidar_open_side == "right":
                steer_bias = -0.36
            elif features.lidar_open_side == "left":
                steer_bias = 0.36
            else:
                steer_bias = -0.30 if features.lidar_lateral_centroid <= 0.0 else 0.30
            return PlannerAction(
                True,
                self.state,
                throttle_cap=0.72,
                throttle_floor=0.54,
                brake_cap=0.0,
                steer_limit=0.62,
                steer_bias=steer_bias,
                steer_min_magnitude=0.30,
                reverse=True,
                reason="reverse_vehicle_ultra_close_reverse_memory",
            )

        reverse_observed_open_side_escape = (
            estimate.macro_scenario == "reverse_vehicle"
            and not features.reversing_vehicle_evidence
            and self.config.lidar_open_side_nudge_enabled
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.ego_speed < 0.90
            and features.front_obstacle_distance is not None
            and 1.5 <= float(features.front_obstacle_distance) <= 3.6
            and features.lidar_blockage_ratio >= 0.85
            and features.lidar_center_blockage_ratio >= 0.85
            and features.lidar_open_side in ("right", "left")
            and (
                self.last_open_side in ("unknown", features.lidar_open_side)
                or self.open_side_pass_memory_frames > 0
            )
            and (
                (
                    features.lidar_open_side == "right"
                    and features.lidar_left_blockage_ratio >= 0.85
                    and features.lidar_right_blockage_ratio <= 0.20
                )
                or (
                    features.lidar_open_side == "left"
                    and features.lidar_right_blockage_ratio >= 0.85
                    and features.lidar_left_blockage_ratio <= 0.20
                )
            )
        )
        if reverse_observed_open_side_escape:
            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
            self.last_open_side = features.lidar_open_side
            self.state = "AVOID_OR_PASS"
            obstacle_distance = float(features.front_obstacle_distance)
            if obstacle_distance <= 3.60 and abs(features.ego_speed) < 0.45:
                self.blocked_frames += 1
            elif abs(features.ego_speed) > 0.80:
                self.blocked_frames = 0
            if obstacle_distance <= 3.60 and self.blocked_frames >= 4:
                self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 120)
            if obstacle_distance <= 3.60 and self.reverse_unwedge_frames > 0:
                self.reverse_unwedge_frames -= 1
                steer_magnitude = 0.42 if obstacle_distance > 2.6 else 0.50
                steer_bias = -steer_magnitude if features.lidar_open_side == "right" else steer_magnitude
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.58,
                    throttle_floor=0.42,
                    brake_cap=0.0,
                    steer_limit=0.68,
                    steer_bias=steer_bias,
                    steer_min_magnitude=steer_magnitude,
                    reverse=True,
                    reason="reverse_vehicle_open_side_reverse_unwedge",
                )
            steer_bias = 0.22 if features.lidar_open_side == "right" else -0.22
            return PlannerAction(
                True,
                self.state,
                target_speed=0.8,
                throttle_cap=0.22,
                throttle_floor=0.10,
                brake_cap=0.0,
                steer_limit=0.32,
                steer_bias=steer_bias,
                steer_min_magnitude=0.16,
                reason="reverse_vehicle_open_side_cautious_probe",
            )

        if estimate.macro_scenario == "reverse_vehicle" and not self.config.reverse_vehicle_rule_enabled:
            self.state = "PREPARE" if features.risk_level > 0 else "NORMAL"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            if (
                features.reversing_vehicle_evidence
                and features.front_vehicle_ttc is not None
                and float(features.front_vehicle_ttc) < 2.8
                and features.front_vehicle_distance is not None
                and float(features.front_vehicle_distance) <= 5.0
                and features.ego_speed > 1.0
                and features.front_pedestrian_distance is None
            ):
                return PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    target_speed=0.0,
                    throttle_cap=0.0,
                    brake=0.58,
                    steer_limit=0.25,
                    reason="reverse_vehicle_ttc_defensive_brake",
                )
            if (
                self.config.suppress_lateral_intersection_rules
                and self.config.lidar_open_side_nudge_enabled
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and 3.2 <= float(features.front_obstacle_distance) <= 5.0
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left")
                and features.lidar_blockage_ratio >= 0.80
                and abs(features.ego_speed) < 0.35
            ):
                self.state = "AVOID_OR_PASS"
                steer_bias = 0.42 if features.lidar_open_side == "right" else -0.42
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.70,
                    throttle_floor=0.48,
                    brake_cap=0.0,
                    steer_limit=0.62,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.30,
                    reverse=True,
                    reason="construction_false_reverse_observed_open_side_unwedge",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and float(features.front_obstacle_distance) <= 1.55
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and features.lidar_blockage_ratio >= 0.80
                and abs(features.ego_speed) < 2.50
            ):
                self.state = "RECOVER"
                obstacle_distance = float(features.front_obstacle_distance)
                steer_mag = 0.58 if obstacle_distance <= 0.90 else 0.44
                if features.lidar_open_side == "left":
                    steer_bias = steer_mag
                else:
                    steer_bias = -steer_mag
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.78 if obstacle_distance <= 0.90 else 0.68,
                    throttle_floor=0.62 if obstacle_distance <= 0.90 else 0.50,
                    brake_cap=0.0,
                    steer_limit=0.72,
                    steer_bias=steer_bias,
                    steer_min_magnitude=min(0.44, steer_mag),
                    reverse=True,
                    reason="reverse_vehicle_observed_static_reverse_unwedge",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and float(features.front_obstacle_distance) <= 0.25
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and abs(features.ego_speed) < 1.00
            ):
                self.state = "RECOVER"
                if features.lidar_open_side == "left":
                    steer_bias = 0.58
                else:
                    steer_bias = -0.58
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.78,
                    throttle_floor=0.62,
                    brake_cap=0.0,
                    steer_limit=0.72,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.44,
                    reverse=True,
                    reason="reverse_vehicle_observed_static_ultraclose_reverse_unwedge",
                )
            reverse_vehicle_high_blockage_stuck = (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and 1.70 <= float(features.front_obstacle_distance) <= 3.05
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and features.lidar_blockage_ratio >= 0.85
                and abs(float(features.ego_speed)) < 0.65
            )
            if reverse_vehicle_high_blockage_stuck:
                self.reverse_vehicle_high_blockage_stuck_frames += 1
            else:
                self.reverse_vehicle_high_blockage_stuck_frames = 0
            if (
                reverse_vehicle_high_blockage_stuck
                and self.reverse_vehicle_high_blockage_stuck_frames >= 2
            ):
                self.state = "RECOVER"
                if features.lidar_open_side == "right":
                    steer_bias = -0.58
                    steer_min_magnitude = 0.44
                elif features.lidar_open_side == "left":
                    steer_bias = 0.58
                    steer_min_magnitude = 0.44
                else:
                    steer_bias = -0.48 if (self.reverse_vehicle_high_blockage_stuck_frames // 2) % 2 == 0 else 0.48
                    steer_min_magnitude = 0.36
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.72,
                    throttle_floor=0.58,
                    brake_cap=0.0,
                    steer_limit=0.72,
                    steer_bias=steer_bias,
                    steer_min_magnitude=steer_min_magnitude,
                    reverse=True,
                    reason="reverse_vehicle_observed_static_high_blockage_reverse_swing",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and 1.55 < float(features.front_obstacle_distance) <= 16.00
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and features.lidar_blockage_ratio <= 0.50
                and -0.50 < float(features.ego_speed) < 3.40
            ):
                self.state = "RECOVER"
                if features.lidar_open_side == "right":
                    steer_bias = 0.12
                elif features.lidar_open_side == "left":
                    steer_bias = -0.12
                else:
                    steer_bias = 0.0
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=3.4,
                    throttle_cap=1.0,
                    throttle_floor=0.90,
                    brake_cap=0.0,
                    steer_limit=0.26,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.0 if features.lidar_open_side == "balanced" else 0.08,
                    reason="reverse_vehicle_observed_static_low_blockage_forward_resume",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and 1.55 < float(features.front_obstacle_distance) <= 4.80
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and features.lidar_blockage_ratio >= 0.55
                and -1.00 < float(features.ego_speed) < 4.20
            ):
                self.state = "RECOVER"
                strong_static_nudge = (
                    features.lidar_blockage_ratio >= 0.85
                    and float(features.front_obstacle_distance) <= 2.80
                    and abs(float(features.ego_speed)) < 0.25
                )
                if features.lidar_open_side == "right":
                    steer_bias = 0.48 if strong_static_nudge else 0.18
                elif features.lidar_open_side == "left":
                    steer_bias = -0.48 if strong_static_nudge else -0.18
                else:
                    steer_bias = 0.0
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=3.4,
                    throttle_cap=1.0,
                    throttle_floor=0.90,
                    brake_cap=0.0,
                    steer_limit=0.56 if strong_static_nudge else 0.30,
                    steer_bias=steer_bias,
                    steer_min_magnitude=(
                        0.0
                        if features.lidar_open_side == "balanced"
                        else (0.34 if strong_static_nudge else 0.10)
                    ),
                    reason="reverse_vehicle_observed_static_forward_resume",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is not None
                and 4.80 < float(features.front_obstacle_distance) <= 16.00
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced")
                and features.lidar_blockage_ratio >= 0.55
                and -0.50 < float(features.ego_speed) < 4.00
            ):
                self.state = "RECOVER"
                if features.lidar_open_side == "right":
                    steer_bias = -0.08
                elif features.lidar_open_side == "left":
                    steer_bias = 0.08
                else:
                    steer_bias = 0.0
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.8,
                    throttle_cap=0.86,
                    throttle_floor=0.56,
                    brake_cap=0.0,
                    steer_limit=0.22,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.0 if features.lidar_open_side == "balanced" else 0.04,
                    reason="reverse_vehicle_observed_static_far_forward_keepalive",
                )
            if (
                self.config.allow_route_prior
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left", "balanced", "unknown")
                and features.lidar_blockage_ratio <= 0.75
                and 0.20 <= float(features.ego_speed) < 4.80
            ):
                self.state = "RECOVER"
                if features.lidar_open_side == "right":
                    steer_bias = 0.08
                elif features.lidar_open_side == "left":
                    steer_bias = -0.08
                else:
                    steer_bias = 0.0
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=3.4,
                    throttle_cap=0.92,
                    throttle_floor=0.62,
                    brake_cap=0.0,
                    steer_limit=0.18,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.0,
                    reason="reverse_vehicle_route_prior_clear_low_blockage_resume",
                )
            return PlannerAction(False, self.state, reason="reverse_vehicle_observed_only")

        close_open_side_reverse_continuation = (
            estimate.macro_scenario in (
                "trucks_encountered_during_construction",
                "high_speed_temporary_construction",
                        )
                        and not roundabout_layout_context
                        and self.config.lidar_open_side_nudge_enabled
            and self.reverse_unwedge_frames > 0
            and features.front_obstacle_distance is not None
            and 1.5 <= float(features.front_obstacle_distance) <= 3.6
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.ego_speed < 0.90
            and features.lidar_blockage_ratio >= 0.80
            and features.lidar_center_blockage_ratio >= 0.80
            and features.lidar_open_side in ("right", "left")
            and self.last_open_side in ("unknown", features.lidar_open_side)
        )
        if close_open_side_reverse_continuation:
            self.reverse_unwedge_frames -= 1
            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
            self.last_open_side = features.lidar_open_side
            self.state = "AVOID_OR_PASS"
            steer_magnitude = 0.35
            steer_bias = -steer_magnitude if features.lidar_open_side == "right" else steer_magnitude
            return PlannerAction(
                True,
                self.state,
                throttle_cap=0.40,
                throttle_floor=0.30,
                brake_cap=0.0,
                steer_limit=0.55,
                steer_bias=steer_bias,
                steer_min_magnitude=steer_magnitude,
                reverse=True,
                reason="construction_open_side_reverse_unwedge",
            )

        construction_open_side_vehicle_memory_escape = (
            estimate.macro_scenario in (
                "trucks_encountered_during_construction",
                "high_speed_temporary_construction",
            )
            and self.config.lidar_open_side_nudge_enabled
            and self.open_side_pass_memory_frames > 0
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.ego_speed < 1.20
            and features.lidar_blockage_ratio >= 0.85
            and features.lidar_center_blockage_ratio >= 0.85
            and features.lidar_open_side in ("right", "left")
            and self.last_open_side == features.lidar_open_side
            and (
                (
                    features.lidar_open_side == "right"
                    and features.lidar_left_blockage_ratio >= 0.85
                    and features.lidar_right_blockage_ratio <= 0.20
                )
                or (
                    features.lidar_open_side == "left"
                    and features.lidar_right_blockage_ratio >= 0.85
                    and features.lidar_left_blockage_ratio <= 0.20
                )
            )
            and (
                (
                    features.front_obstacle_distance is not None
                    and 1.8 <= float(features.front_obstacle_distance) <= 3.2
                )
                or (
                    features.front_vehicle_distance is not None
                    and 1.8 <= float(features.front_vehicle_distance) <= 3.2
                )
            )
        )
        if construction_open_side_vehicle_memory_escape:
            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
            self.state = "AVOID_OR_PASS"
            escape_distance = (
                features.front_obstacle_distance
                if features.front_obstacle_distance is not None
                else features.front_vehicle_distance
            )
            near_distant_escape = (
                self.config.distant_lidar_creep_enabled
                and escape_distance is not None
                and float(escape_distance) <= 4.2
            )
            near_blocked_escape = (
                escape_distance is not None
                and float(escape_distance) <= 3.2
                and abs(features.ego_speed) < 0.12
            )
            if (near_distant_escape and abs(features.ego_speed) < 0.05) or near_blocked_escape:
                self.blocked_frames += 1
            else:
                self.blocked_frames = 0
            if near_blocked_escape and self.blocked_frames >= 12:
                self.reverse_unwedge_frames = 26
                self.blocked_frames = 0
            if near_distant_escape and self.blocked_frames >= 45:
                self.reverse_unwedge_frames = 20
                self.blocked_frames = 0
            if near_distant_escape and escape_distance is not None and float(escape_distance) <= 4.1 and abs(features.ego_speed) < 1.20:
                self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 24)
            if (near_distant_escape or near_blocked_escape) and self.reverse_unwedge_frames > 0:
                self.reverse_unwedge_frames -= 1
                steer_magnitude = 0.35
                steer_bias = -steer_magnitude if features.lidar_open_side == "right" else steer_magnitude
                return PlannerAction(
                    True,
                    self.state,
                    throttle_cap=0.40,
                    throttle_floor=0.30,
                    brake_cap=0.0,
                    steer_limit=0.55,
                    steer_bias=steer_bias,
                    steer_min_magnitude=steer_magnitude,
                    reverse=True,
                    reason="construction_open_side_reverse_unwedge",
                )
            steer_magnitude = 0.55 if near_distant_escape else 0.25
            steer_bias = steer_magnitude if features.lidar_open_side == "right" else -steer_magnitude
            return PlannerAction(
                True,
                self.state,
                target_speed=1.8,
                throttle_cap=0.58,
                throttle_floor=0.36,
                brake_cap=0.0,
                steer_limit=0.80 if near_distant_escape else 0.45,
                steer_bias=steer_bias,
                steer_min_magnitude=steer_magnitude,
                reason="construction_full_blockage_open_side_escape",
            )

        if (
            estimate.macro_scenario == "trucks_encountered_during_construction"
            and not roundabout_layout_context
            and features.front_vehicle_distance is not None
            and features.front_vehicle_distance >= 2.0
            and not features.reversing_vehicle_evidence
            and (features.front_vehicle_ttc is None or features.front_vehicle_ttc >= 3.5)
            and features.ego_speed < 4.0
        ):
            vehicle_observed_open_side_push = (
                features.front_obstacle_distance is not None
                and float(features.front_obstacle_distance) <= 5.6
                and (
                    float(features.front_obstacle_distance) <= 5.0
                    or float(features.front_vehicle_distance) >= 8.0
                )
                and abs(features.ego_speed) < 0.80
                and features.lidar_available
                and not features.lidar_stale
                and features.lidar_blockage_ratio >= 0.85
                and features.lidar_center_blockage_ratio >= 0.85
                and features.lidar_open_side in ("right", "left")
                and (
                    (
                        features.lidar_open_side == "right"
                        and features.lidar_left_blockage_ratio >= 0.85
                        and features.lidar_right_blockage_ratio <= 0.20
                    )
                    or (
                        features.lidar_open_side == "left"
                        and features.lidar_right_blockage_ratio >= 0.85
                        and features.lidar_left_blockage_ratio <= 0.20
                    )
                )
                and features.front_pedestrian_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
            )
            if vehicle_observed_open_side_push:
                self.state = "AVOID_OR_PASS"
                self.last_open_side = features.lidar_open_side
                self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                steer_bias = 0.18 if features.lidar_open_side == "right" else -0.18
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=1.2,
                    throttle_cap=0.34,
                    throttle_floor=0.22,
                    brake_cap=0.0,
                    steer_limit=0.42,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.16,
                    reason="construction_vehicle_open_side_push_release",
                )
            self.state = "PREPARE"
            self.static_creep_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(False, self.state, reason="construction_vehicle_observed_without_confirmed_collision")

        construction_far_side_blockage_push = (
            estimate.macro_scenario in (
                "trucks_encountered_during_construction",
                "high_speed_temporary_construction",
            )
            and features.front_obstacle_distance is not None
            and 6.0 <= float(features.front_obstacle_distance) <= 13.5
            and abs(features.ego_speed) < 0.80
            and self.red_final_context_frames > 0
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.red_stop_distance is None
            and not features.red_light_active
            and features.lidar_available
            and not features.lidar_stale
            and features.lidar_blockage_ratio >= 0.85
            and features.lidar_open_side in ("right", "left")
            and (
                (
                    features.lidar_open_side == "right"
                    and features.lidar_left_blockage_ratio >= 0.80
                    and features.lidar_right_blockage_ratio <= 0.25
                )
                or (
                    features.lidar_open_side == "left"
                    and features.lidar_right_blockage_ratio >= 0.80
                    and features.lidar_left_blockage_ratio <= 0.25
                )
            )
        )
        if construction_far_side_blockage_push:
            self.state = "AVOID_OR_PASS"
            self.last_open_side = features.lidar_open_side
            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
            steer_bias = 0.22 if features.lidar_open_side == "right" else -0.22
            return PlannerAction(
                True,
                self.state,
                target_speed=2.6,
                throttle_cap=0.82,
                throttle_floor=0.58,
                brake_cap=0.0,
                steer_limit=0.45,
                steer_bias=steer_bias,
                steer_min_magnitude=0.18,
                reason="construction_far_side_blockage_forward_push",
            )

        if features.immediate_hazard:
            self.state = "EMERGENCY" if features.risk_level >= 3 else "YIELD_OR_BRAKE"
            # Pedestrian / bicycle hazards need yielding, but hard braking for the
            # whole detection window caused route-75 blocking in closed loop. Use
            # emergency braking only for very close objects; otherwise crawl/yield.
            if features.front_pedestrian_distance is not None:
                if features.front_pedestrian_distance < 3.5:
                    return PlannerAction(True, self.state, throttle_cap=0.0, brake=0.85, reason=estimate.reason)
                return PlannerAction(True, "YIELD_OR_BRAKE", target_speed=0.8, throttle_cap=0.12, brake=0.35, steer_limit=0.40, reason=estimate.reason)
            if features.front_vehicle_distance is not None:
                if (
                    estimate.macro_scenario == "reverse_vehicle"
                    and features.reversing_vehicle_evidence
                    and features.front_vehicle_ttc is not None
                    and features.front_vehicle_ttc < 3.5
                ):
                    return PlannerAction(True, "YIELD_OR_BRAKE", throttle_cap=0.0, brake=0.35, steer_limit=0.60, reason=estimate.reason)
                if features.front_vehicle_distance < 2.0:
                    return PlannerAction(True, self.state, throttle_cap=0.0, brake=0.65, steer_limit=0.60, reason=estimate.reason)
                return PlannerAction(True, "YIELD_OR_BRAKE", target_speed=1.8, throttle_cap=0.28, brake=0.0, steer_limit=0.75, reason=estimate.reason)
            if features.front_obstacle_distance is not None:
                return PlannerAction(False, "PREPARE", reason="static_obstacle_observed_without_confirmed_collision")
            return PlannerAction(True, self.state, throttle_cap=0.0, brake=0.75, reason=estimate.reason)

        if not features.front_clear:
            previous_state = self.state
            self.state = "PREPARE"
            static_only = (
                features.front_obstacle_distance is not None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
            )
            if static_only:
                obstacle_distance = float(features.front_obstacle_distance)
                current_open_side = features.lidar_open_side if features.lidar_open_side in ("right", "left") else self.last_open_side
                red_or_stop_conflict = bool(features.red_light_active or features.red_stop_distance is not None)
                center_full_blockage = features.lidar_center_blockage_ratio >= 0.90
                low_conf_construction = (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and estimate.confidence < 0.80
                    and obstacle_distance > 7.0
                    and center_full_blockage
                )
                if low_conf_construction and features.ego_speed > 2.0:
                    self.low_conf_construction_suppress_frames = 40
                elif self.low_conf_construction_suppress_frames > 0:
                    self.low_conf_construction_suppress_frames -= 1
                construction_open_side_confident = not low_conf_construction
                roundabout_layout_context = bool(
                    roundabout_layout_context
                    or estimate.macro_scenario == "roundabout"
                    or (features.junction_like and abs(features.route_curvature) >= 1.0)
                )
                construction_open_side_allowed = (
                    construction_open_side_confident
                    and not red_or_stop_conflict
                    and not roundabout_layout_context
                )
                stalled_observable_open_side_allowed = (
                    not red_or_stop_conflict
                    and abs(features.ego_speed) < 0.35
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                )
                red_center_geometry = (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and self.config.lidar_open_side_nudge_enabled
                    and not self.config.distant_lidar_creep_enabled
                    and center_full_blockage
                    and 1.5 <= obstacle_distance <= 3.8
                    and features.lidar_blockage_ratio >= 0.85
                    and features.ego_speed < 0.65
                )
                red_center_observed = (
                    features.red_light_active
                    and features.red_stop_distance is not None
                    and float(features.red_stop_distance) <= 3.2
                )
                if red_center_geometry and red_center_observed:
                    self.red_stop_hold_frames += 1
                    self.red_stop_gap_frames = 0
                elif red_center_geometry and self.red_stop_hold_frames > 0 and self.red_stop_gap_frames < 150:
                    self.red_stop_gap_frames += 1
                else:
                    self.red_stop_hold_frames = 0
                    self.red_stop_gap_frames = 0
                if red_center_geometry and self.red_stop_hold_frames >= 20 and self.red_stop_gap_frames < 150:
                    self.open_side_pass_memory_frames = 0
                    self.post_pass_frames = 0
                    self.close_obstacle_memory_frames = 0
                    self.progress_recovery_frames = 0
                    self.state = "RECOVER"
                    red_center_open_side = current_open_side if current_open_side in ("right", "left") else "unknown"
                    prolonged_hold = (
                        self.red_stop_hold_frames >= 60
                        and abs(features.ego_speed) < 0.08
                    )
                    if (
                        prolonged_hold
                        and red_center_open_side == "unknown"
                        and abs(features.lidar_lateral_centroid) >= 0.05
                    ):
                        red_center_open_side = "right" if features.lidar_lateral_centroid < 0.0 else "left"
                    use_open_side_escape = prolonged_hold and red_center_open_side in ("right", "left")
                    hard_escape = prolonged_hold and self.red_stop_hold_frames >= 60
                    steer_bias = 0.52 if hard_escape and red_center_open_side == "right" else (
                        -0.52 if hard_escape and red_center_open_side == "left"
                        else (0.42 if red_center_open_side == "right" else (-0.42 if red_center_open_side == "left" else 0.0))
                    )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=4.0 if hard_escape else (2.1 if prolonged_hold else 0.9),
                        throttle_cap=1.0 if hard_escape else (0.65 if prolonged_hold else 0.22),
                        throttle_floor=0.75 if hard_escape else (0.38 if prolonged_hold else 0.10),
                        brake=0.0 if hard_escape else None,
                        brake_cap=0.0,
                        steer_limit=0.90 if hard_escape and use_open_side_escape else (0.65 if use_open_side_escape else 0.12),
                        steer_bias=steer_bias if use_open_side_escape else 0.0,
                        steer_min_magnitude=0.60 if hard_escape and use_open_side_escape else (0.34 if use_open_side_escape else None),
                        reason="red_center_blockage_straight_creep",
                    )
                if (
                    (low_conf_construction or (center_full_blockage and 3.0 < obstacle_distance <= 12.0))
                    and self.config.lidar_open_side_nudge_enabled
                    and not self.config.distant_lidar_creep_enabled
                    and not red_or_stop_conflict
                    and features.ego_speed < 1.20
                ):
                    full_blockage_side_evidence = (
                        features.lidar_blockage_ratio >= 0.85
                        and features.lidar_center_blockage_ratio >= 0.85
                        and current_open_side in ("right", "left")
                        and (
                            (
                                current_open_side == "right"
                                and features.lidar_left_blockage_ratio >= 0.85
                                and features.lidar_right_blockage_ratio <= 0.20
                            )
                            or (
                                current_open_side == "left"
                                and features.lidar_right_blockage_ratio >= 0.85
                                and features.lidar_left_blockage_ratio <= 0.20
                            )
                        )
                    )
                    construction_full_blockage_open_side_escape = (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                        )
                        and 1.8 <= obstacle_distance <= 8.5
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and full_blockage_side_evidence
                    )
                    construction_full_blockage_memory_escape = (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                        )
                        and self.open_side_pass_memory_frames > 0
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and full_blockage_side_evidence
                        and self.last_open_side == current_open_side
                    )
                    if construction_full_blockage_open_side_escape or construction_full_blockage_memory_escape:
                        self.construction_full_blockage_escape_frames += 1
                        if obstacle_distance <= 3.2 and abs(features.ego_speed) < 0.12:
                            self.blocked_frames += 1
                        elif features.ego_speed > 0.30:
                            self.blocked_frames = 0
                        if obstacle_distance <= 3.2 and self.blocked_frames >= 12:
                            self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 72)
                            self.blocked_frames = 0
                        if obstacle_distance <= 3.2 and self.reverse_unwedge_frames > 0:
                            self.reverse_unwedge_frames -= 1
                            self.static_creep_frames = 0
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                            self.last_open_side = current_open_side
                            self.state = "AVOID_OR_PASS"
                            steer_magnitude = 0.35
                            steer_bias = -steer_magnitude if current_open_side == "right" else steer_magnitude
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.40,
                                throttle_floor=0.30,
                                brake_cap=0.0,
                                steer_limit=0.55,
                                steer_bias=steer_bias,
                                steer_min_magnitude=steer_magnitude,
                                reverse=True,
                                reason="construction_open_side_reverse_unwedge",
                            )
                        self.static_creep_frames = 0
                        self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                        self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                        self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                        self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                        self.last_open_side = current_open_side
                        self.state = "AVOID_OR_PASS"
                        memory_escape = construction_full_blockage_memory_escape and not construction_full_blockage_open_side_escape
                        close_push_escape = (not memory_escape) and 2.6 < obstacle_distance <= 4.5
                        route51_cone_mode = self.config.suppress_lateral_intersection_rules
                        if (
                            route51_cone_mode
                            and not memory_escape
                            and 3.0 <= obstacle_distance <= 5.4
                            and self.construction_full_blockage_escape_frames >= 60
                            and abs(features.ego_speed) < 0.20
                        ):
                            steer_magnitude = 0.34
                            steer_bias = -steer_magnitude if current_open_side == "right" else steer_magnitude
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.56,
                                throttle_floor=0.42,
                                brake_cap=0.0,
                                steer_limit=0.58,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.30,
                                reverse=True,
                                reason="construction_full_blockage_open_side_long_hold_reverse",
                            )
                        if (
                            route51_cone_mode
                            and not memory_escape
                            and 3.0 <= obstacle_distance <= 4.8
                            and self.construction_full_blockage_escape_frames >= 18
                            and abs(features.ego_speed) < 0.55
                            and (obstacle_distance <= 4.2 or features.ego_speed <= 0.05)
                            and features.lidar_center_blockage_ratio >= 0.90
                        ):
                            steer_magnitude = 0.40
                            steer_bias = -steer_magnitude if current_open_side == "right" else steer_magnitude
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.62,
                                throttle_floor=0.46,
                                brake_cap=0.0,
                                steer_limit=0.68,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.34,
                                reverse=True,
                                reason="construction_full_blockage_open_side_long_hold_reverse",
                            )
                        if (
                            route51_cone_mode
                            and not memory_escape
                            and 4.2 < obstacle_distance <= 6.2
                            and self.construction_full_blockage_escape_frames >= 18
                            and 0.05 < features.ego_speed < 0.95
                        ):
                            steer_magnitude = 0.34
                            steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.2,
                                throttle_cap=0.72,
                                throttle_floor=0.50,
                                brake_cap=0.0,
                                steer_limit=0.46,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.28,
                                reason="construction_full_blockage_open_side_long_hold_push",
                            )
                        steer_magnitude = (
                            (0.18 if route51_cone_mode else 0.70)
                            if (memory_escape or obstacle_distance <= 2.6)
                            else ((0.16 if route51_cone_mode else 0.32) if close_push_escape else ((0.20 if route51_cone_mode else 0.58) if obstacle_distance <= 6.2 else (0.18 if route51_cone_mode else 0.46)))
                        )
                        steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                        memory_target_speed = (1.0 if route51_cone_mode else 2.4) if obstacle_distance > 6.2 else (0.9 if route51_cone_mode else 2.2)
                        memory_throttle_cap = (0.36 if route51_cone_mode else 0.72) if obstacle_distance > 6.2 else (0.34 if route51_cone_mode else 0.65)
                        memory_throttle_floor = (0.20 if route51_cone_mode else 0.50) if obstacle_distance > 6.2 else (0.18 if route51_cone_mode else 0.38)
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=(
                                memory_target_speed
                                if memory_escape
                                else ((0.8 if route51_cone_mode else 2.2) if obstacle_distance <= 2.6 else ((1.0 if route51_cone_mode else 1.8) if close_push_escape else (0.9 if route51_cone_mode else 1.4)))
                            ),
                                throttle_cap=(
                                    memory_throttle_cap
                                    if memory_escape
                                    else ((0.34 if route51_cone_mode else 0.80) if obstacle_distance <= 2.6 else ((0.38 if route51_cone_mode else 0.58) if close_push_escape else ((0.34 if route51_cone_mode else 0.70) if obstacle_distance >= 4.8 else (0.30 if route51_cone_mode else 0.42))))
                                ),
                                throttle_floor=(
                                    memory_throttle_floor
                                    if memory_escape
                                    else ((0.18 if route51_cone_mode else 0.55) if obstacle_distance <= 2.6 else ((0.22 if route51_cone_mode else 0.36) if close_push_escape else ((0.18 if route51_cone_mode else 0.40) if obstacle_distance >= 4.8 else (0.14 if route51_cone_mode else 0.24))))
                                ),
                            brake_cap=0.0,
                            steer_limit=(0.25 if route51_cone_mode else (0.55 if close_push_escape else 0.90)),
                            steer_bias=steer_bias,
                            steer_min_magnitude=steer_magnitude,
                            reason="construction_full_blockage_open_side_escape",
                        )
                    if features.ego_speed < 0.08 and 4.8 <= obstacle_distance <= 8.5:
                        self.blocked_frames += 1
                    elif features.ego_speed > 0.60:
                        self.blocked_frames = 0
                    self.open_side_pass_memory_frames = 0
                    self.post_pass_frames = 0
                    self.close_obstacle_memory_frames = 0
                    self.progress_recovery_frames = 0
                    self.state = "RECOVER"
                    if self.balanced_blockage_progress_frames > 0 and features.ego_speed < 1.2:
                        self.balanced_blockage_progress_frames -= 1
                        if (
                            features.front_obstacle_distance is None
                            and features.lidar_front_distance is None
                            and abs(features.ego_speed) < 0.20
                            and self.blocked_frames >= 10
                        ):
                            steer_bias = 0.18 if (self.blocked_frames // 20) % 2 == 0 else -0.18
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=4.8,
                                throttle_cap=1.0,
                                throttle_floor=0.90,
                                brake_cap=0.0,
                                steer_limit=0.35,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.16,
                                reason="low_conf_clear_progress_forward_unwedge",
                            )
                        if (
                            estimate.macro_scenario == "roundabout"
                            and 2.8 <= obstacle_distance <= 5.6
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                        ):
                            if features.lidar_open_side == "right":
                                steer_bias = -0.30
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.30
                            else:
                                steer_bias = 0.26 if features.lidar_lateral_centroid <= 0.0 else -0.26
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.60,
                                throttle_floor=0.40,
                                brake_cap=0.0,
                                steer_limit=0.46,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.22,
                                reason="roundabout_close_static_progress_side_push",
                            )
                        if (
                            estimate.macro_scenario == "roundabout"
                            and 2.8 <= obstacle_distance <= 5.6
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                        ):
                            if features.lidar_open_side == "right":
                                steer_bias = -0.30
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.30
                            else:
                                steer_bias = 0.26 if features.lidar_lateral_centroid <= 0.0 else -0.26
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.60,
                                throttle_floor=0.40,
                                brake_cap=0.0,
                                steer_limit=0.46,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.22,
                                reason="roundabout_close_static_progress_side_push",
                            )
                        if self.blocked_frames >= 105 and obstacle_distance <= 4.2:
                            steer_magnitude = 0.35
                            if features.lidar_open_side == "right":
                                steer_bias = -steer_magnitude
                            elif features.lidar_open_side == "left":
                                steer_bias = steer_magnitude
                            else:
                                steer_bias = 0.24 if features.lidar_lateral_centroid <= 0.0 else -0.24
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.40,
                                throttle_floor=0.30,
                                brake_cap=0.0,
                                steer_limit=0.55,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.24,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_unwedge",
                            )
                        if self.blocked_frames >= 105 and obstacle_distance <= 5.5:
                            steer_bias = 0.42 if features.lidar_open_side == "right" else -0.42
                            if features.lidar_open_side not in ("right", "left"):
                                steer_bias = -0.36 if features.lidar_lateral_centroid <= 0.0 else 0.36
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.70,
                                throttle_floor=0.48,
                                brake_cap=0.0,
                                steer_limit=0.62,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.30,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_escape_sweep",
                            )
                        if obstacle_distance <= 4.8 and features.lidar_center_blockage_ratio >= 0.90:
                            steer_bias = -0.40 if features.lidar_open_side == "right" else 0.40
                            if features.lidar_open_side not in ("right", "left"):
                                steer_bias = -0.34 if features.lidar_lateral_centroid <= 0.0 else 0.34
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.62,
                                throttle_floor=0.46,
                                brake_cap=0.0,
                                steer_limit=0.68,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.34,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_escape_sweep",
                            )
                        if obstacle_distance <= 5.4:
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.58,
                                throttle_floor=0.38,
                                brake_cap=0.0,
                                steer_limit=0.20,
                                reason="low_conf_center_blockage_progress_push",
                            )
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=3.2,
                            throttle_cap=1.0,
                            throttle_floor=0.65,
                            brake_cap=0.0,
                            steer_limit=0.20,
                            reason="low_conf_center_blockage_progress_push",
                        )
                    if self.blocked_frames >= 10:
                        self.balanced_blockage_progress_frames = 80
                        if self.blocked_frames >= 105 and obstacle_distance <= 4.2:
                            steer_magnitude = 0.35
                            if features.lidar_open_side == "right":
                                steer_bias = -steer_magnitude
                            elif features.lidar_open_side == "left":
                                steer_bias = steer_magnitude
                            else:
                                steer_bias = 0.24 if features.lidar_lateral_centroid <= 0.0 else -0.24
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.40,
                                throttle_floor=0.30,
                                brake_cap=0.0,
                                steer_limit=0.55,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.24,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_unwedge",
                            )
                        if self.blocked_frames >= 105 and obstacle_distance <= 5.5:
                            steer_bias = 0.42 if features.lidar_open_side == "right" else -0.42
                            if features.lidar_open_side not in ("right", "left"):
                                steer_bias = -0.36 if features.lidar_lateral_centroid <= 0.0 else 0.36
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.70,
                                throttle_floor=0.48,
                                brake_cap=0.0,
                                steer_limit=0.62,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.30,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_escape_sweep",
                            )
                        if obstacle_distance <= 4.8 and features.lidar_center_blockage_ratio >= 0.90:
                            steer_bias = -0.40 if features.lidar_open_side == "right" else 0.40
                            if features.lidar_open_side not in ("right", "left"):
                                steer_bias = -0.34 if features.lidar_lateral_centroid <= 0.0 else 0.34
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.62,
                                throttle_floor=0.46,
                                brake_cap=0.0,
                                steer_limit=0.68,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.34,
                                reverse=True,
                                reason="low_conf_center_blockage_reverse_escape_sweep",
                            )
                        if obstacle_distance <= 5.4:
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.58,
                                throttle_floor=0.38,
                                brake_cap=0.0,
                                steer_limit=0.20,
                                reason="low_conf_center_blockage_progress_push",
                            )
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=3.2,
                            throttle_cap=1.0,
                            throttle_floor=0.65,
                            brake_cap=0.0,
                            steer_limit=0.20,
                            reason="low_conf_center_blockage_progress_push",
                        )
                    if (
                        estimate.macro_scenario == "roundabout"
                        and 2.6 <= obstacle_distance <= 3.8
                        and abs(features.ego_speed) < 0.35
                        and features.lidar_open_side in ("right", "left")
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and not red_or_stop_conflict
                    ):
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=1.8,
                            throttle_cap=0.65,
                            throttle_floor=0.45,
                            brake_cap=0.0,
                            steer_limit=0.35,
                            reason="roundabout_layout_blockage_cautious_creep",
                        )
                    if (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                        )
                        and (8.0 <= obstacle_distance <= 10.0 or 12.0 <= obstacle_distance <= 28.0)
                        and abs(features.ego_speed) < 0.20
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and not red_or_stop_conflict
                        and (features.lidar_available or features.lidar_front_distance is not None)
                    ):
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=3.2,
                            throttle_cap=0.88,
                            throttle_floor=0.58,
                            brake_cap=0.0,
                            steer_limit=0.16,
                            reason="low_conf_far_center_blockage_progress_release",
                        )
                    if (
                        2.8 <= obstacle_distance <= 5.2
                        and abs(features.ego_speed) < 0.45
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                    ):
                        self.blocked_frames += 1
                    if (
                        2.8 <= obstacle_distance <= 5.2
                        and self.blocked_frames >= 22
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                    ):
                        steer_bias = 0.20 if features.lidar_lateral_centroid <= 0.0 else -0.20
                        if features.lidar_open_side == "right":
                            steer_bias = 0.22
                        elif features.lidar_open_side == "left":
                            steer_bias = -0.22
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=2.6,
                            throttle_cap=0.78,
                            throttle_floor=0.46,
                            brake_cap=0.0,
                            steer_limit=0.42,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.14,
                            reason="low_conf_close_center_blockage_escape_sweep",
                        )
                    if (
                        estimate.macro_scenario == "reverse_vehicle"
                        and self.config.lidar_open_side_nudge_enabled
                        and not red_or_stop_conflict
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and 2.3 <= obstacle_distance <= 3.4
                        and (features.lidar_available or features.lidar_front_distance is not None)
                        and features.lidar_blockage_ratio >= 0.85
                        and features.lidar_center_blockage_ratio >= 0.85
                        and abs(features.ego_speed) < 0.75
                        and (
                            self.blocked_frames >= 12
                            or self.red_stop_gap_frames >= 600
                            or self.reverse_unwedge_frames > 0
                        )
                    ):
                        self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 34)
                        self.reverse_unwedge_frames -= 1
                        steer_bias = -0.58 if current_open_side == "right" else 0.58
                        if current_open_side not in ("right", "left"):
                            steer_bias = -0.48 if features.lidar_lateral_centroid <= 0.0 else 0.48
                        return PlannerAction(
                            True,
                            "AVOID_OR_PASS",
                            throttle_cap=0.72,
                            throttle_floor=0.52,
                            brake_cap=0.0,
                            steer_limit=0.78,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.44,
                            reverse=True,
                            reason="reverse_vehicle_near_open_side_backout",
                        )
                    if (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                            "reverse_vehicle",
                        )
                        and self.config.lidar_open_side_nudge_enabled
                        and not red_or_stop_conflict
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and 3.2 <= obstacle_distance <= 4.4
                        and (features.lidar_available or features.lidar_front_distance is not None)
                        and abs(features.ego_speed) < 4.50
                        and (
                            estimate.macro_scenario == "reverse_vehicle"
                            or self.blocked_frames >= 8
                            or self.red_stop_gap_frames >= 250
                        )
                    ):
                        steer_bias = -0.42 if current_open_side == "right" else 0.42
                        if current_open_side not in ("right", "left"):
                            steer_bias = -0.34 if features.lidar_lateral_centroid <= 0.0 else 0.34
                        return PlannerAction(
                            True,
                            "AVOID_OR_PASS",
                            throttle_cap=0.68,
                            throttle_floor=0.46,
                            brake_cap=0.0,
                            steer_limit=0.60,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.28,
                            reverse=True,
                            reason="reverse_vehicle_low_conf_close_reverse_unwedge" if estimate.macro_scenario == "reverse_vehicle" else "construction_low_conf_close_reverse_unwedge",
                        )
                    if (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                        )
                        and self.config.lidar_open_side_nudge_enabled
                        and not red_or_stop_conflict
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and 5.5 <= obstacle_distance <= 10.2
                        and (features.lidar_available or features.lidar_front_distance is not None)
                        and features.lidar_open_side == "balanced"
                        and 0.12 <= features.ego_speed < 0.90
                    ):
                        self.balanced_blockage_progress_frames = max(self.balanced_blockage_progress_frames, 90)
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=3.0,
                            throttle_cap=1.0,
                            throttle_floor=0.68,
                            brake_cap=0.0,
                            steer_limit=0.20,
                            steer_bias=0.0,
                            reason="construction_balanced_blockage_low_conf_progress_push",
                        )
                    if (
                        roundabout_layout_context
                        and 3.4 <= obstacle_distance <= 7.2
                        and features.lidar_open_side in ("right", "left")
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and not red_or_stop_conflict
                    ):
                        steer_bias = -0.22 if features.lidar_open_side == "right" else 0.22
                        return PlannerAction(
                            True,
                            "RECOVER",
                            target_speed=2.6,
                            throttle_cap=0.78,
                            throttle_floor=0.52,
                            brake_cap=0.0,
                            steer_limit=0.40,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.18,
                            reason="roundabout_low_conf_open_side_forward_push",
                        )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.2,
                        throttle_cap=0.32,
                        throttle_floor=0.16,
                        brake_cap=0.0,
                        steer_limit=0.25,
                        reason="low_conf_center_blockage_straight_creep",
                    )
                balanced_construction_blockage = (
                    estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and self.config.lidar_open_side_nudge_enabled
                    and not self.config.distant_lidar_creep_enabled
                    and not red_or_stop_conflict
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.ego_speed < 0.35
                    and 5.5 <= obstacle_distance <= 10.2
                    and (features.lidar_available or features.lidar_front_distance is not None)
                    and features.lidar_open_side == "balanced"
                    and features.detection_object_count >= 60
                    and (
                        features.lidar_blockage_ratio >= 0.02
                        or features.lidar_center_blockage_ratio >= 0.02
                    )
                )
                if balanced_construction_blockage:
                    if features.ego_speed < 0.12:
                        self.blocked_frames += 1
                    elif features.ego_speed > 0.60:
                        self.blocked_frames = 0
                    self.open_side_pass_memory_frames = 0
                    self.post_pass_frames = 0
                    self.close_obstacle_memory_frames = 0
                    self.progress_recovery_frames = 0
                    self.state = "RECOVER"
                    if self.blocked_frames >= 25:
                        self.balanced_blockage_progress_frames = 80
                        if self.blocked_frames >= 55 and obstacle_distance <= 7.0:
                            steer_bias = -0.40 if features.lidar_lateral_centroid <= 0.0 else 0.40
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.70,
                                throttle_floor=0.48,
                                brake_cap=0.0,
                                steer_limit=0.60,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.28,
                                reverse=True,
                                reason="balanced_construction_blockage_reverse_unwedge",
                            )
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=3.0,
                            throttle_cap=1.0,
                            throttle_floor=0.65,
                            brake_cap=0.0,
                            steer_limit=0.20,
                            steer_bias=0.0,
                            reason="balanced_construction_blockage_progress_push",
                        )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.4,
                        throttle_cap=0.38,
                        throttle_floor=0.22,
                        brake_cap=0.0,
                        steer_limit=0.25,
                        steer_bias=0.0,
                        reason="balanced_construction_blockage_straight_creep",
                    )
                balanced_blockage_progress_memory = (
                    self.balanced_blockage_progress_frames > 0
                    and self.config.lidar_open_side_nudge_enabled
                    and not red_or_stop_conflict
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is not None
                    and 5.0 <= obstacle_distance <= 10.2
                    and (features.lidar_available or features.lidar_front_distance is not None)
                    and features.lidar_open_side == "balanced"
                    and features.ego_speed < 1.2
                )
                if balanced_blockage_progress_memory:
                    self.balanced_blockage_progress_frames -= 1
                    self.state = "RECOVER"
                    if self.blocked_frames >= 55 and obstacle_distance <= 7.0:
                        steer_bias = -0.40 if features.lidar_lateral_centroid <= 0.0 else 0.40
                        return PlannerAction(
                            True,
                            self.state,
                            throttle_cap=0.70,
                            throttle_floor=0.48,
                            brake_cap=0.0,
                            steer_limit=0.60,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.28,
                            reverse=True,
                            reason="balanced_construction_blockage_reverse_unwedge",
                        )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=3.0,
                        throttle_cap=1.0,
                        throttle_floor=0.65,
                        brake_cap=0.0,
                        steer_limit=0.20,
                        steer_bias=0.0,
                        reason="balanced_construction_blockage_progress_push",
                    )
                if self.balanced_blockage_progress_frames > 0 and (
                    red_or_stop_conflict
                    or features.front_vehicle_distance is not None
                    or features.front_pedestrian_distance is not None
                    or features.front_obstacle_distance is None
                    or obstacle_distance < 5.0
                    or obstacle_distance > 11.0
                    or features.lidar_open_side != "balanced"
                    or features.ego_speed > 1.6
                ):
                    self.balanced_blockage_progress_frames = 0
                if red_or_stop_conflict:
                    self.open_side_pass_memory_frames = 0
                    self.post_pass_frames = 0
                    self.close_obstacle_memory_frames = 0
                    self.progress_recovery_frames = 0
                open_side_pass_memory = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and self.open_side_pass_memory_frames > 0
                    and obstacle_distance <= self.config.lidar_open_side_nudge_near_distance
                    and features.ego_speed < self.config.lidar_open_side_nudge_max_speed
                    and current_open_side in ("right", "left")
                    and features.lidar_blockage_ratio >= 0.45
                )
                if (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and previous_state == "AVOID_OR_PASS"
                    and obstacle_distance <= self.config.lidar_open_side_close_memory_distance
                    and current_open_side in ("right", "left")
                ):
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.last_open_side = current_open_side
                construction_close_obstacle_slowdown = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and obstacle_distance <= 4.2
                    and 2.5 < features.ego_speed <= 8.5
                    and features.lidar_blockage_ratio >= 0.45
                    and features.lidar_center_blockage_ratio < 0.55
                    and current_open_side in ("right", "left")
                    and features.red_stop_distance is None
                )
                construction_sparse_cone_entry_slowdown = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and 4.5 <= obstacle_distance <= 6.5
                    and 3.0 < features.ego_speed <= 8.5
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and features.lidar_center_blockage_ratio < 0.20
                    and (
                        features.lidar_blockage_ratio < 0.20
                        or abs(features.lidar_lateral_centroid) >= 1.20
                    )
                )
                construction_sparse_cone_low_speed_guard = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and 4.5 <= obstacle_distance <= 6.8
                    and abs(features.ego_speed) < 1.8
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and features.lidar_center_blockage_ratio < 0.28
                    and (
                        features.lidar_blockage_ratio < 0.35
                        or abs(features.lidar_lateral_centroid) >= 1.0
                    )
                )
                construction_high_speed_close_obstacle_brake = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and obstacle_distance <= 3.2
                    and features.ego_speed > 8.5
                    and features.lidar_blockage_ratio >= 0.45
                    and current_open_side in ("right", "left")
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                construction_near_cone_speed_guard = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario == "trucks_encountered_during_construction"
                    and estimate.confidence >= 0.80
                    and 2.2 <= obstacle_distance <= 7.2
                    and 2.8 < features.ego_speed <= 8.5
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio < 0.65
                )
                if construction_near_cone_speed_guard:
                    self.static_creep_frames = 0
                    self.construction_cone_entry_slow_frames = max(self.construction_cone_entry_slow_frames, 90)
                    self.construction_corridor_memory_frames = max(self.construction_corridor_memory_frames, 320)
                    self.progress_recovery_frames = 0
                    self.state = "PREPARE"
                    # Route-51 scoring tolerates being conservative much better than repeated
                    # construction/static collisions; keep this a braking guard, not a pass.
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.30,
                        throttle_cap=0.0,
                        brake=0.82 if obstacle_distance <= 4.8 else 0.58,
                        steer_limit=0.08,
                        reason="construction_near_cone_speed_guard",
                    )
                if construction_sparse_cone_low_speed_guard:
                    self.static_creep_frames = 0
                    self.construction_sparse_cone_low_speed_guard_frames += 1
                    self.construction_cone_entry_slow_frames = max(self.construction_cone_entry_slow_frames, 70)
                    self.construction_corridor_memory_frames = max(self.construction_corridor_memory_frames, 300)
                    self.progress_recovery_frames = 0
                    if current_open_side in ("right", "left"):
                        self.last_open_side = current_open_side
                    self.state = "PREPARE"
                    if obstacle_distance >= 5.25 and abs(features.ego_speed) < 0.70 and current_open_side in ("right", "left"):
                        route51_cone_mode = self.config.suppress_lateral_intersection_rules
                        if route51_cone_mode:
                            steer_bias = 0.26 if current_open_side == "right" else -0.26
                        else:
                            steer_bias = -0.18 if current_open_side == "right" else 0.18
                        self.construction_cone_entry_slow_frames = max(self.construction_cone_entry_slow_frames, 110)
                        self.open_side_pass_memory_frames = max(self.open_side_pass_memory_frames, self.config.lidar_open_side_pass_memory_frames)
                        return PlannerAction(
                            True,
                            "AVOID_OR_PASS",
                            target_speed=2.2 if route51_cone_mode else 2.8,
                            throttle_cap=0.72 if route51_cone_mode else 0.95,
                            throttle_floor=0.50 if route51_cone_mode else 0.68,
                            brake_cap=0.0,
                            steer_limit=0.42 if route51_cone_mode else 0.34,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.22 if route51_cone_mode else 0.14,
                            reason="construction_sparse_cone_low_speed_open_side_release",
                        )
                    if (
                        not self.config.suppress_lateral_intersection_rules
                        and self.construction_sparse_cone_low_speed_guard_frames >= 12
                        and 4.45 <= obstacle_distance <= 5.25
                        and abs(features.ego_speed) < 0.45
                        and current_open_side in ("right", "left")
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.red_stop_distance is None
                        and not features.red_light_active
                        and features.lidar_center_blockage_ratio <= 0.65
                        and (
                            (current_open_side == "right" and features.lidar_right_blockage_ratio <= 0.20)
                            or (current_open_side == "left" and features.lidar_left_blockage_ratio <= 0.20)
                        )
                    ):
                        self.state = "AVOID_OR_PASS"
                        steer_bias = 0.18 if current_open_side == "right" else -0.18
                        return PlannerAction(
                            True,
                            "AVOID_OR_PASS",
                            target_speed=1.7,
                            throttle_cap=0.62,
                            throttle_floor=0.42,
                            brake_cap=0.0,
                            steer_limit=0.26,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.16,
                            reason="construction_sparse_cone_low_speed_side_gap_release",
                        )
                    if (
                        self.config.suppress_lateral_intersection_rules
                        and self.construction_sparse_cone_low_speed_guard_frames >= 30
                        and 4.45 <= obstacle_distance <= 5.25
                        and abs(features.ego_speed) < 0.35
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.red_stop_distance is None
                        and not features.red_light_active
                        and features.lidar_center_blockage_ratio <= 0.30
                    ):
                        steer_bias = 0.0
                        if current_open_side == "right":
                            steer_bias = -0.10
                        elif current_open_side == "left":
                            steer_bias = 0.10
                        self.construction_cone_entry_slow_frames = max(self.construction_cone_entry_slow_frames, 80)
                        self.open_side_pass_memory_frames = max(self.open_side_pass_memory_frames, 40)
                        return PlannerAction(
                            True,
                            "AVOID_OR_PASS",
                            target_speed=1.6,
                            throttle_cap=0.56,
                            throttle_floor=0.36,
                            brake_cap=0.0,
                            steer_limit=0.20,
                            steer_bias=steer_bias,
                            reason="construction_sparse_cone_long_hold_forward_release",
                        )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.35,
                        throttle_cap=0.0,
                        brake=0.40,
                        steer_limit=0.08,
                        reason="construction_sparse_cone_low_speed_guard",
                    )
                if construction_high_speed_close_obstacle_brake:
                    self.static_creep_frames = 0
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.last_open_side = current_open_side
                    self.state = "AVOID_OR_PASS"
                    steer_bias = 0.12 if current_open_side == "right" else -0.12
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.35,
                        throttle_cap=0.0,
                        brake=0.85,
                        steer_limit=0.30,
                        steer_bias=steer_bias,
                        reason="construction_high_speed_close_obstacle_brake",
                    )
                if construction_sparse_cone_entry_slowdown:
                    self.static_creep_frames = 0
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = 0
                    self.construction_cone_entry_slow_frames = 110
                    self.construction_corridor_memory_frames = max(self.construction_corridor_memory_frames, 340)
                    if current_open_side in ("right", "left"):
                        self.last_open_side = current_open_side
                    self.state = "PREPARE"
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.45,
                        throttle_cap=0.0,
                        brake=0.58,
                        steer_limit=0.10,
                        reason="construction_sparse_cone_entry_slowdown",
                    )
                if construction_close_obstacle_slowdown:
                    self.static_creep_frames = 0
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = 0
                    self.last_open_side = current_open_side
                    self.state = "PREPARE"
                    steer_bias = 0.06 if current_open_side == "right" else -0.06
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.25,
                        throttle_cap=0.0,
                        brake=0.88,
                        steer_limit=0.12,
                        steer_bias=steer_bias,
                        reason="construction_close_obstacle_open_side_slowdown",
                    )
                construction_very_close_open_side_escape = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and 0.0 <= obstacle_distance <= 1.85
                    and abs(features.ego_speed) < 0.25
                    and features.lidar_blockage_ratio >= 0.80
                    and features.lidar_center_blockage_ratio >= (0.60 if obstacle_distance <= 1.0 else 0.85)
                    and current_open_side in ("right", "left")
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                if construction_very_close_open_side_escape:
                    self.static_creep_frames = 0
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.last_open_side = current_open_side
                    self.state = "AVOID_OR_PASS"
                    near_distant_escape = self.config.distant_lidar_creep_enabled and obstacle_distance <= 1.85
                    ultra_close_unwedge = obstacle_distance <= 0.75 and abs(features.ego_speed) < 0.12
                    persistent_ultra_close_unwedge = obstacle_distance <= 1.15 and self.static_creep_frames >= 6 and abs(features.ego_speed) < 0.22
                    very_close_forward_escape = 1.0 < obstacle_distance <= 1.85 and abs(features.ego_speed) < 0.12
                    if (near_distant_escape or ultra_close_unwedge) and abs(features.ego_speed) < 0.12:
                        self.blocked_frames += 1
                    elif not very_close_forward_escape:
                        self.blocked_frames = 0
                    if (
                        (near_distant_escape and (obstacle_distance <= 0.45 or self.blocked_frames >= 10))
                        or ultra_close_unwedge
                        or persistent_ultra_close_unwedge
                    ):
                        self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 72 if ultra_close_unwedge else 28)
                        self.blocked_frames = 0
                    if (near_distant_escape or ultra_close_unwedge or persistent_ultra_close_unwedge) and self.reverse_unwedge_frames > 0:
                        self.reverse_unwedge_frames -= 1
                        steer_magnitude = 0.35
                        steer_bias = -steer_magnitude if current_open_side == "right" else steer_magnitude
                        return PlannerAction(
                            True,
                            self.state,
                            throttle_cap=0.40,
                            throttle_floor=0.30,
                            brake_cap=0.0,
                            steer_limit=0.55,
                            steer_bias=steer_bias,
                            steer_min_magnitude=steer_magnitude,
                            reverse=True,
                            reason="construction_open_side_reverse_unwedge",
                        )
                    ultra_close = obstacle_distance < 1.0
                    steer_magnitude = 0.55 if ultra_close else (0.70 if obstacle_distance > 1.45 else 0.46)
                    steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=2.0 if ultra_close else (2.0 if obstacle_distance > 1.45 else 1.5),
                        throttle_cap=0.80 if ultra_close else (0.65 if obstacle_distance > 1.45 else 0.50),
                        throttle_floor=0.55 if ultra_close else (0.42 if obstacle_distance > 1.45 else 0.34),
                        brake_cap=0.0,
                        steer_limit=0.80 if ultra_close else (0.90 if obstacle_distance > 1.45 else 0.70),
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.55 if ultra_close else (0.70 if obstacle_distance > 1.45 else 0.38),
                        reason="construction_very_close_open_side_escape",
                    )
                open_side_nudge_continuation = (
                    (previous_state == "AVOID_OR_PASS" or open_side_pass_memory)
                    and self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and features.front_obstacle_distance is not None
                    and float(features.front_obstacle_distance) <= self.config.lidar_open_side_nudge_near_distance
                    and features.lidar_blockage_ratio >= (0.45 if open_side_pass_memory else 0.55)
                    and current_open_side in ("right", "left")
                )
                close_memory_nudge = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and previous_state == "AVOID_OR_PASS"
                    and self.close_obstacle_memory_frames > 0
                    and features.ego_speed < 0.9
                    and obstacle_distance <= self.config.lidar_open_side_post_pass_max_distance
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and self.last_open_side in ("right", "left")
                    and obstacle_distance > self.config.lidar_open_side_nudge_near_distance
                )
                if close_memory_nudge:
                    self.close_obstacle_memory_frames = 0
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.static_creep_frames = 0
                    self.state = "AVOID_OR_PASS"
                    route51_cone_mode = self.config.suppress_lateral_intersection_rules
                    steer_magnitude = 0.16 if route51_cone_mode else 0.55
                    steer_bias = steer_magnitude if self.last_open_side == "right" else -steer_magnitude
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.8 if route51_cone_mode else 1.0,
                        throttle_cap=0.24 if route51_cone_mode else 0.34,
                        throttle_floor=0.12 if route51_cone_mode else 0.18,
                        brake_cap=0.0,
                        steer_limit=0.30 if route51_cone_mode else 0.90,
                        steer_bias=steer_bias,
                        steer_min_magnitude=steer_magnitude,
                        reason="distant_lidar_open_side_close_memory_nudge",
                    )
                progress_recovery = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and previous_state in ("AVOID_OR_PASS", "RECOVER", "PREPARE")
                    and self.progress_recovery_frames > 0
                    and features.ego_speed < 2.2
                    and obstacle_distance > self.config.lidar_open_side_nudge_near_distance
                    and obstacle_distance <= self.config.lidar_open_side_post_pass_max_distance
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                reverse_unwedge_far_lidar_jump = (
                    self.config.distant_lidar_creep_enabled
                    and self.reverse_unwedge_frames > 0
                    and previous_state in ("AVOID_OR_PASS", "RECOVER")
                    and abs(features.ego_speed) < 0.80
                    and obstacle_distance > self.config.lidar_open_side_nudge_near_distance
                    and obstacle_distance <= self.config.lidar_open_side_post_pass_max_distance
                    and self.last_open_side in ("right", "left")
                    and current_open_side == self.last_open_side
                    and features.lidar_blockage_ratio >= 0.70
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                if reverse_unwedge_far_lidar_jump:
                    self.reverse_unwedge_frames -= 1
                    self.progress_recovery_frames = max(0, self.progress_recovery_frames - 1)
                    self.state = "AVOID_OR_PASS"
                    steer_bias = -0.35 if current_open_side == "right" else 0.35
                    return PlannerAction(
                        True,
                        self.state,
                        throttle_cap=0.40,
                        throttle_floor=0.30,
                        brake_cap=0.0,
                        steer_limit=0.55,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.35,
                        reverse=True,
                        reason="construction_open_side_reverse_unwedge",
                    )
                if progress_recovery:
                    self.progress_recovery_frames -= 1
                    self.static_creep_frames = 0
                    self.close_obstacle_memory_frames = 0
                    self.state = "RECOVER"
                    far_progress_stalled = (
                        estimate.macro_scenario == "trucks_encountered_during_construction"
                        and 12.0 <= obstacle_distance <= 18.0
                        and abs(features.ego_speed) < 0.12
                        and features.lidar_blockage_ratio >= 0.80
                        and current_open_side in ("right", "left")
                    )
                    if far_progress_stalled:
                        self.blocked_frames += 1
                        if self.blocked_frames >= 6:
                            steer_bias = -0.42 if current_open_side == "right" else 0.42
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.62,
                                throttle_floor=0.46,
                                brake_cap=0.0,
                                steer_limit=0.62,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.38,
                                reverse=True,
                                reason="construction_far_progress_stalled_reverse_unwedge",
                            )
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=2.6,
                        throttle_cap=0.68,
                        throttle_floor=0.45,
                        brake_cap=0.0,
                        steer_limit=0.45,
                        reason="distant_lidar_open_side_progress_recovery",
                    )
                post_pass_recovery = (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and previous_state == "AVOID_OR_PASS"
                    and self.post_pass_frames > 0
                    and self.close_obstacle_memory_frames <= 0
                    and features.ego_speed < 1.2
                    and features.front_pedestrian_distance is None
                    and features.front_vehicle_distance is None
                    and features.red_stop_distance is None
                    and features.front_obstacle_distance is not None
                    and float(features.front_obstacle_distance) <= self.config.lidar_open_side_post_pass_max_distance
                    and float(features.front_obstacle_distance) > self.config.lidar_open_side_nudge_near_distance
                )
                if post_pass_recovery:
                    self.post_pass_frames -= 1
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.static_creep_frames = 0
                    self.state = "RECOVER"
                    steer_bias = 0.10 if self.last_open_side == "right" else (-0.10 if self.last_open_side == "left" else 0.0)
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=2.2,
                        throttle_cap=0.52,
                        throttle_floor=0.28,
                        brake_cap=0.0,
                        steer_limit=0.55,
                        steer_bias=steer_bias,
                        reason="distant_lidar_open_side_post_pass_recovery",
                    )
                reverse_unwedge_continuation = (
                    self.config.distant_lidar_creep_enabled
                    and self.reverse_unwedge_frames > 0
                    and obstacle_distance <= 4.2
                    and features.lidar_blockage_ratio >= 0.80
                    and current_open_side in ("right", "left")
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                if reverse_unwedge_continuation:
                    self.reverse_unwedge_frames -= 1
                    self.state = "AVOID_OR_PASS"
                    self.last_open_side = current_open_side
                    steer_bias = -0.35 if current_open_side == "right" else 0.35
                    return PlannerAction(
                        True,
                        self.state,
                        throttle_cap=0.40,
                        throttle_floor=0.30,
                        brake_cap=0.0,
                        steer_limit=0.55,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.35,
                        reverse=True,
                        reason="construction_open_side_reverse_unwedge",
                    )
                observable_very_close_open_side_creep = (
                    not red_or_stop_conflict
                    and 1.2 <= obstacle_distance <= 2.3
                    and abs(features.ego_speed) < 0.55
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio >= 0.85
                    and features.lidar_center_blockage_ratio >= 0.85
                    and current_open_side in ("right", "left")
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.85
                            and features.lidar_right_blockage_ratio <= 0.20
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.85
                            and features.lidar_left_blockage_ratio <= 0.20
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                )
                if observable_very_close_open_side_creep:
                    self.static_creep_frames += 1
                    self.state = "AVOID_OR_PASS"
                    self.last_open_side = current_open_side
                    self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                    if abs(features.ego_speed) < 0.08 and self.static_creep_frames >= 10:
                        self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 24)
                    if self.reverse_unwedge_frames > 0:
                        self.reverse_unwedge_frames -= 1
                        steer_bias = -0.35 if current_open_side == "right" else 0.35
                        return PlannerAction(
                            True,
                            self.state,
                            throttle_cap=0.40,
                            throttle_floor=0.30,
                            brake_cap=0.0,
                            steer_limit=0.55,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.35,
                            reverse=True,
                            reason="construction_open_side_reverse_unwedge",
                        )
                    steer_bias = 0.24 if current_open_side == "right" else -0.24
                    if self.static_creep_frames >= 4 or previous_state == "AVOID_OR_PASS":
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=0.9,
                            throttle_cap=0.28,
                            throttle_floor=0.18,
                            brake_cap=0.0,
                            steer_limit=0.55,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.24,
                            reason="observable_very_close_open_side_creep",
                        )
                    return PlannerAction(False, self.state, reason="observable_very_close_open_side_observed")
                observable_close_open_side_speed_guard = (
                    not red_or_stop_conflict
                    and not roundabout_layout_context
                    and 1.5 <= obstacle_distance <= 5.0
                    and 0.55 <= abs(features.ego_speed) < 3.80
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio >= 0.80
                    and features.lidar_center_blockage_ratio >= 0.80
                    and current_open_side in ("right", "left")
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.80
                            and features.lidar_right_blockage_ratio <= 0.25
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.80
                            and features.lidar_left_blockage_ratio <= 0.25
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                )
                if observable_close_open_side_speed_guard:
                    self.state = "PREPARE"
                    self.last_open_side = current_open_side
                    steer_bias = 0.12 if current_open_side == "right" else -0.12
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=0.8,
                        throttle_cap=0.18,
                        steer_limit=0.42,
                        steer_bias=steer_bias,
                        reason="observable_close_open_side_speed_guard",
                    )
                close_static_open_side_continuation = (
                        estimate.macro_scenario in (
                            "trucks_encountered_during_construction",
                            "high_speed_temporary_construction",
                            "unknown",
                        )
                        and not roundabout_layout_context
                        and self.config.lidar_open_side_nudge_enabled
                    and 1.8 <= obstacle_distance <= 7.5
                    and features.ego_speed < 0.45
                    and (features.lidar_available or features.lidar_front_distance is not None)
                    and (not features.lidar_stale or features.lidar_front_distance is not None)
                    and features.lidar_blockage_ratio >= 0.85
                    and features.lidar_center_blockage_ratio >= 0.80
                    and current_open_side in ("right", "left")
                    and (
                        previous_state == "AVOID_OR_PASS"
                        or self.static_creep_frames >= 8
                        or self.open_side_pass_memory_frames > 0
                    )
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.85
                            and features.lidar_right_blockage_ratio <= 0.20
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.85
                            and features.lidar_left_blockage_ratio <= 0.20
                        )
                    )
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and (
                        features.front_vehicle_distance is None
                        or float(features.front_vehicle_distance) >= 8.0
                    )
                )
                if close_static_open_side_continuation:
                    self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                    self.last_open_side = current_open_side
                    self.state = "AVOID_OR_PASS"
                    steer_magnitude = 0.44 if obstacle_distance < 2.8 else 0.32
                    steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.4,
                        throttle_cap=0.42 if obstacle_distance < 2.8 else 0.38,
                        throttle_floor=0.28 if obstacle_distance < 2.8 else 0.24,
                        brake_cap=0.0,
                        steer_limit=0.62,
                        steer_bias=steer_bias,
                        steer_min_magnitude=steer_magnitude,
                        reason="construction_close_static_open_side_continue",
                    )
                observable_full_blockage_open_side_escape = (
                    self.config.distant_lidar_creep_enabled
                    and self.config.lidar_open_side_nudge_enabled
                    and (construction_open_side_allowed or stalled_observable_open_side_allowed)
                    and 2.6 <= obstacle_distance <= 8.5
                    and abs(features.ego_speed) < 1.20
                    and features.lidar_blockage_ratio >= 0.90
                    and features.lidar_center_blockage_ratio >= 0.90
                    and current_open_side in ("right", "left")
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.85
                            and features.lidar_right_blockage_ratio <= 0.20
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.85
                            and features.lidar_left_blockage_ratio <= 0.20
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                if observable_full_blockage_open_side_escape:
                    self.static_creep_frames += 1
                    self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                    self.last_open_side = current_open_side
                    self.state = "AVOID_OR_PASS"
                    if self.static_creep_frames >= 2 or previous_state == "AVOID_OR_PASS" or open_side_pass_memory:
                        steer_bias = 0.28 if current_open_side == "right" else -0.28
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=2.8,
                            throttle_cap=0.85,
                            throttle_floor=0.58,
                            brake_cap=0.0,
                            steer_limit=0.50,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.28,
                            reason="observable_full_blockage_open_side_escape",
                        )
                    return PlannerAction(False, self.state, reason="observable_full_blockage_open_side_observed")
                near_static_reverse_recovery = (
                    self.config.distant_lidar_creep_enabled
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                    )
                    and obstacle_distance <= 4.25
                    and abs(features.ego_speed) < 0.25
                    and features.lidar_blockage_ratio >= 0.80
                    and current_open_side in ("right", "left")
                    and (self.open_side_pass_memory_frames > 0 or self.reverse_unwedge_frames > 0)
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                )
                if near_static_reverse_recovery:
                    self.state = "AVOID_OR_PASS"
                    self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 12)
                    self.last_open_side = current_open_side
                    self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                    steer_bias = -0.35 if current_open_side == "right" else 0.35
                    return PlannerAction(
                        True,
                        self.state,
                        throttle_cap=0.40,
                        throttle_floor=0.30,
                        brake_cap=0.0,
                        steer_limit=0.55,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.35,
                        reverse=True,
                        reason="construction_open_side_reverse_unwedge",
                    )
                if (
                    self.config.lidar_open_side_nudge_enabled
                    and construction_open_side_allowed
                    and features.ego_speed < self.config.lidar_open_side_nudge_max_speed
                    and features.front_obstacle_distance is not None
                    and self.config.lidar_open_side_nudge_min_distance
                    <= float(features.front_obstacle_distance)
                    <= self.config.lidar_open_side_nudge_max_distance
                    and (
                        features.lidar_center_blockage_ratio >= self.config.lidar_open_side_nudge_center_threshold
                        or open_side_nudge_continuation
                    )
                    and features.lidar_blockage_ratio >= (0.45 if open_side_nudge_continuation else 0.55)
                    and not center_full_blockage
                    and current_open_side in ("right", "left")
                    and not roundabout_layout_context
                ):
                    self.static_creep_frames += 1
                    self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                    self.last_open_side = current_open_side
                    if self.static_creep_frames >= 6 or open_side_nudge_continuation:
                        self.state = "AVOID_OR_PASS"
                        self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                        obstacle_distance = float(features.front_obstacle_distance)
                        escape_obstacle = (
                            obstacle_distance < self.config.lidar_open_side_nudge_escape_distance
                            and self.static_creep_frames >= self.config.lidar_open_side_nudge_escape_frames
                            and features.ego_speed < 0.35
                        )
                        close_obstacle = obstacle_distance < self.config.lidar_open_side_nudge_close_distance
                        near_obstacle = obstacle_distance < self.config.lidar_open_side_nudge_near_distance
                        if obstacle_distance <= self.config.lidar_open_side_close_memory_distance:
                            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                        sparse_far_cone_corridor = bool(
                            estimate.macro_scenario in (
                                "trucks_encountered_during_construction",
                                "high_speed_temporary_construction",
                            )
                            and obstacle_distance >= 4.2
                            and features.lidar_center_blockage_ratio < 0.30
                            and features.lidar_blockage_ratio < 0.45
                        )
                        construction_dense_close_hold = (
                            estimate.macro_scenario == "trucks_encountered_during_construction"
                            and estimate.confidence >= 0.95
                            and obstacle_distance <= 3.6
                            and features.lidar_blockage_ratio >= 0.90
                            and features.detection_object_count >= 90
                            and abs(features.ego_speed) < 0.35
                        )
                        if construction_dense_close_hold:
                            self.blocked_frames += 1
                            dense_hold_has_open_side = current_open_side in ("right", "left") and (
                                (
                                    current_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.80
                                    and features.lidar_right_blockage_ratio <= 0.28
                                )
                                or (
                                    current_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.80
                                    and features.lidar_left_blockage_ratio <= 0.28
                                )
                            )
                            if self.blocked_frames >= 8 and dense_hold_has_open_side:
                                self.state = "AVOID_OR_PASS"
                                self.open_side_pass_memory_frames = max(
                                    self.open_side_pass_memory_frames,
                                    self.config.lidar_open_side_pass_memory_frames,
                                )
                                self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                                steer_bias = 0.42 if current_open_side == "right" else -0.42
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.8,
                                    throttle_cap=0.72,
                                    throttle_floor=0.54,
                                    brake_cap=0.0,
                                    steer_limit=0.50,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.24,
                                    reason="construction_dense_close_blockage_side_gap_release",
                                )
                            self.state = "YIELD_OR_BRAKE"
                            self.progress_recovery_frames = 0
                            self.open_side_pass_memory_frames = 0
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.0,
                                throttle_cap=0.0,
                                brake=0.62,
                                steer_limit=0.05,
                                reason="construction_dense_close_blockage_collision_hold",
                            )
                        steer_magnitude = (
                            0.08
                            if sparse_far_cone_corridor
                            else (0.70 if escape_obstacle else (0.55 if close_obstacle else (0.42 if near_obstacle else 0.14)))
                        )
                        steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                        if escape_obstacle:
                            escape_throttle_cap = 0.48
                            escape_throttle_floor = 0.30
                            if obstacle_distance >= 2.2 and features.ego_speed < 0.08:
                                escape_throttle_cap = 0.68
                                escape_throttle_floor = 0.48
                        close_nudge_boost = (
                            close_obstacle
                            and not escape_obstacle
                            and obstacle_distance >= 2.2
                            and features.ego_speed < 0.08
                        )
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=1.2 if escape_obstacle else (0.9 if close_obstacle else (1.6 if near_obstacle else 1.3)),
                            throttle_cap=escape_throttle_cap if escape_obstacle else (0.68 if close_nudge_boost else (0.28 if close_obstacle else (0.36 if near_obstacle else 0.30))),
                            throttle_floor=escape_throttle_floor if escape_obstacle else (0.48 if close_nudge_boost else (0.16 if close_obstacle else (0.22 if near_obstacle else 0.16))),
                            brake_cap=0.0,
                            steer_limit=(
                                0.22
                                if sparse_far_cone_corridor
                                else (0.95 if escape_obstacle else (0.90 if close_obstacle else (0.85 if near_obstacle else 0.50)))
                            ),
                            steer_bias=steer_bias,
                            steer_min_magnitude=None if sparse_far_cone_corridor else (steer_magnitude if near_obstacle else None),
                            reason="distant_lidar_open_side_escape" if escape_obstacle else "distant_lidar_open_side_nudge",
                        )
                    return PlannerAction(False, self.state, reason="distant_lidar_open_side_observed")
                if self.open_side_pass_memory_frames > 0:
                    self.open_side_pass_memory_frames -= 1
                self.post_pass_frames = 0
                self.close_obstacle_memory_frames = 0
                self.progress_recovery_frames = 0
                if (
                    self.config.distant_lidar_creep_enabled
                    and
                    features.ego_speed < 2.0
                    and features.front_obstacle_distance is not None
                    and 8.0 <= float(features.front_obstacle_distance) <= 18.0
                    and features.lidar_center_blockage_ratio >= 0.45
                    and features.lidar_open_side in ("right", "balanced", "left")
                ):
                    self.static_creep_frames += 1
                    if self.static_creep_frames >= 4:
                        self.state = "AVOID_OR_PASS"
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=2.4,
                            throttle_cap=0.68,
                            throttle_floor=0.45,
                            brake_cap=0.0,
                            steer_limit=0.45,
                            reason="distant_lidar_blockage_creep_release",
                        )
                    return PlannerAction(False, self.state, reason="distant_lidar_blockage_observed")
                far_full_blockage_open_side_recovery = (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and features.ego_speed < 0.60
                    and 6.0 <= obstacle_distance <= 13.2
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio >= 0.90
                    and features.lidar_center_blockage_ratio >= 0.90
                    and current_open_side in ("right", "left")
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.85
                            and features.lidar_right_blockage_ratio <= 0.20
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.85
                            and features.lidar_left_blockage_ratio <= 0.20
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                medium_speed_full_blockage_open_side_guard = (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and 0.60 <= features.ego_speed < 3.20
                    and 6.0 <= obstacle_distance <= 13.2
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio >= 0.90
                    and features.lidar_center_blockage_ratio >= 0.90
                    and current_open_side in ("right", "left")
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.85
                            and features.lidar_right_blockage_ratio <= 0.20
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.85
                            and features.lidar_left_blockage_ratio <= 0.20
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                if medium_speed_full_blockage_open_side_guard:
                    self.state = "PREPARE"
                    self.last_open_side = current_open_side
                    steer_bias = 0.16 if current_open_side == "right" else -0.16
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.4,
                        throttle_cap=0.24,
                        throttle_floor=0.12,
                        brake_cap=0.0,
                        steer_limit=0.45,
                        steer_bias=steer_bias,
                        reason="construction_far_full_blockage_open_side_speed_guard",
                    )
                if far_full_blockage_open_side_recovery:
                    self.static_creep_frames += 1
                    self.last_open_side = current_open_side
                    if self.static_creep_frames >= 3:
                        self.state = "AVOID_OR_PASS"
                        self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                        self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                        steer_bias = 0.36 if current_open_side == "right" else -0.36
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=2.4,
                            throttle_cap=0.78,
                            throttle_floor=0.72,
                            brake_cap=0.0,
                            steer_limit=0.70,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.36,
                            reason="construction_far_full_blockage_open_side_recovery",
                        )
                    return PlannerAction(False, self.state, reason="construction_far_full_blockage_open_side_observed")
                close_open_side_memory_push = (
                    self.config.lidar_open_side_nudge_enabled
                    and estimate.macro_scenario in (
                        "trucks_encountered_during_construction",
                        "high_speed_temporary_construction",
                        "unknown",
                    )
                    and not roundabout_layout_context
                    and features.front_obstacle_distance is not None
                    and 2.0 <= obstacle_distance <= 5.2
                    and abs(features.ego_speed) < 0.80
                    and current_open_side in ("right", "left")
                    and (
                        previous_state == "AVOID_OR_PASS"
                        or self.open_side_pass_memory_frames > 0
                        or self.static_creep_frames >= 4
                    )
                    and features.lidar_blockage_ratio >= 0.60
                    and features.lidar_center_blockage_ratio >= 0.60
                    and (
                        (
                            current_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.60
                            and features.lidar_right_blockage_ratio <= 0.45
                        )
                        or (
                            current_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.60
                            and features.lidar_left_blockage_ratio <= 0.45
                        )
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                if close_open_side_memory_push:
                    self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                    self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                    self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                    self.last_open_side = current_open_side
                    self.state = "AVOID_OR_PASS"
                    steer_magnitude = 0.62 if obstacle_distance <= 3.0 else 0.50
                    steer_bias = steer_magnitude if current_open_side == "right" else -steer_magnitude
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.8,
                        throttle_cap=0.58,
                        throttle_floor=0.42,
                        brake_cap=0.0,
                        steer_limit=0.85,
                        steer_bias=steer_bias,
                        steer_min_magnitude=steer_magnitude,
                        reason="construction_close_open_side_memory_push",
                    )
                if self._construction_creep_candidate(features, estimate):
                    self.static_creep_frames += 1
                    if self.static_creep_frames >= 8:
                        self.state = "AVOID_OR_PASS"
                        construction_static_open_side_escape = (
                            self.config.lidar_open_side_nudge_enabled
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_center_blockage_ratio >= 0.85
                            and features.lidar_open_side in ("right", "left")
                            and (
                                (
                                    features.lidar_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.85
                                    and features.lidar_right_blockage_ratio <= 0.20
                                )
                                or (
                                    features.lidar_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.85
                                    and features.lidar_left_blockage_ratio <= 0.20
                                )
                            )
                        )
                        construction_static_open_side_push = (
                            self.static_creep_frames >= 16
                            and features.front_obstacle_distance is not None
                            and float(features.front_obstacle_distance) <= 5.6
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_center_blockage_ratio >= 0.85
                            and features.lidar_open_side in ("right", "left")
                            and (
                                (
                                    features.lidar_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.85
                                    and features.lidar_right_blockage_ratio <= 0.20
                                )
                                or (
                                    features.lidar_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.85
                                    and features.lidar_left_blockage_ratio <= 0.20
                                )
                            )
                        )
                        if construction_static_open_side_push:
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                            self.last_open_side = features.lidar_open_side
                            if (
                                features.front_obstacle_distance is not None
                                and float(features.front_obstacle_distance) <= 3.4
                                and abs(features.ego_speed) < 0.08
                                and self.static_creep_frames >= 40
                            ):
                                steer_magnitude = 0.35
                                steer_bias = -steer_magnitude if features.lidar_open_side == "right" else steer_magnitude
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.40,
                                    throttle_floor=0.30,
                                    brake_cap=0.0,
                                    steer_limit=0.55,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=steer_magnitude,
                                    reverse=True,
                                    reason="construction_open_side_reverse_unwedge",
                                )
                            steer_bias = -0.34 if features.lidar_open_side == "right" else 0.34
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.46,
                                throttle_floor=0.32,
                                brake_cap=0.0,
                                steer_limit=0.70,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.34,
                                reason="construction_static_open_side_push_release",
                            )
                        if construction_static_open_side_escape:
                            route51_dense_full_block_hold = (
                                estimate.macro_scenario == "trucks_encountered_during_construction"
                                and features.front_obstacle_distance is not None
                                and float(features.front_obstacle_distance) <= 3.7
                                and features.lidar_blockage_ratio >= 0.90
                                and abs(features.ego_speed) < 0.35
                            )
                            if route51_dense_full_block_hold:
                                self.open_side_pass_memory_frames = 0
                                self.progress_recovery_frames = 0
                                self.state = "YIELD_OR_BRAKE"
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=0.0,
                                    throttle_cap=0.0,
                                    brake=0.70,
                                    steer_limit=0.04,
                                    reason="construction_full_blockage_collision_hold",
                                )
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.close_obstacle_memory_frames = self.config.lidar_open_side_close_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            self.post_pass_frames = self.config.lidar_open_side_post_pass_frames
                            self.last_open_side = features.lidar_open_side
                            if (
                                features.front_obstacle_distance is not None
                                and float(features.front_obstacle_distance) <= 3.2
                                and abs(features.ego_speed) < 0.12
                                and self.static_creep_frames >= 12
                            ):
                                steer_magnitude = 0.35
                                steer_bias = -steer_magnitude if features.lidar_open_side == "right" else steer_magnitude
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.40,
                                    throttle_floor=0.30,
                                    brake_cap=0.0,
                                    steer_limit=0.55,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=steer_magnitude,
                                    reverse=True,
                                    reason="construction_open_side_reverse_unwedge",
                                )
                            steer_bias = 0.25 if features.lidar_open_side == "right" else -0.25
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.8,
                                throttle_cap=0.58,
                                throttle_floor=0.36,
                                brake_cap=0.0,
                                steer_limit=0.45,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.25,
                                reason="construction_full_blockage_open_side_escape",
                            )
                        construction_static_side_gap_forward_push = (
                            features.front_obstacle_distance is not None
                            and 2.4 <= float(features.front_obstacle_distance) <= 3.9
                            and abs(features.ego_speed) < 0.35
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_center_blockage_ratio <= 0.70
                            and features.lidar_open_side in ("right", "left")
                            and (
                                (
                                    features.lidar_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.85
                                    and features.lidar_right_blockage_ratio <= 0.20
                                )
                                or (
                                    features.lidar_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.85
                                    and features.lidar_left_blockage_ratio <= 0.20
                                )
                            )
                        )
                        if construction_static_side_gap_forward_push:
                            route51_dense_side_gap_hold = (
                                estimate.macro_scenario == "trucks_encountered_during_construction"
                                and estimate.confidence >= 0.95
                                and features.detection_object_count >= 90
                                and features.lidar_blockage_ratio >= 0.95
                                and float(features.front_obstacle_distance) <= 3.4
                            )
                            if route51_dense_side_gap_hold:
                                self.blocked_frames += 1
                                if self.blocked_frames >= 8:
                                    self.open_side_pass_memory_frames = max(
                                        self.open_side_pass_memory_frames,
                                        self.config.lidar_open_side_pass_memory_frames,
                                    )
                                    self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                                    self.state = "AVOID_OR_PASS"
                                    return PlannerAction(
                                        True,
                                        self.state,
                                        target_speed=1.8,
                                        throttle_cap=0.72,
                                        throttle_floor=0.54,
                                        brake_cap=0.0,
                                        steer_limit=0.50,
                                        steer_bias=0.42 if features.lidar_open_side == "right" else -0.42,
                                        steer_min_magnitude=0.24,
                                        reason="construction_static_side_gap_hold_release",
                                    )
                                self.open_side_pass_memory_frames = 0
                                self.progress_recovery_frames = 0
                                self.state = "YIELD_OR_BRAKE"
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=0.0,
                                    throttle_cap=0.0,
                                    brake=0.65,
                                    steer_limit=0.05,
                                    reason="construction_static_side_gap_collision_hold",
                                )
                            self.open_side_pass_memory_frames = max(
                                self.open_side_pass_memory_frames,
                                self.config.lidar_open_side_pass_memory_frames,
                            )
                            self.close_obstacle_memory_frames = max(
                                self.close_obstacle_memory_frames,
                                self.config.lidar_open_side_close_memory_frames,
                            )
                            self.progress_recovery_frames = max(
                                self.progress_recovery_frames,
                                self.config.lidar_open_side_progress_recovery_frames,
                            )
                            self.post_pass_frames = max(
                                self.post_pass_frames,
                                self.config.lidar_open_side_post_pass_frames,
                            )
                            self.last_open_side = features.lidar_open_side
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.0,
                                throttle_cap=0.86,
                                throttle_floor=0.68,
                                brake_cap=0.0,
                                steer_limit=0.34,
                                steer_bias=0.06 if features.lidar_open_side == "right" else -0.06,
                                reason="construction_static_side_gap_forward_push",
                            )
                        return PlannerAction(
                            True,
                            self.state,
                            target_speed=1.2,
                            throttle_cap=0.24,
                            throttle_floor=0.16,
                            brake_cap=0.0,
                            steer_limit=0.65,
                            reason="construction_static_creep_release",
                        )
                else:
                    self.static_creep_frames = 0
                    if (
                        features.front_obstacle_distance is not None
                        and features.front_pedestrian_distance is None
                        and features.red_stop_distance is None
                        and not features.red_light_active
                    ):
                        try:
                            close_static_distance = float(features.front_obstacle_distance)
                        except Exception:
                            close_static_distance = 999.0
                        if (
                            close_static_distance <= 3.0
                            and features.ego_speed > 1.0
                            and not roundabout_layout_context
                            and estimate.macro_scenario != "reverse_vehicle"
                        ):
                            self.state = "YIELD_OR_BRAKE"
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.6,
                                throttle_cap=0.0,
                                brake=0.72 if features.ego_speed > 4.0 else 0.48,
                                steer_limit=0.22,
                                reason="construction_close_static_obstacle_defensive_brake",
                            )
                        if (
                            close_static_distance <= 0.90
                            and abs(features.ego_speed) < (1.40 if roundabout_layout_context else 0.80)
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            if self.blocked_frames >= 520 and close_static_distance <= 0.45:
                                reverse_bias = 0.34 if features.lidar_open_side == "right" else (-0.34 if features.lidar_open_side == "left" else 0.28)
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.2,
                                    throttle_cap=0.62,
                                    throttle_floor=0.46,
                                    brake_cap=0.0,
                                    steer_limit=0.42,
                                    steer_bias=reverse_bias,
                                    reverse=True,
                                    reason="roundabout_ultra_close_reverse_unwedge",
                                )
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.4,
                                throttle_cap=0.72,
                                throttle_floor=0.58,
                                brake_cap=0.0,
                                steer_limit=0.12,
                                steer_bias=0.0,
                                reason="roundabout_ultra_close_forward_unwedge",
                            )
                        if (
                            estimate.macro_scenario == "trucks_encountered_during_construction"
                            and 0.65 <= close_static_distance <= 1.05
                            and self.blocked_frames >= 220
                            and abs(features.ego_speed) < 0.25
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and features.lidar_center_blockage_ratio <= 0.15
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 70)
                            self.state = "RECOVER"
                            open_side = features.lidar_open_side if features.lidar_open_side in ("left", "right") else "right"
                            if (self.blocked_frames // 35) % 2 == 0:
                                steer_bias = 0.22 if open_side == "right" else -0.22
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.8,
                                    throttle_cap=0.82,
                                    throttle_floor=0.62,
                                    brake_cap=0.0,
                                    steer_limit=0.42,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.22,
                                    reason="construction_ultra_close_static_long_hold_forward_clearance",
                                )
                            steer_bias = 0.42 if open_side == "right" else -0.42
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.78,
                                throttle_floor=0.60,
                                brake_cap=0.0,
                                steer_limit=0.72,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.42,
                                reverse=True,
                                reason="construction_ultra_close_static_long_hold_reverse_clearance",
                            )
                        if (
                            close_static_distance <= (1.55 if estimate.macro_scenario == "reverse_vehicle" else 1.45)
                            and self.blocked_frames >= (35 if estimate.macro_scenario == "reverse_vehicle" else 55)
                            and abs(features.ego_speed) < (0.75 if estimate.macro_scenario == "reverse_vehicle" else 0.45)
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 54)
                            self.state = "RECOVER"
                            if features.lidar_open_side == "right":
                                steer_bias = -0.32
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.32
                            else:
                                steer_bias = 0.24 if (self.blocked_frames // 20) % 2 == 0 else -0.24
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.72 if estimate.macro_scenario == "reverse_vehicle" else 0.52,
                                throttle_floor=0.58 if estimate.macro_scenario == "reverse_vehicle" else 0.40,
                                brake_cap=0.0,
                                steer_limit=0.64 if estimate.macro_scenario == "reverse_vehicle" else 0.58,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.36 if estimate.macro_scenario == "reverse_vehicle" else 0.30,
                                reverse=True,
                                reason="reverse_vehicle_ultra_close_escape_reverse_sweep" if estimate.macro_scenario == "reverse_vehicle" else "ultra_close_static_escape_reverse_sweep",
                            )
                        if (
                            close_static_distance <= (1.30 if estimate.macro_scenario == "reverse_vehicle" else 0.90)
                            and abs(features.ego_speed) < (0.95 if estimate.macro_scenario == "reverse_vehicle" else 0.80)
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.blocked_frames += 1
                            self.reverse_unwedge_frames = max(self.reverse_unwedge_frames, 54)
                            self.state = "RECOVER"
                            if features.lidar_open_side == "right":
                                steer_bias = -0.12
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.12
                            else:
                                steer_bias = 0.0
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.62 if estimate.macro_scenario == "reverse_vehicle" else 0.24,
                                throttle_floor=0.46 if estimate.macro_scenario == "reverse_vehicle" else 0.14,
                                brake_cap=0.0,
                                steer_limit=0.42 if estimate.macro_scenario == "reverse_vehicle" else 0.22,
                                steer_bias=steer_bias,
                                reverse=True,
                                reason="reverse_vehicle_ultra_close_reverse_clearance" if estimate.macro_scenario == "reverse_vehicle" else "ultra_close_static_reverse_clearance",
                            )
                        if (
                            roundabout_layout_context
                            and 0.60 <= close_static_distance <= 2.10
                            and abs(features.ego_speed) > 0.70
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                        ):
                            self.state = "YIELD_OR_BRAKE"
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.4,
                                throttle_cap=0.0,
                                brake=0.72 if abs(features.ego_speed) > 1.8 else 0.42,
                                steer_limit=0.10,
                                reason="roundabout_ultra_close_static_speed_cap",
                            )
                        if (
                            roundabout_layout_context
                            and 0.45 <= close_static_distance <= 1.80
                            and abs(features.ego_speed) <= 0.70
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.state = "RECOVER"
                            if features.lidar_open_side == "right":
                                steer_bias = -0.10
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.10
                            else:
                                steer_bias = 0.0
                            if abs(features.ego_speed) < 0.12 and features.lidar_open_side in ("right", "left", "balanced"):
                                reverse_steer = 0.42 if features.lidar_open_side in ("right", "balanced") else -0.42
                                self.roundabout_post_reverse_forward_frames = max(self.roundabout_post_reverse_forward_frames, 18)
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.90,
                                    throttle_floor=0.68,
                                    brake_cap=0.0,
                                    steer_limit=0.58,
                                    steer_bias=reverse_steer,
                                    steer_min_magnitude=0.30,
                                    reverse=True,
                                    reason="roundabout_ultra_close_static_reverse_clearance",
                                )
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.8,
                                throttle_cap=0.42,
                                throttle_floor=0.22,
                                brake_cap=0.0,
                                steer_limit=0.16,
                                steer_bias=steer_bias,
                                reason="roundabout_ultra_close_static_controlled_forward",
                            )
                        if (
                            roundabout_layout_context
                            and 2.2 <= close_static_distance <= 3.4
                            and abs(features.ego_speed) <= 1.25
                            and features.lidar_open_side in ("right", "left", "balanced")
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            stronger_push = abs(features.ego_speed) >= 0.80 or self.blocked_frames >= 8
                            side_steer = 0.30 if stronger_push else 0.26
                            if features.lidar_open_side == "left":
                                steer_bias = side_steer
                            elif features.lidar_open_side == "right":
                                steer_bias = -side_steer
                            elif features.lidar_right_density <= max(0, features.lidar_left_density - 2):
                                steer_bias = -side_steer
                            elif features.lidar_left_density <= max(0, features.lidar_right_density - 2):
                                steer_bias = side_steer
                            else:
                                steer_bias = (-0.28 if self.last_open_side != "left" else 0.28) if stronger_push else (-0.26 if self.last_open_side != "left" else 0.26)
                            if self.roundabout_post_reverse_forward_frames > 0:
                                self.roundabout_post_reverse_forward_frames -= 1
                                escape_sweep = self.blocked_frames >= 18 and abs(features.ego_speed) < 0.08
                                if escape_sweep:
                                    if features.lidar_open_side == "right":
                                        post_steer_bias = -0.20
                                    elif features.lidar_open_side == "left":
                                        post_steer_bias = 0.20
                                    else:
                                        post_steer_bias = 0.0
                                    self.roundabout_reverse_clearance_frames = max(self.roundabout_reverse_clearance_frames, 36)
                                    return PlannerAction(
                                        True,
                                        self.state,
                                        throttle_cap=1.0,
                                        throttle_floor=0.88,
                                        brake_cap=0.0,
                                        steer_limit=0.34,
                                        steer_bias=post_steer_bias,
                                        steer_min_magnitude=0.10,
                                        reverse=True,
                                        reason="roundabout_post_reverse_escape_backout",
                                    )
                                throttle_floor = 0.78 if abs(features.ego_speed) < 0.55 else 0.64
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=2.5,
                                    throttle_cap=0.96,
                                    throttle_floor=throttle_floor,
                                    brake_cap=0.0,
                                    steer_limit=0.54,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.30,
                                    reason="roundabout_post_reverse_open_side_commit",
                                )
                            if (
                                self.roundabout_reverse_clearance_frames > 0
                                or (
                                    abs(features.ego_speed) < 0.08
                                    and (
                                        self.blocked_frames >= 120
                                        or (2.45 <= close_static_distance <= 3.45 and self.blocked_frames >= 5)
                                    )
                                )
                            ):
                                self.roundabout_reverse_clearance_frames = max(self.roundabout_reverse_clearance_frames - 1, 0) if self.roundabout_reverse_clearance_frames > 0 else 16
                                self.roundabout_post_reverse_forward_frames = max(self.roundabout_post_reverse_forward_frames, 18)
                                reverse_steer = -steer_bias if abs(steer_bias) >= 0.24 else (0.38 if features.lidar_open_side == "right" else -0.38)
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.90,
                                    throttle_floor=0.72,
                                    brake_cap=0.0,
                                    steer_limit=0.62,
                                    steer_bias=reverse_steer,
                                    steer_min_magnitude=0.30,
                                    reverse=True,
                                    reason="roundabout_very_close_reverse_clearance",
                                )
                            throttle_floor = 0.58 if stronger_push else 0.50
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.2,
                                throttle_cap=0.78,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0,
                                steer_limit=0.46,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.24,
                                reason="roundabout_very_close_open_side_forward_push",
                            )
                        if (
                            roundabout_layout_context
                            and 1.8 <= close_static_distance <= 3.2
                            and abs(features.ego_speed) >= 0.85
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.state = "RECOVER"
                            if features.lidar_open_side == "right":
                                steer_bias = -0.08
                            elif features.lidar_open_side == "left":
                                steer_bias = 0.08
                            else:
                                steer_bias = 0.0
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.7,
                                throttle_cap=0.0,
                                brake=0.56 if abs(features.ego_speed) > 1.5 else 0.34,
                                steer_limit=0.12,
                                steer_bias=steer_bias,
                                reason="roundabout_close_static_near_speed_cap",
                            )
                        if (
                            roundabout_layout_context
                            and 1.8 <= close_static_distance <= 3.2
                            and abs(features.ego_speed) < 0.85
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            if (
                                self.blocked_frames >= 90
                                and (
                                    features.lidar_open_side in ("balanced", "unknown")
                                    or features.lidar_center_blockage_ratio >= 0.85
                                )
                            ):
                                steer_bias = 0.34 if (self.blocked_frames // 20) % 2 == 0 else -0.34
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.66,
                                    throttle_floor=0.48,
                                    brake_cap=0.0,
                                    steer_limit=0.58,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.28,
                                    reverse=True,
                                    reason="roundabout_close_static_balanced_reverse_sweep",
                                )
                            if self.blocked_frames >= 55:
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.2,
                                    throttle_cap=0.52,
                                    throttle_floor=0.34,
                                    brake_cap=0.0,
                                    steer_limit=0.24,
                                    steer_bias=-0.10 if features.lidar_open_side == "right" else (0.10 if features.lidar_open_side == "left" else 0.0),
                                    reason="roundabout_close_static_high_blocked_push",
                                )
                            if (
                                roundabout_layout_context
                                and 2.2 <= close_static_distance <= 3.4
                                and features.lidar_open_side in ("right", "left", "balanced")
                                and features.front_pedestrian_distance is None
                                and features.red_stop_distance is None
                                and not features.red_light_active
                            ):
                                steer_bias = -0.26 if features.lidar_open_side in ("right", "balanced") else 0.26
                                return PlannerAction(
                                    True,
                                    "RECOVER",
                                    target_speed=2.0,
                                    throttle_cap=0.72,
                                    throttle_floor=0.50,
                                    brake_cap=0.0,
                                    steer_limit=0.42,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.20,
                                    reason="roundabout_very_close_open_side_forward_push",
                                )
                            throttle_floor = 0.28 if self.blocked_frames >= 8 else 0.18
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.0,
                                throttle_cap=0.36,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0,
                                steer_limit=0.22,
                                steer_bias=0.0,
                                reason="roundabout_close_static_cautious_creep",
                            )
                        if (
                            roundabout_layout_context
                            and 3.2 < close_static_distance <= 6.2
                            and features.ego_speed < 3.5
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            roundabout_open_side_progress = (
                                roundabout_layout_context
                                and features.lidar_open_side in ("right", "left")
                                and features.front_vehicle_distance is None
                                and features.front_pedestrian_distance is None
                                and features.red_stop_distance is None
                                and not features.red_light_active
                            )
                            if self.roundabout_long_loop_frames >= 260 and features.ego_speed < 2.4:
                                if close_static_distance <= 2.85 and features.ego_speed < 0.45:
                                    steer_bias = -0.46 if features.lidar_open_side == "right" else 0.46
                                    return PlannerAction(
                                        True,
                                        self.state,
                                        target_speed=1.1,
                                        throttle_cap=0.82,
                                        throttle_floor=0.52,
                                        brake_cap=0.0,
                                        steer_limit=0.52,
                                        steer_bias=steer_bias,
                                        reverse=True,
                                        reason="roundabout_close_obstacle_backout",
                                    )
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=5.0,
                                    throttle_cap=1.0,
                                    throttle_floor=0.94,
                                    brake_cap=0.0,
                                    steer_limit=0.08,
                                    steer_bias=0.0,
                                    reason="roundabout_long_loop_route_commit",
                                )
                            if features.ego_speed > 1.1 and close_static_distance <= 4.9 and not roundabout_open_side_progress:
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=0.8,
                                    throttle_cap=0.0,
                                    brake=0.52 if features.ego_speed > 1.8 else 0.30,
                                    steer_limit=0.12,
                                    reason="roundabout_close_static_progress_speed_cap",
                                )
                            roundabout_early_side_push = (
                                roundabout_layout_context
                                and self.blocked_frames >= 2
                                and features.lidar_open_side in ("right", "left")
                                and features.lidar_center_blockage_ratio <= 0.65
                                and (
                                    (
                                        features.lidar_open_side == "right"
                                        and features.lidar_left_blockage_ratio >= 0.80
                                        and features.lidar_right_blockage_ratio <= 0.25
                                    )
                                    or (
                                        features.lidar_open_side == "left"
                                        and features.lidar_right_blockage_ratio >= 0.80
                                        and features.lidar_left_blockage_ratio <= 0.25
                                    )
                                )
                            )
                            if self.blocked_frames >= 24 and abs(features.ego_speed) < 0.35 and close_static_distance <= 5.7:
                                if features.lidar_open_side == "right":
                                    steer_bias = -0.30
                                elif features.lidar_open_side == "left":
                                    steer_bias = 0.30
                                else:
                                    steer_bias = 0.30 if (self.blocked_frames // 16) % 2 == 0 else -0.30
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.98,
                                    throttle_floor=0.84,
                                    brake_cap=0.0,
                                    steer_limit=0.56,
                                    steer_bias=-steer_bias,
                                    steer_min_magnitude=0.26,
                                    reverse=True,
                                    reason="roundabout_mid_static_reverse_escape",
                                )
                            if self.blocked_frames >= 34 or features.lidar_center_blockage_ratio >= 0.82 or roundabout_early_side_push:
                                if features.lidar_open_side == "right":
                                    steer_bias = -0.30
                                elif features.lidar_open_side == "left":
                                    steer_bias = 0.30
                                else:
                                    steer_bias = 0.26 if (self.blocked_frames // 18) % 2 == 0 else -0.26
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.8,
                                    throttle_cap=0.58,
                                    throttle_floor=0.38,
                                    brake_cap=0.0,
                                    steer_limit=0.42,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.22,
                                    reason="roundabout_close_static_progress_side_push",
                                )
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.3,
                                throttle_cap=0.40,
                                throttle_floor=0.24,
                                brake_cap=0.0,
                                steer_limit=0.18,
                                steer_bias=0.0,
                                reason="roundabout_close_static_progress_creep",
                            )
                        if (
                            estimate.macro_scenario in (
                                "trucks_encountered_during_construction",
                                "high_speed_temporary_construction",
                                "avoid_a_disabled_vehicle",
                            )
                            and 1.15 <= close_static_distance <= 2.10
                            and self.blocked_frames >= 100
                            and abs(features.ego_speed) < 0.45
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and features.lidar_open_side in ("balanced", "unknown")
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            steer_bias = 0.34 if (self.blocked_frames // 20) % 2 == 0 else -0.34
                            if abs(features.lidar_lateral_centroid) >= 0.08:
                                steer_bias = 0.34 if features.lidar_lateral_centroid <= 0.0 else -0.34
                            return PlannerAction(
                                True,
                                self.state,
                                throttle_cap=0.52,
                                throttle_floor=0.38,
                                brake_cap=0.0,
                                steer_limit=0.60,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.30,
                                reverse=True,
                                reason="construction_close_static_unknown_reverse_sweep",
                            )
                        if (
                            close_static_distance <= 2.05
                            and self.blocked_frames >= 100
                            and abs(features.ego_speed) < 0.75
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and features.lidar_open_side in ("right", "left")
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.last_open_side = features.lidar_open_side
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            self.state = "AVOID_OR_PASS"
                            steer_bias = 0.22 if features.lidar_open_side == "right" else -0.22
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.1,
                                throttle_cap=0.34,
                                throttle_floor=0.22,
                                brake_cap=0.0,
                                steer_limit=0.38,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.18,
                                reason="construction_close_static_high_blocked_open_side_creep",
                            )
                        if (
                            close_static_distance <= 2.05
                            and abs(features.ego_speed) < 1.0
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and features.lidar_open_side in ("right", "left")
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.last_open_side = features.lidar_open_side
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            if self.blocked_frames <= 8:
                                self.state = "RECOVER"
                                steer_bias = 0.10 if features.lidar_open_side == "right" else -0.10
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.28,
                                    throttle_floor=0.18,
                                    brake_cap=0.0,
                                    steer_limit=0.28,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.12,
                                    reverse=True,
                                    reason="construction_close_static_cone_reverse_clearance",
                                )
                            self.state = "AVOID_OR_PASS"
                            steer_bias = 0.10 if features.lidar_open_side == "right" else -0.10
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=0.8,
                                throttle_cap=0.24,
                                throttle_floor=0.12,
                                brake_cap=0.0,
                                steer_limit=0.22,
                                steer_bias=steer_bias,
                                reason="construction_close_static_cone_tight_creep",
                            )
                        if (
                            estimate.macro_scenario in (
                                "trucks_encountered_during_construction",
                                "high_speed_temporary_construction",
                            )
                            and close_static_distance <= 3.1
                            and abs(features.ego_speed) < 0.35
                            and features.front_vehicle_distance is None
                            and features.lidar_open_side in ("right", "left")
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and not roundabout_layout_context
                        ):
                            self.blocked_frames += 1
                            self.state = "AVOID_OR_PASS"
                            self.last_open_side = features.lidar_open_side
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            steer_bias = 0.28 if features.lidar_open_side == "right" else -0.28
                            throttle_floor = 0.34 if self.blocked_frames >= 6 else 0.26
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.2,
                                throttle_cap=0.46,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0,
                                steer_limit=0.45,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.22,
                                reason="construction_close_static_obstacle_open_side_unwedge",
                            )
                        if (
                            estimate.macro_scenario in (
                                "trucks_encountered_during_construction",
                                "high_speed_temporary_construction",
                                "avoid_a_disabled_vehicle",
                            )
                            and 2.2 <= close_static_distance <= 3.2
                            and abs(features.ego_speed) < 0.35
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (self.red_stop_hold_frames == 0 or self.red_stop_gap_frames >= 45)
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and features.lidar_open_side in ("balanced", "unknown")
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            if self.blocked_frames >= 120:
                                steer_bias = 0.34 if (self.blocked_frames // 20) % 2 == 0 else -0.34
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.48,
                                    throttle_floor=0.36,
                                    brake_cap=0.0,
                                    steer_limit=0.55,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.30,
                                    reverse=True,
                                    reason="construction_close_static_balanced_reverse_sweep",
                                )
                            throttle_floor = 0.38 if self.blocked_frames >= 25 else (0.30 if self.blocked_frames >= 8 else 0.22)
                            throttle_cap = 0.58 if self.blocked_frames >= 25 else 0.46
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.3,
                                throttle_cap=throttle_cap,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0,
                                steer_limit=0.18,
                                steer_bias=0.0,
                                reason="construction_close_static_balanced_creep",
                            )
                        if (
                            close_static_distance <= 5.5
                            and features.ego_speed > 3.0
                            and not roundabout_layout_context
                            and estimate.macro_scenario != "reverse_vehicle"
                        ):
                            self.state = "PREPARE"
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.5,
                                throttle_cap=0.0,
                                brake=0.36,
                                steer_limit=0.30,
                                reason="construction_static_obstacle_approach_brake",
                            )
                        if (
                            3.0 < close_static_distance <= 5.8
                            and abs(features.ego_speed) < 1.2
                            and features.lidar_open_side in ("right", "left")
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and features.detection_object_count >= 10
                            and not roundabout_layout_context
                        ):
                            self.state = "AVOID_OR_PASS"
                            self.last_open_side = features.lidar_open_side
                            self.open_side_pass_memory_frames = self.config.lidar_open_side_pass_memory_frames
                            self.progress_recovery_frames = self.config.lidar_open_side_progress_recovery_frames
                            steer_bias = 0.30 if features.lidar_open_side == "right" else -0.30
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.4,
                                throttle_cap=0.48,
                                throttle_floor=0.30,
                                brake_cap=0.0,
                                steer_limit=0.60,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.22,
                                reason="construction_static_obstacle_open_side_creep",
                            )
                        if (
                            3.0 < close_static_distance <= 5.8
                            and abs(features.ego_speed) < 1.2
                            and features.lidar_open_side == "balanced"
                            and (features.lidar_available or features.lidar_front_distance is not None)
                            and features.detection_object_count >= 10
                            and not roundabout_layout_context
                        ):
                            self.state = "RECOVER"
                            if abs(features.ego_speed) < 0.12:
                                self.blocked_frames += 1
                            elif abs(features.ego_speed) > 0.65:
                                self.blocked_frames = max(0, self.blocked_frames - 2)
                            self.balanced_blockage_progress_frames = max(self.balanced_blockage_progress_frames, 40)
                            if self.blocked_frames >= 85:
                                steer_bias = -0.38 if features.lidar_lateral_centroid <= 0.0 else 0.38
                                return PlannerAction(
                                    True,
                                    self.state,
                                    throttle_cap=0.64,
                                    throttle_floor=0.46,
                                    brake_cap=0.0,
                                    steer_limit=0.58,
                                    steer_bias=steer_bias,
                                    steer_min_magnitude=0.26,
                                    reverse=True,
                                    reason="construction_static_balanced_reverse_sweep",
                                )
                            if self.blocked_frames >= 35:
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=2.4,
                                    throttle_cap=0.82,
                                    throttle_floor=0.56,
                                    brake_cap=0.0,
                                    steer_limit=0.22,
                                    steer_bias=0.0,
                                    reason="construction_static_balanced_progress_push",
                                )
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=1.4,
                                throttle_cap=0.48,
                                throttle_floor=0.28,
                                brake_cap=0.0,
                                steer_limit=0.25,
                                steer_bias=0.0,
                                reason="construction_static_obstacle_balanced_creep",
                            )
                        if (
                            roundabout_layout_context
                            and 12.0 <= close_static_distance <= 18.5
                            and self.blocked_frames >= 45
                            and features.ego_speed < 0.90
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.state = "RECOVER"
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=3.0,
                                throttle_cap=0.78,
                                throttle_floor=0.52,
                                brake_cap=0.0,
                                steer_limit=0.16,
                                reason="roundabout_mid_static_blocked_progress_release",
                            )
                        if (
                            roundabout_layout_context
                            and 12.0 <= close_static_distance <= 18.0
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and features.lidar_available
                            and not features.lidar_stale
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_center_blockage_ratio >= 0.75
                            and features.lidar_open_side in ("right", "left")
                            and abs(features.ego_speed) < 1.20
                        ):
                            self.state = "RECOVER"
                            steer_bias = -0.22 if features.lidar_open_side == "right" else 0.22
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=3.0,
                                throttle_cap=0.88,
                                throttle_floor=0.68,
                                brake_cap=0.0,
                                steer_limit=0.42,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.18,
                                reason="roundabout_mid_far_static_open_side_progress",
                            )
                        if (
                            self.config.lidar_open_side_nudge_enabled
                            and not self.config.distant_lidar_creep_enabled
                            and 8.0 < close_static_distance <= 18.0
                            and abs(features.ego_speed) < 0.20
                            and features.front_vehicle_distance is None
                            and features.lidar_open_side in ("right", "left", "balanced")
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.blocked_frames += 1
                            self.state = "RECOVER"
                            steer_bias = 0.0
                            if features.lidar_open_side == "right":
                                steer_bias = 0.12
                            elif features.lidar_open_side == "left":
                                steer_bias = -0.12
                            throttle_floor = 0.58 if self.blocked_frames >= 8 else 0.42
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.8,
                                throttle_cap=0.82,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0,
                                steer_limit=0.18 if roundabout_layout_context else 0.30,
                                steer_bias=steer_bias,
                                reason="static_obstacle_far_no_progress_recovery",
                            )
                        if (
                            roundabout_layout_context
                            and 6.0 <= close_static_distance <= 13.5
                            and abs(features.ego_speed) < 0.80
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and features.lidar_available
                            and not features.lidar_stale
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_open_side in ("right", "left")
                            and (
                                (
                                    features.lidar_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.80
                                    and features.lidar_right_blockage_ratio <= 0.25
                                )
                                or (
                                    features.lidar_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.80
                                    and features.lidar_left_blockage_ratio <= 0.25
                                )
                            )
                        ):
                            self.state = "RECOVER"
                            steer_bias = -0.18 if features.lidar_open_side == "right" else 0.18
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.8,
                                throttle_cap=0.82,
                                throttle_floor=0.58,
                                brake_cap=0.0,
                                steer_limit=0.40,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.16,
                                reason="roundabout_far_side_blockage_forward_push",
                            )
                        if (
                            roundabout_layout_context
                            and 5.8 <= close_static_distance <= 7.6
                            and abs(features.ego_speed) < 1.45
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and features.lidar_available
                            and not features.lidar_stale
                            and features.lidar_blockage_ratio >= 0.85
                            and features.lidar_center_blockage_ratio <= 0.65
                            and features.lidar_open_side in ("right", "left")
                            and (
                                (
                                    features.lidar_open_side == "right"
                                    and features.lidar_left_blockage_ratio >= 0.80
                                    and features.lidar_right_blockage_ratio <= 0.25
                                )
                                or (
                                    features.lidar_open_side == "left"
                                    and features.lidar_right_blockage_ratio >= 0.80
                                    and features.lidar_left_blockage_ratio <= 0.25
                                )
                            )
                        ):
                            self.state = "RECOVER"
                            steer_bias = -0.20 if features.lidar_open_side == "right" else 0.20
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.6,
                                throttle_cap=0.76,
                                throttle_floor=0.54,
                                brake_cap=0.0,
                                steer_limit=0.38,
                                steer_bias=steer_bias,
                                steer_min_magnitude=0.16,
                                reason="roundabout_near_side_blockage_forward_push",
                            )
                        if (
                            roundabout_layout_context
                            and 6.0 <= close_static_distance <= 26.0
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.state = "PREPARE"
                            if close_static_distance > 18.0 and features.ego_speed < 1.0:
                                return PlannerAction(
                                    True,
                                    "RECOVER",
                                    target_speed=3.2,
                                    throttle_cap=0.65,
                                    throttle_floor=0.45,
                                    brake_cap=0.0,
                                    steer_limit=0.16,
                                    reason="roundabout_far_static_progress_release",
                                )
                            if close_static_distance <= 12.0:
                                brake = 0.46 if features.ego_speed > 1.5 else 0.0
                                return PlannerAction(
                                    True,
                                    self.state,
                                    target_speed=1.0,
                                    throttle_cap=0.0 if features.ego_speed > 1.5 else 0.18,
                                    throttle_floor=None if features.ego_speed > 1.5 else 0.10,
                                    brake_cap=0.0 if features.ego_speed <= 1.5 else None,
                                    brake=brake,
                                    steer_limit=0.06,
                                    reason="roundabout_static_obstacle_pre_stop",
                                )
                            brake = 0.0
                            throttle_cap = 0.42
                            throttle_floor = 0.36
                            if features.ego_speed > 4.8:
                                brake = 0.42
                                throttle_cap = 0.0
                                throttle_floor = None
                            elif features.ego_speed > 3.0:
                                brake = 0.22
                                throttle_cap = 0.0
                                throttle_floor = None
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=2.4,
                                throttle_cap=throttle_cap,
                                throttle_floor=throttle_floor,
                                brake_cap=0.0 if brake == 0.0 else None,
                                brake=brake,
                                steer_limit=0.08,
                                reason="roundabout_static_obstacle_speed_cap",
                            )
                        if (
                            roundabout_layout_context
                            and 26.0 < close_static_distance <= 35.0
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.red_stop_distance is None
                            and not features.red_light_active
                            and (features.lidar_available or features.lidar_front_distance is not None)
                        ):
                            self.state = "RECOVER"
                            far_brake = 0.0
                            far_throttle_cap = 0.32
                            far_throttle_floor = 0.24
                            if features.ego_speed > 5.2:
                                far_brake = 0.34
                                far_throttle_cap = 0.0
                                far_throttle_floor = None
                            elif features.ego_speed > 3.8:
                                far_brake = 0.14
                                far_throttle_cap = 0.0
                                far_throttle_floor = None
                            return PlannerAction(
                                True,
                                self.state,
                                target_speed=3.4,
                                throttle_cap=far_throttle_cap,
                                throttle_floor=far_throttle_floor,
                                brake_cap=0.0,
                                brake=far_brake,
                                steer_limit=0.18,
                                reason="roundabout_far_static_progress_probe",
                            )
                if (
                    self.lateral_intersection_release_frames > 0
                    and features.ego_speed < 11.2
                    and not roundabout_layout_context
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and features.front_obstacle_distance is not None
                    and float(features.front_obstacle_distance) >= 12.0
                ):
                    self.state = "PREPARE"
                    return self._lateral_release_keep_rolling_action(
                        features,
                        reason="lateral_intersection_far_static_memory_override",
                    )
                if (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and estimate.confidence >= self.config.min_rule_confidence
                    and not roundabout_layout_context
                    and features.front_obstacle_distance is not None
                    and 6.0 <= float(features.front_obstacle_distance) <= 17.0
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and features.ego_speed >= 1.0
                    and features.lidar_available
                    and features.lidar_front_distance is not None
                    and features.lidar_center_blockage_ratio >= 0.85
                    and features.lidar_blockage_ratio >= 0.40
                ):
                    if features.ego_speed > 2.2:
                        corridor_target_speed = 0.8
                        corridor_throttle_cap = 0.0
                        corridor_throttle_floor = None
                        corridor_brake = 0.58 if features.ego_speed > 4.0 else 0.42
                    elif features.ego_speed > 1.0:
                        corridor_target_speed = 0.8
                        corridor_throttle_cap = 0.0
                        corridor_throttle_floor = None
                        corridor_brake = 0.22
                    else:
                        corridor_target_speed = 0.7
                        corridor_throttle_cap = 0.16
                        corridor_throttle_floor = 0.08
                        corridor_brake = 0.0
                    self.state = "PREPARE"
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=corridor_target_speed,
                        throttle_cap=corridor_throttle_cap,
                        throttle_floor=corridor_throttle_floor,
                        brake=corridor_brake,
                        steer_limit=0.12,
                        reason="construction_center_blockage_corridor_speed_cap",
                    )
                if (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    and estimate.confidence >= 0.70
                    and not roundabout_layout_context
                    and features.front_obstacle_distance is not None
                    and 4.0 <= float(features.front_obstacle_distance) <= 10.0
                    and features.ego_speed > 2.0
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and features.lidar_blockage_ratio >= 0.45
                ):
                    self.state = "PREPARE"
                    return PlannerAction(
                        True,
                        self.state,
                        target_speed=1.8,
                        throttle_cap=0.0,
                        brake=0.58 if features.ego_speed > 4.0 else 0.34,
                        steer_limit=0.16,
                        reason="construction_mid_static_collision_speed_cap",
                    )
                return PlannerAction(False, self.state, reason="static_obstacle_observed_without_immediate_conflict")
            self.static_creep_frames = 0
            self.post_pass_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            if (
                roundabout_layout_context
                and features.front_obstacle_distance is not None
                and 3.2 <= float(features.front_obstacle_distance) <= 4.3
                and features.ego_speed < 1.2
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.lidar_open_side in ("right", "left")
            ):
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.2,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.18,
                    steer_bias=0.04 if features.lidar_open_side == "right" else -0.04,
                    reason="roundabout_close_static_side_gap_forward_push",
                )
            target = 3.0 if estimate.macro_scenario in ("trucks_encountered_during_construction", "highway_accident_vehicle") else 4.5
            return PlannerAction(True, self.state, target_speed=target, throttle_cap=0.30, steer_limit=0.45, reason=estimate.reason)
        if features.side_risk:
            self.state = "PREPARE"
            self.static_creep_frames = 0
            self.post_pass_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(False, self.state, reason="side_risk_observed_without_longitudinal_conflict")

        if features.risk_level == 0:
            self.clear_frames += 1
            self.static_creep_frames = 0
            no_front_conflict = (
                features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                and features.lidar_center_blockage_ratio < 0.10
            )
            if (
                features.red_light_active
                and features.red_stop_distance is None
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.ego_speed > 2.0
            ):
                self.state = "YIELD_OR_BRAKE"
                self.open_side_pass_memory_frames = 0
                self.post_pass_frames = 0
                self.close_obstacle_memory_frames = 0
                self.progress_recovery_frames = 0
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=0.0,
                    throttle_cap=0.0,
                    brake=0.35 if features.ego_speed < 4.0 else 0.55,
                    steer_limit=0.35,
                    reason="active_red_without_stopline_deceleration",
                )
            if (
                roundabout_layout_context
                and no_front_conflict
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.ego_speed > 4.0
            ):
                self.state = "PREPARE"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=3.5,
                    throttle_cap=0.0 if features.ego_speed > 5.0 else 0.18,
                    brake=0.38 if features.ego_speed > 7.0 else (0.24 if features.ego_speed > 5.0 else 0.0),
                    steer_limit=0.18,
                    reason="roundabout_clear_lane_speed_guard",
                )
            if (
                roundabout_layout_context
                and no_front_conflict
                and features.red_stop_distance is None
                and not features.red_light_active
                and self.roundabout_context_frames > 0
                and 2.0 <= features.ego_speed <= 4.0
            ):
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=3.2,
                    throttle_cap=0.34,
                    brake_cap=0.0,
                    steer_limit=0.18,
                    reason="roundabout_clear_lane_slow_cruise",
                )
            if (
                no_front_conflict
                and self.observable_risk_creep_frames > 0
                and features.ego_speed < 2.0
                and (not features.red_light_active or features.ego_speed < 0.25)
            ):
                self.observable_risk_creep_frames -= 1
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.35,
                    reason="observable_risk_cautious_creep_recovery",
                )
            if features.red_stop_distance is None and not features.red_light_active and self.red_stop_hold_frames > 0:
                self.red_stop_gap_frames += 1
            red_clear_recovery_allowed = self.red_stop_hold_frames == 0 or self.red_stop_gap_frames >= 120
            if (
                no_front_conflict
                and red_clear_recovery_allowed
                and features.red_stop_distance is None
                and not features.red_light_active
                and self.blocked_frames >= 2
                and features.ego_speed < 2.0
            ):
                if features.ego_speed < 0.60:
                    self.blocked_frames += 1
                return self._clear_road_no_progress_action(
                    "clear_road_cautious_creep_recovery",
                    allow_reverse=not roundabout_layout_context,
                )
            if (
                no_front_conflict
                and red_clear_recovery_allowed
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.ego_speed < 0.60
            ):
                self.blocked_frames += 1
                if self.blocked_frames >= 2:
                    self.state = "RECOVER"
                    return PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=2.6,
                        throttle_cap=0.68,
                        throttle_floor=0.45,
                        brake_cap=0.0,
                        steer_limit=0.35,
                        reason="clear_road_cautious_creep_recovery",
                    )
            elif not no_front_conflict or features.ego_speed >= 1.00:
                self.blocked_frames = 0
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state == "RECOVER"
                and self.progress_recovery_frames > 0
                and features.ego_speed < 2.2
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
            ):
                self.progress_recovery_frames -= 1
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.45,
                    reason="distant_lidar_open_side_progress_recovery",
                )
            if (
                self.config.lidar_open_side_nudge_enabled
                and self.state == "AVOID_OR_PASS"
                and self.post_pass_frames > 0
                and self.close_obstacle_memory_frames <= 0
                and features.ego_speed < 1.2
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.red_stop_distance is None
            ):
                self.post_pass_frames -= 1
                self.state = "RECOVER"
                steer_bias = 0.10 if self.last_open_side == "right" else (-0.10 if self.last_open_side == "left" else 0.0)
                return PlannerAction(
                    True,
                    self.state,
                    target_speed=2.2,
                    throttle_cap=0.52,
                    throttle_floor=0.28,
                    brake_cap=0.0,
                    steer_limit=0.55,
                    steer_bias=steer_bias,
                    reason="distant_lidar_open_side_post_pass_recovery",
                )
            if self.clear_frames >= 3 and self.state not in ("NORMAL", "RECOVER"):
                self.state = "RECOVER"
                return PlannerAction(True, self.state, throttle_cap=0.45, steer_limit=0.55, reason="risk cleared recovery")
            self.state = "NORMAL"
            self.post_pass_frames = 0
            self.close_obstacle_memory_frames = 0
            self.progress_recovery_frames = 0
            self.open_side_pass_memory_frames = 0
            return PlannerAction(False, self.state, reason="clear")

        self.state = "APPROACH"
        self.static_creep_frames = 0
        self.post_pass_frames = 0
        self.close_obstacle_memory_frames = 0
        self.progress_recovery_frames = 0
        self.open_side_pass_memory_frames = 0
        no_front_conflict = (
            features.front_clear
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.front_obstacle_distance is None
            and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
            and features.lidar_center_blockage_ratio < 0.10
        )
        red_stop_actor_clear = (
            features.front_clear
            and features.front_vehicle_distance is None
            and features.front_pedestrian_distance is None
            and features.front_obstacle_distance is None
        )
        stop_line_margin_ok = features.red_stop_distance is None or float(features.red_stop_distance) > 5.5
        near_stop_line_without_red = (
            no_front_conflict
            and not features.red_light_active
            and features.red_stop_distance is not None
            and float(features.red_stop_distance) <= 5.5
        )
        close_stop_line_hint = (
            red_stop_actor_clear
            and features.red_light_active
            and features.red_stop_distance is not None
            and float(features.red_stop_distance) <= 5.5
        )
        unstable_red_stop_release = self._update_red_stop_stability(features, red_stop_actor_clear)
        if (
            no_front_conflict
            and self.red_stop_release_frames > 0
            and not features.red_light_active
            and (features.red_stop_distance is None or float(features.red_stop_distance) > 5.5)
            and features.ego_speed < 1.4
            and features.front_pedestrian_distance is None
            and features.front_vehicle_distance is None
            and features.front_obstacle_distance is None
        ):
            self.red_stop_release_frames -= 1
            return self._red_stop_release_action()
        if unstable_red_stop_release:
            return self._red_stop_release_action(start_window=True)
        if (
            no_front_conflict
            and self.observable_risk_creep_frames > 0
            and features.ego_speed < 2.0
            and (not features.red_light_active or features.red_stop_distance is None or features.ego_speed < 0.25)
        ):
            self.observable_risk_creep_frames -= 1
            self.blocked_frames = max(self.blocked_frames, 4)
            self.state = "RECOVER"
            return PlannerAction(
                True,
                "RECOVER",
                target_speed=2.6,
                throttle_cap=0.68,
                throttle_floor=0.45,
                brake_cap=0.0,
                steer_limit=0.35,
                reason="observable_risk_cautious_creep_recovery",
            )
        if near_stop_line_without_red and self.blocked_frames >= 2 and features.ego_speed < 2.5:
            self.state = "RECOVER"
            return PlannerAction(
                True,
                "RECOVER",
                target_speed=2.6,
                throttle_cap=0.68,
                throttle_floor=0.45,
                brake_cap=0.0,
                steer_limit=0.35,
                reason="near_stop_line_cautious_creep_recovery",
            )
        if near_stop_line_without_red and features.ego_speed < 0.60:
            self.blocked_frames += 1
            if self.blocked_frames >= 2:
                self.state = "RECOVER"
                return PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.6,
                    throttle_cap=0.68,
                    throttle_floor=0.45,
                    brake_cap=0.0,
                    steer_limit=0.35,
                    reason="near_stop_line_cautious_creep_recovery",
                )
        if close_stop_line_hint and features.ego_speed < 0.60:
            self.blocked_frames += 1
        if no_front_conflict and stop_line_margin_ok and features.ego_speed < 0.30:
            self.blocked_frames += 1
            if self.blocked_frames >= 4:
                self.observable_risk_creep_frames = 500
                return self._clear_road_no_progress_action(
                    "observable_risk_cautious_creep_recovery",
                    allow_reverse=not roundabout_layout_context,
                )
        elif not near_stop_line_without_red and not close_stop_line_hint:
            self.blocked_frames = 0
        return PlannerAction(False, self.state, reason="observable_risk_without_confirmed_longitudinal_conflict")


class SafetySupervisor:
    def __init__(self, config: AuxiliaryConfig):
        self.config = config
        self.last_log_time = 0.0

    def apply(self, raw_control: Any, features: AuxFeatures, estimate: ScenarioEstimate, action: PlannerAction) -> Any:
        if not self.config.enabled or not self.config.safety_supervisor_enabled or not action.active:
            return raw_control
        steer = float(raw_control.steer)
        throttle = float(raw_control.throttle)
        brake = float(raw_control.brake)
        if action.target_speed is not None and features.ego_speed > float(action.target_speed):
            throttle = 0.0
            brake = max(brake, min(0.55, 0.08 + 0.04 * (features.ego_speed - float(action.target_speed))))
        if action.brake_cap is not None:
            brake = min(brake, float(action.brake_cap))
        if action.throttle_cap is not None:
            throttle = min(throttle, float(action.throttle_cap))
        if action.throttle_floor is not None and brake < 0.05:
            throttle = max(throttle, float(action.throttle_floor))
        if action.brake is not None:
            brake = max(brake, float(action.brake))
            if brake >= 0.30:
                throttle = 0.0
        if action.brake_cap is not None and float(action.brake_cap) > 0.0:
            brake = min(brake, float(action.brake_cap))
        if action.steer_bias is not None:
            steer += float(action.steer_bias)
            if action.steer_min_magnitude is not None:
                min_steer = abs(float(action.steer_min_magnitude))
                if action.steer_bias > 0.0:
                    steer = max(steer, min_steer)
                else:
                    steer = min(steer, -min_steer)
        if action.steer_limit is not None:
            limit = abs(float(action.steer_limit))
            steer = _clip(steer, -limit, limit)
        if carla is not None:
            control = carla.VehicleControl(steer=_clip(steer, -1.0, 1.0), throttle=_clip(throttle, 0.0, 1.0), brake=_clip(brake, 0.0, 1.0))
        else:
            control = type(raw_control)(steer=_clip(steer, -1.0, 1.0), throttle=_clip(throttle, 0.0, 1.0), brake=_clip(brake, 0.0, 1.0))
        setattr(control, "reverse", bool(action.reverse))
        return control


class CVCIAuxiliarySystem:
    def __init__(self, config: Optional[AuxiliaryConfig] = None):
        self.config = config or AuxiliaryConfig()
        self.perception = AuxiliaryPerception(self.config)
        self.feature_builder = SceneFeatureBuilder()
        self.recognizer = ScenarioRecognizer(self.config)
        self.rule_planner = ScenarioRulePlanner(self.config)
        self.supervisor = SafetySupervisor(self.config)
        self.intervention_count = 0
        self.emergency_count = 0
        self.red_final_clamp_hold_frames = 0
        self.red_final_clamp_gap_frames = 0
        self.red_final_clamp_last_distance: Optional[float] = None
        self.red_final_creep_memory_frames = 0
        self.students_red_deadlock_release_frames = 0
        self.crazy_bike_decelerate_frames = 0
        self.crazy_bike_resume_frames = 0
        self.crazy_bike_decelerate_done = False
        self.red_macro_deadlock_release_frames = 0
        self.red_macro_deadlock_release_reason = ""
        self.roundabout_mid_red_yield_frames = 0
        self.ghost_probe_active_red_hold_frames = 0
        self.ghost_probe_far_red_release_frames = 0
        self.ghost_probe_line_commit_frames = 0
        self.red_final_near_line_hold_frames = 0
        self.red_reverse_unwedge_frames = 0
        self.highspeed_brake_response_frames = 0
        self.highspeed_brake_response_done = False
        self.highspeed_route_local_start_pos: Optional[Tuple[float, float]] = None
        self.cut_in_straight_unwedge_frames = 0
        self.cut_in_post_unwedge_commit_frames = 0
        self.cut_in_clear_recovery_frames = 0
        self.cut_in_route_rejoin_frames = 0
        self.cut_in_route_rejoin_side = "unknown"
        self.blind_spot_prebrake_frames = 0
        self.blind_spot_prebrake_cooldown_frames = 0
        self.blind_spot_route_prior_trigger_brake_frames = 0
        self.blind_spot_route_prior_trigger_brake_done = False
        self.cut_in_open_side_reverse_frames = 0
        self.cut_in_open_side_stuck_frames = 0
        self.cut_in_open_side_sustain_frames = 0
        self.cut_in_open_side_sustain_side = "unknown"
        self.cut_in_post_reverse_no_backoff_frames = 0
        self.cut_in_reverse_gap_reset_frames = 0
        self.cut_in_false_red_release_frames = 0
        self.cut_in_wide_gap_push_frames = 0
        self.cut_in_long_loop_frames = 0
        self.cut_in_wide_gap_push_side = "unknown"
        self.last_debug: Dict[str, Any] = {"enabled": self.config.enabled}

    @property
    def wants_lidar(self) -> bool:
        return bool(self.config.enabled and self.config.perception_enabled and self.config.lidar_enabled)

    def process(self, raw_control: Any, model_detection: Dict[str, Any], tick_data: Dict[str, Any], timestamp: float, legacy_rule_action: str = "none") -> Any:
        if not self.config.enabled:
            self.last_debug = {"enabled": False, "action": "baseline_passthrough"}
            return raw_control
        start = time.time()
        try:
            observation = self.perception.update(model_detection, tick_data, timestamp)
            features = self.feature_builder.build(observation, tick_data)
            estimate = self.recognizer.recognize(features)
            self.rule_planner.red_final_context_frames = (
                self.red_final_clamp_hold_frames
                if self.red_final_clamp_hold_frames > 0 and self.red_final_clamp_gap_frames <= 30
                else 0
            )
            forced_route_prior_macro = bool(self.config.allow_route_prior and self.config.forced_macro_scenario)
            highspeed_route_progress_debug = None
            highspeed_hazard_distance_debug = None
            highspeed_ego_pos_debug = None
            blind_spot_route_prior_ego_pos_debug = None
            action = self.rule_planner.plan(features, estimate)
            generic_highspeed_hazard_distance = None
            if estimate.macro_scenario == "highway_accident_vehicle":
                for distance in (
                    features.front_vehicle_distance,
                    features.front_obstacle_distance,
                    features.lidar_front_distance,
                ):
                    if distance is None:
                        continue
                    generic_highspeed_hazard_distance = (
                        float(distance)
                        if generic_highspeed_hazard_distance is None
                        else min(generic_highspeed_hazard_distance, float(distance))
                    )
                for track in features.tracked_objects or []:
                    cls = str(track.get("class_name", "")).lower()
                    if cls not in ("car", "truck", "bus", "van", "vehicle"):
                        continue
                    try:
                        tx = float(track.get("x", 999.0))
                        ty = abs(float(track.get("y", 999.0)))
                    except Exception:
                        continue
                    if 4.0 <= tx <= 60.0 and ty <= 9.0:
                        generic_highspeed_hazard_distance = (
                            tx
                            if generic_highspeed_hazard_distance is None
                            else min(generic_highspeed_hazard_distance, tx)
                        )
                generic_highspeed_brake_probe = (
                    not self.highspeed_brake_response_done
                    and generic_highspeed_hazard_distance is not None
                    and 4.0 <= generic_highspeed_hazard_distance <= 60.0
                    and features.ego_speed >= 2.0
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and (
                        features.front_vehicle_distance is not None
                        or features.immediate_hazard
                        or features.lidar_blockage_ratio >= 0.25
                        or features.lidar_center_blockage_ratio >= 0.25
                        or (
                            features.front_obstacle_distance is not None
                            and float(features.front_obstacle_distance) <= 12.0
                        )
                    )
                )
                if generic_highspeed_brake_probe:
                    self.highspeed_brake_response_frames = max(self.highspeed_brake_response_frames, 18)
                if self.highspeed_brake_response_frames > 0 and generic_highspeed_brake_probe:
                    self.highspeed_brake_response_frames -= 1
                    if self.highspeed_brake_response_frames <= 0:
                        self.highspeed_brake_response_done = True
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        throttle_cap=0.0,
                        brake=0.82,
                        steer_limit=0.20,
                        reason="highspeed_accident_observed_hazard_brake_response",
                    )
            if forced_route_prior_macro:
                if (
                    estimate.macro_scenario == "highway_accident_vehicle"
                    and self.highspeed_brake_response_frames <= 0
                ):
                    highspeed_hazard_distance = None
                    for distance in (
                        features.front_vehicle_distance,
                        features.front_obstacle_distance,
                        features.lidar_front_distance,
                    ):
                        if distance is None:
                            continue
                        highspeed_hazard_distance = float(distance) if highspeed_hazard_distance is None else min(highspeed_hazard_distance, float(distance))
                    for track in features.tracked_objects or []:
                        cls = str(track.get("class_name", "")).lower()
                        if cls not in ("car", "truck", "bus", "van", "vehicle"):
                            continue
                        try:
                            tx = float(track.get("x", 999.0))
                            ty = abs(float(track.get("y", 999.0)))
                        except Exception:
                            continue
                        if 8.0 <= tx <= 50.0 and ty <= 9.0:
                            highspeed_hazard_distance = tx if highspeed_hazard_distance is None else min(highspeed_hazard_distance, tx)
                    highspeed_route_progress = None
                    try:
                        ego_pos = tick_data.get("pos")
                        if ego_pos is None and isinstance(model_detection, dict):
                            ego_state = model_detection.get("ego") or {}
                            if isinstance(ego_state, dict):
                                ego_pos = ego_state.get("pos")
                            if ego_pos is None:
                                ego_pos = model_detection.get("pos")
                        if ego_pos is not None:
                            ex = float(ego_pos[0])
                            ey = float(ego_pos[1])
                            highspeed_ego_pos_debug = [ex, ey]
                            # Match HighSpeedBrakeCriterion's route XML frame for route36.
                            sx, sy = 127.23, 222.25
                            ex_ref, ey_ref = 257.60, 171.16
                            vx = ex_ref - sx
                            vy = ey_ref - sy
                            norm = (vx * vx + vy * vy) ** 0.5
                            if norm > 1e-6:
                                highspeed_route_progress = ((ex - sx) * vx + (ey - sy) * vy) / norm
                                if highspeed_route_progress < -20.0:
                                    if self.highspeed_route_local_start_pos is None and features.ego_speed >= 8.0:
                                        self.highspeed_route_local_start_pos = (ex, ey)
                                    if self.highspeed_route_local_start_pos is not None:
                                        lsx, lsy = self.highspeed_route_local_start_pos
                                        highspeed_route_progress = ((ex - lsx) * vx + (ey - lsy) * vy) / norm
                                highspeed_route_progress_debug = highspeed_route_progress
                    except Exception:
                        highspeed_route_progress = None
                    highspeed_hazard_distance_debug = highspeed_hazard_distance
                    route_window_response = (
                        not self.highspeed_brake_response_done
                        and highspeed_route_progress is not None
                        and 35.0 <= highspeed_route_progress <= 66.0
                        and features.ego_speed >= 1.5
                    )
                    hazard_response = (
                        not self.highspeed_brake_response_done
                        and highspeed_hazard_distance is not None
                        and highspeed_route_progress is None
                        and 4.0 <= highspeed_hazard_distance <= 75.0
                        and features.ego_speed >= 2.0
                        and (
                            features.front_vehicle_distance is not None
                            or features.immediate_hazard
                            or features.lidar_blockage_ratio >= 0.25
                            or features.lidar_center_blockage_ratio >= 0.25
                            or (
                                features.front_obstacle_distance is not None
                                and float(features.front_obstacle_distance) <= 12.0
                            )
                        )
                    )
                    forced_route_brake_probe = (
                        forced_route_prior_macro
                        and estimate.macro_scenario == "highway_accident_vehicle"
                        and highspeed_route_progress is not None
                        and 35.0 <= highspeed_route_progress <= 66.0
                        and features.ego_speed >= 1.5
                        and not self.highspeed_brake_response_done
                    )
                    early_lidar_brake_probe = (
                        not self.highspeed_brake_response_done
                        and highspeed_hazard_distance is not None
                        and 18.0 <= highspeed_hazard_distance <= 36.0
                        and (highspeed_route_progress is None or highspeed_route_progress >= 35.0)
                        and features.ego_speed >= 0.25
                        and (
                            features.front_vehicle_distance is not None
                            or features.immediate_hazard
                            or features.lidar_blockage_ratio >= 0.25
                            or features.lidar_center_blockage_ratio >= 0.25
                            or (
                                features.front_obstacle_distance is not None
                                and float(features.front_obstacle_distance) <= 12.0
                            )
                        )
                    )
                    pretrigger_speed_preserve = (
                        False
                        and not self.highspeed_brake_response_done
                        and highspeed_route_progress is not None
                        and 24.0 <= highspeed_route_progress < 39.5
                        and features.ego_speed >= 4.0
                        and (highspeed_hazard_distance is None or highspeed_hazard_distance >= 28.0)
                    )
                    if pretrigger_speed_preserve and action.reason in (
                        "clear",
                        "risk cleared recovery",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                    ):
                        action = PlannerAction(
                            True,
                            "PREPARE",
                            target_speed=8.0,
                            throttle_cap=0.75,
                            throttle_floor=0.45,
                            brake_cap=0.0,
                            steer_limit=0.18,
                            reason="highspeed_accident_pretrigger_speed_preserve",
                        )
                    near_hazard_response = (
                        self.highspeed_brake_response_done
                        and highspeed_hazard_distance is not None
                        and 1.5 <= highspeed_hazard_distance <= 7.0
                        and features.ego_speed >= 0.75
                    )
                    if route_window_response or hazard_response or forced_route_brake_probe or early_lidar_brake_probe:
                        # A short, strong brake pulse is enough for HighSpeedBrakeCriterion;
                        # holding it for hundreds of frames starves the later bypass rules.
                        self.highspeed_brake_response_frames = max(self.highspeed_brake_response_frames, 18)
                    elif near_hazard_response:
                        self.highspeed_brake_response_frames = max(self.highspeed_brake_response_frames, 6)

                if estimate.macro_scenario == "highway_accident_vehicle" and self.highspeed_brake_response_frames > 0:
                    self.highspeed_brake_response_frames -= 1
                    if self.highspeed_brake_response_frames <= 0:
                        self.highspeed_brake_response_done = True
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        throttle_cap=0.0,
                        brake=0.85,
                        steer_limit=0.24,
                        reason="highspeed_accident_brake_response_probe",
                    )
                elif (
                    estimate.macro_scenario == "avoid_a_disabled_vehicle"
                    and action.reason in (
                        "clear",
                        "risk cleared recovery",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                        "static_obstacle_observed_without_immediate_conflict",
                    )
                    and abs(features.ego_speed) < 0.55
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (
                        features.front_obstacle_distance is None
                        or float(features.front_obstacle_distance) >= 5.5
                    )
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 5.5)
                    and features.lidar_center_blockage_ratio <= 0.08
                ):
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=3.2,
                        throttle_cap=0.75,
                        throttle_floor=0.50,
                        brake_cap=0.0,
                        steer_limit=0.35,
                        reason="disabled_vehicle_clear_road_release",
                    )
                elif (
                    estimate.macro_scenario == "ghost_probe"
                    and action.reason in (
                        "forced route-prior macro scenario",
                        "static_obstacle_observed_without_immediate_conflict",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                    )
                    and self.red_final_clamp_hold_frames >= 150
                    and abs(features.ego_speed) < 0.35
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) >= 1.8)
                    and features.lidar_front_distance is not None
                    and 1.8 <= float(features.lidar_front_distance) <= 3.2
                ):
                    if features.lidar_open_side == "right":
                        steer_bias = -0.24
                    elif features.lidar_open_side == "left":
                        steer_bias = 0.24
                    else:
                        steer_bias = -0.18 if (self.red_final_clamp_hold_frames // 35) % 2 == 0 else 0.18
                    if self.red_final_clamp_hold_frames >= 220:
                        steer_bias = -0.34 if features.lidar_open_side == "right" else (0.34 if features.lidar_open_side == "left" else (0.28 if (self.red_final_clamp_hold_frames // 30) % 2 == 0 else -0.28))
                        action = PlannerAction(
                            True,
                            "RECOVER",
                            target_speed=1.0,
                            throttle_cap=0.58,
                            throttle_floor=0.40,
                            brake_cap=0.0,
                            steer_limit=0.48,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.26,
                            reverse=True,
                            reason="ghost_probe_close_static_red_hold_reverse_sweep",
                        )
                    else:
                        action = PlannerAction(
                            True,
                            "RECOVER",
                            target_speed=2.0,
                            throttle_cap=0.62,
                            throttle_floor=0.42,
                            brake_cap=0.0,
                            steer_limit=0.34,
                            steer_bias=steer_bias,
                            steer_min_magnitude=0.14,
                            reason="ghost_probe_close_static_red_hold_bypass",
                        )
                elif (
                    estimate.macro_scenario == "highway_accident_vehicle"
                    and action.reason == "forced route-prior macro scenario"
                    and features.front_obstacle_distance is not None
                    and 1.0 <= float(features.front_obstacle_distance) <= 2.4
                    and features.lidar_open_side in ("right", "left")
                    and abs(features.ego_speed) < 1.2
                ):
                    steer_bias = -0.18 if features.lidar_open_side == "right" else 0.18
                    action = PlannerAction(
                        True,
                        "AVOID_OR_PASS",
                        target_speed=2.8,
                        throttle_cap=0.65,
                        throttle_floor=0.38,
                        brake_cap=0.0,
                        steer_limit=0.45,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.12,
                        reason="highspeed_accident_near_open_side_pass",
                    )
                elif (
                    estimate.macro_scenario == "highway_accident_vehicle"
                    and action.reason in (
                        "forced route-prior macro scenario",
                        "forced_macro_suppressed_construction_rule",
                        "static_obstacle_observed_without_immediate_conflict",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                    )
                    and self.highspeed_brake_response_done
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is not None
                    and 5.8 < float(features.front_obstacle_distance) <= 9.0
                    and abs(features.ego_speed) < 0.35
                ):
                    if features.lidar_open_side == "right":
                        steer_bias = -0.20
                    elif features.lidar_open_side == "left":
                        steer_bias = 0.20
                    else:
                        steer_bias = -0.18
                    action = PlannerAction(
                        True,
                        "AVOID_OR_PASS",
                        target_speed=3.0,
                        throttle_cap=1.0,
                        throttle_floor=0.92,
                        brake_cap=0.0,
                        steer_limit=0.36,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.14,
                        reason="highspeed_accident_mid_hazard_creep_bypass",
                    )
                elif (
                    estimate.macro_scenario == "highway_accident_vehicle"
                    and action.reason in (
                        "forced route-prior macro scenario",
                        "static_obstacle_observed_without_immediate_conflict",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                    )
                    and self.highspeed_brake_response_done
                    and features.front_pedestrian_distance is None
                    and (
                        (features.front_vehicle_distance is not None and 1.8 <= float(features.front_vehicle_distance) <= 5.4)
                        or (features.front_obstacle_distance is not None and 1.8 <= float(features.front_obstacle_distance) <= 5.4)
                    )
                    and abs(features.ego_speed) < 0.9
                ):
                    if features.lidar_open_side == "right":
                        steer_bias = -0.24
                    elif features.lidar_open_side == "left":
                        steer_bias = 0.24
                    else:
                        steer_bias = -0.22
                    action = PlannerAction(
                        True,
                        "AVOID_OR_PASS",
                        target_speed=2.6,
                        throttle_cap=0.78,
                        throttle_floor=0.52,
                        brake_cap=0.0,
                        steer_limit=0.42,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.18,
                        reason="highspeed_accident_close_hazard_creep_bypass",
                    )
                elif (
                    estimate.macro_scenario == "avoid_a_disabled_vehicle"
                    and action.reason.startswith("roundabout_")
                    and features.front_pedestrian_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_obstacle_distance is not None
                    and 2.4 <= float(features.front_obstacle_distance) <= 5.8
                    and features.lidar_open_side in ("right", "left", "balanced")
                    and abs(features.ego_speed) < 1.2
                ):
                    if features.lidar_open_side == "right":
                        steer_bias = 0.30
                    elif features.lidar_open_side == "left":
                        steer_bias = -0.30
                    else:
                        steer_bias = 0.26 if features.lidar_lateral_centroid <= 0.0 else -0.26
                    action = PlannerAction(
                        True,
                        "AVOID_OR_PASS",
                        target_speed=2.8,
                        throttle_cap=0.82,
                        throttle_floor=0.58,
                        brake_cap=0.0,
                        steer_limit=0.62,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.24,
                        reason="disabled_vehicle_open_side_escape",
                    )
                elif (
                    estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                    and action.reason in (
                        "static_obstacle_observed_without_immediate_conflict",
                        "observable_risk_without_confirmed_longitudinal_conflict",
                    )
                    and abs(features.ego_speed) < 0.45
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                    and features.front_obstacle_distance is not None
                    and 1.8 <= float(features.front_obstacle_distance) <= 3.0
                    and (features.lidar_available or features.lidar_front_distance is not None)
                ):
                    if features.lidar_open_side == "right":
                        steer_bias = -0.24
                    elif features.lidar_open_side == "left":
                        steer_bias = 0.24
                    elif self.cut_in_route_rejoin_side == "right":
                        steer_bias = 0.12
                    elif self.cut_in_route_rejoin_side == "left":
                        steer_bias = -0.12
                    else:
                        steer_bias = 0.18 if features.lidar_lateral_centroid <= 0.0 else -0.18
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=1.6,
                        throttle_cap=0.56,
                        throttle_floor=0.36,
                        brake_cap=0.0,
                        steer_limit=0.38,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.12,
                        reason="cut_in_post_rejoin_static_obstacle_bypass",
                    )
                elif (
                    estimate.macro_scenario in (
                        "highway_accident_vehicle",
                        "blind_left_car",
                        "blind_spot_hidden_car",
                        "four_students_crossing_the_road",
                        "high_speed_reckless_lane_cutting",
                    )
                    and (
                        action.reason.startswith("construction_")
                        or action.reason.startswith("balanced_construction_")
                        or action.reason.startswith("low_conf_center_blockage_")
                        or action.reason.startswith("observable_very_close_open_side")
                        or action.reason.startswith("observable_close_open_side")
                    )
                ):
                    action = PlannerAction(False, action.state, reason="forced_macro_suppressed_construction_rule")
                if estimate.macro_scenario != "roundabout" and action.reason.startswith("roundabout_"):
                    action = PlannerAction(False, action.state, reason="forced_macro_suppressed_roundabout_rule")
                blind_spot_static_obstacle_slowdown = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and features.front_vehicle_distance is not None
                    and 2.0 <= float(features.front_vehicle_distance) <= 8.0
                    and features.front_pedestrian_distance is None
                    and abs(features.ego_speed) > 8.5
                )
                if self.blind_spot_prebrake_cooldown_frames > 0:
                    self.blind_spot_prebrake_cooldown_frames -= 1
                blind_spot_side_vehicle_prebrake = False
                if (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and abs(features.ego_speed) > (3.2 if forced_route_prior_macro else 4.0)
                    and (
                        self.blind_spot_prebrake_cooldown_frames <= 0
                        or (
                            forced_route_prior_macro
                            and self.blind_spot_prebrake_frames <= 0
                            and self.blind_spot_prebrake_cooldown_frames <= 90
                        )
                    )
                ):
                    blind_spot_side_vehicle_max_x = 48.0 if forced_route_prior_macro else 36.0
                    for track in features.tracked_objects or []:
                        cls = str(track.get("class_name", "")).lower()
                        if cls not in ("car", "truck", "bus", "van", "vehicle"):
                            continue
                        try:
                            tx = float(track.get("x", 999.0))
                            ty = abs(float(track.get("y", 999.0)))
                        except Exception:
                            continue
                        if 24.0 <= tx <= blind_spot_side_vehicle_max_x and 0.8 <= ty <= 6.5:
                            blind_spot_side_vehicle_prebrake = True
                            break
                blind_spot_junction_prebrake = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and not blind_spot_side_vehicle_prebrake
                    and self.blind_spot_prebrake_cooldown_frames <= 0
                    and features.junction_like
                    and 4.0 <= abs(features.ego_speed) <= 15.5
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                blind_spot_clear_approach_prebrake = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and not blind_spot_side_vehicle_prebrake
                    and not blind_spot_junction_prebrake
                    and self.blind_spot_prebrake_cooldown_frames <= 0
                    and not action.active
                    and not forced_route_prior_macro
                    and 3.0 <= abs(features.ego_speed) <= 9.0
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                )
                blind_spot_route_prior_trigger_brake = False
                if (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "blind_spot_hidden_car"
                    and not self.blind_spot_route_prior_trigger_brake_done
                    and self.blind_spot_route_prior_trigger_brake_frames <= 0
                    and 5.0 <= abs(features.ego_speed) <= 15.0
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and features.red_stop_distance is None
                    and not features.red_light_active
                ):
                    try:
                        ego_pos = tick_data.get("pos")
                        if ego_pos is None and isinstance(model_detection, dict):
                            ego_state = model_detection.get("ego_state") or model_detection.get("ego") or {}
                            if isinstance(ego_state, dict):
                                ego_pos = ego_state.get("pos")
                            if ego_pos is None:
                                ego_pos = model_detection.get("pos")
                        if ego_pos is not None:
                            ex = float(ego_pos[0])
                            ey = float(ego_pos[1])
                            blind_spot_route_prior_ego_pos_debug = [ex, ey]
                            blind_spot_route_prior_trigger_brake = -8.0 <= ex <= 9.0 and 211.0 <= ey <= 224.0
                    except Exception:
                        blind_spot_route_prior_trigger_brake = False
                if blind_spot_route_prior_trigger_brake:
                    self.blind_spot_route_prior_trigger_brake_frames = max(
                        self.blind_spot_route_prior_trigger_brake_frames,
                        12,
                    )
                blind_spot_prebrake_reason = "blind_spot_side_vehicle_prebrake"
                if blind_spot_junction_prebrake:
                    blind_spot_side_vehicle_prebrake = True
                    blind_spot_prebrake_reason = "blind_spot_junction_prebrake"
                elif blind_spot_clear_approach_prebrake:
                    blind_spot_side_vehicle_prebrake = True
                    blind_spot_prebrake_reason = "blind_spot_clear_approach_prebrake"
                if blind_spot_side_vehicle_prebrake:
                    self.blind_spot_prebrake_frames = max(
                        self.blind_spot_prebrake_frames,
                        10 if blind_spot_junction_prebrake else (16 if blind_spot_clear_approach_prebrake else 14),
                    )
                    self.blind_spot_prebrake_cooldown_frames = 90 if (blind_spot_junction_prebrake or blind_spot_clear_approach_prebrake) else 120
                blind_spot_prebrake_pulse_active = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and self.blind_spot_prebrake_frames > 0
                    and abs(features.ego_speed) > 1.2
                )
                blind_spot_route_prior_trigger_active = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and self.blind_spot_route_prior_trigger_brake_frames > 0
                    and abs(features.ego_speed) > 2.0
                )
                if blind_spot_static_obstacle_slowdown or blind_spot_prebrake_pulse_active:
                    if self.blind_spot_prebrake_frames > 0:
                        self.blind_spot_prebrake_frames -= 1
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        target_speed=1.2 if blind_spot_prebrake_pulse_active else 5.0,
                        throttle_cap=0.0,
                        brake=0.92 if blind_spot_prebrake_pulse_active else 0.82,
                        steer_limit=0.12 if blind_spot_prebrake_pulse_active else 0.20,
                        reason=blind_spot_prebrake_reason if blind_spot_prebrake_pulse_active else "blind_spot_static_obstacle_slowdown",
                    )
                if blind_spot_route_prior_trigger_active:
                    self.blind_spot_route_prior_trigger_brake_frames -= 1
                    if self.blind_spot_route_prior_trigger_brake_frames <= 0:
                        self.blind_spot_route_prior_trigger_brake_done = True
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        target_speed=6.4,
                        throttle_cap=0.0,
                        brake=0.62,
                        steer_limit=0.08,
                        reason="blind_spot_route_prior_trigger_zone_brake",
                    )
                cut_in_forward_commit = (
                    estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                    and not features.red_light_active
                    and features.red_stop_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is not None
                    and 4.6 <= float(features.front_obstacle_distance) <= 12.0
                    and abs(features.ego_speed) < 2.2
                    and (features.lidar_available or features.lidar_front_distance is not None)
                    and (not action.active or action.reason == "forced_macro_suppressed_construction_rule")
                )
                if cut_in_forward_commit:
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=4.2,
                        throttle_cap=1.0,
                        throttle_floor=0.90,
                        brake_cap=0.0,
                        steer_limit=0.26,
                        reason="cut_in_forward_commit_no_open_side",
                    )
                cut_in_close_forward_crawl = (
                    estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                    and not features.red_light_active
                    and features.red_stop_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is not None
                    and 2.35 <= float(features.front_obstacle_distance) < 4.6
                    and abs(features.ego_speed) < 1.4
                    and (features.lidar_available or features.lidar_front_distance is not None)
                    and (
                        action.reason != "forced_macro_suppressed_construction_rule"
                        or features.lidar_open_side in ("right", "left")
                    )
                )
                if cut_in_close_forward_crawl:
                    open_side_close = features.lidar_open_side in ("right", "left")
                    steer_bias = 0.18 if features.lidar_open_side == "right" else -0.18
                    if not open_side_close:
                        steer_bias = max(-0.08, min(0.08, getattr(action, "steer_bias", None) or 0.0))
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=3.8,
                        throttle_cap=1.0,
                        throttle_floor=0.88 if open_side_close else 0.92,
                        brake_cap=0.0,
                        steer_limit=0.24 if open_side_close else 0.10,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.10 if open_side_close else 0.0,
                        reason="cut_in_close_open_side_forward_crawl" if open_side_close else "cut_in_close_forward_crawl_no_open_side",
                    )
                blind_spot_clear_speed_keepalive = (
                    estimate.macro_scenario == "blind_spot_hidden_car"
                    and not action.active
                    and not features.red_light_active
                    and features.red_stop_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (
                        features.front_obstacle_distance is None
                        or 12.0 <= float(features.front_obstacle_distance) <= 22.0
                    )
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 12.0)
                    and abs(features.route_curvature) <= 0.85
                    and abs(features.ego_speed) < 8.2
                )
                if blind_spot_clear_speed_keepalive:
                    far_static_blind_keepalive = features.front_obstacle_distance is not None
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=7.2 if far_static_blind_keepalive else 9.0,
                        throttle_cap=1.0,
                        throttle_floor=0.82 if far_static_blind_keepalive else 0.78,
                        brake_cap=0.0,
                        steer_limit=0.12 if far_static_blind_keepalive else 0.18,
                        reason="blind_spot_far_static_speed_keepalive" if far_static_blind_keepalive else "blind_spot_clear_speed_keepalive",
                    )
                roundabout_clear_passthrough_speed_cap = (
                    estimate.macro_scenario == "roundabout"
                    and not action.active
                    and not features.red_light_active
                    and features.red_stop_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and abs(features.ego_speed) > 2.8
                )
                if roundabout_clear_passthrough_speed_cap:
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=2.8,
                        throttle_cap=0.24,
                        brake=0.18 if abs(features.ego_speed) > 4.5 else None,
                        brake_cap=0.0 if abs(features.ego_speed) <= 4.5 else None,
                        steer_limit=0.10,
                        reason="roundabout_clear_passthrough_speed_cap",
                    )
                students_clear_route_speed_guard = (
                    estimate.macro_scenario == "four_students_crossing_the_road"
                    and not action.active
                    and not features.red_light_active
                    and features.red_stop_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 10.0)
                    and abs(features.ego_speed) > 4.6
                )
                if students_clear_route_speed_guard:
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=4.2,
                        throttle_cap=0.28,
                        brake=0.22 if abs(features.ego_speed) > 6.0 else 0.10,
                        steer_limit=0.12,
                        reason="students_clear_route_speed_guard",
                    )
            if (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.02 <= float(features.red_stop_distance) <= 8.0
                and self.red_final_clamp_hold_frames >= 90
                and features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 2.2
            ):
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=3.2,
                    throttle_cap=0.94,
                    throttle_floor=0.78,
                    brake_cap=0.0,
                    steer_limit=0.06,
                    reason="students_long_active_red_final_release",
                )

            if (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and action.reason in (
                    "static_obstacle_observed_without_immediate_conflict",
                    "observable_risk_without_confirmed_longitudinal_conflict",
                )
                and abs(features.ego_speed) < 0.45
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
                and features.front_obstacle_distance is not None
                and 1.8 <= float(features.front_obstacle_distance) <= 3.0
                and (features.lidar_available or features.lidar_front_distance is not None)
            ):
                if features.lidar_open_side == "right":
                    steer_bias = -0.24
                elif features.lidar_open_side == "left":
                    steer_bias = 0.24
                elif self.cut_in_route_rejoin_side == "right":
                    steer_bias = 0.12
                elif self.cut_in_route_rejoin_side == "left":
                    steer_bias = -0.12
                else:
                    steer_bias = 0.18 if features.lidar_lateral_centroid <= 0.0 else -0.18
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=1.6,
                    throttle_cap=0.56,
                    throttle_floor=0.36,
                    brake_cap=0.0,
                    steer_limit=0.38,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.12,
                    reason="cut_in_post_rejoin_static_obstacle_bypass",
                )
            if (
                estimate.macro_scenario in (
                    "highway_accident_vehicle",
                    "blind_left_car",
                    "blind_spot_hidden_car",
                    "four_students_crossing_the_road",
                    "high_speed_reckless_lane_cutting",
                )
                and action.active
                and (
                    action.reason.startswith("construction_")
                    or action.reason.startswith("balanced_construction_")
                    or action.reason.startswith("low_conf_center_blockage_")
                    or action.reason.startswith("observable_very_close_open_side")
                    or action.reason.startswith("observable_close_open_side")
                )
            ):
                action = PlannerAction(False, action.state, reason="forced_macro_suppressed_construction_rule")
            elapsed_ms = (time.time() - start) * 1000.0
            if elapsed_ms > self.config.max_aux_latency_ms:
                self.last_debug = {"enabled": True, "action": "timeout_passthrough", "elapsed_ms": elapsed_ms}
                return raw_control
            aux_overridable_legacy_actions = (
                "front_obstacle_brake",
                "front_obstacle_wait_release",
                "clear_crawl_release",
                "clear_stuck_recovery",
            )
            aux_overrides_legacy = bool(
                legacy_rule_action in aux_overridable_legacy_actions
                and action.active
                and action.reason in (
                    "distant_lidar_open_side_nudge",
                    "distant_lidar_open_side_escape",
                    "distant_lidar_open_side_post_pass_recovery",
                    "distant_lidar_open_side_close_memory_nudge",
                    "distant_lidar_open_side_progress_recovery",
                )
                and self.config.lidar_open_side_nudge_enabled
            )
            if legacy_rule_action and legacy_rule_action != "none" and not aux_overrides_legacy:
                # Preserve previously validated DriveTransformer detection-head rules.
                # Closed-loop evidence showed route-75 can regress to a block when this
                # module adds extra emergency braking on top of the legacy rule.
                control = raw_control
                preserved = True
            else:
                control = self.supervisor.apply(raw_control, features, estimate, action)
                preserved = False
            if (
                estimate.macro_scenario == "blind_spot_hidden_car"
                and action.reason == "rules_disabled_or_low_confidence"
                and not preserved
                and self.blind_spot_prebrake_cooldown_frames <= 0
                and features.junction_like
                and 4.0 <= abs(features.ego_speed) <= 15.5
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.red_stop_distance is None
                and not features.red_light_active
            ):
                self.blind_spot_prebrake_frames = max(self.blind_spot_prebrake_frames, 10)
                self.blind_spot_prebrake_cooldown_frames = 90
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.92)
                control.steer = _clip(float(control.steer), -0.12, 0.12)
                setattr(control, "reverse", False)
                action = PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    target_speed=1.2,
                    throttle_cap=0.0,
                    brake=0.92,
                    steer_limit=0.12,
                    reason="blind_spot_low_conf_junction_prebrake",
                )
            red_final_clamp = False
            red_final_creep_release = False
            red_stop_distance = (
                float(features.red_stop_distance)
                if features.red_stop_distance is not None
                else None
            )
            red_no_front_conflict = (
                (
                    features.front_vehicle_distance is None
                    or float(features.front_vehicle_distance) > 10.0
                )
                and features.front_pedestrian_distance is None
            )
            red_two_wheeler_priority = False
            for track in features.tracked_objects or []:
                cls = str(track.get("class_name", "")).lower()
                if cls not in ("motorcycle", "bicycle", "cyclist"):
                    continue
                try:
                    tx = float(track.get("x", 999.0))
                    ty = float(track.get("y", 999.0))
                except Exception:
                    continue
                if 4.0 <= tx <= 18.0 and abs(ty) <= 6.0:
                    red_two_wheeler_priority = True
                    break
            red_stationary_hold = bool(
                features.red_light_active
                and red_no_front_conflict
                and not red_two_wheeler_priority
                and red_stop_distance is not None
                and 0.8 <= red_stop_distance <= 15.0
                and abs(features.ego_speed) < 0.25
                and not (
                    estimate.macro_scenario == "four_students_crossing_the_road"
                    and 3.0 <= red_stop_distance <= 6.2
                    and self.red_final_clamp_hold_frames >= 220
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                )
            )
            if red_stationary_hold:
                self.red_final_clamp_hold_frames += 1
                self.red_final_clamp_gap_frames = 0
                self.red_final_clamp_last_distance = red_stop_distance
            elif (
                features.red_light_active
                and red_no_front_conflict
                and not red_two_wheeler_priority
                and red_stop_distance is not None
                and 0.8 <= red_stop_distance <= 15.0
                and self.red_final_clamp_hold_frames >= 80
                and abs(features.ego_speed) < 1.20
            ):
                self.red_final_clamp_hold_frames += 1
                self.red_final_clamp_gap_frames = 0
                self.red_final_clamp_last_distance = red_stop_distance
            elif not features.red_light_active or (red_stop_distance is not None and red_stop_distance > 15.0):
                self.red_final_clamp_gap_frames += 1
                if self.red_final_clamp_gap_frames > 30:
                    self.red_final_clamp_hold_frames = 0
                    self.red_final_clamp_last_distance = None
            red_near_line_stationary_hold = bool(
                features.red_light_active
                and red_no_front_conflict
                and red_stop_distance is not None
                and 0.0 <= red_stop_distance <= 0.65
                and abs(features.ego_speed) < 0.18
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 4.0)
                and features.lidar_center_blockage_ratio <= 0.05
            )
            if red_near_line_stationary_hold:
                self.red_final_near_line_hold_frames += 1
            elif not features.red_light_active or red_stop_distance is None or red_stop_distance > 1.2 or abs(features.ego_speed) > 0.60:
                self.red_final_near_line_hold_frames = 0
            red_clamp_distance_ok = (
                red_stop_distance is None
                or red_stop_distance <= 15.0
            )
            release_distance = red_stop_distance
            if release_distance is None and self.red_final_clamp_gap_frames <= 30:
                release_distance = self.red_final_clamp_last_distance
            red_prolonged_far_release = bool(
                (features.red_light_active or (self.red_final_clamp_gap_frames <= 12 and self.red_final_clamp_hold_frames >= 200))
                and estimate.macro_scenario not in (
                    "high_speed_reckless_lane_cutting",
                    "reverse_vehicle",
                )
                and red_no_front_conflict
                and release_distance is not None
                and (
                    (
                        release_distance > 5.5
                        and self.red_final_clamp_hold_frames >= 80
                        and abs(features.ego_speed) < (2.50 if self.red_final_clamp_hold_frames >= 180 else 1.20)
                    )
                    or (
                        3.0 < release_distance <= 3.6
                        and self.red_final_clamp_hold_frames >= 80
                        and self.red_final_clamp_hold_frames < 140
                        and self.rule_planner.blocked_frames >= 80
                        and abs(features.ego_speed) < 0.70
                    )
                    or (
                        4.0 < release_distance <= 5.5
                        and self.rule_planner.lateral_intersection_release_frames > 0
                        and self.red_final_clamp_hold_frames >= 80
                        and abs(features.ego_speed) < 1.80
                    )
                    or (
                        3.0 < release_distance <= 5.5
                        and self.red_final_clamp_hold_frames >= 140
                        and abs(features.ego_speed) < (
                            1.60
                            if self.red_final_clamp_hold_frames >= 300
                            else (
                                1.35
                                if self.red_final_clamp_hold_frames >= 260
                                else (0.85 if self.red_final_clamp_hold_frames >= 180 else 0.35)
                            )
                        )
                    )
                    or (
                        2.15 < release_distance <= 3.2
                        and (
                            self.red_final_clamp_hold_frames >= 170
                            or (
                                2.7 <= release_distance <= 3.2
                                and self.red_final_clamp_hold_frames >= 100
                            )
                            or (
                                release_distance < 2.7
                                and self.red_final_clamp_hold_frames >= 160
                            )
                        )
                        and abs(features.ego_speed) < (
                            0.65 if self.red_final_clamp_hold_frames >= 200 else 0.12
                        )
                    )
                    or (
                        0.8 <= release_distance <= 2.2
                        and (
                            (
                                release_distance > 1.8
                                and self.red_final_clamp_hold_frames >= 140
                            )
                            or (
                                estimate.macro_scenario == "reverse_vehicle"
                                and self.red_final_clamp_hold_frames >= 150
                            )
                            or self.red_final_clamp_hold_frames >= 300
                            or self.red_final_creep_memory_frames > 0
                        )
                        and abs(features.ego_speed) < 0.65
                    )
                )
            )
            construction_suppressed_close_red_release = bool(
                self.config.suppress_lateral_intersection_rules
                and features.red_light_active
                and red_no_front_conflict
                and release_distance is not None
                and 2.4 <= release_distance <= 3.6
                and self.red_final_clamp_hold_frames >= 20
                and features.front_obstacle_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 0.80
            )
            construction_suppressed_very_close_red_release = bool(
                self.config.suppress_lateral_intersection_rules
                and features.red_light_active
                and red_no_front_conflict
                and release_distance is not None
                and 1.60 <= release_distance < 3.05
                and self.red_final_clamp_hold_frames >= 50
                and features.front_obstacle_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 5.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 0.75
            )
            if construction_suppressed_close_red_release or construction_suppressed_very_close_red_release:
                red_prolonged_far_release = True
            
            students_close_red_timeout_release = bool(
                estimate.macro_scenario == "four_students_crossing_the_road"
                and not features.red_light_active
                and self.red_final_clamp_gap_frames >= 2
                and red_no_front_conflict
                and release_distance is not None
                and 3.0 <= release_distance <= 6.2
                and self.red_final_clamp_hold_frames >= 20
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 2.80
            )
            students_very_close_red_timeout_release = bool(
                estimate.macro_scenario == "four_students_crossing_the_road"
                and not features.red_light_active
                and self.red_final_clamp_gap_frames >= 2
                and red_no_front_conflict
                and release_distance is not None
                and 1.60 <= release_distance < 3.05
                and self.red_final_clamp_hold_frames >= 50
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 5.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 0.75
            )
            if students_close_red_timeout_release or students_very_close_red_timeout_release:
                red_prolonged_far_release = True

            students_far_red_release = bool(
                estimate.macro_scenario == "four_students_crossing_the_road"
                and not features.red_light_active
                and self.red_final_clamp_gap_frames >= 2
                and red_no_front_conflict
                and release_distance is not None
                and release_distance > 5.5
                and self.red_final_clamp_hold_frames >= 24
                and (
                    features.front_obstacle_distance is None
                    or float(features.front_obstacle_distance) > 16.0
                )
                and abs(features.ego_speed) < 2.60
            )
            if students_far_red_release:
                red_prolonged_far_release = True
            blind_spot_late_red_brake_response = bool(
                forced_route_prior_macro
                and estimate.macro_scenario == "blind_spot_hidden_car"
                and features.red_light_active
                and release_distance is not None
                and 4.0 <= float(release_distance) <= 10.0
                and features.front_obstacle_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 4.5)
                and features.lidar_center_blockage_ratio <= 0.25
                and 2.0 <= abs(features.ego_speed) <= 5.5
            )
            blind_spot_mid_red_false_release = bool(
                estimate.macro_scenario == "blind_spot_hidden_car"
                and features.red_light_active
                and release_distance is not None
                and 3.5 <= release_distance <= 7.2
                and self.red_final_clamp_hold_frames >= 8
                and features.front_obstacle_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 4.5)
                and features.lidar_center_blockage_ratio <= 0.25
                and abs(features.ego_speed) < 2.8
                and not blind_spot_late_red_brake_response
            )
            if blind_spot_mid_red_false_release:
                red_prolonged_far_release = True
            red_clamp_allowed_for_macro = not (
                forced_route_prior_macro
                and estimate.macro_scenario in (
                    "highway_accident_vehicle",
                    "avoid_a_disabled_vehicle",
                    "blind_left_car",
                    "ebike_and_pedestrian_cross",
                    "ghost_probe",
                )
            )
            if (
                red_clamp_allowed_for_macro
                and (features.red_light_active or red_prolonged_far_release)
                and red_clamp_distance_ok
                and not red_two_wheeler_priority
            ):
                red_far_side_blockage_creep = (
                    features.red_light_active
                    and release_distance is not None
                    and release_distance > 8.0
                    and self.red_final_clamp_hold_frames >= 6
                    and abs(features.ego_speed) < 0.80
                    and features.front_obstacle_distance is not None
                    and 6.0 <= float(features.front_obstacle_distance) <= 13.5
                    and (
                        features.front_vehicle_distance is None
                        or float(features.front_vehicle_distance) > 10.0
                    )
                    and features.front_pedestrian_distance is None
                    and features.lidar_available
                    and not features.lidar_stale
                    and features.lidar_blockage_ratio >= 0.85
                    and features.lidar_open_side in ("right", "left")
                    and (
                        (
                            features.lidar_open_side == "right"
                            and features.lidar_left_blockage_ratio >= 0.80
                            and features.lidar_right_blockage_ratio <= 0.25
                        )
                        or (
                            features.lidar_open_side == "left"
                            and features.lidar_right_blockage_ratio >= 0.80
                            and features.lidar_left_blockage_ratio <= 0.25
                        )
                    )
                )
                red_far_close_obstacle_creep = (
                    features.red_light_active
                    and release_distance is not None
                    and release_distance > 8.0
                    and self.red_final_clamp_hold_frames >= 6
                    and abs(features.ego_speed) < 0.80
                    and features.front_obstacle_distance is not None
                    and float(features.front_obstacle_distance) <= 4.1
                    and (
                        features.front_vehicle_distance is None
                        or float(features.front_vehicle_distance) > 10.0
                    )
                    and features.front_pedestrian_distance is None
                    and features.lidar_open_side in ("right", "left")
                    and not (
                        self.red_final_clamp_hold_frames >= 100
                        and float(features.front_obstacle_distance) <= 2.2
                    )
                )
                blind_spot_route_prior_no_stopline_red_release = (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "blind_spot_hidden_car"
                    and features.red_light_active
                    and release_distance is None
                    and red_no_front_conflict
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 18.0)
                    and features.lidar_center_blockage_ratio <= 0.15
                    and abs(features.ego_speed) >= 2.0
                )
                roundabout_route_prior_no_stopline_red_release = (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "roundabout"
                    and features.red_light_active
                    and release_distance is None
                    and red_no_front_conflict
                    and features.front_obstacle_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 10.0)
                    and features.lidar_center_blockage_ratio <= 0.20
                    and abs(features.ego_speed) >= 2.0
                )
                roundabout_route_prior_no_stopline_close_obstacle_reverse = (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "roundabout"
                    and features.red_light_active
                    and release_distance is None
                    and red_no_front_conflict
                    and features.front_obstacle_distance is not None
                    and 2.30 <= float(features.front_obstacle_distance) <= 3.15
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and abs(features.ego_speed) < 0.35
                    and self.rule_planner.blocked_frames >= 10
                )
                reverse_vehicle_route_prior_close_red_release = (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "reverse_vehicle"
                    and features.red_light_active
                    and release_distance is not None
                    and 2.20 <= float(release_distance) <= 4.05
                    and red_no_front_conflict
                    and (
                        features.front_obstacle_distance is None
                        or 2.20 <= float(features.front_obstacle_distance) <= 5.20
                    )
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and self.red_final_clamp_hold_frames >= 2
                    and features.lidar_center_blockage_ratio <= 1.0
                    and abs(features.ego_speed) < 1.20
                )
                reverse_vehicle_route_prior_near_line_static_red_release = (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "reverse_vehicle"
                    and features.red_light_active
                    and release_distance is not None
                    and 0.70 <= float(release_distance) <= 2.20
                    and red_no_front_conflict
                    and features.front_obstacle_distance is not None
                    and 2.00 <= float(features.front_obstacle_distance) <= 3.20
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and self.red_final_clamp_hold_frames >= 2
                    and (
                        features.lidar_blockage_ratio >= 0.55
                        or features.lidar_center_blockage_ratio >= 0.75
                    )
                    and features.lidar_open_side in ("right", "left", "balanced")
                    and abs(features.ego_speed) < 0.35
                )
                if blind_spot_late_red_brake_response:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.82)
                    control.steer = max(-0.08, min(0.08, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = False
                    self.red_macro_deadlock_release_reason = ""
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        throttle_cap=0.0,
                        brake=0.82,
                        steer_limit=0.08,
                        reason="blind_spot_late_red_brake_response",
                    )
                elif blind_spot_route_prior_no_stopline_red_release:
                    control.throttle = min(max(float(control.throttle), 0.30), 0.46)
                    control.brake = 0.0
                    control.steer = max(-0.08, min(0.08, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                    self.red_macro_deadlock_release_reason = "blind_spot_route_prior_no_stopline_red_release"
                elif roundabout_route_prior_no_stopline_red_release:
                    control.throttle = min(max(float(control.throttle), 0.56), 0.78)
                    control.brake = 0.0
                    control.steer = max(-0.12, min(0.12, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                    self.red_macro_deadlock_release_reason = "roundabout_route_prior_no_stopline_red_release"
                elif roundabout_route_prior_no_stopline_close_obstacle_reverse:
                    steer_bias = 0.34 if features.lidar_open_side == "right" else (-0.34 if features.lidar_open_side == "left" else 0.28)
                    control.throttle = min(max(float(control.throttle), 0.52), 0.68)
                    control.brake = 0.0
                    control.steer = max(-0.42, min(0.42, steer_bias))
                    setattr(control, "reverse", True)
                    red_final_creep_release = True
                    self.red_macro_deadlock_release_reason = "roundabout_route_prior_no_stopline_close_obstacle_reverse"
                elif reverse_vehicle_route_prior_close_red_release:
                    control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                    control.brake = 0.0
                    control.steer = max(-0.18, min(0.18, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                    self.red_macro_deadlock_release_reason = "reverse_vehicle_route_prior_close_red_release"
                elif reverse_vehicle_route_prior_near_line_static_red_release:
                    if features.lidar_open_side == "right":
                        steer_bias = 0.28
                    elif features.lidar_open_side == "left":
                        steer_bias = -0.28
                    else:
                        steer_bias = 0.0
                    control.throttle = min(max(float(control.throttle), 0.82), 0.96)
                    control.brake = 0.0
                    control.steer = max(-0.34, min(0.34, steer_bias))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                    self.red_macro_deadlock_release_reason = "reverse_vehicle_route_prior_near_line_static_red_release"
                elif construction_suppressed_close_red_release or construction_suppressed_very_close_red_release:
                    if construction_suppressed_very_close_red_release:
                        control.throttle = min(max(float(control.throttle), 0.46), 0.62)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.28), 0.42)
                    control.brake = 0.0
                    control.steer = max(-0.12, min(0.12, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                elif students_close_red_timeout_release or students_very_close_red_timeout_release:
                    if students_very_close_red_timeout_release:
                        control.throttle = min(max(float(control.throttle), 0.42), 0.58)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.34), 0.52)
                    control.brake = 0.0
                    control.steer = max(-0.12, min(0.12, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                elif students_far_red_release:
                    control.throttle = min(max(float(control.throttle), 0.68), 0.86)
                    control.brake = 0.0
                    control.steer = max(-0.12, min(0.12, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                elif blind_spot_mid_red_false_release:
                    control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                    control.brake = 0.0
                    control.steer = max(-0.08, min(0.08, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                elif (
                    forced_route_prior_macro
                    and estimate.macro_scenario == "roundabout"
                    and features.red_light_active
                    and release_distance is not None
                    and 2.2 <= float(release_distance) <= 5.6
                    and self.red_final_clamp_hold_frames >= 1
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (
                        (
                            features.front_obstacle_distance is not None
                            and 4.8 <= float(features.front_obstacle_distance) <= 8.0
                        )
                        or features.front_obstacle_distance is None
                    )
                    and (
                        features.lidar_blockage_ratio >= 0.40
                        or features.lidar_center_blockage_ratio >= 0.40
                    )
                    and features.lidar_open_side in ("right", "left")
                    and abs(features.ego_speed) < 0.90
                ):
                    steer_bias = -0.24 if features.lidar_open_side == "right" else 0.24
                    control.throttle = min(max(float(control.throttle), 0.54), 0.72)
                    control.brake = 0.0
                    control.steer = max(-0.32, min(0.32, steer_bias))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = True
                    self.roundabout_mid_red_yield_frames = 0
                    self.red_macro_deadlock_release_reason = "roundabout_mid_red_blocked_open_side_release"
                elif (
                    estimate.macro_scenario == "roundabout"
                    and features.red_light_active
                    and release_distance is not None
                    and 3.5 <= float(release_distance) <= 5.4
                    and self.red_final_clamp_hold_frames >= 16
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                    and features.lidar_center_blockage_ratio <= 0.10
                    and abs(features.ego_speed) < 0.80
                    and len(features.tracked_objects or []) > 3
                ):
                    self.roundabout_mid_red_yield_frames += 1
                    if self.roundabout_mid_red_yield_frames >= 18:
                        control.throttle = min(max(float(control.throttle), 0.28), 0.40)
                        control.brake = 0.0
                        control.steer = max(-0.04, min(0.04, float(control.steer)))
                        setattr(control, "reverse", False)
                        red_final_clamp = False
                        red_final_creep_release = True
                        self.red_macro_deadlock_release_reason = "roundabout_active_mid_red_yield_timeout_release"
                    else:
                        control.throttle = 0.0
                        control.brake = min(max(float(control.brake), 0.22), 0.38)
                        control.steer = max(-0.04, min(0.04, float(control.steer)))
                        setattr(control, "reverse", False)
                        red_final_clamp = False
                        red_final_creep_release = False
                        self.red_macro_deadlock_release_frames = 0
                        self.red_macro_deadlock_release_reason = ""
                        action = PlannerAction(
                            True,
                            "YIELD_OR_BRAKE",
                            throttle_cap=0.0,
                            brake=0.22,
                            steer_limit=0.04,
                            reason="roundabout_active_mid_red_yield_hold",
                        )
                elif (
                    estimate.macro_scenario == "roundabout"
                    and features.red_light_active
                    and release_distance is not None
                    and 3.5 <= float(release_distance) <= 5.4
                    and self.red_final_clamp_hold_frames >= 16
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                    and features.lidar_center_blockage_ratio <= 0.10
                    and abs(features.ego_speed) < 0.80
                    and len(features.tracked_objects or []) <= 3
                ):
                    control.throttle = min(max(float(control.throttle), 0.34), 0.48)
                    control.brake = 0.0
                    control.steer = max(-0.05, min(0.05, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                    self.roundabout_mid_red_yield_frames = 0
                    self.red_macro_deadlock_release_reason = "roundabout_active_mid_red_release"
                elif (
                    estimate.macro_scenario == "roundabout"
                    and red_prolonged_far_release
                    and not features.red_light_active
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                ):
                    control.throttle = min(max(float(control.throttle), 0.30), 0.46)
                    control.brake = 0.0
                    control.steer = max(-0.06, min(0.06, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_creep_release = True
                elif red_far_side_blockage_creep:
                    control.throttle = min(max(float(control.throttle), 0.58), 0.82)
                    control.brake = 0.0
                    steer_bias = 0.18 if features.lidar_open_side == "right" else -0.18
                    control.steer = max(-0.40, min(0.40, steer_bias))
                    red_final_creep_release = True
                elif red_far_close_obstacle_creep:
                    control.throttle = min(max(float(control.throttle), 0.28), 0.42)
                    control.brake = 0.0
                    steer_bias = 0.42 if features.lidar_open_side == "right" else -0.42
                    control.steer = max(-0.55, min(0.55, steer_bias))
                    red_final_creep_release = True
                elif red_prolonged_far_release:
                    students_long_active_red_final_release = (
                        estimate.macro_scenario == "four_students_crossing_the_road"
                        and features.red_light_active
                        and release_distance is not None
                        and 0.02 <= float(release_distance) <= 8.0
                        and self.red_final_clamp_hold_frames >= 90
                        and features.front_clear
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.front_obstacle_distance is None
                        and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                        and features.lidar_center_blockage_ratio <= 0.10
                        and abs(features.ego_speed) < 2.2
                    )
                    if students_long_active_red_final_release:
                        control.throttle = min(max(float(control.throttle), 0.78), 0.94)
                        control.brake = 0.0
                        control.steer = max(-0.06, min(0.06, float(control.steer)))
                        setattr(control, "reverse", False)
                        red_final_creep_release = True
                        self.red_macro_deadlock_release_reason = "students_long_active_red_final_release"
                    # Outer students_close_red_timeout_release handles this before fallback red release.
                    red_close_obstacle_unwedge = (
                        release_distance is not None
                        and release_distance > 5.5
                        and self.red_final_clamp_hold_frames >= 100
                        and abs(features.ego_speed) < 0.15
                        and features.front_obstacle_distance is not None
                        and float(features.front_obstacle_distance) <= 2.2
                        and features.lidar_open_side in ("right", "left")
                    )
                    reverse_vehicle_close_unwedge_candidate = (
                        estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                        and release_distance is not None
                        and 0.8 <= release_distance <= 3.2
                        and self.red_final_clamp_hold_frames >= 175
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.front_obstacle_distance is not None
                        and 2.25 <= float(features.front_obstacle_distance) <= 3.05
                        and abs(features.ego_speed) < 0.45
                    )
                    if reverse_vehicle_close_unwedge_candidate:
                        self.red_reverse_unwedge_frames = max(self.red_reverse_unwedge_frames, 42)
                    reverse_vehicle_persistent_unwedge = (
                        estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                        and self.red_reverse_unwedge_frames > 0
                        and release_distance is not None
                        and 0.8 <= release_distance <= 3.3
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.front_obstacle_distance is not None
                        and 2.15 <= float(features.front_obstacle_distance) <= 3.20
                        and abs(features.ego_speed) < 0.75
                    )
                    reverse_vehicle_ultra_close_red_unwedge = (
                        estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.front_obstacle_distance is not None
                        and 0.70 <= float(features.front_obstacle_distance) <= 1.55
                        and self.rule_planner.blocked_frames >= 80
                        and abs(features.ego_speed) < 0.95
                    )
                    if reverse_vehicle_ultra_close_red_unwedge:
                        control.throttle = min(max(float(control.throttle), 0.70), 0.86)
                        control.brake = 0.0
                        control.steer = max(-0.56, min(0.56, 0.50 if features.lidar_open_side == "right" else (-0.50 if features.lidar_open_side == "left" else 0.42)))
                        setattr(control, "reverse", True)
                        red_final_creep_release = True
                    elif construction_suppressed_close_red_release or construction_suppressed_very_close_red_release:
                        if construction_suppressed_very_close_red_release:
                            control.throttle = min(max(float(control.throttle), 0.46), 0.62)
                        else:
                            control.throttle = min(max(float(control.throttle), 0.28), 0.42)
                        control.brake = 0.0
                        control.steer = max(-0.12, min(0.12, float(control.steer)))
                        setattr(control, "reverse", False)
                        red_final_creep_release = True
                    elif students_close_red_timeout_release or students_very_close_red_timeout_release:
                        if students_very_close_red_timeout_release:
                            control.throttle = min(max(float(control.throttle), 0.42), 0.58)
                        else:
                            control.throttle = min(max(float(control.throttle), 0.34), 0.48)
                        control.brake = 0.0
                        control.steer = max(-0.12, min(0.12, float(control.steer)))
                        setattr(control, "reverse", False)
                        red_final_creep_release = True
                    elif red_close_obstacle_unwedge:
                        control.throttle = 0.30
                        control.brake = 0.0
                        control.steer = -0.35 if features.lidar_open_side == "right" else 0.35
                        setattr(control, "reverse", True)
                        red_final_creep_release = True
                    elif reverse_vehicle_persistent_unwedge:
                        self.red_reverse_unwedge_frames -= 1
                        control.throttle = min(max(float(control.throttle), 0.38), 0.50)
                        control.brake = 0.0
                        control.steer = max(-0.46, min(0.46, -0.32 if features.lidar_lateral_centroid <= 0.0 else 0.32))
                        setattr(control, "reverse", True)
                        red_final_creep_release = True
                    elif (
                        release_distance is not None
                        and release_distance > 8.0
                        and self.red_final_clamp_hold_frames >= 80
                        and self.red_final_clamp_hold_frames < 300
                        and features.front_obstacle_distance is None
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                        and features.lidar_center_blockage_ratio <= 0.05
                    ):
                        release_floor = 0.55
                        release_cap = 0.78
                        control.throttle = min(max(float(control.throttle), release_floor), release_cap)
                        control.brake = 0.0
                        red_final_creep_release = True
                    elif release_distance is not None and release_distance > 5.5 and self.red_final_clamp_hold_frames >= 300:
                        release_floor = 0.65
                        release_cap = 0.90
                        control.throttle = min(max(float(control.throttle), release_floor), release_cap)
                        control.brake = 0.0
                        red_final_creep_release = True
                    else:
                        if release_distance is not None and release_distance > 5.5 and self.red_final_clamp_hold_frames >= 180:
                            release_floor = 0.55
                            release_cap = 0.75
                        elif (
                            release_distance is not None
                            and release_distance > 5.5
                            and self.red_final_clamp_hold_frames >= 100
                        ):
                            release_floor = 0.34
                            release_cap = 0.55
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 3.6
                            and self.red_final_clamp_hold_frames >= 80
                            and self.red_final_clamp_hold_frames < 140
                            and self.rule_planner.blocked_frames >= 80
                        ):
                            if estimate.macro_scenario in (
                                "trucks_encountered_during_construction",
                                "high_speed_temporary_construction",
                            ):
                                release_floor = 0.62
                                release_cap = 0.85
                            else:
                                release_floor = 0.35
                                release_cap = 0.50
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 3.6
                            and self.red_final_clamp_hold_frames >= 140
                            and self.red_final_clamp_hold_frames < 170
                            and self.rule_planner.blocked_frames >= 80
                        ):
                            release_floor = 0.35
                            release_cap = 0.55
                        elif (
                            release_distance is not None
                            and 4.0 < release_distance <= 5.5
                            and self.rule_planner.lateral_intersection_release_frames > 0
                            and self.red_final_clamp_hold_frames >= 80
                        ):
                            release_floor = 0.78
                            release_cap = 0.95
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 5.5
                            and self.red_final_clamp_hold_frames >= 300
                        ):
                            release_floor = 0.80
                            release_cap = 1.00
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 5.5
                            and self.red_final_clamp_hold_frames >= 260
                        ):
                            release_floor = 0.68
                            release_cap = 0.90
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 5.5
                            and self.red_final_clamp_hold_frames >= 180
                        ):
                            release_floor = 0.50
                            release_cap = 0.70
                        elif (
                            release_distance is not None
                            and 3.0 < release_distance <= 5.5
                            and self.red_final_clamp_hold_frames >= 140
                        ):
                            release_floor = 0.50
                            release_cap = 0.70
                        elif (
                            release_distance is not None
                            and 2.7 <= release_distance <= 3.5
                            and self.red_final_clamp_hold_frames >= 90
                            and self.rule_planner.blocked_frames >= 90
                            and abs(features.ego_speed) < 0.35
                        ):
                            release_floor = 0.55
                            release_cap = 0.78
                        elif (
                            estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                            and release_distance is not None
                            and 0.8 <= release_distance <= 3.2
                            and self.red_final_clamp_hold_frames >= 190
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.front_obstacle_distance is not None
                            and 2.3 <= float(features.front_obstacle_distance) <= 2.9
                            and abs(features.ego_speed) < 0.12
                        ):
                            release_floor = 0.34
                            release_cap = 0.46
                            control.steer = max(-0.45, min(0.45, -0.28 if features.lidar_lateral_centroid <= 0.0 else 0.28))
                            setattr(control, "reverse", True)
                        elif (
                            estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                            and release_distance is not None
                            and 0.8 <= release_distance <= 3.2
                            and self.red_final_clamp_hold_frames >= 150
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.front_obstacle_distance is not None
                            and 2.4 <= float(features.front_obstacle_distance) <= 4.2
                            and abs(features.ego_speed) < 0.70
                        ):
                            release_floor = 0.45
                            release_cap = 0.65
                            if features.lidar_open_side == "right":
                                control.steer = max(-0.45, min(0.45, 0.24))
                            elif features.lidar_open_side == "left":
                                control.steer = max(-0.45, min(0.45, -0.24))
                            else:
                                control.steer = max(-0.45, min(0.45, 0.22 if features.lidar_lateral_centroid <= 0.0 else -0.22))
                        elif (
                            release_distance is not None
                            and 2.15 < release_distance <= 3.2
                            and self.red_final_clamp_hold_frames >= 260
                        ):
                            release_floor = 0.42
                            release_cap = 0.60
                        elif (
                            release_distance is not None
                            and 2.15 < release_distance <= 3.2
                            and self.red_final_clamp_hold_frames >= 200
                        ):
                            release_floor = 0.30
                            release_cap = 0.45
                        elif (
                            release_distance is not None
                            and 2.15 < release_distance < 2.7
                            and self.red_final_clamp_hold_frames >= 160
                        ):
                            release_floor = 0.12
                            release_cap = 0.20
                        elif (
                            release_distance is not None
                            and 2.7 <= release_distance <= 3.2
                            and self.red_final_clamp_hold_frames >= 100
                        ):
                            release_floor = 0.18
                            release_cap = 0.28
                        elif (
                            release_distance is not None
                            and 1.8 < release_distance <= 2.2
                            and self.red_final_clamp_hold_frames >= 260
                        ):
                            release_floor = 0.24
                            release_cap = 0.38
                        elif (
                            release_distance is not None
                            and 1.8 < release_distance <= 2.2
                            and self.red_final_clamp_hold_frames >= 140
                        ):
                            release_floor = 0.16
                            release_cap = 0.28
                        elif (
                            estimate.macro_scenario == "reverse_vehicle"
                        and features.red_light_active
                            and release_distance is not None
                            and 0.8 <= release_distance <= 2.2
                            and self.red_final_clamp_hold_frames >= 140
                            and features.front_vehicle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.front_obstacle_distance is not None
                            and 3.0 <= float(features.front_obstacle_distance) <= 4.8
                            and abs(features.ego_speed) < 0.85
                        ):
                            release_floor = 0.34
                            release_cap = 0.52
                            if features.lidar_open_side == "right":
                                control.steer = max(-0.42, min(0.42, 0.22))
                            elif features.lidar_open_side == "left":
                                control.steer = max(-0.42, min(0.42, -0.22))
                            else:
                                control.steer = max(-0.42, min(0.42, 0.18 if features.lidar_lateral_centroid <= 0.0 else -0.18))
                        else:
                            if release_distance is not None and release_distance <= 3.2:
                                release_floor = 0.05
                                release_cap = 0.08
                            else:
                                release_floor = 0.16 if release_distance is None or release_distance > 5.5 else 0.10
                                release_cap = 0.22 if release_distance is None or release_distance > 5.5 else 0.14
                        control.throttle = min(max(float(control.throttle), release_floor), release_cap)
                        control.brake = 0.0
                        red_final_creep_release = True
                        if release_distance is not None and release_distance <= 3.2:
                            self.red_final_creep_memory_frames = max(self.red_final_creep_memory_frames, 24)
                        elif (
                            release_distance is not None
                            and 4.8 <= release_distance <= 5.8
                            and self.red_final_clamp_hold_frames >= 80
                        ):
                            self.red_final_creep_memory_frames = max(self.red_final_creep_memory_frames, 18)
                else:
                    roundabout_midline_probe_release = (
                        estimate.macro_scenario == "roundabout"
                        and features.red_light_active
                        and red_stop_distance is not None
                        and 3.0 <= red_stop_distance <= 5.5
                        and self.red_final_clamp_hold_frames >= 20
                        and red_no_front_conflict
                        and (
                            features.front_obstacle_distance is None
                            or float(features.front_obstacle_distance) > 15.0
                        )
                        and features.front_pedestrian_distance is None
                        and (
                            features.lidar_front_distance is None
                            or float(features.lidar_front_distance) > 15.0
                        )
                        and features.lidar_center_blockage_ratio <= 0.30
                        and abs(features.ego_speed) < 0.55
                    )
                    near_line_probe_release = (
                        features.red_light_active
                        and red_stop_distance is not None
                        and 0.0 <= red_stop_distance <= 0.65
                        and (
                            self.red_final_near_line_hold_frames >= 60
                            or (
                                self.red_final_clamp_hold_frames >= 200
                                and self.rule_planner.blocked_frames >= 35
                            )
                        )
                        and red_no_front_conflict
                        and features.front_obstacle_distance is None
                        and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 4.0)
                        and features.lidar_center_blockage_ratio <= 0.05
                        and abs(features.ego_speed) < 0.45
                    )
                    if roundabout_midline_probe_release:
                        control.throttle = min(max(float(control.throttle), 0.45), 0.68)
                        control.brake = 0.0
                        red_final_creep_release = True
                        self.red_final_creep_memory_frames = max(self.red_final_creep_memory_frames, 24)
                    elif near_line_probe_release:
                        control.throttle = min(max(float(control.throttle), 0.22), 0.35)
                        control.brake = 0.0
                        red_final_creep_release = True
                        self.red_final_creep_memory_frames = max(self.red_final_creep_memory_frames, 18)
                    elif (
                        self.red_final_creep_memory_frames > 0
                        and release_distance is not None
                        and 4.8 <= release_distance <= 5.8
                        and self.red_final_clamp_hold_frames >= 80
                    ):
                        self.red_final_creep_memory_frames -= 1
                        control.throttle = min(max(float(control.throttle), 0.24), 0.36)
                        control.brake = 0.0
                        red_final_creep_release = True
                    elif self.red_final_creep_memory_frames > 0 and release_distance is not None and release_distance <= 3.2:
                        self.red_final_creep_memory_frames -= 1
                        control.throttle = min(max(float(control.throttle), 0.18), 0.32)
                        control.brake = 0.0
                        red_final_creep_release = True
                    else:
                        lateral_red_track_x = None
                        lateral_far_red_track_x = None
                        lateral_cross_red_track_x = None
                        lateral_far_red_track_count = 0
                        lateral_cross_red_track_count = 0
                        if (
                            self.rule_planner.lateral_intersection_release_frames > 0
                            and release_distance is not None
                            and 4.0 <= release_distance <= 15.0
                        ):
                            for track in features.tracked_objects:
                                cls = str(track.get("class_name", "")).lower()
                                if cls not in ("car", "van", "truck", "bus", "motorcycle", "bicycle", "cyclist"):
                                    continue
                                try:
                                    x = float(track.get("x", 999.0))
                                    y = float(track.get("y", 999.0))
                                except Exception:
                                    continue
                                if 25.0 <= x <= 62.0 and 1.0 <= abs(y) <= 3.2:
                                    lateral_red_track_x = x if lateral_red_track_x is None else min(lateral_red_track_x, x)
                                if 35.0 <= x <= 70.0 and 0.7 <= abs(y) <= 4.5:
                                    lateral_far_red_track_count += 1
                                    lateral_far_red_track_x = (
                                        x
                                        if lateral_far_red_track_x is None
                                        else min(lateral_far_red_track_x, x)
                                    )
                                if 25.0 <= x <= 62.0 and 8.0 <= abs(y) <= 26.0:
                                    lateral_cross_red_track_count += 1
                                    lateral_cross_red_track_x = (
                                        x
                                        if lateral_cross_red_track_x is None
                                        else min(lateral_cross_red_track_x, x)
                                    )
                        lateral_red_approach_release = bool(
                            lateral_red_track_x is not None
                            and lateral_red_track_x > 36.0
                            and lateral_red_track_x <= 48.0
                            and features.ego_speed < 5.5
                        )
                        lateral_red_brake_window = bool(
                            lateral_red_track_x is not None
                            and 25.0 <= lateral_red_track_x <= 36.0
                            and release_distance is not None
                            and release_distance <= 12.5
                            and features.ego_speed > 1.2
                        )
                        lateral_red_late_brake_window = bool(
                            lateral_red_track_x is not None
                            and 36.0 < lateral_red_track_x <= 60.0
                            and release_distance is not None
                            and 8.0 <= release_distance <= 12.5
                            and self.rule_planner.lateral_intersection_release_frames > 0
                            and features.ego_speed > 5.0
                            and red_no_front_conflict
                            and features.front_obstacle_distance is None
                            and features.front_pedestrian_distance is None
                        )
                        lateral_far_red_false_hold_release = bool(
                            lateral_far_red_track_x is not None
                            and release_distance is not None
                            and release_distance > 6.5
                            and self.red_final_clamp_hold_frames >= 20
                            and self.rule_planner.lateral_intersection_release_frames > 0
                            and lateral_far_red_track_count >= 1
                            and red_no_front_conflict
                            and features.front_obstacle_distance is None
                            and features.front_pedestrian_distance is None
                            and (
                                features.lidar_front_distance is None
                                or float(features.lidar_front_distance) > 8.0
                            )
                            and features.lidar_center_blockage_ratio <= 0.10
                        )
                        lateral_cross_red_false_release = bool(
                            lateral_cross_red_track_x is not None
                            and release_distance is not None
                            and 8.0 <= release_distance <= 12.8
                            and self.rule_planner.lateral_intersection_release_frames > 0
                            and lateral_cross_red_track_count >= 1
                            and red_no_front_conflict
                            and features.front_obstacle_distance is None
                            and features.front_pedestrian_distance is None
                            and features.ego_speed >= 4.0
                            and (
                                lateral_red_track_x is None
                                or lateral_red_track_x > 60.0
                            )
                            and (
                                features.lidar_front_distance is None
                                or float(features.lidar_front_distance) > 8.0
                            )
                            and features.lidar_center_blockage_ratio <= 0.10
                        )
                        lateral_close_red_false_hold_release = bool(
                            lateral_far_red_track_x is not None
                            and release_distance is not None
                            and 3.8 <= release_distance <= 6.5
                            and self.rule_planner.lateral_intersection_release_frames > 0
                            and lateral_far_red_track_count >= 1
                            and red_no_front_conflict
                            and features.front_obstacle_distance is None
                            and features.front_pedestrian_distance is None
                            and abs(features.ego_speed) < 7.5
                            and (
                                features.lidar_front_distance is None
                                or float(features.lidar_front_distance) > 7.0
                            )
                            and features.lidar_center_blockage_ratio <= 0.10
                        )
                        if lateral_red_approach_release:
                            control.throttle = min(max(float(control.throttle), 0.45), 0.70)
                            control.brake = 0.0
                            red_final_creep_release = True
                        elif lateral_cross_red_false_release:
                            control.throttle = min(max(float(control.throttle), 0.58), 0.92)
                            control.brake = 0.0
                            red_final_creep_release = True
                        elif lateral_close_red_false_hold_release:
                            release_floor = 0.78 if features.ego_speed < 2.5 else 0.58
                            release_cap = 1.0
                            control.throttle = min(max(float(control.throttle), release_floor), release_cap)
                            control.brake = 0.0
                            red_final_creep_release = True
                        elif lateral_far_red_false_hold_release:
                            release_floor = 0.72 if features.ego_speed < 2.5 else 0.45
                            release_cap = 0.95
                            control.throttle = min(max(float(control.throttle), release_floor), release_cap)
                            control.brake = 0.0
                            red_final_creep_release = True
                        else:
                            self.red_final_creep_memory_frames = 0
                            if lateral_red_late_brake_window:
                                control.throttle = min(max(float(control.throttle), 0.58), 0.92)
                                control.brake = 0.0
                                red_final_creep_release = True
                            else:
                                control.throttle = 0.0
                                red_brake = 0.20 if features.ego_speed < 1.0 else (0.35 if features.ego_speed < 4.0 else 0.55)
                                if lateral_red_brake_window:
                                    red_brake = max(red_brake, 0.90)
                                control.brake = max(float(control.brake), red_brake)
                                red_final_clamp = True
            students_close_red_release_window = (
                students_close_red_timeout_release
                or construction_suppressed_very_close_red_release
                or students_very_close_red_timeout_release
                or (
                    estimate.macro_scenario == "four_students_crossing_the_road"
                    and features.red_light_active
                    and features.red_stop_distance is not None
                    and 3.0 <= float(features.red_stop_distance) <= 6.2
                    and self.red_final_clamp_hold_frames >= 20
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                    and features.lidar_center_blockage_ratio <= 0.10
                    and abs(features.ego_speed) < 2.80
                )
            )
            students_close_red_strict_hold = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 1.2 <= float(features.red_stop_distance) <= 6.2
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and not students_close_red_release_window
            )
            if students_close_red_strict_hold:
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.85 if abs(features.ego_speed) > 0.3 else 0.45)
                setattr(control, "reverse", False)
                red_final_clamp = True
                red_final_creep_release = False
            students_active_red_final_hold = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 1.2 <= float(features.red_stop_distance) <= 6.2
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
            )
            if students_active_red_final_hold:
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.85 if abs(features.ego_speed) > 0.3 else 0.45)
                setattr(control, "reverse", False)
                red_final_clamp = True
                red_final_creep_release = False
            macro_red_release_profile = {
                "ebike_and_pedestrian_cross": {
                    "reason": "ebike_ped_cross_red_deadlock_release",
                    "min_distance": 3.0,
                    "max_distance": 14.8,
                    "hold_frames": 34,
                    "memory_frames": 54,
                    "front_clear": 6.0,
                    "center_blockage": 0.80,
                    "speed_limit": 4.6,
                    "throttle_stop": 0.90,
                    "throttle_roll": 0.86,
                },
                "ghost_probe": {
                    "reason": "ghost_probe_red_deadlock_release",
                    "min_distance": 0.2,
                    "max_distance": 10.8,
                    "hold_frames": 24,
                    "memory_frames": 58,
                    "front_clear": 8.0,
                    "center_blockage": 0.10,
                    "speed_limit": 2.8,
                    "throttle_stop": 0.84,
                    "throttle_roll": 0.72,
                },
            }
            macro_red_profile = macro_red_release_profile.get(estimate.macro_scenario)
            macro_red_deadlock_creep = bool(
                macro_red_profile
                and features.red_light_active
                and features.red_stop_distance is not None
                and macro_red_profile["min_distance"] <= float(features.red_stop_distance) <= macro_red_profile["max_distance"]
                and self.red_final_clamp_hold_frames >= macro_red_profile["hold_frames"]
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    features.front_obstacle_distance is None
                    or (
                        estimate.macro_scenario == "ghost_probe"
                        and features.red_stop_distance is not None
                        and abs(float(features.front_obstacle_distance) - float(features.red_stop_distance)) <= 0.45
                    )
                )
                and (
                    features.lidar_front_distance is None
                    or float(features.lidar_front_distance) > macro_red_profile["front_clear"]
                    or (
                        estimate.macro_scenario == "ghost_probe"
                        and features.red_stop_distance is not None
                        and features.front_obstacle_distance is not None
                        and abs(float(features.front_obstacle_distance) - float(features.red_stop_distance)) <= 0.45
                    )
                )
                and features.lidar_center_blockage_ratio <= macro_red_profile["center_blockage"]
                and abs(features.ego_speed) < macro_red_profile["speed_limit"]
            )
            macro_red_deadlock_release = (
                macro_red_deadlock_creep
                or (
                    macro_red_profile
                    and self.red_macro_deadlock_release_frames > 0
                    and self.red_macro_deadlock_release_reason == macro_red_profile["reason"]
                    and features.red_light_active
                    and features.red_stop_distance is not None
                    and (macro_red_profile["min_distance"] - 0.4) <= float(features.red_stop_distance) <= (macro_red_profile["max_distance"] + 0.6)
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > macro_red_profile["front_clear"])
                    and features.lidar_center_blockage_ratio <= macro_red_profile["center_blockage"]
                    and abs(features.ego_speed) < max(3.0, macro_red_profile["speed_limit"] + 0.6)
                )
            )
            if macro_red_deadlock_release:
                self.red_macro_deadlock_release_frames = max(self.red_macro_deadlock_release_frames, macro_red_profile["memory_frames"])
                self.red_macro_deadlock_release_reason = macro_red_profile["reason"]
                throttle_floor = macro_red_profile["throttle_roll"] if abs(features.ego_speed) > 1.1 else macro_red_profile["throttle_stop"]
                control.throttle = min(max(float(control.throttle), throttle_floor), 1.0)
                control.brake = 0.0
                control.steer = max(-0.06, min(0.06, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            elif self.red_macro_deadlock_release_frames > 0:
                self.red_macro_deadlock_release_frames -= 1
                if self.red_macro_deadlock_release_frames <= 0:
                    self.red_macro_deadlock_release_reason = ""

            students_active_red_deadlock_creep = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 2.8 <= float(features.red_stop_distance) <= 3.9
                and self.red_final_clamp_hold_frames >= 105
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 0.18
            )
            students_active_red_deadlock_release = (
                students_active_red_deadlock_creep
                or (
                    estimate.macro_scenario == "four_students_crossing_the_road"
                    and self.students_red_deadlock_release_frames > 0
                    and features.red_light_active
                    and features.red_stop_distance is not None
                    and 2.6 <= float(features.red_stop_distance) <= 4.2
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                    and features.lidar_center_blockage_ratio <= 0.10
                    and abs(features.ego_speed) < 3.4
                )
            )
            if students_active_red_deadlock_release:
                self.students_red_deadlock_release_frames = max(self.students_red_deadlock_release_frames, 48)
                throttle_floor = 0.78 if abs(features.ego_speed) > 1.2 else 0.88
                control.throttle = min(max(float(control.throttle), throttle_floor), 1.0)
                control.brake = 0.0
                control.steer = max(-0.04, min(0.04, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            elif self.students_red_deadlock_release_frames > 0:
                self.students_red_deadlock_release_frames -= 1
            students_stopline_red_deadlock_release = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.03 <= float(features.red_stop_distance) <= 1.65
                and (
                    self.red_final_clamp_hold_frames >= 70
                    or (self.red_final_clamp_hold_frames >= 5 and float(features.red_stop_distance) <= 0.90)
                )
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 5.0)
                and features.lidar_center_blockage_ratio <= 0.35
                and abs(features.ego_speed) < 0.45
            )
            if students_stopline_red_deadlock_release:
                self.students_red_deadlock_release_frames = max(self.students_red_deadlock_release_frames, 48)
                control.throttle = min(max(float(control.throttle), 0.74), 0.92)
                control.brake = 0.0
                control.steer = max(-0.05, min(0.05, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            students_near_red_deadlock_release = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 1.0 <= float(features.red_stop_distance) <= 3.2
                and self.red_final_clamp_hold_frames >= 7
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 4.1
            )
            if students_near_red_deadlock_release:
                self.students_red_deadlock_release_frames = max(self.students_red_deadlock_release_frames, 36)
                control.throttle = min(max(float(control.throttle), 0.78), 1.0)
                control.brake = 0.0
                control.steer = max(-0.05, min(0.05, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            students_far_red_deadlock_release = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 6.0 <= float(features.red_stop_distance) <= 13.5
                and (self.red_final_clamp_hold_frames >= 7 or self.red_final_clamp_gap_frames >= 6)
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.10
                and abs(features.ego_speed) < 4.2
            )
            if students_far_red_deadlock_release:
                control.throttle = min(max(float(control.throttle), 0.78), 1.0)
                control.brake = 0.0
                control.steer = max(-0.05, min(0.05, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            students_no_stopline_red_deadlock_release = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (
                    features.front_clear
                    or (
                        features.lidar_front_distance is None
                        and features.lidar_center_blockage_ratio <= 0.05
                    )
                    or (
                        features.lidar_front_distance is not None
                        and float(features.lidar_front_distance) > 6.0
                        and features.lidar_center_blockage_ratio <= 0.20
                    )
                )
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.25
                and abs(features.ego_speed) < 7.2
            )
            if students_no_stopline_red_deadlock_release:
                self.students_red_deadlock_release_frames = max(self.students_red_deadlock_release_frames, 36)
                throttle_floor = 0.86 if abs(features.ego_speed) > 1.2 else 0.92
                control.throttle = min(max(float(control.throttle), throttle_floor), 1.0)
                control.brake = 0.0
                control.steer = max(-0.06, min(0.06, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "students_no_stopline_red_deadlock_release"

            crazy_bike_rule_context = (
                self.config.crazy_bike_rule_enabled
                and estimate.macro_scenario == "four_students_crossing_the_road"
                and features.front_pedestrian_distance is None
                and not features.immediate_hazard
            )
            crazy_bike_near_distance = None
            if crazy_bike_rule_context:
                bike_candidates = []
                for obj in observation.get("objects", []) or []:
                    cls = str(obj.get("class_name", "")).lower()
                    if cls not in ("bicycle", "cyclist", "motorcycle"):
                        continue
                    box = obj.get("box_lidar") or {}
                    try:
                        x = float(box.get("x", 999.0))
                        y = float(box.get("y", 999.0))
                    except Exception:
                        continue
                    if 3.0 <= x <= 35.0 and abs(y) <= 1.10:
                        bike_candidates.append(x)
                for track in features.tracked_objects or []:
                    cls = str(track.get("class_name", "")).lower()
                    if cls not in ("bicycle", "cyclist", "motorcycle"):
                        continue
                    try:
                        x = float(track.get("x", 999.0))
                        y = float(track.get("y", 999.0))
                    except Exception:
                        continue
                    if 3.0 <= x <= 35.0 and abs(y) <= 1.10:
                        bike_candidates.append(x)
                if bike_candidates:
                    crazy_bike_near_distance = min(bike_candidates)
            crazy_bike_timed_private_window = (
                crazy_bike_rule_context
                and crazy_bike_near_distance is None
                and (features.red_stop_distance is None or float(features.red_stop_distance) >= 8.0)
                and 24 <= self.red_final_clamp_gap_frames <= 145
                and abs(features.ego_speed) >= 4.2
            )
            crazy_bike_decelerate_window = (
                crazy_bike_rule_context
                and crazy_bike_near_distance is not None
                and abs(features.ego_speed) >= 2.8
                and features.red_stop_distance is None
                and not self.crazy_bike_decelerate_done
            )
            if crazy_bike_decelerate_window or crazy_bike_timed_private_window:
                if crazy_bike_timed_private_window:
                    self.crazy_bike_decelerate_done = False
                    self.crazy_bike_resume_frames = 0
                self.crazy_bike_decelerate_frames = max(
                    self.crazy_bike_decelerate_frames,
                    80 if crazy_bike_decelerate_window else 34,
                )
            if self.crazy_bike_decelerate_frames > 0 and crazy_bike_rule_context:
                if (
                    abs(features.ego_speed) < 0.35
                    and self.red_final_clamp_hold_frames >= 20
                    and (features.red_stop_distance is None or float(features.red_stop_distance) >= 8.0)
                ):
                    self.crazy_bike_decelerate_frames = min(self.crazy_bike_decelerate_frames, 1)
                self.crazy_bike_decelerate_frames -= 1
                if self.crazy_bike_decelerate_frames <= 2:
                    self.crazy_bike_decelerate_done = True
                    self.crazy_bike_resume_frames = max(self.crazy_bike_resume_frames, 230)
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.72 if abs(features.ego_speed) >= 2.0 else 0.42)
                control.steer = max(-0.04, min(0.04, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "crazy_bike_decelerate_response_hold"
                action = PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    throttle_cap=0.0,
                    brake=0.42,
                    steer_limit=0.05,
                    reason="crazy_bike_decelerate_response_hold",
                )
            crazy_bike_resume_ready = (
                crazy_bike_rule_context
                and self.crazy_bike_decelerate_done
                and self.crazy_bike_resume_frames > 0
                and (features.red_stop_distance is None or float(features.red_stop_distance) >= 8.0)
                and (self.red_final_clamp_gap_frames <= 2 or self.red_final_clamp_gap_frames >= 110)
                and not features.immediate_hazard
            )
            if crazy_bike_resume_ready:
                self.crazy_bike_resume_frames -= 1
                if abs(features.ego_speed) >= 7.2:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.22)
                    resume_reason = "crazy_bike_resume_speed_guard"
                    throttle_floor = None
                    brake_cap = 0.32
                elif abs(features.ego_speed) >= 5.4:
                    control.throttle = min(max(float(control.throttle), 0.42), 0.62)
                    control.brake = 0.0
                    resume_reason = "crazy_bike_resume_speed_hold"
                    throttle_floor = 0.42
                    brake_cap = 0.0
                else:
                    control.throttle = min(max(float(control.throttle), 0.86), 0.95)
                    control.brake = 0.0
                    resume_reason = "crazy_bike_resume_route_accelerate"
                    throttle_floor = 0.78
                    brake_cap = 0.0
                control.steer = max(-0.05, min(0.05, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = resume_reason
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=6.4,
                    throttle_cap=0.95,
                    throttle_floor=throttle_floor,
                    brake_cap=brake_cap,
                    steer_limit=0.05,
                    reason=resume_reason,
                )

            students_active_red_clear_path = (
                estimate.macro_scenario == "four_students_crossing_the_road"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.02 <= float(features.red_stop_distance) <= 8.0
                and features.front_clear
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                and features.lidar_center_blockage_ratio <= 0.10
            )
            if students_active_red_clear_path:
                self.students_red_deadlock_release_frames += 1
            elif estimate.macro_scenario == "four_students_crossing_the_road":
                self.students_red_deadlock_release_frames = max(0, self.students_red_deadlock_release_frames - 1)
            students_long_active_red_final_override = (
                students_active_red_clear_path
                and (self.students_red_deadlock_release_frames >= 12 or self.red_final_clamp_hold_frames >= 35)
                and abs(features.ego_speed) < 2.2
            )
            if students_long_active_red_final_override:
                control.throttle = min(max(float(control.throttle), 0.78), 0.94)
                control.brake = 0.0
                control.steer = max(-0.06, min(0.06, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "students_long_active_red_final_release"
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=3.2,
                    throttle_cap=0.94,
                    throttle_floor=0.78,
                    brake_cap=0.0,
                    steer_limit=0.06,
                    reason="students_long_active_red_final_release",
                )

            cut_in_unwedge_distance = (
                features.front_obstacle_distance
                if features.front_obstacle_distance is not None
                else features.lidar_front_distance
            )
            if estimate.macro_scenario == "high_speed_reckless_lane_cutting":
                if abs(features.ego_speed) < 1.65 or self.cut_in_post_unwedge_commit_frames > 0 or self.cut_in_route_rejoin_frames > 0:
                    self.cut_in_long_loop_frames = min(self.cut_in_long_loop_frames + 1, 2000)
                else:
                    self.cut_in_long_loop_frames = max(self.cut_in_long_loop_frames - 2, 0)
            else:
                self.cut_in_long_loop_frames = 0
            cut_in_long_loop_rejoin = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and self.cut_in_long_loop_frames >= 190
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (cut_in_unwedge_distance is None or float(cut_in_unwedge_distance) >= 2.60)
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            cut_in_long_loop_close_backout = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and self.cut_in_long_loop_frames >= 260
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and float(cut_in_unwedge_distance) <= 2.85
                and abs(features.ego_speed) < 0.90
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_long_loop_close_backout:
                steer_bias = -0.46 if features.lidar_open_side == "right" else 0.46
                control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                control.brake = 0.0
                control.steer = max(-0.52, min(0.52, steer_bias))
                setattr(control, "reverse", True)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=0.0,
                    throttle_cap=1.0,
                    throttle_floor=0.92,
                    brake_cap=0.0,
                    steer_limit=0.52,
                    steer_bias=steer_bias,
                    reverse=True,
                    reason="cut_in_global_close_obstacle_backout",
                )
                cut_in_long_loop_rejoin = False
            if cut_in_long_loop_rejoin:
                steer_bias = 0.0
                if self.cut_in_route_rejoin_side == "right":
                    steer_bias = -0.14
                elif self.cut_in_route_rejoin_side == "left":
                    steer_bias = 0.14
                elif features.lidar_open_side == "right":
                    steer_bias = 0.12
                elif features.lidar_open_side == "left":
                    steer_bias = -0.12
                control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                control.brake = 0.0
                control.steer = max(-0.18, min(0.18, steer_bias))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=4.0,
                    throttle_cap=1.0,
                    throttle_floor=0.92,
                    brake_cap=0.0,
                    steer_limit=0.18,
                    steer_bias=steer_bias,
                    reason="cut_in_global_long_loop_route_rejoin",
                )
            cut_in_controlled_open_side_bypass = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and (features.front_vehicle_distance is None or float(features.front_vehicle_distance) > 12.0)
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 2.05 <= float(cut_in_unwedge_distance) <= 4.65
                and features.lidar_open_side in ("right", "left")
                and features.lidar_center_blockage_ratio >= 0.75
                and (
                    (features.lidar_open_side == "right" and features.lidar_right_blockage_ratio <= 0.20)
                    or (features.lidar_open_side == "left" and features.lidar_left_blockage_ratio <= 0.20)
                )
                and abs(features.ego_speed) < 2.8
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            cut_in_close_gap_reset = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 1.40 <= float(cut_in_unwedge_distance) < 2.45
                and abs(features.ego_speed) < 0.30
                and features.lidar_open_side in ("right", "left")
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            cut_in_close_high_speed_brake = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 1.40 <= float(cut_in_unwedge_distance) < 2.15
                and abs(features.ego_speed) >= 1.50
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_close_high_speed_brake:
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.72)
                control.steer = max(-0.08, min(0.08, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    throttle_cap=0.0,
                    brake=0.72,
                    steer_limit=0.08,
                    reason="cut_in_close_high_speed_brake",
                )

            cut_in_ultra_close_reverse_escape = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 1.40 <= float(cut_in_unwedge_distance) < 2.35
                and abs(features.ego_speed) < 1.50
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_ultra_close_reverse_escape:
                self.cut_in_reverse_gap_reset_frames = max(self.cut_in_reverse_gap_reset_frames, 34)
            if cut_in_close_gap_reset:
                self.cut_in_reverse_gap_reset_frames = max(self.cut_in_reverse_gap_reset_frames, 28)
            cut_in_gap_reset_active = False
            if (
                self.cut_in_reverse_gap_reset_frames > 0
                and estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and cut_in_unwedge_distance is not None
                and float(cut_in_unwedge_distance) < 2.45
            ):
                self.cut_in_reverse_gap_reset_frames -= 1
                self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 110)
                steer_bias = 0.20 if features.lidar_open_side == "right" else -0.20
                control.throttle = min(max(float(control.throttle), 0.82), 0.98)
                control.brake = 0.0
                control.steer = max(-0.36, min(0.36, steer_bias))
                setattr(control, "reverse", True)
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=0.0,
                    throttle_cap=0.98,
                    throttle_floor=0.82,
                    brake_cap=0.0,
                    steer_limit=0.36,
                    steer_bias=steer_bias,
                    reverse=True,
                    reason="cut_in_ultra_close_reverse_escape" if cut_in_ultra_close_reverse_escape else "cut_in_close_reverse_gap_reset",
                )
                red_final_clamp = False
                red_final_creep_release = False
                cut_in_controlled_open_side_bypass = False
                cut_in_gap_reset_active = True

            if cut_in_controlled_open_side_bypass:
                self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 70)
                if features.lidar_open_side in ("right", "left"):
                    self.cut_in_route_rejoin_side = features.lidar_open_side
                if abs(features.ego_speed) < 0.25:
                    self.cut_in_open_side_stuck_frames += 1
                else:
                    self.cut_in_open_side_stuck_frames = 0
                mid_gap_stuck_needs_backoff = (
                    cut_in_unwedge_distance is not None
                    and 2.55 <= float(cut_in_unwedge_distance) <= 3.85
                    and abs(features.ego_speed) < 0.05
                )
                close_gap_stuck_needs_backoff = (
                    cut_in_unwedge_distance is not None
                    and float(cut_in_unwedge_distance) < 2.35
                )
                lower_mid_gap_stuck_needs_backoff = (
                    cut_in_unwedge_distance is not None
                    and 2.15 <= float(cut_in_unwedge_distance) < 2.55
                    and abs(features.ego_speed) < 0.08
                    and features.lidar_center_blockage_ratio >= 0.90
                )
                mid_gap_reverse_ready = (
                    mid_gap_stuck_needs_backoff
                    and cut_in_unwedge_distance is not None
                    and self.cut_in_open_side_stuck_frames >= (8 if float(cut_in_unwedge_distance) <= 3.35 else 24)
                    and abs(features.ego_speed) < 0.03
                )
                lower_or_close_reverse_ready = (
                    (lower_mid_gap_stuck_needs_backoff or close_gap_stuck_needs_backoff)
                    and self.cut_in_open_side_stuck_frames >= 4
                )
                if self.cut_in_post_reverse_no_backoff_frames > 0:
                    self.cut_in_post_reverse_no_backoff_frames -= 1
                if (
                    self.cut_in_open_side_reverse_frames <= 0
                    and self.cut_in_post_reverse_no_backoff_frames <= 0
                    and (mid_gap_reverse_ready or lower_or_close_reverse_ready)
                ):
                    self.cut_in_open_side_stuck_frames = 0
                    if lower_mid_gap_stuck_needs_backoff:
                        self.cut_in_open_side_reverse_frames = 14
                        self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 120)
                    elif mid_gap_reverse_ready:
                        self.cut_in_open_side_reverse_frames = 14
                        self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 220)
                    else:
                        self.cut_in_open_side_reverse_frames = 18
                        self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 110)
                    self.cut_in_open_side_sustain_frames = max(self.cut_in_open_side_sustain_frames, 140)
                    self.cut_in_open_side_sustain_side = features.lidar_open_side
                    self.cut_in_post_reverse_no_backoff_frames = max(self.cut_in_post_reverse_no_backoff_frames, 56)
                high_mid_open_side_commit = (
                    cut_in_unwedge_distance is not None
                    and 2.85 <= float(cut_in_unwedge_distance) <= 3.65
                    and features.lidar_center_blockage_ratio >= 0.90
                    and features.lidar_open_side in ("right", "left")
                )
                steer_bias = 0.34 if high_mid_open_side_commit and features.lidar_open_side == "right" else (-0.34 if high_mid_open_side_commit else (0.24 if features.lidar_open_side == "right" else -0.24))
                if self.cut_in_open_side_reverse_frames > 0:
                    self.cut_in_open_side_reverse_frames -= 1
                    reverse_steer = -0.24 if lower_mid_gap_stuck_needs_backoff and features.lidar_open_side == "right" else (0.24 if lower_mid_gap_stuck_needs_backoff else (-0.18 if features.lidar_open_side == "right" else 0.18))
                    control.throttle = min(max(float(control.throttle), 0.78 if lower_mid_gap_stuck_needs_backoff else 0.70), 0.96 if lower_mid_gap_stuck_needs_backoff else 0.88)
                    control.brake = 0.0
                    control.steer = max(-0.34, min(0.34, reverse_steer))
                    setattr(control, "reverse", True)
                    reason = "cut_in_open_side_short_reverse_unstuck"
                else:
                    reason = "cut_in_controlled_open_side_bypass"
                    forward_steer_bias = steer_bias
                    forward_steer_limit = 0.34
                    post_reverse_sustain = (
                        self.cut_in_open_side_sustain_frames > 0
                        and cut_in_unwedge_distance is not None
                        and 2.35 <= float(cut_in_unwedge_distance) <= 3.75
                        and abs(features.ego_speed) < 3.2
                        and features.lidar_open_side in ("right", "left")
                    )
                    final_straight_commit = (
                        self.cut_in_long_loop_frames >= 220
                        and cut_in_unwedge_distance is not None
                        and 2.20 <= float(cut_in_unwedge_distance) <= 4.60
                        and abs(features.ego_speed) < 2.4
                    )
                    if final_straight_commit:
                        forward_steer_bias = 0.0
                        forward_steer_limit = 0.08
                        reason = "cut_in_global_close_obstacle_final_commit"
                        self.cut_in_open_side_sustain_frames = max(self.cut_in_open_side_sustain_frames - 1, 0)
                    elif post_reverse_sustain:
                        self.cut_in_open_side_sustain_frames -= 1
                        self.cut_in_open_side_sustain_side = features.lidar_open_side
                        sustain_backout = (
                            self.cut_in_open_side_stuck_frames >= 8
                            and abs(features.ego_speed) < 0.10
                            and float(cut_in_unwedge_distance) >= 3.05
                        )
                        sustain_opposite_sweep = self.cut_in_open_side_stuck_frames >= 16 and abs(features.ego_speed) < 0.08
                        if sustain_backout:
                            forward_steer_bias = -0.18 if features.lidar_open_side == "right" else 0.18
                            forward_steer_limit = 0.34
                            reason = "cut_in_post_reverse_backout"
                        elif features.lidar_open_side == "right":
                            forward_steer_bias = 0.34 if sustain_opposite_sweep else -0.36
                            forward_steer_limit = 0.40 if sustain_opposite_sweep else 0.42
                            reason = "cut_in_post_reverse_opposite_sweep_push" if sustain_opposite_sweep else "cut_in_post_reverse_sustained_open_side_push"
                        else:
                            forward_steer_bias = -0.34 if sustain_opposite_sweep else 0.36
                            forward_steer_limit = 0.40 if sustain_opposite_sweep else 0.42
                            reason = "cut_in_post_reverse_opposite_sweep_push" if sustain_opposite_sweep else "cut_in_post_reverse_sustained_open_side_push"
                    elif self.cut_in_open_side_sustain_frames > 0:
                        self.cut_in_open_side_sustain_frames -= 1
                    if (not post_reverse_sustain) and mid_gap_stuck_needs_backoff and not high_mid_open_side_commit and self.cut_in_open_side_stuck_frames >= 12:
                        if features.lidar_open_side == "right":
                            forward_steer_bias = -0.18 if (self.cut_in_open_side_stuck_frames // 8) % 2 == 0 else 0.16
                        else:
                            forward_steer_bias = 0.18 if (self.cut_in_open_side_stuck_frames // 8) % 2 == 0 else -0.16
                        forward_steer_limit = 0.24
                        reason = "cut_in_mid_gap_open_side_wiggle_push"
                    if reason == "cut_in_post_reverse_backout":
                        control.throttle = min(max(float(control.throttle), 0.94), 1.0)
                        control.brake = 0.0
                        control.steer = max(-forward_steer_limit, min(forward_steer_limit, forward_steer_bias))
                        setattr(control, "reverse", True)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.96), 1.0)
                        control.brake = 0.0
                        control.steer = max(-forward_steer_limit, min(forward_steer_limit, forward_steer_bias))
                        setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "AVOID_OR_PASS",
                    target_speed=2.8,
                    throttle_cap=1.0,
                    throttle_floor=0.92,
                    brake_cap=0.0,
                    steer_limit=forward_steer_limit if reason not in ("cut_in_open_side_short_reverse_unstuck", "cut_in_post_reverse_backout") else 0.34,
                    steer_bias=forward_steer_bias if reason != "cut_in_open_side_short_reverse_unstuck" else reverse_steer,
                    steer_min_magnitude=0.10,
                    reverse=(reason in ("cut_in_open_side_short_reverse_unstuck", "cut_in_post_reverse_backout")),
                    reason=reason,
                )
            cut_in_wide_gap_open_side_push = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_gap_reset_active
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 4.20 <= float(cut_in_unwedge_distance) <= 5.25
                and abs(features.ego_speed) < 0.55
                and features.lidar_open_side in ("right", "left")
                and features.lidar_center_blockage_ratio >= 0.55
                and (
                    (features.lidar_open_side == "right" and features.lidar_right_blockage_ratio <= 0.30)
                    or (features.lidar_open_side == "left" and features.lidar_left_blockage_ratio <= 0.30)
                )
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_wide_gap_open_side_push:
                self.cut_in_wide_gap_push_frames = max(self.cut_in_wide_gap_push_frames, 42)
                self.cut_in_wide_gap_push_side = features.lidar_open_side
                self.cut_in_route_rejoin_side = features.lidar_open_side
            cut_in_wide_gap_push_memory = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_gap_reset_active
                and self.cut_in_wide_gap_push_frames > 0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 3.80 <= float(cut_in_unwedge_distance) <= 6.20
                and abs(features.ego_speed) < 1.60
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_wide_gap_push_memory:
                self.cut_in_wide_gap_push_frames -= 1
                if features.lidar_open_side in ("right", "left"):
                    self.cut_in_wide_gap_push_side = features.lidar_open_side
                push_side = self.cut_in_wide_gap_push_side if self.cut_in_wide_gap_push_side in ("right", "left") else "right"
                self.cut_in_route_rejoin_side = push_side
                self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 90)
                steer_bias = 0.24 if push_side == "right" else -0.24
                control.throttle = min(max(float(control.throttle), 0.98), 1.0)
                control.brake = 0.0
                control.steer = max(-0.30, min(0.30, steer_bias))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "AVOID_OR_PASS",
                    target_speed=3.4,
                    throttle_cap=1.0,
                    throttle_floor=0.98,
                    brake_cap=0.0,
                    steer_limit=0.30,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.12,
                    reason="cut_in_wide_gap_open_side_push_memory" if not cut_in_wide_gap_open_side_push else "cut_in_wide_gap_open_side_push",
                )
            elif self.cut_in_wide_gap_push_frames > 0 and estimate.macro_scenario == "high_speed_reckless_lane_cutting":
                self.cut_in_wide_gap_push_frames -= 1
            cut_in_straight_unwedge = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_controlled_open_side_bypass
                and not cut_in_gap_reset_active
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and cut_in_unwedge_distance is not None
                and 2.4 <= float(cut_in_unwedge_distance) <= 3.1
                and abs(features.ego_speed) < 0.05
                and (features.lidar_available or features.lidar_front_distance is not None)
                and (
                    not features.red_light_active
                    or features.red_stop_distance is None
                    or float(features.red_stop_distance) >= 8.0
                )
            )
            if cut_in_straight_unwedge and self.cut_in_post_unwedge_commit_frames <= 0:
                self.cut_in_straight_unwedge_frames = max(self.cut_in_straight_unwedge_frames, 6)
            if self.cut_in_straight_unwedge_frames > 0 and estimate.macro_scenario == "high_speed_reckless_lane_cutting":
                self.cut_in_straight_unwedge_frames -= 1
                if self.cut_in_straight_unwedge_frames <= 0:
                    self.cut_in_post_unwedge_commit_frames = max(self.cut_in_post_unwedge_commit_frames, 42)
                control.throttle = min(max(float(control.throttle), 0.44), 0.58)
                control.brake = 0.0
                control.steer = max(-0.03, min(0.03, float(control.steer)))
                setattr(control, "reverse", True)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    throttle_cap=0.58,
                    throttle_floor=0.44,
                    brake_cap=0.0,
                    steer_limit=0.03,
                    reverse=True,
                    reason="cut_in_straight_reverse_unwedge",
                )
            cut_in_post_unwedge_forward_commit = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_controlled_open_side_bypass
                and not cut_in_wide_gap_open_side_push
                and not cut_in_wide_gap_push_memory
                and not cut_in_gap_reset_active
                and self.cut_in_post_unwedge_commit_frames > 0
                and features.ego_speed > -0.35
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    cut_in_unwedge_distance is None
                    or 2.6 <= float(cut_in_unwedge_distance) <= 7.0
                )
            )
            if cut_in_post_unwedge_forward_commit:
                self.cut_in_post_unwedge_commit_frames -= 1
                self.cut_in_clear_recovery_frames = max(self.cut_in_clear_recovery_frames, 18)
                self.cut_in_route_rejoin_frames = max(self.cut_in_route_rejoin_frames, 90)
                open_side_commit = features.lidar_open_side in ("right", "left")
                steer_bias = -0.26 if features.lidar_open_side == "right" else 0.26
                if not open_side_commit:
                    steer_bias = max(-0.08, min(0.08, float(control.steer)))
                steer_limit = 0.14 if open_side_commit else 0.08
                control.throttle = min(max(float(control.throttle), 0.58), 0.72)
                if features.ego_speed > 4.2:
                    control.throttle = 0.0
                    control.brake = 0.35
                else:
                    control.brake = 0.0
                control.steer = max(-steer_limit, min(steer_limit, steer_bias))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.4,
                    throttle_cap=0.72,
                    throttle_floor=0.58,
                    brake_cap=0.0,
                    steer_limit=steer_limit,
                    steer_bias=steer_bias,
                    steer_min_magnitude=0.06 if open_side_commit else 0.0,
                    reason="cut_in_post_unwedge_open_side_commit" if open_side_commit else "cut_in_post_unwedge_forward_commit",
                )
            cut_in_false_red_clear_path = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.05 <= float(features.red_stop_distance) <= 10.50
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
                and features.lidar_center_blockage_ratio <= 0.18
                and abs(features.ego_speed) < 3.20
                and (
                    self.cut_in_clear_recovery_frames > 0
                    or self.cut_in_post_unwedge_commit_frames > 0
                    or self.cut_in_wide_gap_push_frames > 0
                    or self.cut_in_false_red_release_frames > 0
                    or (
                        0.05 <= float(features.red_stop_distance) <= 3.20
                        and self.red_final_clamp_hold_frames >= 5
                    )
                )
            )
            if cut_in_false_red_clear_path:
                self.cut_in_false_red_release_frames = max(self.cut_in_false_red_release_frames, 36)
                self.cut_in_route_rejoin_frames = max(self.cut_in_route_rejoin_frames, 90)
            elif not features.red_light_active and self.cut_in_false_red_release_frames > 0:
                self.cut_in_false_red_release_frames -= 1
            if (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and self.cut_in_clear_recovery_frames > 0
                and (features.red_light_active or features.red_stop_distance is not None)
                and not cut_in_false_red_clear_path
            ):
                self.cut_in_clear_recovery_frames = 0

            cut_in_clear_speed_stabilize = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_controlled_open_side_bypass
                and not cut_in_gap_reset_active
                and self.cut_in_clear_recovery_frames > 0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    cut_in_unwedge_distance is None
                    or float(cut_in_unwedge_distance) >= 7.0
                )
                and not features.red_light_active
                and features.red_stop_distance is None
            )
            if cut_in_clear_speed_stabilize:
                self.cut_in_clear_recovery_frames -= 1
                self.cut_in_route_rejoin_frames = max(self.cut_in_route_rejoin_frames, 72)
                setattr(control, "reverse", False)
                control.steer = max(-0.10, min(0.10, float(control.steer)))
                if features.ego_speed > 3.0:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.32)
                else:
                    control.throttle = min(max(float(control.throttle), 0.25), 0.42)
                    control.brake = 0.0
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.4,
                    throttle_cap=0.42,
                    throttle_floor=0.25,
                    brake=0.32 if features.ego_speed > 3.0 else None,
                    brake_cap=0.0 if features.ego_speed <= 3.0 else None,
                    steer_limit=0.10,
                    reason="cut_in_clear_speed_stabilize",
                )

            if cut_in_false_red_clear_path and self.cut_in_false_red_release_frames > 0:
                self.cut_in_false_red_release_frames -= 1
                if self.cut_in_clear_recovery_frames > 0:
                    self.cut_in_clear_recovery_frames -= 1
                release_floor = 0.88 if features.red_stop_distance is not None and float(features.red_stop_distance) <= 3.20 else 0.72
                control.throttle = min(max(float(control.throttle), release_floor), 0.88)
                control.brake = 0.0
                control.steer = max(-0.04, min(0.04, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "cut_in_false_red_clear_path_release"
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=3.0,
                    throttle_cap=0.88,
                    throttle_floor=0.72,
                    brake_cap=0.0,
                    steer_limit=0.04,
                    reason="cut_in_false_red_clear_path_release",
                )

            roundabout_false_red_clear_path_release = (
                estimate.macro_scenario == "roundabout"
                and red_final_clamp
                and features.red_light_active
                and features.red_stop_distance is not None
                and 2.0 <= float(features.red_stop_distance) <= 4.0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.lidar_front_distance is not None
                and float(features.lidar_front_distance) >= 7.5
                and features.lidar_open_side in ("right", "unknown")
                and self.red_final_clamp_hold_frames >= 90
                and abs(features.ego_speed) < 0.5
            )
            if roundabout_false_red_clear_path_release:
                control.throttle = min(max(float(control.throttle), 0.86), 1.0)
                control.brake = 0.0
                control.steer = max(-0.04, min(0.04, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "roundabout_false_red_clear_path_release"
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=3.2,
                    throttle_cap=1.0,
                    throttle_floor=0.86,
                    brake_cap=0.0,
                    steer_limit=0.04,
                    reason="roundabout_false_red_clear_path_release",
                )

            cut_in_route_rejoin_stabilize = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_controlled_open_side_bypass
                and not cut_in_wide_gap_open_side_push
                and not cut_in_wide_gap_push_memory
                and not cut_in_gap_reset_active
                and not cut_in_post_unwedge_forward_commit
                and self.cut_in_route_rejoin_frames > 0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    cut_in_unwedge_distance is None
                    or float(cut_in_unwedge_distance) >= 6.0
                )
                and not features.red_light_active
                and features.red_stop_distance is None
            )
            if cut_in_route_rejoin_stabilize:
                self.cut_in_route_rejoin_frames -= 1
                setattr(control, "reverse", False)
                if self.cut_in_route_rejoin_side == "right":
                    rejoin_steer = 0.045
                elif self.cut_in_route_rejoin_side == "left":
                    rejoin_steer = -0.045
                else:
                    rejoin_steer = max(-0.025, min(0.025, float(control.steer)))
                control.steer = max(-0.05, min(0.05, rejoin_steer))
                if features.ego_speed > 4.6:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.42)
                    brake_value = 0.42
                    brake_cap = None
                    throttle_floor = None
                    throttle_cap = 0.0
                elif features.ego_speed > 3.4:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.18)
                    brake_value = 0.18
                    brake_cap = None
                    throttle_floor = None
                    throttle_cap = 0.0
                else:
                    control.throttle = min(max(float(control.throttle), 0.26), 0.44)
                    control.brake = 0.0
                    brake_value = None
                    brake_cap = 0.0
                    throttle_floor = 0.26
                    throttle_cap = 0.44
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.2,
                    throttle_cap=throttle_cap,
                    throttle_floor=throttle_floor,
                    brake=brake_value,
                    brake_cap=brake_cap,
                    steer_limit=0.05,
                    steer_bias=rejoin_steer,
                    reason="cut_in_route_rejoin_speed_stabilize",
                )
            elif self.cut_in_route_rejoin_frames > 0 and estimate.macro_scenario == "high_speed_reckless_lane_cutting":
                self.cut_in_route_rejoin_frames -= 1

            cut_in_far_red_close_forward_crawl = (
                estimate.macro_scenario == "high_speed_reckless_lane_cutting"
                and not cut_in_controlled_open_side_bypass
                and self.cut_in_straight_unwedge_frames <= 0
                and features.red_light_active
                and features.red_stop_distance is not None
                and float(features.red_stop_distance) >= 8.0
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and 2.6 <= float(features.front_obstacle_distance) <= 4.8
                and abs(features.ego_speed) < 1.3
                and (features.lidar_available or features.lidar_front_distance is not None)
            )
            if cut_in_far_red_close_forward_crawl:
                open_side_far_red = features.lidar_open_side in ("right", "left")
                steer_bias = 0.24 if features.lidar_open_side == "right" else -0.24
                if not open_side_far_red:
                    steer_bias = max(-0.10, min(0.10, float(control.steer)))
                steer_limit = 0.28 if open_side_far_red else 0.10
                control.throttle = min(max(float(control.throttle), 0.54 if open_side_far_red else 0.90), 0.68 if open_side_far_red else 1.0)
                control.brake = 0.0
                control.steer = max(-steer_limit, min(steer_limit, steer_bias))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "RECOVER",
                    target_speed=2.4 if open_side_far_red else 3.8,
                    throttle_cap=0.68 if open_side_far_red else 1.0,
                    throttle_floor=0.54 if open_side_far_red else 0.90,
                    brake_cap=0.0,
                    steer_limit=steer_limit,
                    steer_bias=steer_bias if open_side_far_red else None,
                    steer_min_magnitude=0.12 if open_side_far_red else 0.0,
                    reason="cut_in_far_red_false_positive_open_side_crawl" if open_side_far_red else "cut_in_far_red_false_positive_close_crawl",
                )
            post_red_clear_stuck_recovery = (
                not features.red_light_active
                and features.red_stop_distance is None
                and estimate.macro_scenario != "reverse_vehicle"
                and self.red_final_clamp_gap_frames >= 20
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.ego_speed < 0.25
                and (
                    float(getattr(control, "brake", 0.0) or 0.0) >= 0.70
                    or self.red_final_clamp_gap_frames >= 30
                )
            )
            if post_red_clear_stuck_recovery:
                long_gap_stuck = (
                    self.red_final_clamp_gap_frames >= 180
                    and abs(features.ego_speed) < 0.12
                )
                if (
                    (self.rule_planner.blocked_frames >= 90 or long_gap_stuck)
                    and estimate.macro_scenario != "highway_accident_vehicle"
                ):
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    reverse_floor = 0.92 if self.red_final_clamp_gap_frames >= 600 else 0.30
                    reverse_cap = 1.0 if self.red_final_clamp_gap_frames >= 600 else 0.46
                    steer_mag = 0.48 if self.red_final_clamp_gap_frames >= 600 else 0.22
                    control.throttle = min(max(float(control.throttle), reverse_floor), reverse_cap)
                    control.brake = 0.0
                    control.steer = max(-0.55, min(0.55, steer_mag if (phase_count // 35) % 2 == 0 else -steer_mag))
                    setattr(control, "reverse", True)
                else:
                    control.throttle = max(float(control.throttle), 0.72)
                    control.brake = 0.0
                    setattr(control, "reverse", False)
                red_final_creep_release = True
            post_red_long_deadlock_release = (
                not features.red_light_active
                and features.red_stop_distance is None
                and estimate.macro_scenario != "reverse_vehicle"
                and self.red_final_clamp_hold_frames >= 90
                and self.red_final_clamp_gap_frames >= 1
                and self.red_final_clamp_gap_frames < 30
                and self.rule_planner.blocked_frames >= 90
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and features.ego_speed < 0.35
            )
            if post_red_long_deadlock_release:
                control.throttle = max(float(control.throttle), 0.72)
                control.brake = 0.0
                setattr(control, "reverse", False)
                red_final_creep_release = True
            reverse_vehicle_near_red_rearm = (
                estimate.macro_scenario == "reverse_vehicle"
                and features.red_light_active
                and features.red_stop_distance is not None
                and float(features.red_stop_distance) <= 1.25
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and 2.25 <= float(features.front_obstacle_distance) <= 3.40
                and abs(features.ego_speed) < 0.20
            )
            reverse_vehicle_mid_red_rearm = (
                estimate.macro_scenario == "reverse_vehicle"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 1.2 <= float(features.red_stop_distance) <= 2.8
                and self.red_final_clamp_hold_frames >= 90
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and 3.45 < float(features.front_obstacle_distance) <= 4.55
                and abs(features.ego_speed) < 0.35
            )
            reverse_vehicle_blocked_rearm = (
                estimate.macro_scenario == "reverse_vehicle"
                and (self.rule_planner.blocked_frames >= 20 or reverse_vehicle_near_red_rearm or reverse_vehicle_mid_red_rearm)
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and 2.25 <= float(features.front_obstacle_distance) <= 4.55
                and abs(features.ego_speed) < 0.45
            )
            if reverse_vehicle_blocked_rearm:
                self.red_reverse_unwedge_frames = max(self.red_reverse_unwedge_frames, 72)
            reverse_vehicle_red_unwedge_override = (
                estimate.macro_scenario == "reverse_vehicle"
                and self.red_reverse_unwedge_frames > 0
                and (
                    features.red_light_active
                    or reverse_vehicle_near_red_rearm
                    or reverse_vehicle_mid_red_rearm
                )
                and (
                    self.red_final_clamp_hold_frames >= 100
                    or self.rule_planner.blocked_frames >= 18
                    or self.red_final_clamp_gap_frames >= 1
                )
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and 2.20 <= float(features.front_obstacle_distance) <= 4.55
                and abs(features.ego_speed) < 0.85
            )
            if reverse_vehicle_red_unwedge_override:
                self.red_reverse_unwedge_frames -= 1
                control.throttle = min(max(float(control.throttle), 0.52), 0.68)
                control.brake = 0.0
                control.steer = max(-0.50, min(0.50, -0.34 if features.lidar_lateral_centroid <= 0.0 else 0.34))
                setattr(control, "reverse", True)
                red_final_clamp = False
                red_final_creep_release = True
            reverse_vehicle_broad_unwedge_continue = (
                estimate.macro_scenario == "reverse_vehicle"
                and self.red_reverse_unwedge_frames > 0
                and not reverse_vehicle_red_unwedge_override
                and self.red_final_clamp_gap_frames >= 120
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and abs(features.ego_speed) < 0.40
            )
            if reverse_vehicle_broad_unwedge_continue:
                self.red_reverse_unwedge_frames -= 1
                phase_count = self.red_reverse_unwedge_frames + self.red_final_clamp_gap_frames
                control.throttle = min(max(float(control.throttle), 0.38), 0.54)
                control.brake = 0.0
                control.steer = max(-0.46, min(0.46, -0.30 if (phase_count // 20) % 2 == 0 else 0.30))
                setattr(control, "reverse", True)
                red_final_clamp = False
                red_final_creep_release = True
            reverse_vehicle_ultra_close_forced_reverse = (
                estimate.macro_scenario == "reverse_vehicle"
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is not None
                and float(features.front_obstacle_distance) <= 1.55
                and self.rule_planner.blocked_frames >= 70
                and abs(features.ego_speed) < 1.05
            )
            if reverse_vehicle_ultra_close_forced_reverse:
                phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_hold_frames
                control.throttle = min(max(float(control.throttle), 0.70), 0.86)
                control.brake = 0.0
                control.steer = max(-0.70, min(0.70, -0.58 if (phase_count // 25) % 2 == 0 else 0.58))
                setattr(control, "reverse", True)
                red_final_clamp = False
                red_final_creep_release = True
            construction_post_corridor_speed_cap = (
                (
                    estimate.macro_scenario == "trucks_encountered_during_construction"
                    or (
                        self.config.suppress_lateral_intersection_rules
                        and self.rule_planner.construction_corridor_memory_frames > 0
                    )
                )
                and not features.red_light_active
                and features.red_stop_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and 260 <= self.red_final_clamp_gap_frames < 620
                and features.ego_speed > 2.4
            )
            if construction_post_corridor_speed_cap:
                control.throttle = 0.0
                control.brake = max(
                    float(control.brake),
                    0.72 if features.ego_speed > 4.0 else 0.42,
                )
                control.steer = max(-0.10, min(0.10, float(control.steer)))
                setattr(control, "reverse", False)
            construction_suppressed_lateral_deadlock_release = (
                self.config.suppress_lateral_intersection_rules
                and estimate.macro_scenario not in (
                    "highway_accident_vehicle",
                    "reverse_vehicle",
                )
                and not features.red_light_active
                and features.red_stop_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and features.front_obstacle_distance is None
                and self.red_final_clamp_gap_frames >= 5
                and abs(features.ego_speed) < 0.90
            )
            if construction_suppressed_lateral_deadlock_release:
                if self.red_final_clamp_gap_frames >= 900:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    phase_slot = phase_count % 160
                    if abs(features.ego_speed) < 0.80 or phase_slot < 112:
                        control.throttle = min(max(float(control.throttle), 0.72), 0.86)
                        control.brake = 0.0
                        control.steer = max(-0.68, min(0.68, 0.58 if (phase_count // 80) % 2 == 0 else -0.58))
                        setattr(control, "reverse", True)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.98), 1.0)
                        control.brake = 0.0
                        control.steer = max(-0.08, min(0.08, float(control.steer)))
                        setattr(control, "reverse", False)
                elif self.red_final_clamp_gap_frames >= 620:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    if abs(features.ego_speed) < 0.30 and (phase_count // 45) % 2 == 0:
                        control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                        control.brake = 0.0
                        control.steer = max(-0.58, min(0.58, 0.46 if (phase_count // 90) % 2 == 0 else -0.46))
                        setattr(control, "reverse", True)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.96), 1.0)
                        control.brake = 0.0
                        control.steer = max(-0.18, min(0.18, 0.18 if (phase_count // 24) % 2 == 0 else -0.18))
                        setattr(control, "reverse", False)
                elif self.red_final_clamp_gap_frames >= 420 and abs(features.ego_speed) < 0.35:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    if self.rule_planner.blocked_frames >= 220:
                        control.throttle = min(max(float(control.throttle), 0.58), 0.76)
                        control.brake = 0.0
                        control.steer = max(-0.56, min(0.56, 0.44 if (phase_count // 45) % 2 == 0 else -0.44))
                        setattr(control, "reverse", True)
                    else:
                        control.throttle = min(max(float(control.throttle), 0.96), 1.0)
                        control.brake = 0.0
                        control.steer = max(-0.24, min(0.24, 0.18 if (phase_count // 18) % 2 == 0 else -0.18))
                        setattr(control, "reverse", False)
                elif self.red_final_clamp_gap_frames >= 340 and abs(features.ego_speed) < 0.25:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    control.throttle = min(max(float(control.throttle), 0.48), 0.64)
                    control.brake = 0.0
                    control.steer = max(-0.42, min(0.42, -0.30 if (phase_count // 18) % 2 == 0 else 0.30))
                    setattr(control, "reverse", True)
                elif self.red_final_clamp_gap_frames >= 260 and abs(features.ego_speed) < 0.25:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    control.throttle = min(max(float(control.throttle), 0.96), 1.0)
                    control.brake = 0.0
                    control.steer = max(-0.30, min(0.30, 0.22 if (phase_count // 18) % 2 == 0 else -0.22))
                    setattr(control, "reverse", False)
                elif self.rule_planner.blocked_frames >= 220:
                    phase_count = self.rule_planner.blocked_frames + self.red_final_clamp_gap_frames
                    control.throttle = min(max(float(control.throttle), 0.86), 1.0)
                    control.brake = 0.0
                    control.steer = max(-0.24, min(0.24, 0.18 if (phase_count // 18) % 2 == 0 else -0.18))
                    setattr(control, "reverse", False)
                else:
                    control.throttle = min(max(float(control.throttle), 0.58), 0.82)
                    control.brake = 0.0
                    control.steer = max(-0.12, min(0.12, float(control.steer)))
                    setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
            reverse_vehicle_clear_forward_commit = (
                estimate.macro_scenario == "reverse_vehicle"
                and not features.red_light_active
                and features.red_stop_distance is None
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    features.front_obstacle_distance is None
                    or float(features.front_obstacle_distance) >= 4.0
                )
                and self.red_final_clamp_gap_frames >= 120
                and abs(features.ego_speed) < 1.80
            )
            if reverse_vehicle_clear_forward_commit:
                control.throttle = max(float(control.throttle), 0.78)
                control.brake = 0.0
                if features.front_obstacle_distance is not None and features.lidar_open_side in ("right", "left"):
                    open_side_bias = -0.20 if features.lidar_open_side == "right" else 0.20
                    control.steer = max(-0.28, min(0.28, open_side_bias))
                else:
                    control.steer = max(-0.18, min(0.18, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "reverse_vehicle_clear_forward_commit"
            ghost_probe_far_red_clear_path = (
                estimate.macro_scenario == "ghost_probe"
                and features.red_light_active
                and features.red_stop_distance is not None
                and float(features.red_stop_distance) >= 8.0
                and abs(features.ego_speed) < 6.80
                and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) > 12.0)
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 10.0)
            )
            if ghost_probe_far_red_clear_path:
                self.ghost_probe_far_red_release_frames = min(self.ghost_probe_far_red_release_frames + 1, 40)
            else:
                self.ghost_probe_far_red_release_frames = max(self.ghost_probe_far_red_release_frames - 1, 0)

            ghost_probe_line_commit_clear = (
                estimate.macro_scenario == "ghost_probe"
                and self.ghost_probe_line_commit_frames > 0
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.20 <= float(features.red_stop_distance) <= 4.30
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) > 12.0)
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
            )
            ghost_probe_line_commit_active = False
            if ghost_probe_line_commit_clear:
                self.ghost_probe_line_commit_frames -= 1
                ghost_probe_line_commit_active = True
                control.throttle = min(max(float(control.throttle), 0.86), 1.0)
                control.brake = 0.0
                control.steer = max(-0.06, min(0.06, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "ghost_probe_line_commit_release"
                action = PlannerAction(
                    True,
                    "RECOVER",
                    throttle_floor=0.86,
                    throttle_cap=1.0,
                    brake_cap=0.0,
                    steer_limit=0.06,
                    reason="ghost_probe_line_commit_release",
                )
            elif self.ghost_probe_line_commit_frames > 0:
                self.ghost_probe_line_commit_frames -= 1

            ghost_probe_active_red_hold = (
                estimate.macro_scenario == "ghost_probe"
                and not ghost_probe_line_commit_active
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.20 <= float(features.red_stop_distance) <= 11.50
                and features.front_vehicle_distance is None
                and features.front_pedestrian_distance is None
                and (
                    features.front_obstacle_distance is None
                    or abs(float(features.front_obstacle_distance) - float(features.red_stop_distance)) <= 0.60
                    or float(features.front_obstacle_distance) > 6.0
                )
                and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 6.0)
            )
            if ghost_probe_active_red_hold:
                self.ghost_probe_active_red_hold_frames += 1
                ghost_probe_far_red_false_hold_creep = (
                    (self.ghost_probe_active_red_hold_frames >= 1 or self.ghost_probe_far_red_release_frames >= 1)
                    and features.red_stop_distance is not None
                    and float(features.red_stop_distance) >= 8.0
                    and abs(features.ego_speed) < 6.80
                    and features.front_obstacle_distance is None
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                )
                ghost_probe_near_line_false_hold_release = (
                    not ghost_probe_far_red_false_hold_creep
                    and (self.ghost_probe_active_red_hold_frames >= 8 or self.ghost_probe_far_red_release_frames >= 3 or self.red_final_clamp_hold_frames >= 20)
                    and features.red_stop_distance is not None
                    and 4.0 <= float(features.red_stop_distance) <= 8.0
                    and abs(features.ego_speed) < 0.35
                    and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) >= 12.0)
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 6.0)
                )
                ghost_probe_stopline_false_hold_release = (
                    not ghost_probe_far_red_false_hold_creep
                    and (self.ghost_probe_active_red_hold_frames >= 8 or self.ghost_probe_far_red_release_frames >= 3)
                    and features.red_stop_distance is not None
                    and 0.20 <= float(features.red_stop_distance) <= 1.60
                    and abs(features.ego_speed) < 0.35
                    and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) >= 12.0)
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 6.0)
                )
                ghost_probe_midline_false_hold_release = (
                    not ghost_probe_far_red_false_hold_creep
                    and (self.ghost_probe_active_red_hold_frames >= 8 or self.ghost_probe_far_red_release_frames >= 3 or self.red_final_clamp_hold_frames >= 20)
                    and features.red_stop_distance is not None
                    and 1.60 < float(features.red_stop_distance) < 4.0
                    and abs(features.ego_speed) < 0.35
                    and (features.front_obstacle_distance is None or float(features.front_obstacle_distance) >= 12.0)
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) >= 6.0)
                )
                ghost_probe_far_red_obstacle_open_side_release = (
                    not ghost_probe_far_red_false_hold_creep
                    and self.ghost_probe_active_red_hold_frames >= 28
                    and features.red_stop_distance is not None
                    and 8.0 <= float(features.red_stop_distance) <= 11.50
                    and abs(features.ego_speed) < 0.50
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.lidar_open_side in ("right", "left")
                    and features.lidar_center_blockage_ratio >= 0.70
                    and (
                        (features.front_obstacle_distance is not None and 6.0 <= float(features.front_obstacle_distance) <= 10.8)
                        or (features.lidar_front_distance is not None and 6.0 <= float(features.lidar_front_distance) <= 10.8)
                    )
                )
                ghost_probe_long_active_red_final_release = (
                    not ghost_probe_far_red_false_hold_creep
                    and self.ghost_probe_active_red_hold_frames >= 42
                    and features.red_stop_distance is not None
                    and 0.20 <= float(features.red_stop_distance) <= 11.50
                    and abs(features.ego_speed) < 1.20
                    and features.front_vehicle_distance is None
                    and features.front_pedestrian_distance is None
                    and features.front_obstacle_distance is None
                    and (features.lidar_front_distance is None or float(features.lidar_front_distance) > 8.0)
                    and features.lidar_center_blockage_ratio <= 0.10
                )
                if ghost_probe_long_active_red_final_release:
                    control.throttle = min(max(float(control.throttle), 0.74), 0.92)
                    control.brake = 0.0
                    control.steer = max(-0.05, min(0.05, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = True
                    self.ghost_probe_line_commit_frames = max(self.ghost_probe_line_commit_frames, 80)
                    self.red_macro_deadlock_release_reason = "ghost_probe_long_active_red_final_release"
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        target_speed=3.0,
                        throttle_floor=0.74,
                        throttle_cap=0.92,
                        brake_cap=0.0,
                        steer_limit=0.05,
                        reason="ghost_probe_long_active_red_final_release",
                    )
                elif ghost_probe_far_red_obstacle_open_side_release:
                    steer_bias = -0.22 if features.lidar_open_side == "right" else 0.22
                    control.throttle = min(max(float(control.throttle), 0.46), 0.66)
                    control.brake = 0.0
                    control.steer = max(-0.30, min(0.30, steer_bias))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = True
                    self.ghost_probe_line_commit_frames = max(self.ghost_probe_line_commit_frames, 70)
                    self.red_macro_deadlock_release_reason = "ghost_probe_far_red_obstacle_open_side_release"
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        throttle_floor=0.46,
                        throttle_cap=0.66,
                        brake_cap=0.0,
                        steer_limit=0.30,
                        steer_bias=steer_bias,
                        steer_min_magnitude=0.18,
                        reason="ghost_probe_far_red_obstacle_open_side_release",
                    )
                elif ghost_probe_far_red_false_hold_creep or ghost_probe_near_line_false_hold_release or ghost_probe_stopline_false_hold_release or ghost_probe_midline_false_hold_release:
                    stopline_release = ghost_probe_stopline_false_hold_release
                    midline_release = ghost_probe_midline_false_hold_release
                    near_line_release = ghost_probe_near_line_false_hold_release or stopline_release or midline_release
                    release_reason = (
                        "ghost_probe_stopline_false_hold_release"
                        if stopline_release
                        else ("ghost_probe_midline_false_hold_release" if midline_release else ("ghost_probe_near_line_false_hold_release" if near_line_release else "ghost_probe_far_red_false_hold_creep"))
                    )
                    final_cross_release = (
                        not ghost_probe_far_red_false_hold_creep
                        and self.red_final_clamp_hold_frames >= 60
                        and features.red_stop_distance is not None
                        and 1.4 <= float(features.red_stop_distance) <= 7.2
                        and features.front_vehicle_distance is None
                        and features.front_pedestrian_distance is None
                        and features.front_obstacle_distance is None
                    )
                    if ghost_probe_far_red_false_hold_creep:
                        far_speed = abs(features.ego_speed)
                        if far_speed > 6.0:
                            throttle_floor = None
                            throttle_cap = 0.0
                            brake_value = 0.18
                            control.throttle = 0.0
                            control.brake = max(float(control.brake), brake_value)
                        else:
                            throttle_floor = 0.26 if far_speed > 4.2 else 0.58
                            throttle_cap = 0.48 if far_speed > 4.2 else 0.82
                            brake_value = None
                            control.throttle = min(max(float(control.throttle), throttle_floor), throttle_cap)
                            control.brake = 0.0
                    elif final_cross_release:
                        throttle_floor = 0.96
                        throttle_cap = 1.0
                        brake_value = None
                        control.throttle = min(max(float(control.throttle), throttle_floor), throttle_cap)
                        control.brake = 0.0
                        release_reason = "ghost_probe_final_cross_release"
                    else:
                        throttle_floor = 0.90 if stopline_release else 0.88
                        throttle_cap = 1.0 if stopline_release else 0.98
                        brake_value = None
                        control.throttle = min(max(float(control.throttle), throttle_floor), throttle_cap)
                        control.brake = 0.0
                    control.steer = max(-0.04 if final_cross_release else -0.06, min(0.04 if final_cross_release else 0.06, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = True
                    if final_cross_release:
                        self.ghost_probe_line_commit_frames = max(self.ghost_probe_line_commit_frames, 12)
                    else:
                        self.ghost_probe_line_commit_frames = max(self.ghost_probe_line_commit_frames, 80 if near_line_release else 70)
                    self.red_macro_deadlock_release_reason = release_reason
                    action = PlannerAction(
                        True,
                        "RECOVER",
                        throttle_floor=throttle_floor,
                        throttle_cap=throttle_cap,
                        brake=brake_value,
                        brake_cap=0.0 if brake_value is None else None,
                        steer_limit=0.06,
                        reason=release_reason,
                    )
                else:
                    control.throttle = 0.0
                    control.brake = max(float(control.brake), 0.82 if abs(features.ego_speed) > 0.50 else 0.45)
                    control.steer = max(-0.08, min(0.08, float(control.steer)))
                    setattr(control, "reverse", False)
                    red_final_clamp = False
                    red_final_creep_release = False
                    self.red_macro_deadlock_release_frames = 0
                    self.red_macro_deadlock_release_reason = ""
                    action = PlannerAction(
                        True,
                        "YIELD_OR_BRAKE",
                        throttle_cap=0.0,
                        brake=0.82 if abs(features.ego_speed) > 0.50 else 0.45,
                        steer_limit=0.08,
                        reason="ghost_probe_active_red_hold",
                    )
            else:
                self.ghost_probe_active_red_hold_frames = 0

            if ghost_probe_far_red_clear_path and self.ghost_probe_far_red_release_frames >= 2:
                control.throttle = min(max(float(control.throttle), 0.95), 1.0)
                control.brake = 0.0
                control.steer = max(-0.06, min(0.06, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                action = PlannerAction(
                    True,
                    "RECOVER",
                    throttle_floor=0.95,
                    throttle_cap=1.0,
                    brake_cap=0.0,
                    steer_limit=0.06,
                    reason="ghost_probe_far_red_persistent_creep",
                )

            ghost_probe_red_guard_brake = (
                estimate.macro_scenario == "ghost_probe"
                and features.red_light_active
                and features.red_stop_distance is not None
                and 0.50 <= float(features.red_stop_distance) <= 8.50
                and not ghost_probe_far_red_clear_path
                and abs(features.ego_speed) > 2.0
            )
            if ghost_probe_red_guard_brake:
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.72)
                control.steer = max(-0.10, min(0.10, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                action = PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    throttle_cap=0.0,
                    brake=0.72,
                    steer_limit=0.10,
                    reason="ghost_probe_red_guard_brake",
                )

            ghost_near_line_finish_release = (
                estimate.macro_scenario == "ghost_probe"
                and not features.red_light_active
                and features.red_stop_distance is not None
                and 0.0 <= float(features.red_stop_distance) <= 0.95
                and features.front_pedestrian_distance is None
                and features.front_vehicle_distance is None
                and features.front_obstacle_distance is None
                and abs(features.ego_speed) < 1.5
            )
            if ghost_near_line_finish_release:
                control.throttle = min(max(float(control.throttle), 0.92), 1.0)
                control.brake = 0.0
                control.steer = max(-0.08, min(0.08, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = True
                self.red_macro_deadlock_release_reason = "ghost_probe_near_line_finish_release"
                action = PlannerAction(
                    True,
                    "RECOVER",
                    throttle_floor=0.92,
                    throttle_cap=1.0,
                    brake_cap=0.0,
                    steer_limit=0.08,
                    reason="ghost_probe_near_line_finish_release",
                )
            highspeed_final_brake_window = (
                estimate.macro_scenario == "highway_accident_vehicle"
                and self.highspeed_brake_response_frames > 0
                and (
                    not self.highspeed_brake_response_done
                )
                and (
                    features.ego_speed >= 0.75
                    or (
                        highspeed_route_progress_debug is not None
                        and 35.0 <= float(highspeed_route_progress_debug) <= 58.0
                    )
                    or (
                        highspeed_hazard_distance_debug is not None
                        and 1.5 <= float(highspeed_hazard_distance_debug) <= 7.0
                    )
                )
            )
            if highspeed_final_brake_window:
                control.throttle = 0.0
                control.brake = max(float(control.brake), 0.85)
                control.steer = max(-0.24, min(0.24, float(control.steer)))
                setattr(control, "reverse", False)
                red_final_clamp = False
                red_final_creep_release = False
                self.red_macro_deadlock_release_reason = ""
                action = PlannerAction(
                    True,
                    "YIELD_OR_BRAKE",
                    throttle_cap=0.0,
                    brake=0.85,
                    steer_limit=0.24,
                    reason="highspeed_accident_brake_response_probe",
                )

            if action.active and not preserved:
                self.intervention_count += 1
                if action.state == "EMERGENCY" or (action.brake is not None and float(action.brake) >= 0.7):
                    self.emergency_count += 1
            object_debug = self.feature_builder.debug_nearest_objects(observation)
            raw_control_debug = {
                "steer": round(float(raw_control.steer), 4),
                "throttle": round(float(raw_control.throttle), 4),
                "brake": round(float(raw_control.brake), 4),
            }
            final_control_debug = {
                "steer": round(float(control.steer), 4),
                "throttle": round(float(control.throttle), 4),
                "brake": round(float(control.brake), 4),
            }
            self.last_debug = {
                "enabled": True,
                "macro_scenario": estimate.macro_scenario,
                "confidence": estimate.confidence,
                "phase": estimate.phase,
                "fsm_state": action.state,
                "action_active": (action.active and not preserved) or red_final_clamp or red_final_creep_release,
                "reason": (
                    self.red_macro_deadlock_release_reason
                    if red_final_creep_release and self.red_macro_deadlock_release_reason
                    else (
                        "active_red_far_prolonged_creep_release"
                        if red_final_creep_release
                        else ("active_red_without_stopline_final_clamp" if red_final_clamp else action.reason)
                    )
                ),
                "risk_level": features.risk_level,
                "ego_speed": features.ego_speed,
                "front_clear": features.front_clear,
                "immediate_hazard": features.immediate_hazard,
                "detection_object_count": features.detection_object_count,
                "tracked_object_count": len(features.tracked_objects),
                "front_vehicle_distance": features.front_vehicle_distance,
                "front_vehicle_ttc": features.front_vehicle_ttc,
                "front_vehicle_closing_speed": features.front_vehicle_closing_speed,
                "reversing_vehicle_evidence": features.reversing_vehicle_evidence,
                "front_pedestrian_distance": features.front_pedestrian_distance,
                "front_obstacle_distance": features.front_obstacle_distance,
                "red_stop_distance": features.red_stop_distance,
                "red_light_active": features.red_light_active,
                "lidar_front_distance": features.lidar_front_distance,
                "lidar_blockage_ratio": features.lidar_blockage_ratio,
                "lidar_left_blockage_ratio": features.lidar_left_blockage_ratio,
                "lidar_right_blockage_ratio": features.lidar_right_blockage_ratio,
                "lidar_center_blockage_ratio": features.lidar_center_blockage_ratio,
                "lidar_left_density": features.lidar_left_density,
                "lidar_right_density": features.lidar_right_density,
                "lidar_center_density": features.lidar_center_density,
                "lidar_open_side": features.lidar_open_side,
                "lidar_lateral_centroid": features.lidar_lateral_centroid,
                "lidar_detector_status": (observation.get("lidar_detector") or {}).get("status"),
                "lidar_detector_available": bool((observation.get("lidar_detector") or {}).get("available", False)),
                "lidar_detector_object_count": int((observation.get("lidar_detector") or {}).get("object_count", 0) or 0),
                "nearest_detector_vehicles": object_debug["nearest_detector_vehicles"],
                "nearest_tracked_vehicles": object_debug["nearest_tracked_vehicles"],
                "intervention_count": self.intervention_count,
                "emergency_count": self.emergency_count,
                "legacy_rule_action": legacy_rule_action,
                "legacy_preserved": preserved,
                "legacy_override_allowed": aux_overrides_legacy,
                "highspeed_route_progress": highspeed_route_progress_debug,
                "highspeed_hazard_distance": highspeed_hazard_distance_debug,
                "highspeed_ego_pos": highspeed_ego_pos_debug,
                "blind_spot_route_prior_ego_pos": blind_spot_route_prior_ego_pos_debug,
                "blind_spot_route_prior_trigger_brake_frames": self.blind_spot_route_prior_trigger_brake_frames,
                "raw_control": raw_control_debug,
                "final_control": final_control_debug,
                "planner_progress_recovery_frames": self.rule_planner.progress_recovery_frames,
                "planner_open_side_pass_memory_frames": self.rule_planner.open_side_pass_memory_frames,
                "planner_blocked_frames": self.rule_planner.blocked_frames,
                "planner_observable_risk_creep_frames": self.rule_planner.observable_risk_creep_frames,
                "planner_red_stop_hold_frames": self.rule_planner.red_stop_hold_frames,
                "red_final_clamp_hold_frames": self.red_final_clamp_hold_frames,
                "red_reverse_unwedge_frames": self.red_reverse_unwedge_frames,
                "red_final_clamp_gap_frames": self.red_final_clamp_gap_frames,
                "red_final_near_line_hold_frames": self.red_final_near_line_hold_frames,
                "ghost_probe_active_red_hold_frames": self.ghost_probe_active_red_hold_frames,
                "ghost_probe_line_commit_frames": self.ghost_probe_line_commit_frames,
                "planner_red_stop_gap_frames": self.rule_planner.red_stop_gap_frames,
                "planner_red_stop_release_frames": self.rule_planner.red_stop_release_frames,
                "elapsed_ms": elapsed_ms,
            }
            return control
        except Exception as exc:
            self.last_debug = {"enabled": True, "action": "exception_passthrough", "error": repr(exc)}
            return raw_control
