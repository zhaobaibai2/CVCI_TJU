import os
import re
import xml.etree.ElementTree as ET


def _snake(text):
    text = (text or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def _macro_from_class(value):
    aliases = {
        "missing car": "missing_car",
        "high speed temporary construction": "high_speed_temporary_construction",
        "drive into the roundabout": "roundabout",
    }
    key = (value or "").strip().lower()
    return aliases.get(key, _snake(value))


class ScenarioClassifier:
    def __init__(self):
        self.route_xml = os.environ.get("ROUTES", "")
        self.allow_route_prior = os.environ.get("CVCI_ALLOW_ROUTE_PRIOR", "0").lower() in ("1", "true", "yes", "on")
        self.forced_macro = os.environ.get("CVCI_FORCE_MACRO_SCENARIO", "").strip()
        self.route_macro = "unknown"
        self.confidence = 0.0
        self.scenario_names = []
        self.route_ids = []
        self._load_route_xml()

    def _load_route_xml(self):
        if not self.allow_route_prior:
            return
        if self.forced_macro:
            self.route_macro = self.forced_macro
            self.confidence = 1.0
            return
        if not self.route_xml or not os.path.exists(self.route_xml):
            return
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        try:
            root = ET.parse(self.route_xml, parser=parser).getroot()
        except Exception:
            return
        current_class = ""
        macros = set()
        for child in list(root):
            if child.tag is ET.Comment:
                text = (child.text or "").strip()
                m = re.search(r"scenario_class\s+\d+\s*:\s*([^=]+)", text, flags=re.I)
                if m:
                    current_class = m.group(1).strip()
                continue
            if child.tag != "route":
                continue
            self.route_ids.append(child.attrib.get("id", ""))
            scenario = child.find("./scenarios/scenario")
            if scenario is not None:
                self.scenario_names.append(scenario.attrib.get("name", ""))
            macro = _macro_from_class(current_class)
            if macro != "unknown":
                macros.add(macro)
        if len(macros) == 1:
            self.route_macro = next(iter(macros))
            self.confidence = 1.0

    def classify(self, detection_context, tick_data, frame_idx):
        speed = float(tick_data.get("speed", 0.0))
        risk_flags = {
            "front_vehicle": detection_context.get("front_vehicle_distance") is not None,
            "front_pedestrian": detection_context.get("front_pedestrian_distance") is not None,
            "front_obstacle": detection_context.get("front_obstacle_distance") is not None,
            "side_risk": bool(detection_context.get("side_risk", False)),
            "immediate_hazard": bool(detection_context.get("immediate_hazard", False)),
            "front_clear": bool(detection_context.get("front_clear", True)),
            "left_clear": bool(detection_context.get("left_clear", True)),
            "right_clear": bool(detection_context.get("right_clear", True)),
        }
        phase = "approach"
        if risk_flags["immediate_hazard"] or not risk_flags["front_clear"]:
            phase = "interaction"
        elif speed < 0.8:
            phase = "recovery"
        return {
            "macro_scenario": self.route_macro,
            "confidence": self.confidence,
            "phase": phase,
            "risk_flags": risk_flags,
            "route_id": ",".join(self.route_ids[:3]) if len(self.route_ids) <= 3 else "",
            "scenario_name": ",".join(sorted(set(self.scenario_names))[:3]),
        }
