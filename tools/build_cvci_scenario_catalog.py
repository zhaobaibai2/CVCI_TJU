#!/usr/bin/env python3
"""Build an offline CVCI scenario catalog from route XML and scenario code."""
import argparse
import ast
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

SCENARIO_LABELS = {
    "missing car": "missing_car",
    "car encountered during construction": "trucks_encountered_during_construction",
    "highway accident vehicle": "highway_accident_vehicle",
    "children crossing the road": "four_students_crossing_the_road",
    "reverse vehicle": "reverse_vehicle",
    "drive into the roundabout": "roundabout",
    "high speed reckless lane cutting": "high_speed_reckless_lane_cutting",
    "blind spot hidden car": "blind_spot_hidden_car",
    "high speed temporary construction": "high_speed_temporary_construction",
}

WEATHER_NAMES = ["weather_0", "weather_1", "weather_2", "weather_3"]

RULE_HINTS = {
    "trucks_encountered_during_construction": (
        "static/dynamic construction blockage",
        "APPROACH -> PREPARE -> YIELD_OR_BRAKE -> AVOID_OR_PASS -> RECOVER",
    ),
    "highway_accident_vehicle": (
        "highway stopped/accident vehicle ahead",
        "APPROACH -> PREPARE -> YIELD_OR_BRAKE -> AVOID_OR_PASS -> RECOVER",
    ),
    "four_students_crossing_the_road": (
        "pedestrians crossing from road edge",
        "APPROACH -> YIELD_OR_BRAKE -> RECOVER",
    ),
    "reverse_vehicle": (
        "oncoming or reversing vehicle intrudes into ego lane",
        "APPROACH -> YIELD_OR_BRAKE -> AVOID_OR_PASS -> RECOVER",
    ),
    "roundabout": (
        "roundabout entering/yield topology",
        "APPROACH -> PREPARE -> YIELD_OR_BRAKE -> AVOID_OR_PASS -> RECOVER",
    ),
    "high_speed_reckless_lane_cutting": (
        "fast adjacent vehicle cut-in",
        "NORMAL -> PREPARE -> YIELD_OR_BRAKE -> RECOVER",
    ),
    "blind_spot_hidden_car": (
        "occluded crossing or hidden vehicle",
        "APPROACH -> PREPARE -> YIELD_OR_BRAKE -> RECOVER",
    ),
}


def snake(text):
    text = (text or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def macro_from_comment(text):
    m = re.search(r"scenario_class\s+\d+\s*:\s*([^=]+)", text or "", flags=re.I)
    if not m:
        return None, None
    raw = (m.group(1).strip() if m else text or "unknown").strip().lower()
    return SCENARIO_LABELS.get(raw, snake(raw)), raw


def scenario_classes(scenario_dir):
    out = {}
    if not scenario_dir or not Path(scenario_dir).is_dir():
        return out
    for path in sorted(Path(scenario_dir).glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out.setdefault(node.name.lower(), {"class": node.name, "file": str(path)})
        out.setdefault(path.stem.lower(), {"class": path.stem, "file": str(path)})
    return out


def child_value(elem):
    if elem is None:
        return None
    if elem.attrib:
        return dict(elem.attrib)
    text = (elem.text or "").strip()
    return text or None


def parse_routes(xml_path, class_index):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(xml_path, parser=parser).getroot()
    current_macro = "unknown"
    current_raw = "unknown"
    catalog = OrderedDict()
    for child in list(root):
        if child.tag is ET.Comment:
            macro, raw = macro_from_comment(child.text or "")
            if macro is None:
                continue
            current_macro, current_raw = macro, raw
            catalog.setdefault(
                current_macro,
                OrderedDict(
                    [
                        ("user_score_name", current_macro),
                        ("scenario_class_comment", current_raw),
                        ("xml_scenario_types", OrderedDict()),
                        ("python_scenario_classes", OrderedDict()),
                        ("routes", []),
                        ("actors", OrderedDict()),
                        ("trigger_parameters", OrderedDict()),
                        ("dynamic_parameters", OrderedDict()),
                    ]
                ),
            )
            continue
        if child.tag != "route":
            continue

        route_id = str(child.attrib.get("id", ""))
        try:
            rid_int = int(route_id)
        except ValueError:
            rid_int = 0
        local_idx = rid_int % 12
        scenario = child.find("./scenarios/scenario")
        scenario_type = scenario.attrib.get("type", "unknown") if scenario is not None else "unknown"
        scenario_name = scenario.attrib.get("name", "unknown") if scenario is not None else "unknown"
        item = catalog[current_macro]
        item["xml_scenario_types"].setdefault(scenario_type, 0)
        item["xml_scenario_types"][scenario_type] += 1
        class_info = class_index.get(scenario_type.lower()) or class_index.get(snake(scenario_type)) or {}
        if class_info:
            item["python_scenario_classes"].setdefault(class_info["class"], class_info["file"])
        route_info = OrderedDict(
            [
                ("id", route_id),
                ("town", child.attrib.get("town", "")),
                ("difficulty_level", local_idx // 4 + 1),
                ("weather", WEATHER_NAMES[local_idx % 4]),
                ("scenario_name", scenario_name),
                ("scenario_type", scenario_type),
            ]
        )
        if scenario is not None:
            for sub in list(scenario):
                if sub.tag is ET.Comment:
                    continue
                val = child_value(sub)
                if sub.tag == "trigger_point":
                    route_info["trigger_point"] = val
                    item["trigger_parameters"].setdefault("trigger_point", val)
                elif sub.tag in ("other_actor", "actor", "vehicle", "walker") or "actor" in sub.tag:
                    model = sub.attrib.get("model", sub.tag)
                    item["actors"].setdefault(model, 0)
                    item["actors"][model] += 1
                else:
                    item["dynamic_parameters"].setdefault(sub.tag, val)
        item["routes"].append(route_info)
    return catalog


def finalize(catalog):
    for macro, item in catalog.items():
        risk, fsm = RULE_HINTS.get(
            macro,
            (
                "scenario-specific dynamic obstacle or topology conflict",
                "NORMAL -> APPROACH -> PREPARE -> YIELD_OR_BRAKE -> AVOID_OR_PASS -> RECOVER -> EMERGENCY",
            ),
        )
        if "pedestrian" in macro or "student" in macro or "children" in item.get("scenario_class_comment", ""):
            sensor = "front/side camera pedestrian detections, LiDAR small moving clusters, decreasing TTC"
        elif "roundabout" in macro:
            sensor = "route curvature, junction topology, side vehicle detections, low-speed merge gap"
        elif "construction" in macro or "truck" in macro:
            sensor = "front LiDAR corridor blockage, cones/barriers/trucks, lane narrowing geometry"
        else:
            sensor = "front/side object detections, LiDAR corridor occupancy, route command/topology, ego speed trend"
        item["expected_sensor_features"] = sensor
        item["main_risk_type"] = risk
        item["recommended_rule_state_machine"] = fsm
        item["route_ids"] = [r["id"] for r in item["routes"]]
        item["towns"] = sorted(set(r["town"] for r in item["routes"] if r.get("town")))
    return catalog


def yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + s + '"'


def yaml_dump(value, indent=0):
    pad = " " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(yaml_dump(v, indent + 2))
            else:
                lines.append(f"{pad}{k}: {yaml_scalar(v)}")
    elif isinstance(value, list):
        for v in value:
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(yaml_dump(v, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(v)}")
    return lines


def write_markdown(catalog, path, xml_path, scenario_dir):
    lines = [
        "# CVCI Scenario Catalog",
        "",
        "This catalog is generated for offline analysis, rule design, and test selection. It is not runtime route truth for the agent.",
        "",
        f"- XML source: `{xml_path}`",
        f"- Scenario source: `{scenario_dir}`",
        f"- Scenario families: {len(catalog)}",
        "",
        "| Family | XML types | Python classes | Routes | Towns | Main risk | FSM hint |",
        "|---|---|---|---:|---|---|---|",
    ]
    for macro, item in catalog.items():
        xml_types = ", ".join(item["xml_scenario_types"].keys()) or "--"
        py_classes = ", ".join(item["python_scenario_classes"].keys()) or "--"
        towns = ", ".join(item["towns"]) or "--"
        lines.append(
            f"| {macro} | {xml_types} | {py_classes} | {len(item['routes'])} | {towns} | "
            f"{item['main_risk_type']} | {item['recommended_rule_state_machine']} |"
        )
    for macro, item in catalog.items():
        lines += [
            "",
            f"## {macro}",
            "",
            f"- User score name: `{item['user_score_name']}`",
            f"- XML scenario types: {', '.join(item['xml_scenario_types'].keys()) or '--'}",
            f"- Python scenario classes: {', '.join(item['python_scenario_classes'].keys()) or '--'}",
            f"- Towns: {', '.join(item['towns']) or '--'}",
            f"- Expected sensor features: {item['expected_sensor_features']}",
            f"- Main risk type: {item['main_risk_type']}",
            f"- Suggested FSM: {item['recommended_rule_state_machine']}",
            "",
            "| Route | Town | Difficulty | Weather | Scenario name | Scenario type |",
            "|---:|---|---:|---|---|---|",
        ]
        for r in item["routes"]:
            lines.append(
                f"| {r['id']} | {r['town']} | {r['difficulty_level']} | {r['weather']} | "
                f"{r['scenario_name']} | {r['scenario_type']} |"
            )
        if item["actors"]:
            lines += ["", "Actors observed in XML:"]
            for model, count in item["actors"].items():
                lines.append(f"- `{model}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark/scenario_runner/srunner/data/CVCI_BenchMark.xml")
    ap.add_argument("--scenario-dir", default="/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark/scenario_runner/srunner/scenarios")
    ap.add_argument("--yaml-out", default="configs/cvci_scenario_catalog.yaml")
    ap.add_argument("--md-out", default="reports/cvci_scenario_catalog.md")
    args = ap.parse_args()
    class_index = scenario_classes(args.scenario_dir)
    catalog = finalize(parse_routes(args.xml, class_index))
    wrapped = OrderedDict(
        [
            (
                "metadata",
                OrderedDict(
                    [
                        ("xml_source", args.xml),
                        ("scenario_source", args.scenario_dir),
                        ("runtime_use", "offline_analysis_only_do_not_use_as_agent_truth"),
                        ("num_families", len(catalog)),
                    ]
                ),
            ),
            ("scenario_families", catalog),
        ]
    )
    yaml_path = Path(args.yaml_out)
    md_path = Path(args.md_out)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(yaml_dump(wrapped)) + "\n", encoding="utf-8")
    write_markdown(catalog, md_path, args.xml, args.scenario_dir)
    print(f"wrote {yaml_path} and {md_path} ({len(catalog)} families)")


if __name__ == "__main__":
    main()
