#!/usr/bin/env python3
import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


WEATHER_KEYS = [
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "fog_density",
    "wetness",
    "sun_altitude_angle",
    "sun_azimuth_angle",
]


def fvalue(attrs, key, default=None):
    if key not in attrs:
        return default
    try:
        return float(attrs[key])
    except (TypeError, ValueError):
        return attrs[key]


def snake(text):
    text = (text or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def weather_bucket(weather):
    precip = float(weather.get("precipitation") or 0)
    wet = float(weather.get("wetness") or 0)
    fog = float(weather.get("fog_density") or 0)
    if precip >= 30 or wet >= 40:
        return "rain_or_wet"
    if fog >= 30:
        return "fog"
    return "clear"


def light_bucket(weather):
    sun = float(weather.get("sun_altitude_angle") or 90)
    return "night" if sun < 5 else "day"


def macro_from(scenario_class, scenario_name, scenario_type):
    source = scenario_class or scenario_name or scenario_type or "unknown"
    aliases = {
        "missing car": "missing_car",
        "high speed temporary construction": "high_speed_temporary_construction",
        "drive into the roundabout": "roundabout",
        "roundabout": "roundabout",
    }
    key = source.strip().lower()
    return aliases.get(key, snake(source))


def parse_xml(xml_path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(xml_path, parser=parser).getroot()
    rows = []
    scenario_class = ""
    difficulty_level = ""
    for child in list(root):
        if child.tag is ET.Comment:
            text = (child.text or "").strip()
            m = re.search(r"scenario_class\s+\d+\s*:\s*([^=]+)", text, flags=re.I)
            if m:
                scenario_class = m.group(1).strip()
                difficulty_level = ""
            m = re.search(r"\blevel\s*-\s*(\d+)\b", text, flags=re.I)
            if m:
                difficulty_level = f"level-{m.group(1)}"
            continue
        if child.tag != "route":
            continue
        route = child
        route_id = route.attrib.get("id", "")
        town = route.attrib.get("town", "")
        waypoints = route.find("waypoints")
        waypoint_count = len(list(waypoints)) if waypoints is not None else 0

        scenario = route.find("./scenarios/scenario")
        scenario_name = scenario.attrib.get("name", "") if scenario is not None else ""
        scenario_type = scenario.attrib.get("type", "") if scenario is not None else ""
        trigger = scenario.find("trigger_point") if scenario is not None else None
        trigger_attrs = trigger.attrib if trigger is not None else {}

        params = {}
        actors = []
        if scenario is not None:
            for node in list(scenario):
                if node.tag == "trigger_point":
                    continue
                if node.tag == "other_actor":
                    actors.append(dict(node.attrib))
                    continue
                if node.tag is ET.Comment:
                    continue
                if "value" in node.attrib:
                    params[node.tag] = node.attrib["value"]
                elif node.attrib:
                    params[node.tag] = dict(node.attrib)

        weather_node = route.find("./weathers/weather")
        weather = {k: fvalue(weather_node.attrib, k, 0.0) for k in WEATHER_KEYS} if weather_node is not None else {k: 0.0 for k in WEATHER_KEYS}

        actor_models = [a.get("model", "") for a in actors]
        actor_prefix = Counter(m.split(".")[0] if "." in m else m for m in actor_models)
        row = {
            "route_id": route_id,
            "town": town,
            "scenario_class": scenario_class,
            "difficulty_level": difficulty_level,
            "scenario_name": scenario_name,
            "scenario_type": scenario_type,
            "trigger_point_x": trigger_attrs.get("x", ""),
            "trigger_point_y": trigger_attrs.get("y", ""),
            "trigger_point_z": trigger_attrs.get("z", ""),
            "trigger_point_yaw": trigger_attrs.get("yaw", ""),
            "waypoint_count": waypoint_count,
            "cloudiness": weather["cloudiness"],
            "precipitation": weather["precipitation"],
            "precipitation_deposits": weather["precipitation_deposits"],
            "wind_intensity": weather["wind_intensity"],
            "fog_density": weather["fog_density"],
            "wetness": weather["wetness"],
            "sun_altitude_angle": weather["sun_altitude_angle"],
            "sun_azimuth_angle": weather["sun_azimuth_angle"],
            "init_speed": params.get("init_speed", ""),
            "trigger_distance": params.get("trigger_distance", ""),
            "npc_speed": params.get("npc_speed", params.get("lead_vehicle_speed", "")),
            "adversary_speed": params.get("adversary_speed", params.get("other_actor_speed", "")),
            "other_actor_count": len(actors),
            "other_actor_types": ";".join(f"{k}:{v}" for k, v in sorted(actor_prefix.items()) if k),
            "raw_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "weather_bucket": weather_bucket(weather),
            "light_bucket": light_bucket(weather),
            "macro_scenario": macro_from(scenario_class, scenario_name, scenario_type),
        }
        rows.append(row)
    return rows


def write_outputs(rows, out_dir, stem_date):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cvci_route_scenario_table_{stem_date}.csv"
    json_path = out_dir / f"cvci_route_scenario_table_{stem_date}.json"
    md_path = out_dir / f"cvci_macro_scenario_summary_{stem_date}.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    by_macro = defaultdict(list)
    for row in rows:
        by_macro[row["macro_scenario"]].append(row)

    lines = [
        "# CVCI Macro Scenario Summary",
        "",
        f"- route_count: {len(rows)}",
        f"- macro_count: {len(by_macro)}",
        "",
    ]
    for macro, items in sorted(by_macro.items(), key=lambda kv: (min(int(r["route_id"]) for r in kv[1]), kv[0])):
        route_ids = [int(r["route_id"]) for r in items]
        scenario_types = Counter(r["scenario_type"] for r in items)
        names = Counter(r["scenario_name"] for r in items)
        difficulties = Counter(r["difficulty_level"] for r in items)
        weather = Counter(r["weather_bucket"] for r in items)
        light = Counter(r["light_bucket"] for r in items)
        actors = Counter(r["other_actor_types"] for r in items if r["other_actor_types"])
        trigger_dist = Counter(r["trigger_distance"] for r in items if r["trigger_distance"])
        init_speed = Counter(r["init_speed"] for r in items if r["init_speed"])
        lines += [
            f"## {macro}",
            "",
            f"- route_count: {len(items)}",
            f"- routes: {','.join(str(i) for i in sorted(route_ids))}",
            f"- scenario_class: {items[0]['scenario_class']}",
            f"- scenario_name_distribution: {dict(names)}",
            f"- scenario_type_distribution: {dict(scenario_types)}",
            f"- difficulty_distribution: {dict(difficulties)}",
            f"- weather_distribution: {dict(weather)}",
            f"- light_distribution: {dict(light)}",
            f"- typical_init_speed: {dict(init_speed.most_common(5))}",
            f"- typical_trigger_distance: {dict(trigger_dist.most_common(5))}",
            f"- typical_other_actor: {dict(actors.most_common(5))}",
            f"- likely_failure_modes: {guess_failure_modes(macro)}",
            f"- candidate_rule_direction: {guess_rule_direction(macro)}",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def guess_failure_modes(macro):
    if "missing" in macro or "car" in macro:
        return "front/side vehicle collision, late braking, timeout from over-caution"
    if "construction" in macro or "closure" in macro or "barrier" in macro:
        return "layout collision, lane invasion, blocked recovery"
    if "roundabout" in macro:
        return "yield failure, route deviation, blocked entry"
    if "pedestrian" in macro or "bike" in macro:
        return "pedestrian/bike collision or excessive waiting"
    if "cut" in macro or "merge" in macro or "traffic" in macro:
        return "vehicle collision, unsafe gap, route deviation"
    return "collision, blocked, timeout, route deviation"


def guess_rule_direction(macro):
    if "missing" in macro:
        return "approach speed cap, vehicle-risk hysteresis, TTC brake"
    if "construction" in macro or "closure" in macro or "barrier" in macro:
        return "obstacle-sensitive slow zone, steer smoothing, bounded creep recovery"
    if "roundabout" in macro:
        return "approach/yield/enter phases with vehicle-risk exit conditions"
    if "pedestrian" in macro or "bike" in macro:
        return "future-path corridor risk, conservative brake, short clear hysteresis"
    if "cut" in macro or "merge" in macro or "traffic" in macro:
        return "dynamic following gap, lateral overlap risk, yield hysteresis"
    return "keep v3 unless closed-loop failure analysis indicates a narrow override"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/home/carla/cvci_back/detection_rules_v3_full144_video_20260623/split_routes/CVCI_BenchMark.xml")
    ap.add_argument("--out-dir", default="/home/carla/cvci_back")
    ap.add_argument("--date", default="20260624")
    args = ap.parse_args()
    rows = parse_xml(Path(args.xml))
    paths = write_outputs(rows, Path(args.out_dir), args.date)
    print(f"routes={len(rows)}")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
