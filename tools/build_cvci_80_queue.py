#!/usr/bin/env python3
"""Build an 80-pass CVCI route queue from XML metadata and existing results."""

import argparse
import csv
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


PASS_STATES = ("PASS_80", "YELLOW_PASS_80", "NEEDS_FIX", "TODO")


def snake(text):
    text = (text or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def macro_from(scenario_class, scenario_name, scenario_type):
    source = scenario_class or scenario_name or scenario_type or "unknown"
    aliases = {
        "missing car": "car_disappear_accident",
        "high speed temporary construction": "lane_closure_with_truck",
        "drive into the roundabout": "roundabout",
        "roundabout": "roundabout",
        "ghost probe": "ghost_probe",
    }
    key = source.strip().lower()
    return aliases.get(key, snake(source))


def parse_routes(xml_path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(xml_path, parser=parser).getroot()
    routes = {}
    scenario_class = ""
    difficulty = ""
    for child in list(root):
        if child.tag is ET.Comment:
            text = (child.text or "").strip()
            m = re.search(r"scenario_class\s+\d+\s*:\s*([^=]+)", text, flags=re.I)
            if m:
                scenario_class = m.group(1).strip()
                difficulty = ""
            m = re.search(r"\blevel\s*-\s*(\d+)\b", text, flags=re.I)
            if m:
                difficulty = f"level-{m.group(1)}"
            continue
        if child.tag != "route":
            continue
        rid = str(child.attrib.get("id", "")).strip()
        if not rid:
            continue
        scenario = child.find("./scenarios/scenario")
        sname = scenario.attrib.get("name", "") if scenario is not None else ""
        stype = scenario.attrib.get("type", "") if scenario is not None else ""
        weather = child.find("./weathers/weather")
        sun = float(weather.attrib.get("sun_altitude_angle", 90.0)) if weather is not None else 90.0
        wet = float(weather.attrib.get("wetness", 0.0)) if weather is not None else 0.0
        rain = float(weather.attrib.get("precipitation", 0.0)) if weather is not None else 0.0
        routes[rid] = {
            "route_id": rid,
            "town": child.attrib.get("town", ""),
            "scenario_class": scenario_class,
            "difficulty_level": difficulty,
            "scenario_name": sname,
            "scenario_type": stype,
            "light_bucket": "night" if sun < 5 else "day",
            "weather_bucket": "rain_or_wet" if rain >= 30 or wet >= 40 else "clear",
            "waypoint_count": str(len(list(child.find("waypoints") or []))),
            "macro_scenario": macro_from(scenario_class, sname, stype),
        }
    return routes


def route_id_from_text(*values):
    for value in values:
        if value is None:
            continue
        text = str(value)
        for pattern in (r"RouteScenario[_-](\d+)", r"\broute[_-]?(\d+)\b", r"/(\d+)_route"):
            m = re.search(pattern, text, flags=re.I)
            if m:
                return str(int(m.group(1)))
    return ""


def records_from_json(path):
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return []
    found = []

    def walk(node):
        if isinstance(node, dict):
            scores = node.get("scores")
            if isinstance(scores, dict) and any(k in scores for k in ("score_challenge", "score_composed", "score_route")):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    rows = []
    for node in found:
        scores = node.get("scores", {}) or {}
        infractions = node.get("infractions", {}) or {}
        meta = node.get("meta", {}) or {}
        rid = route_id_from_text(
            node.get("route_id"),
            node.get("name"),
            meta.get("route_id"),
            meta.get("name"),
            path,
        )
        if not rid:
            continue
        score = as_float(scores.get("score_challenge"))
        rows.append({
            "route_id": rid,
            "score_challenge": score,
            "score_composed": as_float(scores.get("score_composed")),
            "score_route": as_float(scores.get("score_route")),
            "score_penalty": as_float(scores.get("score_penalty")),
            "status": str(node.get("status") or ""),
            "completed": str(node.get("status") or "").lower() in ("completed", "perfect"),
            "collision": infraction_count(infractions, "collision"),
            "timeout": contains_any(node, ("TickRuntime", "timeout")),
            "blocked": contains_any(node, ("blocked", "vehicle_blocked")),
            "red_light": infraction_count(infractions, "red_light"),
            "route_deviation": infraction_count(infractions, "route_dev") + infraction_count(infractions, "outside_route_lanes"),
            "lane_invasion": infraction_count(infractions, "outside_route_lanes"),
            "result_json_path": str(path),
            "timestamp": path.stat().st_mtime,
        })
    return rows


def as_float(value):
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def infraction_count(infractions, needle):
    total = 0
    for key, value in (infractions or {}).items():
        if needle.lower() not in str(key).lower():
            continue
        if isinstance(value, list):
            total += len(value)
        elif value:
            total += 1
    return total


def contains_any(node, needles):
    text = json.dumps(node, ensure_ascii=False, default=str).lower()
    return int(any(str(n).lower() in text for n in needles))


def collect_results(search_roots):
    rows = []
    seen = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name in {"acceptance_policy.json", "cvci_acceptance_policy.json"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            rows.extend(records_from_json(path))
    return rows


def best_by_route(records):
    grouped = defaultdict(list)
    for row in records:
        if math.isnan(row["score_challenge"]):
            continue
        grouped[row["route_id"]].append(row)
    best = {}
    for rid, rows in grouped.items():
        best[rid] = max(rows, key=lambda r: (r["score_challenge"], int(r["completed"]), r["timestamp"]))
    return best, grouped


def pass_state(best, threshold):
    if not best:
        return "TODO"
    score = best.get("score_challenge", math.nan)
    if not math.isnan(score) and score >= threshold:
        return "PASS_80" if best.get("completed") else "YELLOW_PASS_80"
    return "NEEDS_FIX"


def priority(row, macro_stats, grouped):
    state = row["pass_state"]
    best = as_float(row.get("best_score"))
    attempts = len(grouped.get(row["route_id"], []))
    macro_mean = macro_stats.get(row["macro_scenario"], {}).get("mean", -1)
    state_rank = {"TODO": 0, "NEEDS_FIX": 1, "YELLOW_PASS_80": 5, "PASS_80": 9}.get(state, 4)
    if str(row.get("environment_deprioritized", "")).lower() == "true":
        state_rank = 8
    score_rank = best if not math.isnan(best) else -1.0
    return (state_rank, macro_mean if macro_mean >= 0 else -1, score_rank, attempts)


def read_route_id_file(path):
    if not path or not Path(path).exists():
        return set()
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-root", required=True)
    ap.add_argument("--cvci-root", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", default="/home/carla/cvci_back")
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--environment-deprioritized-routes", default="", help="Optional newline route-id file for likely environment/startup artifacts that should go to the end of the queue.")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    env_file = args.environment_deprioritized_routes or str(run_root / "environment_deprioritized_routes.txt")
    environment_deprioritized = read_route_id_file(env_file)
    routes = parse_routes(Path(args.xml))
    records = collect_results([args.dt_root, args.cvci_root, args.report_root, args.run_root, "/root/autodl-tmp/projects/cvci_supervised_runs"])
    best, grouped = best_by_route(records)

    route_rows = []
    for rid, meta in sorted(routes.items(), key=lambda kv: int(kv[0])):
        b = best.get(rid)
        state = pass_state(b, args.threshold)
        row = dict(meta)
        row.update({
            "best_score": "" if not b else f"{b['score_challenge']:.6g}",
            "best_status": "" if not b else b["status"],
            "best_completed": "" if not b else str(bool(b["completed"])),
            "best_result_json": "" if not b else b["result_json_path"],
            "attempt_count": str(len(grouped.get(rid, []))),
            "pass_state": state,
            "frozen": str(state in ("PASS_80", "YELLOW_PASS_80")),
            "environment_deprioritized": str(rid in environment_deprioritized),
        })
        route_rows.append(row)

    macro_groups = defaultdict(list)
    for row in route_rows:
        macro_groups[row["macro_scenario"]].append(row)
    macro_rows = []
    macro_stats = {}
    for macro, rows in sorted(macro_groups.items()):
        scores = [as_float(r["best_score"]) for r in rows if r["best_score"]]
        valid = [s for s in scores if not math.isnan(s)]
        mean = sum(valid) / len(valid) if valid else -1.0
        pass_count = sum(1 for r in rows if r["pass_state"] in ("PASS_80", "YELLOW_PASS_80"))
        completed_pass = sum(1 for r in rows if r["pass_state"] == "PASS_80")
        macro_stats[macro] = {"mean": mean, "pass_count": pass_count}
        macro_rows.append({
            "macro_scenario": macro,
            "route_count": len(rows),
            "scored_routes": len(valid),
            "mean_best_score": "" if mean < 0 else f"{mean:.6g}",
            "min_best_score": "" if not valid else f"{min(valid):.6g}",
            "pass_80_count": pass_count,
            "completed_pass_80_count": completed_pass,
            "macro_pass_80": str(pass_count == len(rows) and mean >= args.threshold),
            "routes": ",".join(r["route_id"] for r in rows),
        })

    queue = [dict(r) for r in route_rows if r["pass_state"] in ("TODO", "NEEDS_FIX")]
    queue.sort(key=lambda r: priority(r, macro_stats, grouped))
    for i, row in enumerate(queue):
        row["queue_priority"] = str(i)
        row["max_attempts"] = "3"
        row["next_action"] = "RETRY_LATER_ENVIRONMENT" if row.get("environment_deprioritized") == "True" else "RUN_CLOSED_LOOP"

    route_fields = list(route_rows[0].keys()) if route_rows else []
    queue_fields = list(queue[0].keys()) if queue else route_fields + ["queue_priority", "max_attempts", "next_action"]
    write_csv(run_root / "route_best_scores.csv", route_rows, route_fields)
    write_csv(run_root / "macro_best_scores.csv", macro_rows, list(macro_rows[0].keys()) if macro_rows else ["macro_scenario"])
    write_csv(run_root / "parallel_80_queue.csv", queue, queue_fields)
    (run_root / "parallel_80_queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "current_queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "frozen_routes.txt").write_text("\n".join(r["route_id"] for r in route_rows if r["frozen"] == "True") + "\n", encoding="utf-8")
    (run_root / "needs_fix_routes.txt").write_text("\n".join(r["route_id"] for r in route_rows if r["pass_state"] in ("TODO", "NEEDS_FIX")) + "\n", encoding="utf-8")
    (run_root / "regression_routes.txt").write_text("\n".join(r["route_id"] for r in route_rows if r["pass_state"] == "PASS_80") + "\n", encoding="utf-8")

    lines = [
        "# CVCI Parallel 80 Queue Progress",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- threshold: {args.threshold}",
        f"- routes: {len(route_rows)}",
        f"- queue_routes: {len(queue)}",
        f"- frozen_routes: {sum(1 for r in route_rows if r['frozen'] == 'True')}",
        "",
        "## Pass State Counts",
        "",
    ]
    counts = Counter(r["pass_state"] for r in route_rows)
    for state in PASS_STATES:
        lines.append(f"- {state}: {counts.get(state, 0)}")
    lines += ["", "## Next Queue", ""]
    for row in queue[:30]:
        suffix = " | env_deprioritized=True" if row.get("environment_deprioritized") == "True" else ""
        lines.append(f"- route {row['route_id']} | {row['macro_scenario']} | best={row['best_score'] or 'NA'} | state={row['pass_state']}{suffix}")
    (run_root / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"routes={len(route_rows)} queue={len(queue)} frozen={sum(1 for r in route_rows if r['frozen'] == 'True')}")
    print(run_root / "route_best_scores.csv")
    print(run_root / "macro_best_scores.csv")
    print(run_root / "parallel_80_queue.json")


if __name__ == "__main__":
    main()
