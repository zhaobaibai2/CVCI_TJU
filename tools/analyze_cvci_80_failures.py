#!/usr/bin/env python3
"""Summarize CVCI parallel-80 route failures from run_matrix result JSONs."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FAILURE_FIELDS = ["collision", "timeout", "blocked", "red_light", "route_deviation", "lane_invasion"]
COLLISION_KEYS = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")


def has_entries(value):
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return any(has_entries(v) for v in value.values())
    return bool(value)


def latest_record(path):
    try:
        data = json.loads(Path(path).read_text(errors="ignore"))
    except Exception:
        return None
    records = []

    def walk(node):
        if isinstance(node, dict):
            scores = node.get("scores")
            if isinstance(scores, dict) and "score_challenge" in scores:
                records.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return records[-1] if records else None


def classify(record):
    if not record:
        return {field: False for field in FAILURE_FIELDS}
    status = str(record.get("status", "") or "").lower()
    infractions = record.get("infractions", {}) or {}
    return {
        "collision": any(has_entries(infractions.get(k)) for k in COLLISION_KEYS),
        "timeout": has_entries(infractions.get("scenario_timeouts")) or has_entries(infractions.get("route_timeout")) or "tickruntime" in status or "timeout" in status,
        "blocked": has_entries(infractions.get("vehicle_blocked")) or "blocked" in status,
        "red_light": has_entries(infractions.get("red_light")) or "red light" in status,
        "route_deviation": has_entries(infractions.get("route_dev")) or "deviated" in status,
        "lane_invasion": has_entries(infractions.get("outside_route_lanes")) or "outside route" in status,
    }


def first_collision_detail(record):
    if not record:
        return ""
    infractions = record.get("infractions", {}) or {}
    for key in COLLISION_KEYS:
        entries = infractions.get(key) or []
        if isinstance(entries, list) and entries:
            return f"{key}: {entries[0]}"
    return ""


def first_route_deviation_detail(record):
    if not record:
        return ""
    infractions = record.get("infractions", {}) or {}
    entries = infractions.get("route_dev") or []
    if isinstance(entries, list) and entries:
        return str(entries[0])
    return ""


def challenge_private_facts(record):
    if not record:
        return {}
    facts = record.get("private_facts") or {}
    return facts if isinstance(facts, dict) else {}


def collision_bucket(detail):
    text = detail.lower()
    if not text:
        return "none"
    z_match = re.search(r"z=([-+]?\d+(?:\.\d+)?)", text)
    high_static_layout = bool("static." in text and z_match and float(z_match.group(1)) >= 4.0)
    if ("static.unknown" in text and "z=10" in text) or high_static_layout:
        return "overhead_or_spawn_static_unknown"
    if "constructioncone" in text or "guardrail" in text:
        return "construction_layout"
    if "vehicle." in text:
        return "vehicle"
    if "pedestrian" in text or "walker" in text:
        return "pedestrian"
    if "static." in text:
        return "static_layout"
    return "other_collision"


def route_issue_bucket(record, flags, collision_detail):
    collision = collision_bucket(collision_detail)
    if collision != "none":
        return collision
    if not record:
        return "no_result_record"
    facts = challenge_private_facts(record)
    rec_score = score(record)
    if rec_score != "" and float(rec_score) < 80 and not any(flags.values()):
        if facts.get("brake_response") is False and facts.get("safe_bypass") is True:
            return "challenge_brake_response_missing"
        if facts:
            return "challenge_private_goal_missing"
    if flags.get("route_deviation"):
        status = str(record.get("status", "") or "")
        detail = first_route_deviation_detail(record)
        duration_game = None
        try:
            duration_game = float((record.get("meta") or {}).get("duration_game"))
        except Exception:
            duration_game = None
        early = duration_game is not None and duration_game <= 25.0
        high_elevation = False
        z_match = re.search(r"z=([-+]?\d+(?:\.\d+)?)", detail.lower())
        if z_match:
            try:
                high_elevation = float(z_match.group(1)) >= 4.0
            except Exception:
                high_elevation = False
        if early and high_elevation:
            return "early_high_elevation_route_deviation"
        if "deviated" in status.lower():
            return "route_deviation"
    if flags.get("blocked"):
        return "blocked_no_collision"
    if flags.get("timeout"):
        return "timeout_no_collision"
    return "low_score_no_collision"


def score(record):
    if not record:
        return ""
    try:
        return float((record.get("scores") or {}).get("score_challenge", ""))
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="/home/carla/cvci_back/v5_parallel_80_runs")
    parser.add_argument("--backfill-run-matrix", action="store_true", help="Rewrite run_matrix.csv failure-type columns from per-attempt result JSONs.")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    run_matrix = run_root / "run_matrix.csv"
    rows = []
    if run_matrix.exists():
        rows = list(csv.DictReader(run_matrix.open(newline="", encoding="utf-8")))

    enriched = []
    macro_counts = defaultdict(Counter)
    collision_bucket_counts = defaultdict(Counter)
    route_latest = {}
    for row in rows:
        result_path = row.get("result_json_path") or ""
        record = latest_record(result_path) if result_path else None
        flags = classify(record)
        rec_score = score(record)
        out = dict(row)
        out["score_from_json"] = rec_score
        out["status_from_json"] = record.get("status", "") if record else "NO_RECORD"
        out["first_collision"] = first_collision_detail(record)
        out["collision_bucket"] = collision_bucket(out["first_collision"])
        out["first_route_deviation"] = first_route_deviation_detail(record)
        private_facts = challenge_private_facts(record)
        out["private_brake_response"] = private_facts.get("brake_response", "")
        out["private_safe_bypass"] = private_facts.get("safe_bypass", "")
        out["route_issue_bucket"] = route_issue_bucket(record, flags, out["first_collision"])
        for key, value in flags.items():
            out[key] = value
            if value:
                macro_counts[row.get("macro_scenario", "unknown")][key] += 1
        if flags.get("collision"):
            collision_bucket_counts[row.get("macro_scenario", "unknown")][out["collision_bucket"]] += 1
        if not any(flags.values()) and rec_score != "" and float(rec_score) < 80:
            macro_counts[row.get("macro_scenario", "unknown")]["low_score_no_flag"] += 1
        if not record:
            macro_counts[row.get("macro_scenario", "unknown")]["no_result_record"] += 1
        enriched.append(out)
        rid = row.get("route_id", "")
        if rid:
            route_latest[rid] = out

    if args.backfill_run_matrix and rows:
        original_fields = list(rows[0].keys())
        for row in enriched:
            for field in FAILURE_FIELDS:
                row[field] = str(row.get(field, ""))
        tmp_path = run_matrix.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=original_fields)
            writer.writeheader()
            for row in enriched:
                writer.writerow({field: row.get(field, "") for field in original_fields})
        tmp_path.replace(run_matrix)

    csv_path = run_root / "cvci_80_failure_diagnostics.csv"
    fields = list(rows[0].keys()) if rows else []
    for extra in ["score_from_json", "status_from_json", "first_collision", "collision_bucket", "first_route_deviation", "private_brake_response", "private_safe_bypass", "route_issue_bucket"] + FAILURE_FIELDS:
        if extra not in fields:
            fields.append(extra)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)

    md = ["# CVCI 80-Pass Failure Diagnostics", "", f"- run_root: `{run_root}`", f"- run_matrix_rows: {len(rows)}", ""]
    md += ["## Macro Failure Counts", "", "| macro | collision | timeout | blocked | route_deviation | lane_invasion | red_light | no_record | low_score_no_flag |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for macro in sorted(macro_counts):
        c = macro_counts[macro]
        md.append(f"| {macro} | {c['collision']} | {c['timeout']} | {c['blocked']} | {c['route_deviation']} | {c['lane_invasion']} | {c['red_light']} | {c['no_result_record']} | {c['low_score_no_flag']} |")
    md += ["", "## Collision Buckets", "", "| macro | bucket | count |", "|---|---|---:|"]
    for macro in sorted(collision_bucket_counts):
        for bucket, count in sorted(collision_bucket_counts[macro].items()):
            md.append(f"| {macro} | {bucket} | {count} |")
    issue_bucket_counts = defaultdict(Counter)
    for row in enriched:
        try:
            sc = float(row.get("score_from_json") or row.get("score_challenge") or 0)
        except Exception:
            sc = 0
        if sc < 80:
            issue_bucket_counts[row.get("macro_scenario", "unknown")][row.get("route_issue_bucket", "unknown")] += 1
    md += ["", "## Route Issue Buckets", "", "| macro | bucket | count |", "|---|---|---:|"]
    for macro in sorted(issue_bucket_counts):
        for bucket, count in sorted(issue_bucket_counts[macro].items()):
            md.append(f"| {macro} | {bucket} | {count} |")
    md += ["", "## Latest Route Attempts Below 80", "", "| route | macro | score | status | failure_types | issue_bucket | collision_bucket | first_collision_or_route_dev | result |", "|---:|---|---:|---|---|---|---|---|---|"]
    def sort_key(item):
        rid, row = item
        try:
            score_key = float(row.get("score_from_json") or row.get("score_challenge") or 999)
        except Exception:
            score_key = 999
        try:
            rid_key = int(rid)
        except Exception:
            rid_key = 9999
        return (score_key, rid_key)
    for rid, row in sorted(route_latest.items(), key=sort_key):
        try:
            sc = float(row.get("score_from_json") or row.get("score_challenge") or 0)
        except Exception:
            sc = 0
        if sc >= 80:
            continue
        types = [field for field in FAILURE_FIELDS if str(row.get(field)).lower() == "true"]
        if row.get("status_from_json") == "NO_RECORD":
            types.append("no_result_record")
        if not types:
            types.append("low_score_no_flag")
        detail = str(row.get('first_collision') or row.get('first_route_deviation') or '').replace('|', '/')
        md.append(f"| {rid} | {row.get('macro_scenario','')} | {sc:g} | {row.get('status_from_json','')} | {', '.join(types)} | {row.get('route_issue_bucket','')} | {row.get('collision_bucket','')} | {detail} | {row.get('result_json_path','')} |")
    env_deprioritized = []
    deprioritized_issue_buckets = {
        "overhead_or_spawn_static_unknown",
        "early_high_elevation_route_deviation",
    }
    for rid, row in sorted(route_latest.items(), key=sort_key):
        try:
            sc = float(row.get("score_from_json") or row.get("score_challenge") or 0)
        except Exception:
            sc = 0
        if sc < 80 and row.get("route_issue_bucket") in deprioritized_issue_buckets:
            env_deprioritized.append(str(rid))
    (run_root / "environment_deprioritized_routes.txt").write_text("\n".join(env_deprioritized) + ("\n" if env_deprioritized else ""), encoding="utf-8")

    md_path = run_root / "cvci_80_failure_diagnostics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    if args.backfill_run_matrix and rows:
        print(f"backfilled {run_matrix}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {run_root / 'environment_deprioritized_routes.txt'}")


if __name__ == "__main__":
    main()
