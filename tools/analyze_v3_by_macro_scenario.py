#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


INFRACTION_COLUMNS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def route_num(value):
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def fnum(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-table", default="/home/carla/cvci_back/cvci_route_scenario_table_20260624.csv")
    ap.add_argument("--v3-route-status", default="/home/carla/cvci_back/detection_rules_v3_full144_video_20260623/route_status.csv")
    ap.add_argument("--out-dir", default="/home/carla/cvci_back")
    ap.add_argument("--date", default="20260624")
    args = ap.parse_args()

    route_rows = read_csv(args.route_table)
    status_rows = read_csv(args.v3_route_status)
    route_by_id = {int(r["route_id"]): r for r in route_rows}
    merged = []
    for s in status_rows:
        rid = route_num(s.get("route_id", ""))
        if rid is None or rid not in route_by_id:
            continue
        row = dict(route_by_id[rid])
        row.update({f"v3_{k}": v for k, v in s.items()})
        row["route_id"] = str(rid)
        row["score_challenge"] = fnum(s, "score_challenge")
        row["score_composed"] = fnum(s, "score_composed")
        row["completion"] = fnum(s, "score_route")
        row["status"] = s.get("status", "")
        row["failure_reason"] = s.get("failure_reason", "")
        merged.append(row)

    by_macro = defaultdict(list)
    for row in merged:
        by_macro[row["macro_scenario"]].append(row)

    summary = []
    for macro, items in by_macro.items():
        scores = [r["score_challenge"] for r in items]
        completions = [r["completion"] for r in items]
        infractions = Counter()
        for r in items:
            for col in INFRACTION_COLUMNS:
                val = fnum(r, f"v3_{col}")
                if val:
                    infractions[col] += int(val)
        worst = sorted(items, key=lambda r: (r["score_challenge"], r["completion"]))[:5]
        count_complete = sum(1 for r in items if r["status"] == "Completed")
        summary.append({
            "macro_scenario": macro,
            "route_count": len(items),
            "score_mean": round(statistics.mean(scores), 3) if scores else 0,
            "score_min": round(min(scores), 3) if scores else 0,
            "score_max": round(max(scores), 3) if scores else 0,
            "score_median": round(statistics.median(scores), 3) if scores else 0,
            "count_score_lt_90": sum(1 for s in scores if s < 90),
            "count_complete": count_complete,
            "completion_mean": round(statistics.mean(completions), 3) if completions else 0,
            "collision": infractions["collisions_layout"] + infractions["collisions_pedestrian"] + infractions["collisions_vehicle"],
            "timeout": infractions["scenario_timeouts"] + infractions["route_timeout"],
            "stuck_blocking": infractions["vehicle_blocked"],
            "red_light": infractions["red_light"],
            "lane_invasion": infractions["outside_route_lanes"],
            "off_road": infractions["outside_route_lanes"],
            "route_deviation": infractions["route_dev"],
            "infraction_stats": json.dumps(dict(infractions), ensure_ascii=False, sort_keys=True),
            "worst_routes": ",".join(str(r["route_id"]) for r in worst),
            "sampled_route_candidates": sample_routes(items),
            "needs_rule": "yes" if (statistics.mean(scores) if scores else 0) < 90 else "no",
        })
    summary.sort(key=lambda r: (r["score_mean"], r["score_min"], -r["count_score_lt_90"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cvci_v3_macro_score_analysis_{args.date}.csv"
    md_path = out_dir / f"cvci_v3_macro_score_analysis_{args.date}.md"
    merged_path = out_dir / f"cvci_v3_route_macro_merged_{args.date}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    lines = [
        "# CVCI v3 Macro Scenario Score Analysis",
        "",
        f"- merged_routes: {len(merged)}",
        f"- macro_count: {len(summary)}",
        "",
        "| rank | macro_scenario | routes | mean | min | median | <90 | complete | main_failures | sampled_candidates |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(summary, 1):
        failures = []
        for key in ("collision", "stuck_blocking", "red_light", "lane_invasion", "route_deviation", "timeout"):
            if r[key]:
                failures.append(f"{key}={r[key]}")
        lines.append(
            f"| {i} | {r['macro_scenario']} | {r['route_count']} | {r['score_mean']:.3f} | "
            f"{r['score_min']:.3f} | {r['score_median']:.3f} | {r['count_score_lt_90']} | "
            f"{r['count_complete']} | {'; '.join(failures) or 'none'} | {r['sampled_route_candidates']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(merged_path)


def sample_routes(items):
    # Deterministic seed=20260624 equivalent: choose from worst half after sorting by score,
    # then stable route id order to make the sampled debug set reproducible.
    ordered = sorted(items, key=lambda r: (r["score_challenge"], int(r["route_id"])))
    pool = ordered[: max(3, len(ordered) // 2)]
    # Avoid random dependency in reports; this is deterministic and focuses debugging on failures.
    return ",".join(str(r["route_id"]) for r in pool[:3])


if __name__ == "__main__":
    main()
