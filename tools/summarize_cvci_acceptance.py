#!/usr/bin/env python3
"""Summarize CVCI runs with the project default 80-point acceptance threshold."""
from __future__ import print_function

import argparse
import json
from pathlib import Path


def _load_threshold(repo_root):
    policy_path = repo_root / "configs" / "cvci_acceptance_policy.yaml"
    threshold = 80.0
    if policy_path.exists():
        for line in policy_path.read_text().splitlines():
            if line.strip().startswith("score_challenge_pass_threshold:"):
                threshold = float(line.split(":", 1)[1].strip())
                break
    return threshold


def _summarize_run(run, threshold):
    result = run / "results" / "cvci_0.json"
    summary = {
        "run": run.name,
        "acceptance": "RUNNING_OR_NO_RESULT",
        "score_challenge": None,
        "score_route": None,
        "record_status": None,
        "infractions": {},
    }
    if not result.exists():
        return summary
    data = json.loads(result.read_text())
    records = data.get("_checkpoint", {}).get("records", [])
    if not records:
        return summary
    record = records[-1]
    scores = record.get("scores", {}) or {}
    score = scores.get("score_challenge")
    summary.update(
        {
            "acceptance": "ACCEPTED_80" if score is not None and float(score) >= threshold else "BELOW_80_KEEP_TUNING",
            "score_challenge": score,
            "score_route": scores.get("score_route"),
            "record_status": record.get("status"),
            "infractions": {k: len(v) for k, v in (record.get("infractions", {}) or {}).items() if v},
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_base", type=Path)
    parser.add_argument("--active-file", default="current_parallel3_patch_runs.txt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    threshold = _load_threshold(repo_root)
    active_path = args.run_base / args.active_file
    runs = [Path(x.strip()) for x in active_path.read_text().splitlines() if x.strip()]
    print("PASS_THRESHOLD score_challenge >= %.1f" % threshold)
    for run in runs:
        print(json.dumps(_summarize_run(run, threshold), ensure_ascii=False))


if __name__ == "__main__":
    main()
