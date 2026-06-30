#!/usr/bin/env python3
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path


RUN = Path("/home/carla/cvci_back/v5_parallel_80_runs")
DT = Path("/root/autodl-tmp/projects/code/DriveTransformer")
CVCI = Path("/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark")
XML = CVCI / "runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml"
CKPT = DT / "runs/cvci_drivetransformer_train/work_dirs/cvci_iter20000_unfreeze_last13_to40000/iter_25000.pth"


def read_csv(name):
    path = RUN / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def shell_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def infer_workers(process_snapshot):
    workers = []
    for line in process_snapshot.splitlines():
        if "leaderboard_evaluator.py" not in line:
            continue
        parts = line.split(None, 3)
        pid = parts[0] if parts else ""
        elapsed = parts[2] if len(parts) > 2 else ""
        cmd = parts[3] if len(parts) > 3 else line
        route = ""
        port = ""
        result = ""
        for token in cmd.split():
            if token.startswith("--routes="):
                route_path = token.split("=", 1)[1]
                if "_r" in route_path:
                    route = route_path.rsplit("_r", 1)[-1].split("/", 1)[0]
            elif token.startswith("--port="):
                port = token.split("=", 1)[1]
            elif token.startswith("--checkpoint="):
                result = token.split("=", 1)[1]
        worker = "?"
        gpu_id = "?"
        if port == "45700":
            worker, gpu_id = "0", "0"
        elif port == "45750":
            worker, gpu_id = "1", "1"
        elif port == "45800":
            worker, gpu_id = "2", "2"
        workers.append(
            {
                "worker": worker,
                "gpu": gpu_id,
                "carla_adapter": {"0": "2", "1": "2", "2": "0"}.get(worker, "?"),
                "route": route,
                "pid": pid,
                "elapsed": elapsed,
                "port": port,
                "result": result,
            }
        )
    return workers


def write_gpu_status(now, workers, gpu_snapshot, process_snapshot):
    lines = [
        "# GPU Worker Status",
        "",
        f"- updated_at: {now}",
        "",
        "## GPU Snapshot",
        "",
        "```",
        gpu_snapshot.rstrip(),
        "```",
        "",
        "## Workers",
        "",
        "| worker | gpu | carla_adapter | route | pid | port | elapsed | result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for worker in sorted(workers, key=lambda item: item["worker"]):
        lines.append(
            f"| {worker['worker']} | {worker['gpu']} | {worker.get('carla_adapter', '?')} | {worker['route']} | "
            f"{worker['pid']} | {worker['port']} | {worker['elapsed']} | {worker['result']} |"
        )
    if not workers:
        lines.append("| - | - | - | - | - | - | - | no active evaluator |")
    lines += ["", "## Process Snapshot", "", "```", process_snapshot.rstrip(), "```"]
    (RUN / "gpu_worker_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    route_rows = read_csv("route_best_scores.csv")
    macro_rows = read_csv("macro_best_scores.csv")
    run_rows = read_csv("run_matrix.csv")
    queue_rows = read_csv("parallel_80_queue.csv")

    state_counts = Counter(row.get("pass_state", "") for row in route_rows)
    new_pass = [row for row in run_rows if str(row.get("pass_80", "")).lower() == "true"]
    runtime_pass_routes = {str(row.get("route_id", "")).strip() for row in new_pass}
    failed = [
        row
        for row in run_rows
        if row.get("status") in ("FAILED_RUN", "NEEDS_FIX")
        or (row.get("score_challenge") not in (None, "") and float(row.get("score_challenge") or 0) < 80)
    ]
    below = []
    for row in route_rows:
        try:
            score = float(row.get("best_score") or 0)
        except ValueError:
            score = 0.0
        if str(row.get("route_id", "")).strip() in runtime_pass_routes:
            continue
        if score < 80:
            below.append((score, row.get("route_id", ""), row.get("macro_scenario", ""), row.get("best_status", "")))
    below.sort(key=lambda item: (item[0], int(item[1]) if str(item[1]).isdigit() else 9999))

    gpu_snapshot = shell_output("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
    process_snapshot = shell_output(
        "ps -eo pid,ppid,etime,cmd | grep -E 'run_cvci_parallel_80_queue.py|leaderboard_evaluator.py|CarlaUE4-Linux-Shipping' | grep -v grep"
    )
    workers = infer_workers(process_snapshot)
    write_gpu_status(now, workers, gpu_snapshot, process_snapshot)

    initial_frozen = {
        line.strip()
        for line in (RUN / "frozen_routes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if (RUN / "frozen_routes.txt").exists() else set()
    initial_needs = {
        line.strip()
        for line in (RUN / "needs_fix_routes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if (RUN / "needs_fix_routes.txt").exists() else set()
    merged_frozen = initial_frozen | runtime_pass_routes
    merged_needs = initial_needs - runtime_pass_routes
    numeric_key = lambda value: int(value) if str(value).isdigit() else 10**9
    (RUN / "frozen_routes.txt").write_text(
        "\n".join(sorted(merged_frozen, key=numeric_key)) + "\n",
        encoding="utf-8",
    )
    (RUN / "needs_fix_routes.txt").write_text(
        "\n".join(sorted(merged_needs, key=numeric_key)) + "\n",
        encoding="utf-8",
    )

    failed_lines = ["# Failed Runs", "", f"- updated_at: {now}", f"- failed_or_below80_rows: {len(failed)}", ""]
    if failed:
        failed_lines += ["| time | route | macro | score | status | log |", "|---|---:|---|---:|---|---|"]
        for row in failed:
            failed_lines.append(
                f"| {row.get('timestamp', '')} | {row.get('route_id', '')} | {row.get('macro_scenario', '')} | "
                f"{row.get('score_challenge', '')} | {row.get('status', '')} | {row.get('log_path', '')} |"
            )
    else:
        failed_lines.append("No failed or below-80 attempt has been recorded in this RUN_ROOT yet.")
    (RUN / "failed_runs.md").write_text("\n".join(failed_lines) + "\n", encoding="utf-8")

    full144 = {
        "updated_at": now,
        "status": "not_completed_in_this_run_root",
        "note": "No complete full144 closed-loop result has been produced by v5_parallel_80_runs yet. Current strict historical baselines are recorded in final_80_report.md; current queue results are route-level attempts only.",
        "known_strict_baselines": {
            "original_drivetransformer": {"mean_score_challenge": 31.969686, "completed": "44/144"},
            "iter25000": {"mean_score_challenge": 55.274218, "completed": "47/144"},
            "routewise_selected_evidence": {"mean_score_challenge": 58.998742, "completed": "52/144"},
        },
        "current_parallel80_run_matrix_rows": len(run_rows),
        "current_parallel80_new_pass80": [
            {
                "route_id": row.get("route_id"),
                "macro_scenario": row.get("macro_scenario"),
                "score_challenge": row.get("score_challenge"),
                "status": row.get("status"),
                "completed": row.get("completed"),
            }
            for row in new_pass
        ],
    }
    (RUN / "full144_latest_result.json").write_text(json.dumps(full144, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    macro_table = []
    for row in macro_rows:
        macro_table.append(
            (
                row.get("macro_scenario", ""),
                row.get("mean_best_score", row.get("mean_score", "")),
                row.get("min_best_score", row.get("min_score", "")),
                row.get("pass_80_count", row.get("pass_count", "")),
                row.get("route_count", ""),
                row.get("macro_pass_80", ""),
                row.get("completed_pass_80_count", row.get("completed_pass_count", "")),
            )
        )
    macro_table.sort()

    report = [
        "# CVCI DriveTransformer Parallel 80-Pass Rule Tuning Report",
        "",
        f"- updated_at: {now}",
        "- stage: active parallel queue running; not final full144",
        "",
        "## 1. Objective",
        "",
        "- Threshold: score_challenge >= 80 is route_pass.",
        "- Stop chasing 100 once a route reaches 80+.",
        "- Keep three GPU workers busy with independent closed-loop attempts.",
        "- Formal submit-mode env keeps route prior off and LiDAR/auxiliary perception on.",
        "",
        "## 2. Project Layout",
        "",
        f"- DT_ROOT: `{DT}`",
        f"- CVCI_ROOT: `{CVCI}`",
        f"- CVCI_XML: `{XML}`",
        f"- checkpoint: `{CKPT}`",
        f"- RUN_ROOT: `{RUN}`",
        "",
        "## 3. Current Baseline",
        "",
        "- Original DriveTransformer strict 144-route: mean score_challenge 31.969686, Completed 44/144.",
        "- iter25000 strict 144-route: mean score_challenge 55.274218, Completed 47/144.",
        "- Current selected evidence: mean score_challenge 58.998742, Completed 52/144.",
        f"- Queue initialization states: {dict(state_counts)}",
        f"- Runtime overlay: {len(runtime_pass_routes)} queued routes reached 80+ after scheduler start.",
        "",
        "## 4. Parallel Scheduler",
        "",
        "- Scheduler command and PID are stored in `scheduler.pid` and `scheduler.log.path`.",
        "- Worker ports: 45700/64700, 45750/64750, 45800/64800.",
        "- Freeze policy: PASS_80 or YELLOW_PASS_80 freezes the route for main queue purposes.",
        "- Max attempts per route: 3.",
        "- Current live worker table is in `gpu_worker_status.md`.",
        "",
        "## 5. Rule Changes",
        "",
        "No new scenario-control rule was added in this continuation. The live scheduler was fixed so CARLA uses the host-specific graphicsadapter mapping 2,2,0 instead of sending all render workers to graphicsadapter=0, which caused startup OOM on earlier route 58/59 attempts. The scheduler status write path was also changed so worker starts write per-worker JSON snippets instead of overwriting the aggregated gpu_worker_status.md; the aggregate status is generated by update_cvci_80_reports.py. The scheduler source now also parses evaluator infractions into run_matrix failure-type columns and supports CVCI_AUX_LOG_PERIOD=100 for future attempts; these two additions apply when the scheduler process is next started, because the current live scheduler was already running before the source edit.",
        "",
        "Relevant existing files:",
        "- `team_code/drivetransformer_b2d_agent.py`",
        "- `team_code/cvci_auxiliary_system.py`",
        "- `team_code/auxiliary_perception/pointcloud_geometry.py`",
        "- `team_code/auxiliary_perception/lidar_detector.py`",
        "- `team_code/auxiliary_perception/object_tracker.py`",
        "- `tools/build_cvci_80_queue.py`",
        "- `tools/run_cvci_parallel_80_queue.py`",
        "- `tools/parse_cvci_routes.py`",
        "",
        "## 6. Route Results",
        "",
        f"- New run_matrix rows in this RUN_ROOT: {len(run_rows)}",
        f"- New route attempts reaching 80+: {len(new_pass)}",
        "",
    ]
    if new_pass:
        report += ["| route | macro | score | status | completed | action |", "|---:|---|---:|---|---|---|"]
        for row in new_pass:
            report.append(
                f"| {row.get('route_id', '')} | {row.get('macro_scenario', '')} | {row.get('score_challenge', '')} | "
                f"{row.get('status', '')} | {row.get('completed', '')} | {row.get('next_action', '')} |"
            )
        report.append("")
    report += ["### Still Below 80", "", f"- Count from latest scanned best scores: {len(below)}", ""]
    report += ["| route | macro | best_score | status |", "|---:|---|---:|---|"]
    for score, route_id, macro, status in below[:60]:
        report.append(f"| {route_id} | {macro} | {score:g} | {status} |")
    report += [
        "",
        "## 7. Macro Scenario Results",
        "",
        "| macro | mean | min | pass_count | route_count | macro_pass_80 | completed_pass_count |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for macro, mean, min_score, pass_count, route_count, macro_pass, completed_pass in macro_table:
        report.append(f"| {macro} | {mean} | {min_score} | {pass_count} | {route_count} | {macro_pass} | {completed_pass} |")
    report += [
        "",
        "## 8. Regression",
        "",
        "- No new regression run has completed in this RUN_ROOT yet.",
        "- Frozen route regression is still pending; current queue prioritizes below-80 routes first.",
        "- Regression route list is recorded in `regression_routes.txt`.",
        "",
        "## 9. Full144 Status",
        "",
        "- No complete full144 result has been run inside `v5_parallel_80_runs` yet.",
        "- `full144_latest_result.json` is therefore a status file, not a fabricated full144 score.",
        "- Latest rigorous full144-like baselines remain the historical strict results listed above.",
        "",
        "## 10. Known Risks",
        "",
        "- `test_cvci_auxiliary_system.py` still has known behavior-expectation failures; LiDAR/geometry/tracker tests pass.",
        "- YELLOW_PASS_80 routes reach score_challenge >=80 but are not Completed; they are frozen only under the current throughput policy and require final review.",
        "- Unstable families remain: trucks_encountered_during_construction, reverse_vehicle, high_speed_reckless_lane_cutting, blind_spot_hidden_car, roundabout/cutin/crazy bike variants.",
        "- Route prior remains off by default for submit-mode runs.",
        "",
        "## 11. How To Run",
        "",
        "Single route attempts are represented by per-attempt `command.txt` files under `attempts/*/`.",
        "",
        "Three-GPU queue:",
        "",
        "```bash",
        "cd /root/autodl-tmp/projects/code/DriveTransformer",
        "/root/miniconda3/envs/drivetransformer/bin/python tools/run_cvci_parallel_80_queue.py \\",
        "  --queue /home/carla/cvci_back/v5_parallel_80_runs/parallel_80_queue.json \\",
        "  --gpus 0,1,2 \\",
        "  --carla-graphics-adapters 2,2,0 \\",
        "  --run-root /home/carla/cvci_back/v5_parallel_80_runs \\",
        "  --dt-root /root/autodl-tmp/projects/code/DriveTransformer \\",
        "  --cvci-root /root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark \\",
        "  --xml /root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml \\",
        "  --ckpt /root/autodl-tmp/projects/code/DriveTransformer/runs/cvci_drivetransformer_train/work_dirs/cvci_iter20000_unfreeze_last13_to40000/iter_25000.pth \\",
        "  --threshold 80 --max-attempts-per-route 3 --freeze-on-pass --no-idle",
        "```",
        "",
        "## 12. Final Conclusion",
        "",
        f"- Current frozen under 80-pass policy, including runtime overlay: {len(merged_frozen)} / 144.",
        f"- Current needs-fix list after runtime overlay: {len(merged_needs)} routes.",
        f"- New queue results so far: {len(new_pass)} routes reached 80+ in this RUN_ROOT.",
        "- This is an active phase report; completion requires the queue to finish, failed/below-80 routes to be exhausted or retried, and a final full144 or declared substitute run to be produced.",
        "",
    ]
    (RUN / "final_80_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    append = [
        "",
        "## Progress Update",
        "",
        f"- time: {now}",
        "",
        "### GPU utilization",
        "",
        "| worker | gpu | carla_adapter | route | status | elapsed | next |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for worker in sorted(workers, key=lambda item: item["worker"]):
        append.append(f"| {worker['worker']} | {worker['gpu']} | {worker.get('carla_adapter', '?')} | {worker['route']} | RUNNING | {worker['elapsed']} | continue current attempt |")
    if not workers:
        append.append("| - | - | - | - | IDLE_OR_DONE | - | inspect scheduler |")
    append += ["", "### Newly PASS_80", ""]
    if new_pass:
        for row in new_pass[-10:]:
            append.append(f"- route {row.get('route_id')} | {row.get('macro_scenario')} | score={row.get('score_challenge')} | status={row.get('status')}")
    else:
        append.append("- none in current RUN_ROOT")
    append += ["", "### Newly frozen", ""]
    if new_pass:
        for row in new_pass[-10:]:
            append.append(f"- route {row.get('route_id')} | next={row.get('next_action')}")
    else:
        append.append("- none")
    append += ["", "### Still below 80", ""]
    for score, route_id, macro, status in below[:20]:
        append.append(f"- route {route_id} | {macro} | best={score:g} | status={status}")
    append += [
        "",
        "### Current code changes",
        "",
        "- Scheduler launch uses `--carla-graphics-adapters 2,2,0` for this host.",
        "- Worker status aggregation is owned by `tools/update_cvci_80_reports.py`; future scheduler starts write `worker_status_worker*.json` per worker.",
        "- Future scheduler starts type run_matrix failures from evaluator infractions and print CVCI_AUX debug every 100 frames; the current already-running scheduler predates that source edit.",
        "- Future queue rebuilds read `environment_deprioritized_routes.txt` and postpone likely environment/startup artifact routes such as high-z `static.unknown` collisions.",
        "",
        "### Regression status",
        "",
        "- No completed regression sweep yet in this RUN_ROOT.",
        "",
        "### Next queue",
        "",
    ]
    for row in queue_rows[:12]:
        append.append(f"- route {row.get('route_id')} | {row.get('macro_scenario')} | best={row.get('best_score')} | state={row.get('pass_state')}")
    with (RUN / "progress.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(append) + "\n")

    print(f"updated {RUN}")
    print(f"run_rows={len(run_rows)} new_pass={len(new_pass)} below={len(below)} workers={len(workers)}")


if __name__ == "__main__":
    main()
