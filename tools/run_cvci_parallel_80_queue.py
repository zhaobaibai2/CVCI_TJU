#!/usr/bin/env python3
"""Run a three-GPU CVCI 80-pass queue with per-route locks and logs."""

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = PROJECT_ROOT / "weights" / "iter_25000.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cvci_parallel_80"


def _has_entries(value):
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_has_entries(v) for v in value.values())
    return bool(value)


def summarize_infractions(record):
    status_text = str(record.get("status", "") or "").lower()
    infractions = record.get("infractions", {}) or {}
    collision_keys = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
    collision = any(_has_entries(infractions.get(key)) for key in collision_keys)
    timeout = (
        _has_entries(infractions.get("scenario_timeouts"))
        or _has_entries(infractions.get("route_timeout"))
        or "tickruntime" in status_text
        or "timeout" in status_text
    )
    blocked = _has_entries(infractions.get("vehicle_blocked")) or "blocked" in status_text
    route_deviation = _has_entries(infractions.get("route_dev")) or "deviated" in status_text
    red_light = _has_entries(infractions.get("red_light")) or "red light" in status_text
    lane_invasion = _has_entries(infractions.get("outside_route_lanes")) or "outside route" in status_text
    return {
        "collision": collision,
        "timeout": timeout,
        "blocked": blocked,
        "red_light": red_light,
        "route_deviation": route_deviation,
        "lane_invasion": lane_invasion,
    }


def route_score_from_json(path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(errors="ignore"))
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
    if not records:
        return None
    rec = records[-1]
    scores = rec.get("scores", {}) or {}
    result = {
        "score_challenge": float(scores.get("score_challenge", 0.0) or 0.0),
        "score_composed": scores.get("score_composed", ""),
        "score_route": scores.get("score_route", ""),
        "score_penalty": scores.get("score_penalty", ""),
        "status": rec.get("status", ""),
        "completed": str(rec.get("status", "")).lower() in ("completed", "perfect"),
        "raw": rec,
    }
    result.update(summarize_infractions(rec))
    return result



def carla_adapter_for_worker(args, worker_id, gpu_id):
    adapters = [x.strip() for x in str(args.carla_graphics_adapters).split(",") if x.strip()]
    if worker_id < len(adapters):
        return adapters[worker_id]
    return str(gpu_id)

def git_diff_hash(dt_root):
    try:
        diff = subprocess.check_output(["git", "diff"], cwd=dt_root, stderr=subprocess.DEVNULL)
    except Exception:
        diff = b""
    return hashlib.sha1(diff).hexdigest()[:12]


def select_route_xml(master_xml, route_id, out_xml):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(master_xml, parser=parser).getroot()
    new_root = ET.Element("routes")
    keep_comments = []
    for child in list(root):
        if child.tag is ET.Comment:
            keep_comments.append(child)
            continue
        if child.tag != "route":
            continue
        if str(child.attrib.get("id")) == str(route_id):
            for comment in keep_comments[-3:]:
                new_root.append(comment)
            new_root.append(child)
            break
    if not list(new_root):
        raise RuntimeError(f"route {route_id} not found in {master_xml}")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(new_root).write(out_xml, encoding="utf-8", xml_declaration=True)


def ensure_symlink(link, target):
    try:
        if link.is_symlink():
            try:
                if link.resolve() == target.resolve():
                    return
            except FileNotFoundError:
                pass
            link.unlink()
        elif link.exists():
            return
        link.symlink_to(target)
    except FileExistsError:
        return


def write_csv_row(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def worker(worker_id, gpu_id, tasks, args, lock, stop_event):
    dt_root = Path(args.dt_root)
    cvci_root = Path(args.cvci_root)
    run_root = Path(args.run_root)
    port = args.base_port + worker_id * args.port_stride
    tm_port = args.base_tm_port + worker_id * args.port_stride
    carla_adapter = carla_adapter_for_worker(args, worker_id, gpu_id)
    worker_status = run_root / f"worker_status_worker{worker_id}.json"
    while not stop_event.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            break
        route_id = str(task["route_id"])
        macro = task.get("macro_scenario", "unknown")
        attempt_id = f"{int(time.time())}_w{worker_id}_r{route_id}"
        attempt_dir = run_root / "attempts" / attempt_id
        route_xml = attempt_dir / "route.xml"
        result_json = attempt_dir / "result.json"
        log_path = attempt_dir / "leaderboard.log"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        status = "FAILED_RUN"
        score = ""
        completed = False
        pass_80 = False
        next_action = "RETRY_LATER"
        start = time.time()
        with lock:
            worker_status.write_text(
                json.dumps(
                    {
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "carla_adapter": carla_adapter,
                        "route_id": route_id,
                        "macro_scenario": macro,
                        "status": "RUNNING",
                        "attempt_id": attempt_id,
                        "port": port,
                        "traffic_manager_port": tm_port,
                        "log_path": str(log_path),
                        "result_json_path": str(result_json),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        try:
            select_route_xml(Path(args.xml), route_id, route_xml)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "CARLA_ROOT": args.carla_root,
                "CARLA_SERVER": str(Path(args.carla_root) / "CarlaUE4.sh"),
                "CVCI_CARLA_GRAPHICS_ADAPTER": str(carla_adapter),
                "CVCI_CARLA_CUDA_VISIBLE": str(gpu_id),
                "SCENARIO_RUNNER_ROOT": str(cvci_root / "scenario_runner"),
                "LEADERBOARD_ROOT": str(cvci_root / "leaderboard"),
                "CHALLENGE_TRACK_CODENAME": "SENSORS",
                "IS_BENCH2DRIVE": "True",
                "DISABLE_BEV_SENSOR": args.disable_bev_sensor,
                "CVCI_ALLOW_ROUTE_PRIOR": args.allow_route_prior,
                "CVCI_FORCE_MACRO_SCENARIO": macro if str(args.allow_route_prior).lower() in ("1", "true", "yes", "on") else "",
                "CVCI_LIDAR_ENABLED": "1",
                "CVCI_AUXILIARY_PERCEPTION_ENABLED": "1",
                "CVCI_LEGACY_DETECTION_RULES_ENABLED": "0",
                "CVCI_REVERSE_VEHICLE_RULE_ENABLED": str(args.reverse_vehicle_rule),
                "CVCI_AUX_LOG_PERIOD": str(args.aux_log_period),
                "PYTHONPATH": ":".join([
                    str(cvci_root),
                    str(dt_root),
                    str(dt_root / "adzoo"),
                    str(Path(args.carla_root) / "PythonAPI"),
                    str(Path(args.carla_root) / "PythonAPI/carla"),
                    str(Path(args.carla_root) / "PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"),
                    str(cvci_root / "leaderboard"),
                    str(cvci_root / "scenario_runner"),
                    env.get("PYTHONPATH", ""),
                ]),
            })
            for link_name, target in {
                "DriveTransformer": dt_root,
                "adzoo": dt_root / "adzoo",
                "team_code": dt_root / "team_code",
            }.items():
                ensure_symlink(cvci_root / link_name, target)
            cmd = [
                args.python,
                "leaderboard/leaderboard/leaderboard_evaluator.py",
                f"--routes={route_xml}",
                "--repetitions=1",
                "--track=SENSORS",
                f"--checkpoint={result_json}",
                f"--agent={dt_root / 'team_code/drivetransformer_b2d_agent.py'}",
                f"--agent-config={dt_root / 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py'}+{args.ckpt}",
                "--debug=0",
                "--resume=False",
                f"--port={port}",
                f"--traffic-manager-port={tm_port}",
                "--client-timeout=300",
                "--scenario-timeout=300",
                "--agent-timeout=120",
                "--gpu-rank=0",
            ]
            (attempt_dir / "command.txt").write_text(
                f"CUDA_VISIBLE_DEVICES={gpu_id} CVCI_CARLA_CUDA_VISIBLE={gpu_id} CVCI_CARLA_GRAPHICS_ADAPTER={carla_adapter} CVCI_REVERSE_VEHICLE_RULE_ENABLED={args.reverse_vehicle_rule} CVCI_AUX_LOG_PERIOD={args.aux_log_period} "
                + " ".join(str(x) for x in cmd) + "\n",
                encoding="utf-8",
            )
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=cvci_root, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                rc = proc.wait()
            result = route_score_from_json(result_json)
            if result:
                score = result["score_challenge"]
                completed = result["completed"]
                pass_80 = score >= args.threshold
                status = "PASS_80" if pass_80 and completed else ("YELLOW_PASS_80" if pass_80 else "NEEDS_FIX")
                next_action = "FROZEN" if pass_80 else "RETRY_LATER"
            elif rc == 0:
                status = "FAILED_RUN"
            cleanup_carla(port)
        except Exception as exc:
            (attempt_dir / "worker_exception.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            cleanup_carla(port)
        elapsed = int(time.time() - start)
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "port": port,
            "route_id": route_id,
            "macro_scenario": macro,
            "attempt_id": attempt_id,
            "rule_version": args.rule_version,
            "git_diff_hash": git_diff_hash(dt_root),
            "score_challenge": score,
            "status": status,
            "completed": completed,
            "collision": "",
            "timeout": "",
            "blocked": "",
            "red_light": "",
            "route_deviation": "",
            "lane_invasion": "",
            "pass_80": pass_80,
            "frozen": pass_80,
            "log_path": str(log_path),
            "result_json_path": str(result_json),
            "next_action": next_action,
            "elapsed_sec": elapsed,
        }
        with lock:
            write_csv_row(run_root / "run_matrix.csv", list(row.keys()), row)
            append_progress(run_root, row)
        tasks.task_done()


def cleanup_carla(port):
    try:
        out = subprocess.check_output(["bash", "-lc", f"ps -ef | grep 'carla-rpc-port={port}' | grep -v grep | awk '{{print $2}}'"], text=True)
        pids = [p for p in out.split() if p.isdigit()]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass


def append_progress(run_root, row):
    path = run_root / "progress.md"
    with path.open("a", encoding="utf-8") as f:
        f.write("\n## Progress Update\n\n")
        f.write(f"- time: {row['timestamp']}\n")
        f.write(f"- worker: {row['worker_id']} gpu={row['gpu_id']} route={row['route_id']} macro={row['macro_scenario']}\n")
        f.write(f"- result: status={row['status']} score={row['score_challenge']} completed={row['completed']} pass_80={row['pass_80']}\n")
        f.write(f"- log: {row['log_path']}\n")
        f.write(f"- next: {row['next_action']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--run-root", default=os.environ.get("RUN_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    ap.add_argument("--dt-root", default=os.environ.get("DT_ROOT", str(PROJECT_ROOT)))
    ap.add_argument("--cvci-root", default=os.environ.get("CVCI_ROOT", str(PROJECT_ROOT.parent / "CVCI_Benchmark" / "CVCI_BenchMark")))
    ap.add_argument(
        "--xml",
        default=os.environ.get(
            "BASE_ROUTES",
            str(PROJECT_ROOT.parent / "CVCI_Benchmark" / "CVCI_BenchMark" / "runs" / "drivetransformer_large_cvci_full" / "routes" / "CVCI_BenchMark.xml"),
        ),
    )
    ap.add_argument("--ckpt", default=os.environ.get("CKPT_PATH", str(DEFAULT_CKPT)))
    ap.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", str(PROJECT_ROOT.parent / "carla")))
    ap.add_argument("--python", default=os.environ.get("PYTHON_BIN", "python"))
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--max-attempts-per-route", type=int, default=3)
    ap.add_argument("--max-routes", type=int, default=0)
    ap.add_argument("--base-port", type=int, default=45700)
    ap.add_argument("--base-tm-port", type=int, default=64700)
    ap.add_argument("--port-stride", type=int, default=50)
    ap.add_argument("--rule-version", default="parallel80_initial")
    ap.add_argument("--allow-route-prior", default="0")
    ap.add_argument("--disable-bev-sensor", default="1")
    ap.add_argument("--reverse-vehicle-rule", default="0", help="Set CVCI_REVERSE_VEHICLE_RULE_ENABLED for route-level reverse-vehicle diagnostics/tuning.")
    ap.add_argument("--aux-log-period", default="100", help="Print CVCI auxiliary-system debug state every N frames; 0 disables.")
    ap.add_argument("--carla-graphics-adapters", default="2,2,0", help="Comma-separated CVCI_CARLA_GRAPHICS_ADAPTER values per worker; this host is not physical-index aligned.")
    ap.add_argument("--freeze-on-pass", action="store_true")
    ap.add_argument("--no-idle", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    tasks_data = json.loads(Path(args.queue).read_text())
    selected = []
    attempts = Counter()
    for task in tasks_data:
        rid = str(task["route_id"])
        if attempts[rid] >= args.max_attempts_per_route:
            continue
        selected.append(task)
        attempts[rid] += 1
        if args.max_routes and len(selected) >= args.max_routes:
            break
    q = queue.Queue()
    for task in selected:
        q.put(task)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    lock = threading.Lock()
    stop_event = threading.Event()
    (run_root / "gpu_worker_status.md").write_text("# GPU Worker Status\n\ninitializing; live summaries are maintained by tools/update_cvci_80_reports.py\n", encoding="utf-8")
    threads = []
    for idx, gpu in enumerate(gpus):
        t = threading.Thread(target=worker, args=(idx, gpu, q, args, lock, stop_event), daemon=False)
        t.start()
        threads.append(t)
        time.sleep(3)
    for t in threads:
        t.join()
    with (run_root / "gpu_worker_status.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- finished_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- remaining_queue: {q.qsize()}\n")
    print(f"finished selected_routes={len(selected)} run_matrix={run_root / 'run_matrix.csv'}")


if __name__ == "__main__":
    main()
