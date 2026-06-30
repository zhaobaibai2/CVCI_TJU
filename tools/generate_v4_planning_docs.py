#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-table", default="/home/carla/cvci_back/cvci_route_scenario_table_20260624.csv")
    ap.add_argument("--macro-analysis", default="/home/carla/cvci_back/cvci_v3_macro_score_analysis_20260624.csv")
    ap.add_argument("--merged", default="/home/carla/cvci_back/cvci_v3_route_macro_merged_20260624.csv")
    ap.add_argument("--out-dir", default="/home/carla/cvci_back")
    ap.add_argument("--date", default="20260624")
    args = ap.parse_args()
    routes = read_csv(args.route_table)
    analysis = read_csv(args.macro_analysis)
    merged = read_csv(args.merged)
    by_macro = defaultdict(list)
    for row in routes:
        by_macro[row["macro_scenario"]].append(row)
    score_by_macro = {row["macro_scenario"]: row for row in analysis}
    merged_by_macro = defaultdict(list)
    for row in merged:
        merged_by_macro[row["macro_scenario"]].append(row)

    out = Path(args.out_dir)
    run_root = out / "v4_scenario_rule_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    write_commands(out / f"cvci_v4_closed_loop_commands_{args.date}.md")
    write_plan(out / f"cvci_v4_plan_before_running_{args.date}.md", analysis)
    write_rule_cards(out / f"cvci_v4_scenario_rule_cards_{args.date}.md", by_macro, score_by_macro, merged_by_macro)
    write_run_skeletons(run_root, args.date)
    print(out / f"cvci_v4_plan_before_running_{args.date}.md")
    print(out / f"cvci_v4_scenario_rule_cards_{args.date}.md")


def write_commands(path):
    path.write_text("""# CVCI v4 Closed-loop Commands

## 已定位的真实闭环入口

本机同步脚本来源：

`/home/carla/cvci_back/detection_rules_v3_code_20260622/cvci_drivetransformer_files/run_cvci_drivetransformer_closed_loop.sh`

远端预期路径：

`/root/autodl-tmp/projects/cvci_drivetransformer_files/run_cvci_drivetransformer_closed_loop.sh`

该脚本调用的是 CVCI `leaderboard/leaderboard/leaderboard_evaluator.py`，参数包含 `--routes`、`--checkpoint`、`--agent`、`--agent-config`、CARLA 端口和 resume，因此属于 CVCI closed-loop/CARLA 评测，不是 DriveTransformer 离线评测。

## 单 route / route 子集模板

```bash
export DT_ROOT=/root/autodl-tmp/projects/code/DriveTransformer
export CVCI_ROOT=/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark
export RUN_DIR=/home/carla/cvci_back/v4_scenario_rule_runs/<run_name>
export BASE_ROUTES=/home/carla/cvci_back/v4_scenario_rule_runs/routes/<subset>.xml
export GPU_LIST=0,1,2
export TASK_NUM=3
export FORCE_SPLIT=1
bash /root/autodl-tmp/projects/cvci_drivetransformer_files/run_cvci_drivetransformer_closed_loop.sh
```

## full 144 模板

```bash
export DT_ROOT=/root/autodl-tmp/projects/code/DriveTransformer
export CVCI_ROOT=/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark
export BASE_ROUTES=/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml
export RUN_DIR=/home/carla/cvci_back/v4_scenario_rule_runs/full144_<version>
export GPU_LIST=0,1,2
export TASK_NUM=3
export FORCE_SPLIT=1
bash /root/autodl-tmp/projects/cvci_drivetransformer_files/run_cvci_drivetransformer_closed_loop.sh
```

## 当前限制

本轮本机免密 SSH 到 `root@connect.cqa1.seetacloud.com:45837` 返回 `Permission denied`，所以以上命令尚未在远端实时复核和启动。
""", encoding="utf-8")


def write_plan(path, analysis):
    lowest = analysis[0]
    high = [r for r in analysis if float(r["score_mean"]) >= 90]
    low = [r for r in analysis if float(r["score_mean"]) < 90]
    lines = [
        "# CVCI v4 Plan Before Running",
        "",
        "## 1. 路径确认",
        "",
        "- DriveTransformer 主代码路径: `/root/autodl-tmp/projects/code/DriveTransformer`",
        "- 本机同步镜像: `/home/carla/cvci_back/detection_rules_v3_code_20260622/DriveTransformer`",
        "- CVCI 闭环测试路径: `/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark`",
        "- CVCI XML: `/root/autodl-tmp/projects/CVCI_Benchmark/CVCI_BenchMark/runs/drivetransformer_large_cvci_full/routes/CVCI_BenchMark.xml`",
        "- 本机 XML 证据: `/home/carla/cvci_back/detection_rules_v3_full144_video_20260623/split_routes/CVCI_BenchMark.xml`",
        "",
        "## 2. v3 大类得分排序",
        "",
        "| rank | macro | mean | min | <90 | sampled routes | needs rule |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(analysis, 1):
        lines.append(f"| {i} | {r['macro_scenario']} | {float(r['score_mean']):.3f} | {float(r['score_min']):.3f} | {r['count_score_lt_90']} | {r['sampled_route_candidates']} | {r['needs_rule']} |")
    lines += [
        "",
        "## 3. 暂不加规则的大类",
        "",
        ", ".join(r["macro_scenario"] for r in high) if high else "无。",
        "",
        "## 4. 需要专项规则的大类",
        "",
        ", ".join(r["macro_scenario"] for r in low) if low else "无。",
        "",
        "## 5. 第一轮处理目标",
        "",
        f"- 最低分大类: `{lowest['macro_scenario']}`",
        f"- v3 mean/min: {float(lowest['score_mean']):.3f}/{float(lowest['score_min']):.3f}",
        f"- 固定 seed=20260624 的调试 route 候选: `{lowest['sampled_route_candidates']}`",
        "- 第一版规则方向: 只针对该 macro scenario 加窄触发 override，保留 v3 fallback；优先处理主要 infractions，不按 route id 写死动作。",
        "",
        "## 6. 回归保护",
        "",
        "- 当前大类 sampled routes + 当前大类剩余 routes。",
        "- 已修好大类 sampled routes。",
        "- v3 原本高分大类至少 2 条 route。",
        "- 若 override 误触发导致 90+ route 明显下降，则收紧 macro/phase/risk/confidence/exit 条件或关闭该规则。",
        "",
        "## 7. 需要修改的 DriveTransformer 文件清单",
        "",
        "- `team_code/drivetransformer_b2d_agent.py`: 接入 v4 ScenarioRuleRegistry 或调用点。",
        "- `team_code/cvci_scenario_context.py`: ScenarioContext/RuleAction 数据结构。",
        "- `team_code/cvci_rule_config.py`: 集中配置。",
        "- `team_code/cvci_scenario_rules_v4.py`: 各 macro scenario override。",
        "- `team_code/cvci_scenario_classifier.py`: 运行时 macro/phase/risk 判断。",
        "",
        "## 8. 当前阻断",
        "",
        "本轮本机到远端 `root@connect.cqa1.seetacloud.com:45837` 免密 SSH 不可用，无法把代码写入远端主仓库或启动新的 CVCI closed-loop。已先完成本地镜像侧解析/分析/计划产物。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rule_cards(path, by_macro, score_by_macro, merged_by_macro):
    lines = ["# CVCI v4 Scenario Rule Cards", ""]
    for macro in sorted(by_macro, key=lambda m: min(int(r["route_id"]) for r in by_macro[m])):
        score = score_by_macro.get(macro, {})
        routes = sorted(int(r["route_id"]) for r in by_macro[macro])
        needs = score.get("needs_rule", "unknown")
        failures = []
        for key in ("collision", "stuck_blocking", "red_light", "lane_invasion", "route_deviation", "timeout"):
            if score.get(key) and int(float(score[key])):
                failures.append(f"{key}={score[key]}")
        lines += [
            f"## Macro Scenario: {macro}",
            "",
            "### Routes",
            ",".join(str(r) for r in routes),
            "",
            "### V3 closed-loop score status",
            f"- mean: {score.get('score_mean', 'NA')}",
            f"- min: {score.get('score_min', 'NA')}",
            f"- median: {score.get('score_median', 'NA')}",
            f"- below_90: {score.get('count_score_lt_90', 'NA')}",
            f"- main failures: {'; '.join(failures) or 'none recorded'}",
            "",
            "### Whether extra rule is needed",
            "Yes" if needs == "yes" else "No - keep v3 only unless later full144 regression indicates otherwise.",
            "",
            "### Runtime detection",
            "Allowed signals:",
            "- route metadata if legally available: macro scenario name only, no route-id action sequence",
            "- camera detection: detected vehicle/pedestrian/bike/static obstacle boxes and confidence",
            "- radar/lidar: object distance/relative motion if exposed by the official agent input",
            "- ego speed: yes",
            "- route command: yes",
            "- local waypoint geometry: yes",
            "- weather/light bucket: yes",
            "",
            "Forbidden:",
            "- NPC true transform",
            "- simulator actor list",
            "- ground-truth boxes",
            "- ground-truth labels",
            "",
            "### V4 override",
            rule_direction(macro, needs),
            "",
            "### Activation",
            f"- macro_scenario == {macro}",
            "- confidence >= scenario-specific threshold",
            "- risk flags from legal detections/waypoints/ego state",
            "- phase in approach/interaction/recovery",
            "",
            "### Exit",
            "- risk cleared for N frames",
            "- passed interaction zone or route command changed",
            "- timeout guard",
            "- fallback to v3 when detection is empty or confidence is low",
            "",
            "### Regression risk",
            "- Which good scenarios could be harmed: v3 high-score scenarios sharing vehicle/obstacle/pedestrian detections",
            "- How to protect them: macro-specific activation, confidence threshold, phase gates, short hysteresis, enabled switch",
            "",
            "### Test plan",
            f"- sampled routes: {score.get('sampled_route_candidates', '')}",
            f"- remaining routes: {','.join(str(r) for r in routes if str(r) not in str(score.get('sampled_route_candidates', '')).split(','))}",
            "- regression routes: sampled routes from already solved macros plus at least 2 v3 high-score routes",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def rule_direction(macro, needs):
    if needs != "yes":
        return "- target_speed/brake/throttle/steer: no extra override in first pass; v3 fallback only."
    if "missing" in macro:
        return "- target_speed adjustment: lower approach/interact speed when vehicle risk appears\n- brake/yield condition: TTC or close front vehicle\n- following distance/TTC threshold: conservative vehicle hysteresis\n- lateral behavior: keep v3 steering unless trajectory anomaly\n- creep/recovery: only when front corridor clear\n- special safeguards: cap hold frames to avoid timeout"
    if "construction" in macro or "closure" in macro or "barrier" in macro:
        return "- target_speed adjustment: obstacle-sensitive slow zone\n- brake/yield condition: front obstacle in path corridor\n- following distance/TTC threshold: obstacle distance and speed-scaled stopping\n- lateral behavior: steer smoothing and steer_scale cap\n- creep/recovery: bounded low-speed recovery if blocked and corridor clear\n- special safeguards: no off-road encouragement"
    if "roundabout" in macro:
        return "- target_speed adjustment: approach/yield/enter speed phases\n- brake/yield condition: side/front vehicle risk before entering\n- following distance/TTC threshold: conservative yield TTC\n- lateral behavior: smooth entry, avoid full stop after entry\n- creep/recovery: max-yield wait then creep if clear\n- special safeguards: exit phase restores v3"
    if "bike" in macro or "pedestrian" in macro:
        return "- target_speed adjustment: future-path vulnerable-user speed cap\n- brake/yield condition: pedestrian/bike in path corridor\n- following distance/TTC threshold: short TTC brake plus clear hysteresis\n- lateral behavior: v3 steering with emergency brake priority\n- creep/recovery: clear-only release\n- special safeguards: static non-path detections do not hold forever"
    return "- target_speed adjustment: conservative interaction speed\n- brake/yield condition: legal detection risk only\n- following distance/TTC threshold: dynamic with ego speed\n- lateral behavior: smooth steering under uncertainty\n- creep/recovery: bounded clear-corridor release\n- special safeguards: strict macro/phase/risk gates"


def write_run_skeletons(run_root, date):
    matrix = run_root / f"run_matrix_{date}.csv"
    if not matrix.exists():
        matrix.write_text("timestamp,diff_hash,rule_version,route_id,macro_scenario,run_type,score,completion,infraction,collision,timeout,stuck,off_road,red_light,lane_invasion,log_path,pass_90,failure_reason,next_adjustment\n", encoding="utf-8")
    (run_root / f"progress_{date}.md").write_text("# CVCI v4 Progress\n\n- status: prepared local analysis; remote closed-loop not started in this turn.\n", encoding="utf-8")
    (run_root / f"failures_{date}.md").write_text("# CVCI v4 Failures\n\n待 closed-loop 运行后追加。\n", encoding="utf-8")
    (run_root / f"solved_macro_scenarios_{date}.md").write_text("# Solved Macro Scenarios\n\n待 closed-loop 运行后追加。\n", encoding="utf-8")
    full = run_root / f"full_144_results_{date}.csv"
    if not full.exists():
        full.write_text("route_id,macro_scenario,score,completion,status,main_failure,log_path\n", encoding="utf-8")


if __name__ == "__main__":
    main()
