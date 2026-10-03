"""统计配对干预的实际作用；可以只读分析仍在生成中的完整日志行。"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def read_traces(path):
    if not path.exists():
        return []
    content = path.read_text()
    # 写入进程可能正写最后一行；仅分析已经以换行符结束的记录。
    return [json.loads(line) for line in content.splitlines(keepends=True) if line.endswith("\n")]


def summarize(rows):
    if not rows:
        return {"count": 0}
    improved = [row["best_score"] < row["baseline_score"] - 1e-5 for row in rows]
    failed_before = [row for row in rows if not row["before"]["quality_pass"]]
    relative = [(row["baseline_score"] - row["best_score"]) / max(row["baseline_score"], 1e-8) for row in rows]
    early_better = late_better = local_better = 0
    comparisons = 0
    for row in rows:
        # 同一失败初值的多种干预可有不同来源；按动作家族取各自最优的真实推演分数。
        scores = {}
        for family, score in zip(row["candidate_families"][1:], row["candidate_scores"][1:]):
            scores[family] = min(scores.get(family, float("inf")), score)
        if len(row["candidate_switch_states"]) and row["before"].get("switch_count") == 1:
            early = min(scores["early_steering"], scores["early_speed"])
            late = min(scores["late_steering"], scores["late_speed"])
            local = min(scores["local_steering"], scores["local_speed"])
            comparisons += 1
            early_better += early + 1e-5 < min(late, local)
            late_better += late + 1e-5 < min(early, local)
            local_better += local + 1e-5 < min(early, late)
    newly_passed = sum(row["after"]["quality_pass"] for row in failed_before)
    return {"count": len(rows), "unique_tasks": len({row["sample_id"] for row in rows}),
            "initial_passed": sum(row["before"]["quality_pass"] for row in rows),
            "selected_passed": sum(row["after"]["quality_pass"] for row in rows),
            "failed_initials": len(failed_before), "newly_passed": newly_passed,
            "repair_fraction_among_failed": newly_passed / len(failed_before) if failed_before else None,
            "valid_to_invalid": sum(row["before"]["quality_pass"] and not row["after"]["quality_pass"] for row in rows),
            "proxy_improved": sum(improved), "median_relative_proxy_improvement": float(np.median(relative)),
            "selected_families": dict(Counter(row["selected_family"] for row in rows)),
            "two_phase_comparisons": comparisons,
            "early_strictly_better_than_late_and_local": early_better,
            "late_strictly_better_than_early_and_local": late_better,
            "local_strictly_better_than_early_and_late": local_better,
            "before_collision_failures": sum(not row["before"].get("safe", False) for row in rows),
            "after_collision_failures": sum(not row["after"].get("safe", False) for row in rows),
            "before_terminal_failures": sum(not row["before"].get("terminal_ok", False) for row in rows),
            "after_terminal_failures": sum(not row["after"].get("terminal_ok", False) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/event_repair_data_analysis.json"))
    args = parser.parse_args()
    metadata = json.loads((args.data_dir / "manifest.json").read_text())
    result = {"generation_complete": metadata["complete"], "splits": {},
              "interpretation": "这是离线候选干预的统计，不是神经网络效果；多轮记录共享任务，不能当作独立任务数。"}
    for split in ("train", "val"):
        rows = read_traces(args.data_dir / (split + "_interventions.jsonl"))
        grouped = {"overall": summarize(rows)}
        for key in ("source", "variant", "round"):
            grouped["by_" + key] = {str(value): summarize([row for row in rows if row[key] == value])
                                    for value in sorted({row[key] for row in rows})}
        grouped["source_variant_counts"] = dict(Counter(row["source"] + ":" + row["variant"] for row in rows))
        result["splits"][split] = grouped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"complete": result["generation_complete"],
                      "splits": {key: value["overall"] for key, value in result["splits"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
