"""在独立地图上报告完整规划结果，按场景与机动类别统计。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from config import data_dir
from data_preparation import read_index_records, read_map_data
from train import load_model, choose_device
from plan import plan_task
from planner_backend import optimize_path


def summarize(rows):
    # 第1步：把失败任务计入分母；成功轨迹的长度、间隙另行统计。
    result = {"task_count": len(rows), "success_rate": 0.0, "failure_reasons": {}}
    if not rows:
        return result
    result["success_rate"] = sum(row["quality_pass"] for row in rows) / len(rows)
    for key in ["total_seconds", "network_seconds", "optimizer_seconds", "iterations",
                "position_error_m", "heading_error_deg", "articulation_error_deg",
                "path_length_m", "minimum_clearance_m", "path_heading_error_p95_deg"]:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        if values:
            result[key] = {"count": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    for row in rows:
        if not row["quality_pass"]:
            reason = row["reason"]
            result["failure_reasons"][reason] = result["failure_reasons"].get(reason, 0) + 1
    return result


def main():
    parser = argparse.ArgumentParser(description="在固定测试任务上比较网络初值与原始初值")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=100, help="0表示完整集合")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compare-original", action="store_true")
    parser.add_argument("--network-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args()
    if args.limit < 0 or args.time_limit <= 0:
        parser.error("limit不能为负，time-limit必须为正")
    if args.network_only and args.compare_original:
        parser.error("network-only不执行优化，不能同时比较优化初值")
    torch.set_num_threads(2)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    records = read_index_records(args.data_dir, args.split)
    if args.limit:
        indices = np.linspace(0, len(records) - 1, min(args.limit, len(records)), dtype=int)
        records = [records[index] for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_rows = []
    original_rows = []
    with (args.output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            with np.load(args.data_dir / record["sample_path"], allow_pickle=False) as sample:
                start = sample["start_state"].astype(float)
                goal = sample["goal_state"].astype(float)
            map_data = read_map_data(args.data_dir, record)
            scene = "open_work_area" if "open_work_area" in record["source_task_id"] else "mixing_station"
            row = {"sample_id": record["sample_id"], "map_id": record["map_id"],
                   "scene": scene, "maneuver": record["maneuver"], "quality_pass": False}
            task_started = time.perf_counter()
            try:
                route, states, directions, path, report = plan_task(model, map_data, start, goal, device,
                                                                   not args.network_only, args.time_limit)
                row.update({"quality_pass": report["quality_pass"], "reason": report["reason"],
                            "total_seconds": report["total_seconds"], "network_seconds": report["network_seconds"]})
                raw_rows.append(dict(row, **{"network_quality": report["network_quality"]}))
                if "optimizer" in report:
                    optimizer = report["optimizer"]
                    row.update({"optimizer_seconds": optimizer["optimizer_seconds"], "iterations": optimizer["iterations"]})
                    if path is not None:
                        row.update(optimizer["quality"])
                if args.compare_original:
                    # 同一任务、同一网络产生的引导路线和方向词、同一IPOPT预算，只改变初值。
                    # 此对照测量初值的作用，不等同于整个原生产系统的完整baseline。
                    original_path, original = optimize_path(states, directions, start, goal, map_data, args.time_limit, False)
                    original_row = {"sample_id": row["sample_id"], "map_id": row["map_id"],
                                    "scene": scene, "maneuver": record["maneuver"],
                                    "network_seconds": row["network_seconds"]}
                    original_row.update({"quality_pass": original_path is not None, "reason": original["reason"],
                                         "optimizer_seconds": original["optimizer_seconds"], "iterations": original["iterations"],
                                         "total_seconds": report["total_seconds"] - report["optimizer"]["optimizer_seconds"] + original["optimizer_seconds"]})
                    for key in ["position_error_m", "path_length_m", "minimum_clearance_m"]:
                        original_row.pop(key, None)
                    if original_path is not None:
                        original_row.update(original["quality"])
                    original_rows.append(original_row)
            except ValueError as error:
                row["reason"] = str(error)
                row["total_seconds"] = time.perf_counter() - task_started
                if args.compare_original:
                    original_rows.append(dict(row))
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(index + 1, "/", len(records), record["sample_id"], row["reason"], flush=True)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "split": args.split,
               "checkpoint_sha256": checkpoint["checkpoint_sha256"], "training_epoch": checkpoint["epoch"],
               "use_sequence_directions": checkpoint.get("use_sequence_directions", False),
               "couple_switch": checkpoint["couple_switch"], "clearance_weight": checkpoint["clearance_weight"],
               "scope": "accepted_task_distribution", "point_count": checkpoint["point_count"],
               "overall": summarize(rows), "by_scene": {}, "by_maneuver": {}}
    for scene in sorted({row["scene"] for row in rows}):
        summary["by_scene"][scene] = summarize([row for row in rows if row["scene"] == scene])
    for maneuver in sorted({row["maneuver"] for row in rows}):
        summary["by_maneuver"][maneuver] = summarize([row for row in rows if row["maneuver"] == maneuver])
    network_metrics = {}
    for key in ["path_length_m", "minimum_clearance_m", "path_heading_error_p95_deg",
                "position_error_m", "heading_error_deg", "articulation_error_deg"]:
        values = []
        for row in raw_rows:
            if key in row["network_quality"]:
                values.append(row["network_quality"][key])
        if values:
            network_metrics[key] = {"count": len(values), "median": float(np.median(values)),
                                    "p95": float(np.percentile(values, 95))}
    summary["network_geometry"] = {"evaluated": len(raw_rows), "safe_count": sum(row["network_quality"]["safe"] for row in raw_rows),
                                    "metrics": network_metrics,
                                    "note": "网络输出没有控制和时间，几何安全比例不能称为完整规划成功率；朝向与路径切线误差仅用于诊断"}
    if original_rows:
        summary["original_initialization"] = summarize(original_rows)
        summary["comparison_scope"] = "same_predicted_guide_and_direction_word_different_initialization"
        (args.output_dir / "original_tasks.json").write_text(json.dumps(original_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "network_tasks.json").write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("评估完成：", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
