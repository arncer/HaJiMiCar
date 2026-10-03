"""固定地图、初值和预算，对比神经编辑、消融与搜索修复；失败计入分母。"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from config import data_dir
from event_editor import load_editor
from repair_data import selected_records, read_task, network_seed
from repair_loop import run_repair, synchronize
from repair_physics import RepairProblem, body_variant, KNOT_COUNT
from train import load_model, choose_device


def summarize(rows):
    initial_failed = [row for row in rows if not row.get("initial_quality_pass", False)]
    result = {"count": len(rows), "passed": sum(row["quality_pass"] for row in rows),
              "initial_passed": sum(row.get("initial_quality_pass", False) for row in rows),
              "attempts_without_initial_pass": len(initial_failed),
              "passed_within_time_budget": sum(row.get("success_within_time_budget", False) for row in rows),
              "time_budget_overruns": sum(row.get("time_budget_overrun_seconds", 0) > 0 for row in rows),
              "newly_passed": sum(row["quality_pass"] for row in initial_failed),
              "failure_reasons": dict(Counter(row["reason"] for row in rows if not row["quality_pass"]))}
    result["success_rate"] = result["passed"] / len(rows) if rows else None
    result["success_rate_within_time_budget"] = result["passed_within_time_budget"] / len(rows) if rows else None
    result["repair_rate"] = result["newly_passed"] / len(initial_failed) if initial_failed else None
    for key in ("total_seconds", "repair_seconds", "common_seed_seconds", "setup_seconds",
                "physics_evaluations", "backward_calls", "independent_checks", "initial_proxy_score",
                "final_proxy_score", "position_error_m", "heading_error_deg", "articulation_error_deg",
                "time_budget_overrun_seconds"):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            result[key] = {"count": len(values), "mean": float(np.mean(values)),
                           "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/p104_batch512/best.pt"))
    parser.add_argument("--editor-checkpoint", type=Path, required=True)
    parser.add_argument("--no-evidence-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--variants", nargs="+", default=["nominal", "wide_slow", "long_slow"])
    parser.add_argument("--methods", nargs="+", choices=["neural", "no_evidence", "random", "gradient", "local"],
                        default=["neural", "no_evidence", "random", "gradient", "local"])
    parser.add_argument("--source", choices=["network", "perturbed_reference"], default="network",
                        help="参考轨迹扰动仅作为独立诊断，不能当作端到端部署结果")
    parser.add_argument("--evaluations", type=int, default=97)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--visualize", action="store_true", help="每个方法和身体条件各保存首个成功及失败回放")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="校验协议后保留已提交的逐任务结果，继续剩余任务")
    args = parser.parse_args()
    if args.limit < 0 or min(args.evaluations, args.rounds, args.candidates) < 1 or args.time_limit <= 0:
        parser.error("评估数量不能为负，各预算必须为正")
    if "no_evidence" in args.methods and args.no_evidence_checkpoint is None:
        parser.error("无证据消融需要独立训练的--no-evidence-checkpoint")
    if (args.output_dir / "tasks.jsonl").exists() and not args.resume:
        parser.error("评估目录已有结果，请使用新目录")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "evaluation.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for variant in args.variants:
        body_variant(variant)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    generator, generator_checkpoint = load_model(args.checkpoint, device)
    generator.eval()
    editor, checkpoint = load_editor(args.editor_checkpoint, device)
    models, checkpoints = {"neural": editor}, {"neural": checkpoint}
    if not checkpoint["use_evidence"]:
        raise ValueError("主模型必须使用失败证据")
    if args.no_evidence_checkpoint:
        models["no_evidence"], checkpoints["no_evidence"] = load_editor(args.no_evidence_checkpoint, device)
        if checkpoints["no_evidence"]["use_evidence"]:
            raise ValueError("消融checkpoint没有关闭失败证据")
        if checkpoints["no_evidence"]["manifest_sha256"] != checkpoint["manifest_sha256"]:
            raise ValueError("主模型与消融必须使用相同训练数据")
    for item in checkpoints.values():
        if item["base_checkpoint_sha256"] != generator_checkpoint["checkpoint_sha256"]:
            raise ValueError("生成器与编辑网络训练时使用的版本不同")
    manifest_path = Path(checkpoint["data_dir"]) / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != checkpoint["manifest_sha256"]:
        raise ValueError("编辑训练数据身份发生变化")
    manifest = json.loads(manifest_path.read_text())
    records = selected_records(args.data_dir, args.split, args.limit)
    train_maps = set(manifest["splits"]["train"]["maps"])
    evaluated_maps = {record["map_id"] for record in records}
    if train_maps & evaluated_maps:
        raise ValueError("训练地图进入了评估集合")
    if args.split == "test" and set(manifest["splits"]["val"]["maps"]) & evaluated_maps:
        raise ValueError("选模型用的验证地图进入了测试集合")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update({"generator_sha256": generator_checkpoint["checkpoint_sha256"],
                     "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                     for name in ("config.py", "repair_physics.py", "repair_loop.py", "repair_data.py",
                                                  "event_editor.py", "evaluate_event_editor.py")},
                     "editor_sha256": {key: value["sha256"] for key, value in checkpoints.items()},
                     "manifest_sha256": checkpoint["manifest_sha256"],
                     "sample_ids": [record["sample_id"] for record in records],
                     "map_ids": sorted(evaluated_maps), "torch_version": str(torch.__version__),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "generator_eval": not generator.training,
                     "timing": "每方法总时长=同一初值生成实测时长+该方法问题构建+修复；绘图和存盘另计",
                     "budget": "同一轨迹前向查询上限与修复墙钟上限；反向及独立验收另报",
                     "deadline": "候选批次和最终验收不可中断，可能超过软停止时限；时限内成功另行统计",
                     "ablation_scope": "仅删除编辑网络的车体间隙、违反量和距离梯度输入；共享物理排序与验收",
                     "scope": "network_seed_end_to_end" if args.source == "network" else "reference_corruption_diagnostic"})
    protocol.pop("resume", None)
    protocol_path = args.output_dir / "protocol.json"
    old_rows = []
    tasks_path = args.output_dir / "tasks.jsonl"
    if args.resume:
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise ValueError("恢复评估时模型、代码、任务选择或预算发生变化")
        content = tasks_path.read_bytes()
        if content and not content.endswith(b"\n"):
            tasks_path.with_name("tasks.interrupted_" + str(time.time_ns()) + ".jsonl").write_bytes(content)
            content = content[:content.rfind(b"\n") + 1]
            tasks_path.write_bytes(content)
        old_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if len({row["case_id"] for row in old_rows}) != len(old_rows):
            raise ValueError("已保存任务重复")
        identities = {f"{index:05d}_{variant}_{method}": (record["sample_id"], record["map_id"], variant, method)
                      for index, record in enumerate(records) for variant in args.variants for method in args.methods}
        if any(identities.get(row["case_id"]) != (row["sample_id"], row["map_id"], row["variant"], row["method"])
               for row in old_rows):
            raise ValueError("已保存任务与评估协议身份不一致")
    else:
        save_json(protocol_path, protocol)
    # 只预热网络核；预热不使用任何测试轨迹来更新权重或阈值。
    from repair_physics import TOKEN_COUNT, TOKEN_DIM, CONTEXT_DIM
    with torch.no_grad():
        for model in models.values():
            for _ in range(8):
                model(torch.zeros(1, TOKEN_COUNT, TOKEN_DIM, device=device),
                      torch.zeros(1, CONTEXT_DIM, device=device), torch.zeros(1, 2, 64, 64, device=device))
    synchronize(device)
    rows, visualized = list(old_rows), set()
    completed = {row["case_id"]: row for row in old_rows}
    for row in old_rows:
        if (args.output_dir / "visualizations" / row["case_id"] / "result.json").exists():
            visualized.add((row["method"], row["variant"], row["quality_pass"]))
    started = time.perf_counter()
    with tasks_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            expected_ids = [f"{index:05d}_{variant}_{method}" for variant in args.variants for method in args.methods]
            if all(case_id in completed for case_id in expected_ids):
                continue
            task_started = time.perf_counter()
            task = read_task(args.data_dir, record, reference=args.source == "perturbed_reference")
            seed_error = None
            try:
                if args.source == "network":
                    initial, directions, route = network_seed(generator, task, device)
                else:
                    initial = task["reference_controls"].astype(np.float32)
                    directions = np.where(initial[:, 0] >= 0, 1, -1)
                    route = np.asarray([task["start"][:2], task["goal"][:2]])
                synchronize(device)
            except ValueError as error:
                seed_error = str(error)
            common_seconds = time.perf_counter() - task_started
            for variant in args.variants:
                parameters = body_variant(variant)
                # 轮换方法次序，避免固定顺序造成持续的预热偏差。
                methods = args.methods[index % len(args.methods):] + args.methods[:index % len(args.methods)]
                for method in methods:
                    case_id = f"{index:05d}_{variant}_{method}"
                    if case_id in completed:
                        continue
                    row = {"case_id": case_id, "sample_id": record["sample_id"], "map_id": record["map_id"],
                           "variant": variant, "method": method, "source": args.source,
                           "quality_pass": False, "common_seed_seconds": common_seconds}
                    begin = time.perf_counter()
                    try:
                        if seed_error is not None:
                            raise ValueError(seed_error)
                        previous_case = next((item for item in old_rows if item["sample_id"] == record["sample_id"]
                                              and item["variant"] == variant and
                                              (cases_dir / (item["case_id"] + ".npz")).exists()), None)
                        case_common_seconds = common_seconds
                        case_directions = directions
                        if previous_case is not None:
                            with np.load(cases_dir / (previous_case["case_id"] + ".npz"), allow_pickle=False) as saved:
                                controls = torch.tensor(saved["requested_initial_controls"], device=device)
                                case_directions = saved["directions"].copy()
                            case_common_seconds = previous_case["common_seed_seconds"]
                            row["common_seed_seconds"] = case_common_seconds
                        else:
                            controls = torch.as_tensor(initial, device=device)
                        problem = RepairProblem(task["map_data"], task["start"], task["goal"], case_directions, parameters, device)
                        if args.source == "perturbed_reference" and previous_case is None:
                            noise = np.random.default_rng(args.seed + index).normal(0, .22, (KNOT_COUNT, 2))
                            controls = problem.apply_edit(controls, torch.tensor(noise, device=device).float().clamp(-.7, .7))[0]
                        requested_initial = controls.detach().cpu().numpy()
                        synchronize(device)
                        setup_seconds = time.perf_counter() - begin
                        path, report, artifacts = run_repair(
                            problem, controls, models.get(method), "neural" if method == "no_evidence" else method,
                            args.rounds, args.candidates, args.evaluations, args.time_limit, args.seed + index)
                        row.update({"quality_pass": path is not None, "reason": report["reason"],
                                    "initial_quality_pass": report["initial_quality"]["quality_pass"],
                                    "repair_seconds": report["optimizer_seconds"], "setup_seconds": setup_seconds,
                                    "total_seconds": case_common_seconds + setup_seconds + report["optimizer_seconds"]})
                        for key in ("physics_evaluations", "backward_calls", "independent_checks",
                                    "initial_proxy_score", "final_proxy_score", "iterations",
                                    "time_budget_overrun_seconds", "within_time_budget", "success_within_time_budget"):
                            row[key] = report[key]
                        row.update(report["quality"])
                        row["reason"] = report["reason"]
                        save_json(cases_dir / (case_id + ".json"), dict(report, identity=row))
                        np.savez_compressed(cases_dir / (case_id + ".npz"), **artifacts,
                                            requested_initial_controls=requested_initial,
                                            start=task["start"], goal=task["goal"], map_path=record["map_path"],
                                            body_variant=variant, parameters=json.dumps(parameters))
                        visual_key = (method, variant, path is not None)
                        if args.visualize and visual_key not in visualized:
                            from visualization import save_visualization
                            target = args.output_dir / "visualizations" / case_id
                            save_visualization(target, task["map_data"], task["start"], task["goal"], route,
                                               artifacts["initial_states"], problem.directions, artifacts["final_states"], problem.directions,
                                               prediction_label="Initial rollout", parameters=parameters,
                                               optimized_label="Accepted repair" if path else "Failed repair")
                            save_json(target / "result.json", dict(report, identity=row))
                            visualized.add(visual_key)
                    except (ValueError, FloatingPointError) as error:
                        row.update({"reason": str(error), "total_seconds": common_seconds + time.perf_counter() - begin})
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    print(index + 1, "/", len(records), variant, method, row["reason"], flush=True)
    summary = {"complete": True, "protocol": protocol, "wall_seconds": time.perf_counter() - started,
               "by_method": {method: summarize([r for r in rows if r["method"] == method]) for method in args.methods},
               "by_variant_method": {variant: {method: summarize([r for r in rows if r["variant"] == variant and r["method"] == method])
                                                for method in args.methods} for variant in args.variants},
               "resumed_rows": len(old_rows),
               "wall_time_scope": "仅本次进程；逐任务时长保留各自真实测量，恢复时复用原公共初值生成耗时",
               "note": "身体变化可能使原本任务不可行，失败仍保留；超时和预算耗尽不等于不可行性证明。"}
    save_json(args.output_dir / "summary.json", summary)
    print("评估完成：", args.output_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
