"""实测GPU推理延迟、批量吞吐量，以及可选的完整规划耗时。"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from config import data_dir
from coarse_route import CoarseRoutePlanner
from data_preparation import read_index_records, read_map_data
from loader_model import predict_batch
from train import load_model, move_batch
from trajectory_data import collate_trajectories, prepare_trajectory_input
from trajectory_decoder import decode_directions


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def gpu_status():
    fields = "name,pstate,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw,memory.used"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                             "--format=csv,noheader"], capture_output=True, text=True)
    return {"fields": fields, "value": result.stdout.strip(), "error": result.stderr.strip()}


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def measure(call, repetitions):
    # 同时记录主机实际等待时间与CUDA事件时间，避免只测到异步提交开销。
    wall_ms, event_ms = [], []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    output = None
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        output = call()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000)
        event_ms.append(begin.elapsed_time(end))
    return {"wall_ms": statistics(wall_ms), "cuda_event_ms": statistics(event_ms),
            "wall_samples_ms": wall_ms, "cuda_event_samples_ms": event_ms}, output


def read_inputs(args):
    records = read_index_records(args.data_dir, "test")
    indices = np.linspace(0, len(records) - 1, min(args.tasks, len(records)), dtype=int)
    tasks = []
    for index in indices:
        record = records[index]
        with np.load(args.data_dir / record["sample_path"], allow_pickle=False) as sample:
            start = sample["start_state"].astype(float)
            goal = sample["goal_state"].astype(float)
        map_data = read_map_data(args.data_dir, record)
        route = CoarseRoutePlanner(map_data).plan(start, goal)
        prepared = prepare_trajectory_input(map_data, start, goal, route)
        tasks.append({"record": record, "start": start, "goal": goal,
                      "map_data": map_data, "prepared": prepared,
                      "route_length": len(route)})
    return tasks


def cpu_result(model, batch):
    # 与plan.py一致：状态、方向、模式概率全部取回CPU；不包含地图和粗路线准备。
    output = predict_batch(model, move_batch(batch, torch.device("cuda")))
    return {"states": output["states"].cpu(),
            "directions": decode_directions(output).cpu(),
            "mode_probabilities": torch.softmax(output["mode_logits"], 1).cpu()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/p104_batch512/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output", type=Path, default=Path("runs/p104_gpu_inference/summary.json"))
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    if min(args.tasks, args.warmup, args.repetitions, args.batch_size) < 1 or args.time_limit <= 0:
        parser.error("样本数、预热次数、重复次数、批量和时间预算必须为正")
    if not torch.cuda.is_available():
        parser.error("需要在可访问GPU的环境运行")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "training_epoch": checkpoint["epoch"], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "dtype": "float32", "model_mode": "eval + inference_mode", "cpu_threads": 2,
        "cudnn_benchmark": True, "warmup_per_input": args.warmup,
        "repetitions_per_input": args.repetitions, "gpu_before": gpu_status(),
        "scope": {
            "resident": "输入和输出在GPU，含主机提交与等待；排除数据准备和传输",
            "roundtrip": "CPU张量传入GPU、网络前向、状态/方向/概率取回CPU；排除地图和粗路线准备",
            "graph": "额外测量CUDA Graph重放，固定输入地址与形状；尚未接入plan.py",
            "planning": "plan_task全流程；排除模型与地图文件加载、绘图及文件保存",
            "batch": "大批量以所选真实任务循环组成；不代表同等数量的独立场景或单请求延迟",
        },
        "single_tasks": [],
    }
    print("准备固定测试任务", flush=True)
    tasks = read_inputs(args)
    report["task_count"] = len(tasks)
    report["route_lengths"] = [task["route_length"] for task in tasks]
    single_resident, single_roundtrip, single_graph = [], [], []
    # 保存每次测量而非只保存平均值，后续可重新计算分位数。
    raw = {"resident_wall_ms": [], "roundtrip_wall_ms": [], "graph_wall_ms": []}
    with torch.inference_mode():
        for index, task in enumerate(tasks):
            cpu_batch = collate_trajectories([task["prepared"]])
            batch = move_batch(cpu_batch, device)
            for _ in range(args.warmup):
                predict_batch(model, batch)
            torch.cuda.synchronize()
            resident, reference = measure(lambda: predict_batch(model, batch), args.repetitions)
            for _ in range(5):
                cpu_result(model, cpu_batch)
            roundtrip, _ = measure(lambda: cpu_result(model, cpu_batch), args.repetitions)
            row = {"sample_id": task["record"]["sample_id"], "route_length": task["route_length"],
                   "resident": resident, "roundtrip": roundtrip}
            # 静态图只作为独立加速基准，不改变正式模型或规划入口。
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    predict_batch(model, batch)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = predict_batch(model, batch)
            graph.replay()
            torch.cuda.synchronize()
            error = max(float((graph_output[key] - reference[key]).abs().max()) for key in reference)
            if error > 1e-5 or not all(bool(torch.isfinite(value).all()) for value in graph_output.values()):
                raise ValueError("CUDA Graph输出与普通推理不一致")
            graph_timing, _ = measure(graph.replay, args.repetitions)
            row["cuda_graph"] = graph_timing
            row["graph_max_abs_error"] = error
            report["single_tasks"].append(row)
            single_resident.append(resident["wall_ms"]["median"])
            single_roundtrip.append(roundtrip["wall_ms"]["median"])
            single_graph.append(graph_timing["wall_ms"]["median"])
            raw["resident_wall_ms"].extend(resident["wall_samples_ms"])
            raw["roundtrip_wall_ms"].extend(roundtrip["wall_samples_ms"])
            raw["graph_wall_ms"].extend(graph_timing["wall_samples_ms"])
            print(index + 1, "/", len(tasks), "单任务ms：",
                  round(single_resident[-1], 3), "含传输：", round(single_roundtrip[-1], 3),
                  "CUDA Graph：", round(single_graph[-1], 3), flush=True)
            del graph, graph_output, reference, batch
        report["single_task_median_distribution_ms"] = {
            "resident": statistics(single_resident), "roundtrip": statistics(single_roundtrip),
            "cuda_graph": statistics(single_graph),
            "note": "先计算每个任务重复测量的中位数，再统计各任务之间的分布；各任务P95见single_tasks",
        }
        report["single_all_repetitions_ms"] = {key: statistics(value) for key, value in raw.items()}
        save_report(args.output, report)
        gc.collect()
        torch.cuda.empty_cache()
        prepared = [tasks[index % len(tasks)]["prepared"] for index in range(args.batch_size)]
        batch = move_batch(collate_trajectories(prepared), device)
        torch.cuda.reset_peak_memory_stats()
        print("开始批量吞吐量测量：", args.batch_size, flush=True)
        for _ in range(args.warmup):
            predict_batch(model, batch)
        torch.cuda.synchronize()
        report["gpu_under_batch_load"] = gpu_status()
        bulk, bulk_output = measure(lambda: predict_batch(model, batch), args.repetitions)
        bulk.update({"batch_size": args.batch_size, "padded_route_length": batch["route_padding_mask"].shape[1],
                     "trajectories_per_second": args.batch_size * 1000 / bulk["wall_ms"]["mean"],
                     "amortized_ms_per_trajectory": bulk["wall_ms"]["mean"] / args.batch_size,
                     "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2})
        if not all(bool(torch.isfinite(value).all()) for value in bulk_output.values()):
            raise ValueError("批量推理输出包含非有限数")
        report["batch_inference"] = bulk
        report["gpu_after_batch"] = gpu_status()
        save_report(args.output, report)
        print("批量结果：", json.dumps({key: value for key, value in bulk.items()
                                      if not key.endswith("samples_ms")}, ensure_ascii=False), flush=True)
        del batch, bulk_output
        gc.collect()
        torch.cuda.empty_cache()
        if args.planning:
            from plan import plan_task
            from evaluate import summarize
            rows = []
            task_file = args.output.parent / "planning_tasks.jsonl"
            with task_file.open("w", encoding="utf-8") as handle:
                for index, task in enumerate(tasks):
                    _, _, _, _, result = plan_task(model, task["map_data"], task["start"], task["goal"],
                                                  device, time_limit=args.time_limit)
                    row = {"sample_id": task["record"]["sample_id"], "map_id": task["record"]["map_id"],
                           "quality_pass": result["quality_pass"], "reason": result["reason"],
                           "total_seconds": result["total_seconds"], "network_seconds": result["network_seconds"],
                           "optimizer_seconds": result["optimizer"]["optimizer_seconds"]}
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    print("完整规划", index + 1, "/", len(tasks), row["reason"],
                          round(row["total_seconds"], 3), "秒", flush=True)
            report["planning"] = summarize(rows)
            report["planning"]["total_seconds"].update(statistics([row["total_seconds"] for row in rows]))
            report["planning"]["solver_cpu_budget_seconds"] = args.time_limit
            report["planning"]["note"] = "相同测试任务各规划一次；GPU推理，IPOPT仍在CPU执行，含失败任务"
    report["gpu_end"] = gpu_status()
    save_report(args.output, report)
    print("报告已保存：", args.output, flush=True)


if __name__ == "__main__":
    main()
