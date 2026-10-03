"""从逐轮见证文件恢复被中断的配对数据生成，保留已有记录及随机序列。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from prepare_repair_data import paired_interventions
from repair_data import selected_records, read_task, network_seed
from repair_physics import RepairProblem, body_variant, phase_basis, KNOT_COUNT
from train import load_model, choose_device


FIELDS = ("tokens", "context", "map_image", "delta", "family", "improved")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def completed_lines(path):
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        # 保留中断时尚未完整写入的尾部，然后只恢复已提交的完整行。
        boundary = content.rfind(b"\n") + 1
        backup = path.with_name(path.name + ".interrupted_" + str(time.time_ns()))
        backup.write_bytes(content)
        path.write_bytes(content[:boundary])
        content = content[:boundary]
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/p104_batch512/best.pt"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    args = parser.parse_args()
    directory = args.data_dir
    lock = (directory / "resume.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["complete"]:
        print("数据已经完成，无需恢复", flush=True)
        return
    for name, expected in manifest["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("生成代码已变化，不能静默恢复：" + name)
    torch.set_num_threads(2)
    torch.manual_seed(manifest["seed"])
    rng = np.random.default_rng(manifest["seed"])
    device = choose_device(args.device)
    generator, checkpoint = load_model(args.checkpoint, device)
    generator.eval()
    if checkpoint["checkpoint_sha256"] != manifest["base_checkpoint_sha256"]:
        raise ValueError("原生成器checkpoint发生变化")
    error_path = directory / "recovery_errors.jsonl"
    previous_errors = {(row["split"], row["sample_id"]): row for row in completed_lines(error_path)}
    resumed = {"started_unix_seconds": time.time(), "complete": False, "reused_rows": 0, "generated_rows": 0,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest.setdefault("resumes", []).append(resumed)
    atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    base_directory = Path(manifest["data_dir"])
    for split in ("train", "val"):
        records = selected_records(base_directory, split, manifest[split + "_limit"])
        trace_path = directory / (split + "_interventions.jsonl")
        old_traces = completed_lines(trace_path)
        cached = {}
        for trace in old_traces:
            cached.setdefault(trace["sample_id"], []).append(trace)
        record_indices = {record["sample_id"]: index for index, record in enumerate(records)}
        if not set(cached) <= set(record_indices):
            raise ValueError("缓存任务不属于当前固定选择")
        recorded_indices = [record_indices[name] for name in cached]
        recorded_indices += [record_indices[name] for group, name in previous_errors if group == split]
        last_recorded = max(recorded_indices, default=-1)
        rows, source_ids, maps, failures = [], [], set(), []
        with trace_path.open("a", encoding="utf-8") as handle, error_path.open("a", encoding="utf-8") as errors:
            for index, record in enumerate(records):
                saved = cached.get(record["sample_id"], [])
                saved_error = previous_errors.get((split, record["sample_id"]))
                if index < last_recorded and not saved and saved_error is None:
                    raise ValueError("旧日志存在未记录的任务缺口，不能推断随机状态")
                if [trace["round"] for trace in saved] != list(range(len(saved))) or len(saved) > manifest["rounds"]:
                    raise ValueError("缓存轮次缺失或重复")
                source = "generator" if index % 2 == 0 else "perturbed_reference"
                variant = manifest["variants"][(index // 2) % len(manifest["variants"])]
                perturbation, early_only = None, False
                if source == "perturbed_reference":
                    perturbation = rng.normal(0, .22, (KNOT_COUNT, 2)).astype(np.float32).clip(-.7, .7)
                    early_only = rng.random() < .5
                last_controls, last_directions = None, None
                for trace in saved:
                    if trace["source"] != source or trace["variant"] != variant or trace["map_id"] != record["map_id"]:
                        raise ValueError("缓存样本身份或身体条件不一致")
                    with np.load(directory / trace["witness"], allow_pickle=False) as witness:
                        if hashlib.sha256(witness["base_controls"].tobytes()).hexdigest() != trace["base_controls_sha256"]:
                            raise ValueError("缓存初值指纹不一致")
                        # 旧版未保存RNG状态：逐次消耗相同随机数，并用两次joint随机干预核对。
                        noise = rng.normal(0, .25, (14, KNOT_COUNT, 2)).astype(np.float32).clip(-1, 1)
                        basis, _ = phase_basis(witness["directions"], torch.device("cpu"))
                        active = (basis.sum(0).numpy() > 0)[:, None]
                        for position in (6, 13):
                            observed = witness["candidate_deltas"][-14 + position]
                            if not np.array_equal(observed, noise[position] * active):
                                raise ValueError("恢复随机序列与已保存干预不一致：" + trace["witness"])
                        if "rng_after" in trace and rng.bit_generator.state != trace["rng_after"]:
                            raise ValueError("随机状态重放不一致")
                        rows.append({key: witness[key].copy() for key in FIELDS})
                        source_ids.append(record["sample_id"])
                        maps.add(record["map_id"])
                        last_controls, last_directions = witness["selected_controls"].copy(), witness["directions"].copy()
                    resumed["reused_rows"] += 1
                if saved_error:
                    rng.bit_generator.state = saved_error["rng_after"]
                    failures.append({"sample_id": record["sample_id"], "reason": saved_error["reason"]})
                    continue
                if len(saved) == manifest["rounds"]:
                    continue
                inverse = None
                try:
                    task = read_task(base_directory, record, reference=True)
                    if saved:
                        base, directions = last_controls, last_directions
                    elif source == "generator":
                        base, directions, _ = network_seed(generator, task, device)
                    else:
                        base = task["reference_controls"]
                        directions = np.sign(base[:, 0]).astype(np.int64)
                    problem = RepairProblem(task["map_data"], task["start"], task["goal"], directions, body_variant(variant), device)
                    base = torch.tensor(base, device=device).float()
                    if source == "perturbed_reference" and not saved:
                        perturbation = torch.tensor(perturbation, device=device)
                        if early_only:
                            perturbation[8:] = 0
                            perturbation[:, 0] = 0
                        base = problem.apply_edit(base, perturbation)[0]
                        inverse = -perturbation
                    for iteration in range(len(saved), manifest["rounds"]):
                        with torch.no_grad():
                            base = problem.evaluate(base)["controls"][0].detach()
                        sample, trace, updated = paired_interventions(problem, base, rng, manifest["gradient_steps"], inverse)
                        trace.update({"sample_id": record["sample_id"], "map_id": record["map_id"],
                                      "split": split, "variant": variant, "source": source, "round": iteration,
                                      "rng_after": rng.bit_generator.state,
                                      "base_controls_sha256": hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest()})
                        witness = directory / "episodes" / (split + "_" + record["sample_id"] + "_" + str(iteration) + ".npz")
                        if witness.exists():
                            witness.rename(witness.with_name(witness.name + ".orphan_" + str(time.time_ns())))
                        np.savez_compressed(witness, **sample, base_controls=base.cpu().numpy(), selected_controls=updated.cpu().numpy(),
                                            directions=problem.directions, start=problem.start, goal=problem.goal,
                                            candidate_deltas=np.asarray(trace["candidate_deltas"], dtype=np.float32),
                                            candidate_scores=np.asarray(trace["candidate_scores"], dtype=np.float64),
                                            variant=variant, source=source, map_path=record["map_path"])
                        trace["witness"] = str(witness.relative_to(directory))
                        handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                        rows.append(sample)
                        source_ids.append(record["sample_id"])
                        maps.add(record["map_id"])
                        base, inverse = updated, None
                        resumed["generated_rows"] += 1
                except (ValueError, FloatingPointError) as error:
                    failure = {"sample_id": record["sample_id"], "reason": str(error)}
                    failures.append(failure)
                    errors.write(json.dumps(dict(failure, split=split, rng_after=rng.bit_generator.state)) + "\n")
                    errors.flush()
                if (index + 1) % 8 == 0 or index == len(records) - 1:
                    print(split, index + 1, "/", len(records), "配对样本", len(rows), "失败", len(failures), flush=True)
                    atomic_json(manifest_path, manifest)
        if not rows:
            raise ValueError("当前划分没有有效数据")
        arrays = {key: np.stack([row[key] for row in rows]) for key in FIELDS}
        arrays["sample_ids"] = np.asarray(source_ids)
        data_path = directory / (split + ".npz")
        np.savez_compressed(data_path, **arrays)
        manifest["splits"][split] = {"count": len(rows), "task_count": len(set(source_ids)), "maps": sorted(maps),
                                      "failures": failures, "improvement_fraction": float(arrays["improved"].mean()),
                                      "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
        atomic_json(manifest_path, manifest)
    if set(manifest["splits"]["train"]["maps"]) & set(manifest["splits"]["val"]["maps"]):
        raise ValueError("地图划分发生泄漏")
    resumed.update({"complete": True, "seconds": time.perf_counter() - started})
    manifest.update({"complete": True, "elapsed_seconds": None,
                     "timing_note": "中断前耗时未持久化；resumes分别记录恢复阶段耗时，不能作为全部生成耗时"})
    atomic_json(manifest_path, manifest)
    print("数据恢复并完成：", manifest_path, flush=True)


if __name__ == "__main__":
    main()
