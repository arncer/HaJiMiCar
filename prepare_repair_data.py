"""在固定失败初值上实际推演多种干预，生成事件编辑训练数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from config import data_dir
from repair_data import selected_records, read_task, network_seed
from repair_physics import RepairProblem, body_variant, KNOT_COUNT, FAMILY_NAMES
from trajectory_data import check_map_splits
from train import load_model, choose_device


def gradient_proposal(problem, base, steps, mask=None):
    """离线修复教师只查询当前任务的物理目标，没有参考控制输入。"""
    delta = torch.zeros((1, KNOT_COUNT, 2), device=problem.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.07)
    best_delta = delta.detach().clone()
    with torch.no_grad():
        best_score = float(problem.evaluate(base)["score"][0])
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        used_delta = delta if mask is None else delta * mask
        score = problem.evaluate(problem.apply_edit(base, used_delta))["score"].mean()
        score.backward()
        torch.nn.utils.clip_grad_norm_([delta], 10.0)
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-1, 1)
            candidate = delta if mask is None else delta * mask
            value = float(problem.evaluate(problem.apply_edit(base, candidate))["score"][0])
            if value < best_score:
                best_score, best_delta = value, candidate.detach().clone()
    return best_delta[0]


def paired_interventions(problem, base, rng, steps, inverse=None):
    with torch.no_grad():
        initial = problem.evaluate(base, features=True)
        masks = problem.family_masks(initial)
    gradient = gradient_proposal(problem, base, steps)
    candidates, families = [torch.zeros_like(gradient)], [6]
    # 每个候选均从相同base出发，不把前一个候选的结果传给后一个候选。
    for proposal in [gradient] + ([inverse] if inverse is not None else []):
        for family, mask in enumerate(masks):
            candidates.append(proposal * mask)
            families.append(family)
    random = torch.tensor(rng.normal(0, 0.25, (14, KNOT_COUNT, 2)), device=problem.device).float().clamp(-1, 1)
    for index, proposal in enumerate(random):
        family = index % len(FAMILY_NAMES)
        candidates.append(proposal * masks[family])
        families.append(family)
    candidates = torch.stack(candidates)
    with torch.no_grad():
        evaluated = problem.evaluate(problem.apply_edit(base, candidates))
        # 代价近似相同时优先较小编辑；分数都是实际重新推演所得。
        selection = evaluated["score"] + 1e-4 * candidates.square().mean((1, 2))
        best = int(selection.argmin())
        improved = bool(evaluated["score"][best] < evaluated["score"][0] - 1e-5)
        if not improved:
            best = 0
        sample = {key: initial[key][0].cpu().numpy().astype(np.float16)
                  for key in ("tokens", "context", "map_image")}
        sample.update({"delta": candidates[best].cpu().numpy().astype(np.float32),
                       "family": np.int64(families[best]), "improved": np.float32(improved)})
        _, _, before_check = problem.independent_check(initial["controls"][0].cpu().numpy())
        _, _, after_check = problem.independent_check(evaluated["controls"][best].cpu().numpy())
        trace = {"baseline_score": float(initial["score"][0]), "best_score": float(evaluated["score"][best]),
                 "selected": best, "selected_family": FAMILY_NAMES[families[best]],
                 "candidate_scores": evaluated["score"].cpu().tolist(),
                 "candidate_families": [FAMILY_NAMES[index] for index in families],
                 "candidate_deltas": candidates.cpu().tolist(),
                 "candidate_switch_states": evaluated["switch_state"].cpu().tolist(),
                 "before": before_check, "after": after_check,
                 "gradient_steps": steps, "proxy_evaluations": 1 + steps * 2 + len(candidates),
                 "label_scope": "physical_proxy_improvement_with_independent_selected_candidate_check"}
        return sample, trace, evaluated["controls"][best].detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/p104_batch512/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data/event_repair"))
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--variants", nargs="+", default=["nominal", "wide", "long", "slow"])
    parser.add_argument("--gradient-steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit, args.gradient_steps, args.rounds) < 1:
        parser.error("样本数、迭代数和轮数必须为正")
    if any(name not in ("nominal", "wide", "long", "slow") for name in args.variants):
        parser.error("组合身体条件wide_slow和long_slow必须留给测试")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes").mkdir(exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("输出目录已有数据；请使用新目录保留数据身份")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    generator, checkpoint = load_model(args.checkpoint, device)
    generator.eval()
    manifest = {"format": "loader_repair_pairs_v1", "complete": False,
                "data_dir": str(args.data_dir.resolve()), "base_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "split_audit": check_map_splits(args.data_dir), "variants": args.variants,
                "heldout_variants": ["wide_slow", "long_slow"], "seed": args.seed,
                "gradient_steps": args.gradient_steps, "rounds": args.rounds,
                "train_limit": args.train_limit, "val_limit": args.val_limit,
                "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in ("repair_physics.py", "repair_data.py", "prepare_repair_data.py")},
                "source_types": ["generator", "perturbed_reference"], "splits": {}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    started = time.perf_counter()
    all_maps = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        rows, failures, source_ids, maps = [], [], [], set()
        records = selected_records(args.data_dir, split, limit)
        with (args.output_dir / (split + "_interventions.jsonl")).open("w") as trace_file:
            for index, record in enumerate(records):
                try:
                    task = read_task(args.data_dir, record, reference=True)
                    variant = args.variants[(index // 2) % len(args.variants)]
                    parameters = body_variant(variant)
                    source = "generator" if index % 2 == 0 else "perturbed_reference"
                    inverse = None
                    if source == "generator":
                        base, directions, _ = network_seed(generator, task, device)
                    else:
                        base = task["reference_controls"]
                        directions = np.sign(base[:, 0]).astype(np.int64)
                    problem = RepairProblem(task["map_data"], task["start"], task["goal"], directions, parameters, device)
                    base = torch.tensor(base, device=device).float()
                    if source == "perturbed_reference":
                        perturbation = torch.tensor(rng.normal(0, 0.22, (KNOT_COUNT, 2)), device=device).float().clamp(-0.7, 0.7)
                        # 一半扰动集中在早期转向，产生后半程误差的受控样本。
                        if rng.random() < 0.5:
                            perturbation[8:] = 0
                            perturbation[:, 0] = 0
                        base = problem.apply_edit(base, perturbation)[0]
                        inverse = -perturbation
                    for iteration in range(args.rounds):
                        with torch.no_grad():
                            base = problem.evaluate(base)["controls"][0].detach()
                        sample, trace, updated = paired_interventions(problem, base, rng, args.gradient_steps, inverse)
                        trace.update({"sample_id": record["sample_id"], "map_id": record["map_id"],
                                      "split": split, "variant": variant, "source": source, "round": iteration,
                                      "base_controls_sha256": hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest()})
                        witness = args.output_dir / "episodes" / (split + "_" + record["sample_id"] + "_" + str(iteration) + ".npz")
                        np.savez_compressed(witness, **sample, base_controls=base.cpu().numpy(),
                                            selected_controls=updated.cpu().numpy(), directions=problem.directions,
                                            start=problem.start, goal=problem.goal,
                                            candidate_deltas=np.asarray(trace["candidate_deltas"], dtype=np.float32),
                                            candidate_scores=np.asarray(trace["candidate_scores"], dtype=np.float64),
                                            variant=variant, source=source, map_path=record["map_path"])
                        trace["witness"] = str(witness.relative_to(args.output_dir))
                        trace_file.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
                        trace_file.flush()
                        rows.append(sample)
                        source_ids.append(record["sample_id"])
                        maps.add(record["map_id"])
                        base, inverse = updated, None
                except (ValueError, FloatingPointError) as error:
                    failures.append({"sample_id": record["sample_id"], "reason": str(error)})
                if (index + 1) % 8 == 0 or index == len(records) - 1:
                    print(split, index + 1, "/", len(records), "配对样本", len(rows), "失败", len(failures), flush=True)
        if not rows:
            raise ValueError("没有成功生成配对干预数据：" + str(failures[:3]))
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        arrays["sample_ids"] = np.asarray(source_ids)
        path = args.output_dir / (split + ".npz")
        np.savez_compressed(path, **arrays)
        all_maps[split] = maps
        manifest["splits"][split] = {"count": len(rows), "task_count": len(set(source_ids)),
                                      "maps": sorted(maps), "failures": failures,
                                      "improvement_fraction": float(arrays["improved"].mean()),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if all_maps["train"] & all_maps["val"]:
        raise ValueError("配对干预地图发生泄漏")
    manifest.update({"complete": True, "elapsed_seconds": time.perf_counter() - started})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("干预数据已完成：", manifest_path, flush=True)


if __name__ == "__main__":
    main()
