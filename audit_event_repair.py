"""审计已完成实验的模型、数据划分、逐任务预算和独立物理验收证据。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_preparation import read_index_records, read_map_data
from event_editor import load_editor
from repair_physics import RepairProblem, body_variant


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_training(root, data_directory):
    manifest_path = data_directory / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["complete"], "数据生成未完成"
    base_directory = Path(manifest["data_dir"])
    all_maps = {split: {record["map_id"] for record in read_index_records(base_directory, split)}
                for split in ("train", "val", "test")}
    assert not all_maps["train"] & (all_maps["val"] | all_maps["test"])
    assert not all_maps["val"] & all_maps["test"]
    evidence = {"manifest_sha256": digest(manifest_path), "global_split_map_counts": {k: len(v) for k, v in all_maps.items()},
                "splits": {}, "models": {}}
    for split in ("train", "val"):
        info = manifest["splits"][split]
        data_path = data_directory / (split + ".npz")
        assert digest(data_path) == info["sha256"], (split, "数据指纹不一致")
        traces = read_jsonl(data_directory / (split + "_interventions.jsonl"))
        records = {record["sample_id"]: record for record in read_index_records(base_directory, split)}
        with np.load(data_path, allow_pickle=False) as samples:
            assert len(samples["delta"]) == info["count"] == len(traces)
            assert set(samples["sample_ids"]) <= set(records)
            for index, trace in enumerate(traces):
                assert trace["proxy_evaluations"] == 2 + 2 * trace["gradient_steps"] + len(trace["candidate_scores"])
                assert trace["sample_id"] == samples["sample_ids"][index]
                assert trace["map_id"] == records[trace["sample_id"]]["map_id"]
                assert trace["variant"] in manifest["variants"]
                assert trace["variant"] not in manifest["heldout_variants"]
                with np.load(data_directory / trace["witness"], allow_pickle=False) as witness:
                    assert hashlib.sha256(witness["base_controls"].tobytes()).hexdigest() == trace["base_controls_sha256"]
                    for key in ("tokens", "context", "map_image", "delta", "family", "improved"):
                        assert np.array_equal(witness[key], samples[key][index]), (split, index, key)
                    assert np.array_equal(witness["delta"], witness["candidate_deltas"][trace["selected"]])
                    assert np.array_equal(witness["candidate_scores"], trace["candidate_scores"])
        assert set(info["maps"]) <= all_maps[split]
        evidence["splits"][split] = {"count": info["count"], "tasks": info["task_count"], "maps": len(info["maps"]),
                                     "source_variant_counts": dict(Counter(r["source"] + ":" + r["variant"] for r in traces)),
                                     "witnesses_match_dataset": True}
    for label, use_evidence in (("evidence", True), ("no_evidence", False)):
        summary = read_json(root / label / "summary.json")
        history = read_jsonl(root / label / "history.jsonl")
        _, best = load_editor(root / label / "best.pt", torch.device("cpu"))
        _, last = load_editor(root / label / "last.pt", torch.device("cpu"))
        assert best["manifest_sha256"] == last["manifest_sha256"] == evidence["manifest_sha256"]
        assert best["base_checkpoint_sha256"] == manifest["base_checkpoint_sha256"]
        assert best["batch_size"] == last["batch_size"] == summary["batch_size"] == 512
        assert best["use_evidence"] == last["use_evidence"] == use_evidence
        assert summary["device"].startswith("cuda"), "正式模型必须在GPU训练"
        assert [row["epoch"] for row in history] == list(range(1, summary["completed_epochs"] + 1))
        assert last["epoch"] == summary["completed_epochs"]
        minimum = min(row["val"]["loss"] for row in history)
        selected_loss = next(row["val"]["loss"] for row in history if row["epoch"] == best["epoch"])
        assert abs(minimum - selected_loss) < 1e-8 and abs(minimum - summary["best_val_loss"]) < 1e-8
        for name, sha in best["code_sha256"].items():
            assert digest(Path(name)) == sha, (label, name, "训练代码发生变化，需要注明版本")
        evidence["models"][label] = {"best_sha256": best["sha256"], "best_epoch": best["epoch"],
                                      "last_epoch": last["epoch"], "validation_loss": minimum,
                                      "batch_size": 512, "gpu_training": True, "parameters": summary["parameters"]}
    return manifest, evidence


def audit_evaluation(directory, base_directory, replay_failures):
    summary, protocol = read_json(directory / "summary.json"), read_json(directory / "protocol.json")
    rows = read_jsonl(directory / "tasks.jsonl")
    assert summary["complete"]
    assert protocol["generator_eval"], "初值生成器必须处于推理模式"
    for name, sha in protocol["code_sha256"].items():
        assert digest(Path(name)) == sha, (name, "评估代码发生变化，需要另存版本")
    expected = len(protocol["sample_ids"]) * len(protocol["variants"]) * len(protocol["methods"])
    assert len(rows) == expected and len({row["case_id"] for row in rows}) == expected
    assert set(row["sample_id"] for row in rows) == set(protocol["sample_ids"])
    group_controls, physical_replays, missing_artifacts = defaultdict(set), [], []
    failed_count = 0
    for row in rows:
        assert np.isfinite(row["total_seconds"]) and row["total_seconds"] >= 0
        case_path = directory / "cases" / row["case_id"]
        if not case_path.with_suffix(".npz").exists():
            assert not row["quality_pass"], "成功任务缺失轨迹"
            missing_artifacts.append({"case_id": row["case_id"], "reason": row["reason"]})
            continue
        report = read_json(case_path.with_suffix(".json"))
        assert report["quality_pass"] == report["quality"]["quality_pass"] == row["quality_pass"]
        assert report["physics_evaluations"] <= protocol["evaluations"]
        assert report["within_time_budget"] == (report["optimizer_seconds"] <= protocol["time_limit"])
        assert report["success_within_time_budget"] == (row["quality_pass"] and report["within_time_budget"])
        assert row["success_within_time_budget"] == report["success_within_time_budget"]
        assert report["iterations"] == len(report["trace"]) <= protocol["rounds"]
        assert report["body_parameters"] == body_variant(row["variant"])
        score = report["initial_proxy_score"]
        for trace in report["trace"]:
            assert np.isclose(trace["score_before"], score)
            assert trace["physics_evaluations"] <= protocol["evaluations"]
            if trace["accepted"]:
                assert trace["decision_reason"] == "proxy_improved"
                assert trace["score_after"] < trace["score_before"] - 1e-6
                assert np.isclose(trace["score_after"], min(trace["candidate_scores"]))
            else:
                assert trace["decision_reason"] == "no_proxy_improvement"
                assert trace["score_after"] == trace["score_before"]
            assert len(trace["selected_delta"]) == 16
            score = trace["score_after"]
        assert score == report["final_proxy_score"]
        assert np.isclose(row["total_seconds"], row["common_seed_seconds"] + row["setup_seconds"] + row["repair_seconds"])
        with np.load(case_path.with_suffix(".npz"), allow_pickle=False) as saved:
            group_controls[(row["sample_id"], row["variant"])].add(hashlib.sha256(saved["initial_controls"].tobytes()).hexdigest())
            assert len(saved["final_states"]) == len(saved["final_controls"]) + 1
            assert np.array_equal(saved["start"], saved["initial_states"][0])
            assert np.array_equal(saved["start"], saved["final_states"][0])
            assert np.array_equal(np.sign(saved["final_controls"][:, 0]), saved["directions"])
            if row["quality_pass"] or failed_count < replay_failures:
                if not row["quality_pass"]:
                    failed_count += 1
                scene = read_map_data(base_directory, {"map_path": str(saved["map_path"])})
                problem = RepairProblem(scene, saved["start"], saved["goal"], saved["directions"],
                                        body_variant(row["variant"]), torch.device("cpu"))
                states, controls, quality = problem.independent_check(saved["final_controls"])
                assert quality["quality_pass"] == row["quality_pass"], row["case_id"]
                difference = states - saved["final_states"]
                difference[:, 2] = np.arctan2(np.sin(difference[:, 2]), np.cos(difference[:, 2]))
                assert np.abs(difference).max() < 1e-5
                assert np.max(np.abs(controls - saved["final_controls"])) < 1e-5
                physical_replays.append({"case_id": row["case_id"], "quality_pass": quality["quality_pass"],
                                         "replay_max_error": float(np.abs(difference).max())})
    assert all(len(values) == 1 for values in group_controls.values()), "同一任务的方法初值不一致"
    for method in protocol["methods"]:
        selected = [row for row in rows if row["method"] == method]
        assert len(selected) == len(protocol["sample_ids"]) * len(protocol["variants"])
        assert sum(row["quality_pass"] for row in selected) == summary["by_method"][method]["passed"]
    return {"count": expected, "methods": protocol["methods"], "variants": protocol["variants"],
            "same_initial_controls": True, "budget_and_trace_checks": True,
            "physical_replays": physical_replays, "failed_before_artifact_generation": missing_artifacts,
            "success_counts": {key: value["passed"] for key, value in summary["by_method"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/event_repair_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    parser.add_argument("--replay-failures", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pipeline = read_json(args.output_dir / "pipeline.json")
    assert pipeline["complete"] and all(stage["complete"] and stage["exit_code"] == 0 for stage in pipeline["stages"])
    component_checks = read_json(args.output_dir / "verification.json")
    assert component_checks["passed"] and len(component_checks["witnesses"]) > 0
    manifest, training = audit_training(args.output_dir, args.data_dir)
    result = {"passed": True, "training": training,
              "evaluations": {name: audit_evaluation(args.output_dir / name, Path(manifest["data_dir"]), args.replay_failures)
                              for name in ("test_network", "val_reference_diagnostic")},
              "note": "审计通过证明记录、预算和验收一致；不等同于网络优于基线或论文创新已成立。"}
    path = args.output_dir / "completion_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print("实验审计通过：", path, flush=True)


if __name__ == "__main__":
    main()
