"""复核新闭环的物理一致性、配对证据和预算；不是成功率评估的替代。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from config import integration_dt
from data_preparation import read_map_data
from event_editor import EventEditor
from planner_backend import make_environment, NumpyArticulatedRollout
from repair_loop import run_repair
from repair_physics import (RepairProblem, rollout_controls, body_variant, phase_basis,
                            TOKEN_DIM, TOKEN_COUNT, CONTEXT_DIM, KNOT_COUNT)
from train import choose_device


def open_map(obstacle=False):
    occupancy = np.zeros((120, 120), dtype=np.float32)
    occupancy[[0, -1], :] = 1
    occupancy[:, [0, -1]] = 1
    if obstacle:
        occupancy[42:45, 41:44] = 1
    esdf = (distance_transform_edt(1 - occupancy) - distance_transform_edt(occupancy)) * .5
    return {"map_features": np.stack([occupancy, esdf]).astype(np.float32),
            "origin": np.zeros(2), "origin_yaw": 0., "resolution": .5}


def verify_rollout(device):
    rng = np.random.default_rng(7301)
    errors = []
    for name in ("nominal", "wide_slow", "long_slow"):
        parameters = body_variant(name)
        environment = make_environment(open_map(), parameters)
        teacher = NumpyArticulatedRollout.from_vehicle(environment[1])
        for length in (8, 127, 600):
            controls = rng.normal(0, [.9, .4], (3, length, 2))
            starts = np.array([[20, 20, 3.13, .6], [21, 20, -3.13, -.6], [20, 21, 0, 0]])
            states, applied = rollout_controls(torch.tensor(starts, device=device),
                                                torch.tensor(controls, device=device), parameters)
            output, actions = states.cpu().numpy(), applied.cpu().numpy()
            maximum = 0.
            for batch in range(3):
                state = starts[batch].copy()
                replay = [state]
                for action in actions[batch]:
                    state, _ = teacher.step_rk4(state, action, integration_dt)
                    replay.append(state)
                difference = np.asarray(replay) - output[batch]
                difference[:, 2] = np.arctan2(np.sin(difference[:, 2]), np.cos(difference[:, 2]))
                maximum = max(maximum, float(np.abs(difference).max()))
            assert maximum < 1e-8, (name, length, maximum)
            assert np.abs(actions[..., 1]).max() <= parameters["max_articulation_rate_rps"] + 1e-9
            errors.append({"variant": name, "steps": length, "max_error_float64": maximum})
    return errors


def verify_geometry(device):
    scene = open_map(obstacle=True)
    start, goal = np.array([20., 20., 0, 0]), np.array([20.1, 20., 0, 0])
    directions = np.ones(8, dtype=np.int64)
    problem = RepairProblem(scene, start, goal, directions, body_variant("nominal"), device)
    controls = torch.tensor([[.1, 0]] * 8, device=device)
    evaluation = problem.evaluate(controls, features=True)
    _, _, quality = problem.independent_check(evaluation["controls"][0].cpu().numpy())
    assert not quality["safe"] and not quality["quality_pass"], quality
    assert float(problem._distance(torch.tensor(start[:2], device=device).float())) > 1
    assert float(evaluation["violation"].max()) > 0
    yaw = .63
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translated = dict(scene, origin=np.array([7., -3.]), origin_yaw=yaw)
    moved_start, moved_goal = start.copy(), goal.copy()
    for point in (moved_start, moved_goal):
        point[:2] = point[:2] @ rotation.T + translated["origin"]
        point[2] += yaw
    rotated = RepairProblem(translated, moved_start, moved_goal, directions, body_variant("nominal"), device)
    moved = rotated.evaluate(controls, features=True)
    error = float((evaluation["clearance"] - moved["clearance"]).abs().max())
    assert error < 5e-5, error
    outside = torch.tensor([[-1., -1.]], device=device, requires_grad=True)
    problem._distance(outside).sum().backward()
    assert bool(torch.isfinite(outside.grad).all()) and float(outside.grad.abs().sum()) > 0
    occupancy = np.zeros((400, 400), dtype=np.float32)
    occupancy[[0, -1], :] = 1
    occupancy[:, [0, -1]] = 1
    occupancy[140, :] = 1
    body_scene = {"map_features": np.stack([occupancy, distance_transform_edt(1 - occupancy) * .1]).astype(np.float32),
                  "origin": np.zeros(2), "origin_yaw": 0., "resolution": .1}
    body_checks = {}
    for variant in ("nominal", "wide"):
        changed = RepairProblem(body_scene, np.array([12., 12., 0., 0.]), np.array([12.08, 12., 0., 0.]),
                                directions, body_variant(variant), device)
        _, _, checked = changed.independent_check(controls.cpu().numpy())
        body_checks[variant] = {"quality_pass": checked["quality_pass"], "margin_m": checked["minimum_margin_m"]}
    assert body_checks["nominal"]["quality_pass"] and not body_checks["wide"]["quality_pass"]
    return {"corner_collision_detected": True, "rotated_map_clearance_error": error,
            "outside_map_gradient": outside.grad.cpu().tolist(), "body_change_affects_acceptance": body_checks}


def verify_network(device):
    torch.manual_seed(7301)
    tokens = torch.randn(2, TOKEN_COUNT, TOKEN_DIM, device=device)
    context = torch.randn(2, CONTEXT_DIM, device=device)
    image = torch.randn(2, 2, 64, 64, device=device)
    model = EventEditor(use_evidence=False).to(device).eval()
    altered = tokens.clone()
    altered[..., 10:22] += 100
    first = model(tokens, context, image)
    second = model(altered, context, image)
    for key in first:
        assert torch.equal(first[key], second[key]), key
    (first["delta"] - .2).square().sum().add(first["family_logits"].square().sum()).backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert float(model.delta_head.weight.grad.abs().sum()) > 0
    basis, switch = phase_basis(np.r_[np.ones(8), -np.ones(8)], device)
    assert switch == 8 and bool(torch.allclose(basis.sum(1), torch.ones(16, device=device)))
    assert not bool((basis[:8, 8:] != 0).any()) and not bool((basis[8:, :8] != 0).any())
    try:
        phase_basis(np.array([1, -1, 1]), device)
        raise AssertionError("多次换挡未被拒绝")
    except ValueError:
        pass
    return {"no_evidence_invariance": True, "finite_gradients": True, "phase_basis": True}


def verify_witness(directory, device, limit):
    if limit == 0:
        return {"skipped": True, "reason": "witness-limit=0；配对标签重放另行执行"}
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"], "配对数据未完成"
    errors = []
    for split in ("train", "val"):
        with (directory / (split + "_interventions.jsonl")).open() as handle:
            traces = [json.loads(line) for line in handle]
        indices = np.linspace(0, len(traces) - 1, min(limit, len(traces)), dtype=int)
        for index in indices:
            trace = traces[index]
            with np.load(directory / trace["witness"], allow_pickle=False) as witness:
                map_data = read_map_data(Path(manifest["data_dir"]), {"map_path": str(witness["map_path"])})
                problem = RepairProblem(map_data, witness["start"], witness["goal"], witness["directions"],
                                        body_variant(str(witness["variant"])), device)
                base = torch.tensor(witness["base_controls"], device=device)
                delta = torch.tensor(witness["candidate_deltas"], device=device)
                with torch.no_grad():
                    candidates = problem.evaluate(problem.apply_edit(base, delta))
                scores = candidates["score"].cpu().numpy()
                error = float(np.abs(scores - witness["candidate_scores"]).max())
                assert np.allclose(scores, witness["candidate_scores"], rtol=5e-4, atol=.02), (trace["witness"], error)
                selected = trace["selected"]
                selected_controls = candidates["controls"][selected].cpu().numpy()
                assert np.allclose(selected_controls, witness["selected_controls"], atol=2e-5)
                _, _, quality = problem.independent_check(selected_controls)
                assert quality["quality_pass"] == trace["after"]["quality_pass"]
                assert np.array_equal(witness["delta"], witness["candidate_deltas"][selected])
                errors.append({"witness": trace["witness"], "max_score_error": error,
                               "quality_pass": quality["quality_pass"]})
    assert not set(manifest["splits"]["train"]["maps"]) & set(manifest["splits"]["val"]["maps"])
    return errors


def verify_budget(device):
    problem = RepairProblem(open_map(), np.array([20, 20, 0., 0]), np.array([30, 20, 0., 0]),
                            np.ones(16, dtype=np.int64), body_variant("nominal"), device)
    controls = np.tile([.1, 0], (16, 1))
    editor = EventEditor().to(device).eval()
    reports = []
    for method in ("neural", "random", "gradient", "local"):
        original = problem.evaluate
        query_count = [0]

        def counted(actions, **kwargs):
            query_count[0] += 1 if actions.ndim == 2 else len(actions)
            return original(actions, **kwargs)

        problem.evaluate = counted
        _, report, artifacts = run_repair(problem, controls, editor, method, max_rounds=2, candidates=4,
                                          evaluation_budget=7, time_limit=60)
        problem.evaluate = original
        assert report["physics_evaluations"] <= 7 and report["iterations"] <= 2
        assert report["physics_evaluations"] == query_count[0]
        assert report["initial_proxy_score"] is not None and not report["quality_pass"]
        base = torch.tensor(artifacts["initial_controls"], device=device)
        for step in report["trace"]:
            delta = torch.tensor(step["selected_delta"], device=device)
            with torch.no_grad():
                replayed = problem.evaluate(problem.apply_edit(base, delta))
            assert np.isclose(float(replayed["score"][0]), min(step["candidate_scores"]), rtol=1e-5, atol=1e-4)
            if step["accepted"]:
                assert np.allclose(replayed["switch_state"][0].cpu().numpy(), step["switch_after"], atol=1e-5)
                base = replayed["controls"][0]
        reports.append({key: report[key] for key in ("method", "physics_evaluations", "backward_calls", "reason")})
    already_valid = RepairProblem(open_map(), np.array([20, 20, 0., 0]), np.array([20.16, 20, 0., 0]),
                                  np.ones(16, dtype=np.int64), body_variant("nominal"), device)
    path, late, _ = run_repair(already_valid, controls, editor, time_limit=1e-9)
    assert path is not None and late["quality_pass"]
    assert not late["success_within_time_budget"] and late["reason"] == "passed_after_time_budget"
    reports.append({"method": "deadline_accounting", "quality_pass": True, "success_within_time_budget": False})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_smoke"))
    parser.add_argument("--witness-limit", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/event_editor_verification.json"))
    args = parser.parse_args()
    if args.witness_limit < 0:
        parser.error("witness-limit不能为负")
    torch.set_num_threads(2)
    device = choose_device(args.device)
    result = {"device": str(device), "rollout": verify_rollout(device), "geometry": verify_geometry(device),
              "network": verify_network(device), "witnesses": verify_witness(args.data_dir, device, args.witness_limit),
              "budgets": verify_budget(device), "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print("检查通过：", args.output, flush=True)


if __name__ == "__main__":
    main()
