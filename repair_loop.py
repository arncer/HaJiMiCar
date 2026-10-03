"""物理检查→神经事件编辑→重新检查；所有方法共享验收和预算。"""
import time
from types import SimpleNamespace

import numpy as np
import torch

from repair_physics import KNOT_COUNT, FAMILY_NAMES


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _slice_evaluation(evaluation, index):
    return {key: value[index:index + 1].detach() for key, value in evaluation.items()}


def neural_candidates(editor, current, masks, count, rng):
    with torch.no_grad():
        output = editor(current["tokens"], current["context"], current["map_image"])
        delta = output["delta"][0]
        order = output["family_logits"][0].argsort(descending=True).tolist()
        # 小步长用于接近可行边界时细调；保留部分随机候选，使一次拒绝后仍能探索。
        deterministic_count = count - max(1, count // 4) if count >= 3 else count
        proposals, families = [delta], ["joint"]
        pool = [(delta * scale, "joint_scale_" + str(scale)) for scale in (.5, .25, .125)]
        pool += [(delta * masks[family], FAMILY_NAMES[family]) for family in order]
        for proposal, family in pool:
            if len(proposals) >= deterministic_count:
                break
            if any(torch.allclose(proposal, previous, atol=1e-7, rtol=0) for previous in proposals):
                continue
            proposals.append(proposal)
            families.append(family)
        while len(proposals) < count:
            noise = torch.tensor(rng.normal(0, 0.08, (KNOT_COUNT, 2)), device=delta.device).float()
            proposals.append((delta + noise).clamp(-1, 1))
            families.append("joint_sample")
        return torch.stack(proposals[:count]), families[:count], float(output["improvement_logit"].sigmoid()[0])


def run_repair(problem, initial_controls, editor=None, method="neural", max_rounds=12,
               candidates=8, evaluation_budget=128, time_limit=10.0, seed=1701):
    if method not in ("neural", "random", "gradient", "local"):
        raise ValueError("未知修复方法：" + method)
    if method == "neural" and editor is None:
        raise ValueError("神经编辑需要已训练的checkpoint")
    if min(max_rounds, candidates, evaluation_budget) < 1 or time_limit <= 0:
        raise ValueError("修复预算必须为正")
    if editor is not None:
        editor.eval()
    rng = np.random.default_rng(seed)
    synchronize(problem.device)
    started = time.perf_counter()
    evaluations, backward_calls, checks = 0, 0, 0
    with torch.no_grad():
        current = problem.evaluate(initial_controls, features=True)
    if len(current["score"]) != 1:
        raise ValueError("闭环入口每次处理一个任务")
    if not bool(torch.isfinite(current["score"]).all()):
        raise ValueError("初始控制推演得到非有限物理分数")
    evaluations += 1
    initial_proxy_score = float(current["score"][0])
    baseline_controls = current["controls"][0].detach().cpu().numpy()
    initial_states, _, initial_quality = problem.independent_check(baseline_controls)
    checks += 1
    quality = initial_quality
    final_checked = True
    final_states, final_controls = initial_states, baseline_controls
    trace = []
    reason = "passed" if quality["quality_pass"] else "budget_exhausted"
    for iteration in range(max_rounds):
        synchronize(problem.device)
        if quality["quality_pass"]:
            break
        if evaluations >= evaluation_budget or time.perf_counter() - started >= time_limit:
            break
        count = min(candidates, evaluation_budget - evaluations)
        masks = problem.family_masks(current)
        base = current["controls"][0].detach()
        before_score = float(current["score"][0])
        before_switch = current["switch_state"][0].cpu().tolist()
        confidence = None
        cached_best, cached_delta = None, None
        if method == "neural":
            deltas, families, confidence = neural_candidates(editor, current, masks, count, rng)
        elif method == "random":
            noise = rng.normal(0, 0.22, (count, KNOT_COUNT, 2))
            deltas = torch.tensor(noise, device=problem.device).float().clamp(-1, 1)
            families = [FAMILY_NAMES[index % len(FAMILY_NAMES)] for index in range(count)]
            deltas = deltas * masks[torch.arange(count, device=problem.device) % len(FAMILY_NAMES)]
        else:
            # 一次前向目标查询计一个预算单位；梯度反传另计，墙钟时间包含两者。
            delta = torch.zeros((1, KNOT_COUNT, 2), device=problem.device, requires_grad=True)
            optimizer = torch.optim.Adam([delta], lr=0.07)
            mask = (masks[4] + masks[5]) if method == "local" else torch.ones_like(masks[6])
            best_gradient_score = before_score
            for _ in range(max(0, count - 1)):
                if time.perf_counter() - started >= time_limit:
                    break
                optimizer.zero_grad(set_to_none=True)
                intermediate = problem.evaluate(problem.apply_edit(base, delta * mask), features=True)
                loss = intermediate["score"].mean()
                evaluations += 1
                value = float(loss.detach())
                if value < best_gradient_score - 1e-6:
                    best_gradient_score = value
                    cached_best = _slice_evaluation(intermediate, 0)
                    cached_delta = (delta * mask).detach().clone()
                loss.backward()
                backward_calls += 1
                torch.nn.utils.clip_grad_norm_([delta], 10)
                optimizer.step()
                with torch.no_grad():
                    delta.clamp_(-1, 1)
            deltas = (delta.detach() * mask)
            families = ["gradient_joint" if method == "gradient" else "gradient_local"]
        if time.perf_counter() - started >= time_limit:
            reason = "time_budget_exhausted"
            break
        with torch.no_grad():
            proposed = problem.evaluate(problem.apply_edit(base, deltas), features=True)
            evaluations += len(deltas)
            if not bool(torch.isfinite(proposed["score"]).all()):
                raise ValueError("候选控制推演得到非有限物理分数")
            if cached_best is not None:
                # 中间迭代已经查询过物理模型，直接复用；不重复计算或重复计数。
                proposed = {key: torch.cat([value, cached_best[key]], 0) for key, value in proposed.items()}
                deltas = torch.cat([deltas, cached_delta], 0)
                families.append(families[0] + "_best_intermediate")
            best = int(proposed["score"].argmin())
            accepted = bool(proposed["score"][best] < current["score"][0] - 1e-6)
            candidate_scores = proposed["score"].cpu().tolist()
            if accepted:
                current = _slice_evaluation(proposed, best)
        if accepted:
            final_controls = current["controls"][0].cpu().numpy()
            final_states = current["states"][0].cpu().numpy()
            terminal = current["terminal"][0].cpu().numpy()
            # 终点明显不满足必要条件时不重复运行完整CPU车体检查。
            # 每轮仍重新推演；潜在成功及最终输出必须经过独立验收。
            possibly_terminal = (np.linalg.norm(terminal[:2]) <= .301 and
                                 abs(terminal[2]) <= np.deg2rad(5) + .001 and
                                 abs(terminal[3]) <= np.deg2rad(3) + .001)
            final_checked = bool(possibly_terminal)
            quality = {"quality_pass": False, "reason": "terminal_proxy_failed_pending_final_check"}
            if final_checked:
                final_states, final_controls, quality = problem.independent_check(final_controls)
                checks += 1
        trace.append({"round": iteration + 1, "score_before": before_score,
                      "score_after": float(current["score"][0]), "accepted": accepted,
                      "decision_reason": "proxy_improved" if accepted else "no_proxy_improvement",
                      "selected_family": families[best], "candidate_families": families,
                      "candidate_scores": candidate_scores, "selected_delta": deltas[best].cpu().tolist(),
                      "switch_before": before_switch, "switch_after": current["switch_state"][0].cpu().tolist(),
                      "predicted_improvement": confidence, "quality": quality if final_checked else None,
                      "independent_check_current": final_checked,
                      "terminal_proxy_error": current["terminal"][0].cpu().tolist(),
                      "physics_evaluations": evaluations, "seconds": time.perf_counter() - started})
        if quality["quality_pass"]:
            reason = "passed"
            break
        if not accepted and method in ("gradient", "local"):
            reason = "no_improving_gradient_step"
            break
    if not final_checked:
        final_states, final_controls, quality = problem.independent_check(final_controls)
        checks += 1
    synchronize(problem.device)
    elapsed = time.perf_counter() - started
    if quality["quality_pass"]:
        reason = "passed" if elapsed <= time_limit else "passed_after_time_budget"
    if not quality["quality_pass"] and elapsed >= time_limit:
        reason = "time_budget_exhausted"
    report = {"method": method, "quality_pass": bool(quality["quality_pass"]), "reason": reason,
              "optimizer_seconds": elapsed, "iterations": len(trace), "quality": quality,
              "initial_quality": initial_quality, "initial_proxy_score": initial_proxy_score,
              "final_proxy_score": float(current["score"][0]), "trace": trace,
              "physics_evaluations": evaluations, "backward_calls": backward_calls,
              "independent_checks": checks, "evaluation_budget": evaluation_budget,
              "time_limit_seconds": time_limit, "time_budget_overrun_seconds": max(0., elapsed - time_limit),
              "within_time_budget": elapsed <= time_limit,
              "success_within_time_budget": bool(quality["quality_pass"] and elapsed <= time_limit),
              "body_parameters": problem.parameters, "seed": seed,
              "budget_scope": "单条轨迹物理代理前向计数；梯度反传和独立验收另计，全部计入墙钟时间",
              "scope": "kinematic_zero_or_one_switch_no_gearbox_dwell_or_acceleration_model"}
    path = None
    if quality["quality_pass"]:
        path = SimpleNamespace(states=final_states, controls=final_controls,
                               directions=np.r_[problem.directions[0], problem.directions])
    return path, report, {"initial_states": initial_states, "initial_controls": baseline_controls,
                          "final_states": final_states, "final_controls": final_controls,
                          "directions": problem.directions}
