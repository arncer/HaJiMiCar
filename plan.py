"""自行设置起终状态，查看网络预测以及约束优化后的轨迹。"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from config import data_dir, integration_dt, articulation_limit
from data_preparation import read_index_records, read_map_data
from coarse_route import CoarseRoutePlanner, grid_to_world
from trajectory_data import prepare_trajectory_input, collate_trajectories, restore_states
from trajectory_decoder import decode_directions
from loader_model import predict_batch
from train import load_model, choose_device, move_batch
from visualization import save_visualization


def archive_previous_result(output_dir):
    # 同一输出目录重复规划时，先保留上一轮结果，避免本轮输入错误后误看旧成功轨迹。
    # 成功后旧文件位于previous_runs/时间目录；当前目录只表示这一次任务。
    names = ["trajectory.npz", "trajectory.png", "trajectory.html", "result.json"]
    previous_files = []
    for name in names:
        path = output_dir / name
        if path.is_file():
            previous_files.append(path)
    if previous_files:
        archive = output_dir / "previous_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive.mkdir(parents=True)
        for path in previous_files:
            path.rename(archive / path.name)


def plan_task(model, map_data, start, goal, device, optimize=True, time_limit=10.0, baseline=False,
              backend="ipopt", editor=None, body="nominal", repair_evaluations=97, repair_rounds=12):
    from planner_backend import validate_path, optimize_path, make_environment
    started = time.perf_counter()
    for state in [start, goal]:
        if np.shape(state) != (4,) or not np.isfinite(state).all() or abs(state[3]) > articulation_limit + 1e-6:
            raise ValueError("起终状态必须是有限的[x,y,航向,铰接角]，铰接角不能超过35度")
    if backend not in ("ipopt", "event_editor"):
        raise ValueError("未知规划后端")
    if backend == "ipopt" and body != "nominal":
        raise ValueError("身体变化当前通过event_editor后端评估")
    from repair_physics import body_variant
    parameters = body_variant(body)
    environment = make_environment(map_data, parameters)
    endpoints = environment[3].check(np.asarray([start, goal])[:, None, :], exact_clearance=True)
    if not endpoints.safe_mask.all():
        raise ValueError("起点或终点的完整车体碰撞或安全间隙不足，请调整位置、航向或铰接角")
    route = CoarseRoutePlanner(map_data).plan(start, goal)
    prepared = prepare_trajectory_input(map_data, start, goal, route)
    batch = move_batch(collate_trajectories([prepared]), device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    network_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        output = predict_batch(model, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    network_seconds = time.perf_counter() - network_started
    states = restore_states(output["states"][0].cpu().numpy(), map_data)
    directions = decode_directions(output)[0].cpu().numpy()
    quality = validate_path(states, directions, start, goal, map_data, environment=environment)
    result = {"network_seconds": network_seconds, "network_quality": quality,
              "mode_probabilities": torch.softmax(output["mode_logits"], 1)[0].cpu().tolist(),
              "quality_pass": False, "reason": "network_only"}
    path = None
    if optimize:
        if backend == "event_editor":
            from repair_data import controls_from_states
            from repair_physics import RepairProblem
            from repair_loop import run_repair
            controls, control_directions = controls_from_states(states, directions)
            problem = RepairProblem(map_data, start, goal, control_directions, parameters, device)
            problem.environment = environment
            path, optimizer_report, artifacts = run_repair(
                problem, controls, editor, max_rounds=repair_rounds,
                evaluation_budget=repair_evaluations, time_limit=time_limit)
            result["repair_artifacts"] = artifacts
        else:
            path, optimizer_report = optimize_path(states, directions, start, goal, map_data, time_limit, not baseline)
        result["optimizer"] = optimizer_report
        result["quality_pass"] = path is not None
        result["reason"] = optimizer_report["reason"]
    result["total_seconds"] = time.perf_counter() - started
    result["backend"] = backend
    result["body_variant"] = body
    result["body_parameters"] = parameters
    return route, states, directions, path, result


def select_states(map_data, start_theta, goal_theta):
    # 可选交互入口：依次点击起点、起点朝向、终点、终点朝向。
    # 两个朝向点击只决定方向，不作为路径点；铰接角由命令行参数设置。
    import matplotlib
    import matplotlib.pyplot as plt
    # 没有交互绘图后端时立即说明原因，避免等待不存在的选点窗口。
    if matplotlib.get_backend().lower() in ["agg", "pdf", "svg", "ps", "pgf", "cairo", "template"]:
        raise ValueError("当前绘图后端不能接收鼠标，请在桌面终端选点，或使用--start和--goal输入状态")
    fig, ax = plt.subplots()
    ax.imshow(map_data["map_features"][0], origin="lower", cmap="gray_r")
    ax.set_title("Click: start, start heading, goal, goal heading")
    points = plt.ginput(4, timeout=0)
    plt.close(fig)
    if len(points) != 4:
        raise ValueError("需要完成四次点击")
    world = grid_to_world(points, map_data)
    start_delta = world[1] - world[0]
    goal_delta = world[3] - world[2]
    if min(np.linalg.norm(start_delta), np.linalg.norm(goal_delta)) < 1e-6:
        raise ValueError("朝向点不能与车辆位置重合")
    start = np.array([*world[0], np.arctan2(start_delta[1], start_delta[0]), np.deg2rad(start_theta)])
    goal = np.array([*world[2], np.arctan2(goal_delta[1], goal_delta[0]), np.deg2rad(goal_theta)])
    return start, goal


def main():
    parser = argparse.ArgumentParser(description="设定前桥起终状态并保存轨迹图和交互HTML")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--map-id", help="例如map_000022")
    parser.add_argument("--map-file", type=Path, help="自定义地图npz，包含map_features/origin/origin_yaw/resolution")
    parser.add_argument("--start", nargs=4, type=float, metavar=("X", "Y", "HEADING_DEG", "THETA_DEG"))
    parser.add_argument("--goal", nargs=4, type=float, metavar=("X", "Y", "HEADING_DEG", "THETA_DEG"))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--start-theta", type=float, default=0.0)
    parser.add_argument("--goal-theta", type=float, default=0.0)
    parser.add_argument("--sample-index", type=int, help="只读取示例任务的地图和起终状态")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--reference", action="store_true", help="额外画出参考轨迹，仅用于对比，不传给模型")
    parser.add_argument("--network-only", action="store_true")
    parser.add_argument("--original-initialization", action="store_true")
    parser.add_argument("--backend", choices=["ipopt", "event_editor"], default="ipopt")
    parser.add_argument("--editor-checkpoint", type=Path)
    parser.add_argument("--body-variant", choices=["nominal", "wide", "long", "slow", "wide_slow", "long_slow"], default="nominal")
    parser.add_argument("--repair-evaluations", type=int, default=97)
    parser.add_argument("--repair-rounds", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=10.0,
                        help="IPOPT为CPU求解预算；event_editor为含独立验收的修复墙钟预算；均不含绘图")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("planning_results/manual"))
    args = parser.parse_args()
    if args.time_limit <= 0 or min(args.repair_evaluations, args.repair_rounds) < 1:
        parser.error("求解时间预算必须为正")
    if args.backend == "event_editor" and (args.editor_checkpoint is None or args.original_initialization or args.network_only):
        parser.error("event_editor需要editor-checkpoint，不能同时使用original-initialization或network-only")
    if args.backend == "ipopt" and (args.editor_checkpoint or args.body_variant != "nominal"):
        parser.error("编辑checkpoint和身体变化需要--backend event_editor")
    if args.sample_index is not None and (args.start or args.goal or args.interactive or args.map_id or args.map_file):
        parser.error("示例任务模式与手动起终点模式分别使用")
    if args.map_file and args.map_id:
        parser.error("map-file与map-id只能选择一个")
    reference = None
    map_id = args.map_id or ""
    map_path = ""
    if args.sample_index is not None:
        records = read_index_records(args.data_dir, args.split)
        if args.sample_index < 0 or args.sample_index >= len(records):
            parser.error("sample-index超出集合范围")
        record = records[args.sample_index]
        map_id, map_path = record["map_id"], record["map_path"]
        with np.load(args.data_dir / record["sample_path"], allow_pickle=False) as sample:
            start, goal = sample["start_state"].astype(float), sample["goal_state"].astype(float)
            if args.reference:
                reference = sample["trajectory_states"]
    else:
        if not (args.map_id or args.map_file):
            parser.error("请设置--map-id或--map-file")
        map_path = "maps/" + map_id + ".npz" if map_id else str(args.map_file.resolve())
        if not args.interactive and (args.start is None or args.goal is None):
            parser.error("请同时设置--start和--goal，或者使用--interactive")
        if args.reference:
            parser.error("--reference仅用于示例任务，手动任务没有参考答案")
        if not args.interactive:
            start = np.asarray(args.start)
            goal = np.asarray(args.goal)
            start[2:] = np.deg2rad(start[2:])
            goal[2:] = np.deg2rad(goal[2:])
    map_data = read_map_data(args.data_dir, {"map_path": map_path})
    if args.interactive:
        try:
            start, goal = select_states(map_data, args.start_theta, args.goal_theta)
        except ValueError as error:
            parser.error(str(error))
    torch.set_num_threads(2)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    editor, editor_checkpoint = None, None
    if args.backend == "event_editor":
        from event_editor import load_editor
        editor, editor_checkpoint = load_editor(args.editor_checkpoint, device)
        if editor_checkpoint["base_checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
            parser.error("生成器checkpoint与编辑器训练时使用的版本不一致")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_previous_result(args.output_dir)
    try:
        route, states, directions, path, result = plan_task(
            model, map_data, start, goal, device, not args.network_only, args.time_limit, args.original_initialization,
            args.backend, editor, args.body_variant, args.repair_evaluations, args.repair_rounds,
        )
    except ValueError as error:
        result = {"quality_pass": False, "reason": str(error), "start": start.tolist(), "goal": goal.tolist()}
        (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(str(error))
    result.update({"checkpoint": str(args.checkpoint.resolve()), "training_epoch": checkpoint["epoch"],
                   "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                   "use_sequence_directions": checkpoint.get("use_sequence_directions", False),
                   "map_id": map_id, "map_path": map_path, "data_dir": str(args.data_dir.resolve()),
                   "start": start.tolist(), "goal": goal.tolist(), "integration_dt": integration_dt})
    artifacts = result.pop("repair_artifacts", None)
    if editor_checkpoint is not None:
        result.update({"editor_checkpoint": str(args.editor_checkpoint.resolve()),
                       "editor_sha256": editor_checkpoint["sha256"], "editor_epoch": editor_checkpoint["epoch"]})
    # 成功运行后应看到result.json、trajectory.npz、trajectory.png和trajectory.html。
    saved = {"start_state": start, "goal_state": goal, "route": route,
             "network_states": states, "network_directions": directions,
             "map_id": map_id, "map_path": map_path, "data_dir": str(args.data_dir.resolve()),
             "quality_pass": result["quality_pass"], "integration_dt": integration_dt}
    optimized_states = None
    optimized_directions = None
    if artifacts is not None:
        saved.update({"repair_" + key: value for key, value in artifacts.items()})
    if path is not None:
        saved.update({"trajectory_states": path.states, "trajectory_controls": path.controls,
                      "trajectory_directions": path.directions})
        optimized_states = path.states
        optimized_directions = path.directions[1:]
    np.savez_compressed(args.output_dir / "trajectory.npz", **saved)
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if artifacts is not None:
        save_visualization(args.output_dir, map_data, start, goal, route,
                           artifacts["initial_states"], artifacts["directions"],
                           artifacts["final_states"], artifacts["directions"], reference,
                           prediction_label="Initial rollout", parameters=result["body_parameters"],
                           optimized_label="Accepted repair" if path else "Failed repair")
    else:
        save_visualization(args.output_dir, map_data, start, goal, route, states, directions, optimized_states, optimized_directions, reference)
    print("规划结果：", result["reason"], "通过质量检查：", result["quality_pass"])
    print("请打开：", (args.output_dir / "trajectory.html").resolve())


if __name__ == "__main__":
    main()
