"""运行 python verify_workflow.py 检查关键边界，不会修改原始数据。"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from coarse_route import CoarseRoutePlanner, world_to_grid, grid_to_world
from trajectory_data import resample_reference, normalize_states, restore_states, prepare_trajectory_input, collate_trajectories
from loader_model import build_trajectory_model, predict_batch
from planner_backend import make_environment, optimize_path, validate_path
from pipelines.teacher.fast_dynamics import NumpyArticulatedRollout
from trajectory_loss import trajectory_loss
from trajectory_decoder import decode_directions


def make_map():
    # 检查1：使用矩形地图，避免正方形地图掩盖宽高交换错误。
    occupancy = np.zeros((80, 100), dtype=np.float32)
    occupancy[[0, -1], :] = 1
    occupancy[:, [0, -1]] = 1
    return {"map_features": np.stack([occupancy, distance_transform_edt(1 - occupancy) * 0.5,
                                       np.zeros_like(occupancy)]).astype(np.float32),
            "origin": np.zeros(2), "origin_yaw": 0.0, "resolution": 0.5}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    map_data = make_map()
    environment = make_environment(map_data)
    rollout = NumpyArticulatedRollout.from_vehicle(environment[1])
    start = np.array([18., 20., 0., 0.2])
    controls = np.zeros((100, 2))
    controls[:40, 0] = -1.0
    controls[40:, 0] = 1.0
    states = [start]
    for control in controls:
        state, applied = rollout.step_rk4(states[-1], control, 0.1)
        states.append(state)
    states = np.asarray(states)
    goal = states[-1]
    directions = np.concatenate([[-1], np.sign(controls[:, 0]).astype(int)])
    sample = {"trajectory_states": states, "trajectory_controls": controls,
              "trajectory_directions": directions, "switch_states": states[40:41],
              "phase_directions": np.array([-1, 1])}
    target, edges, mode = resample_reference(sample)
    assert mode == 3 and np.all(edges[:32] == -1) and np.all(edges[32:] == 1)
    np.testing.assert_allclose(target[32], states[40], atol=2e-6)
    assert abs(target[32, 3]) > 0.1

    # 检查2：旋转地图、平移地图时，归一化表示和逆变换必须互相对应。
    rotated_map = dict(map_data, origin=np.array([12., -7.]), origin_yaw=0.8)
    grid = world_to_grid(states[:, :2], map_data)
    rotated_states = states.copy()
    rotated_states[:, :2] = grid_to_world(grid, rotated_map)
    rotated_states[:, 2] += 0.8
    normalized = normalize_states(states, map_data)
    np.testing.assert_allclose(normalized, normalize_states(rotated_states, rotated_map), atol=2e-6)
    np.testing.assert_allclose(restore_states(normalize_states(rotated_states, rotated_map), rotated_map), rotated_states, atol=5e-6)

    # 检查3：跨越pi的航向使用圆周插值，不从179度错误地插到0度。
    from trajectory_data import resample_states
    wrap_states = np.array([[0., 0., np.deg2rad(179), 0.], [1., 0., np.deg2rad(-179), 0.]])
    assert abs(resample_states(wrap_states, 3)[1, 2]) > 3.0

    # 检查4：不同长度输入补齐后，单条推理和批量推理结果一致；反向传播确实更新参数。
    route = CoarseRoutePlanner(map_data).plan(start, goal)
    first = prepare_trajectory_input(map_data, start, goal, route)
    longer_route = np.concatenate([route, route[-1:]], axis=0)
    second = prepare_trajectory_input(map_data, start, goal, longer_route)
    batch = collate_trajectories([first, second])
    model = build_trajectory_model()
    model.eval()
    with torch.no_grad():
        output = predict_batch(model, batch)
        single = predict_batch(model, collate_trajectories([first]))
    torch.testing.assert_close(output["states"][0], single["states"][0], atol=1e-5, rtol=1e-5)
    assert output["states"].shape == (2, 64, 5)
    assert set(decode_directions(output).flatten().tolist()).issubset({-1, 1})
    batch["target_states"] = normalize_states(target, map_data).unsqueeze(0).repeat(2, 1, 1)
    batch["target_directions"] = torch.tensor(edges > 0, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    batch["target_mode"] = torch.tensor([mode, mode])
    batch["map_size_m"] = torch.tensor([[50., 40.], [50., 40.]])
    batch["map_data"] = [map_data, map_data]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    before = model.trajectory_decoder.state_head.weight.detach().clone()
    output = predict_batch(model, batch)
    loss, metrics = trajectory_loss(output, batch, clearance_weight=0.1)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.trajectory_decoder.state_head.weight)
    # 让两个方向分支故意给出相反答案，确认可选序列解码使用区间证据并保持连续挡位。
    with torch.no_grad():
        model.trajectory_decoder.mode_head.weight.zero_()
        model.trajectory_decoder.mode_head.bias.copy_(torch.tensor([0., 0., 0., 5.]))
        model.trajectory_decoder.direction_head.weight.zero_()
        model.trajectory_decoder.direction_head.bias.fill_(10.)
        model.trajectory_decoder.use_sequence_directions = True
        coherent = predict_batch(model, batch)
    assert torch.all(coherent["mode_logits"].argmax(1) == 0)
    assert torch.all(decode_directions(coherent) == 1)

    # 检查5：已知RK4轨迹通过；篡改中间状态或控制会被拒绝，不能因首尾正确就通过。
    valid = validate_path(states, directions[1:], start, goal, map_data, controls, environment)
    assert valid["quality_pass"], valid
    wrong = states.copy()
    wrong[10, 1] += 0.2
    assert not validate_path(wrong, directions[1:], start, goal, map_data, controls, environment)["quality_pass"]
    wrong_controls = controls.copy()
    wrong_controls[0, 0] = 2.0
    assert not validate_path(states, directions[1:], start, goal, map_data, wrong_controls, environment)["quality_pass"]
    collision_states = states.copy()
    collision_states[:, 0] = 3.0
    assert not validate_path(collision_states, directions[1:], collision_states[0], collision_states[-1], map_data)["safe"]

    # 检查6：网络格式的非零铰接换挡初值真正进入IPOPT，求解后再次通过独立检查。
    path, report = optimize_path(target, edges, start, goal, map_data, time_limit=5.0)
    assert path is not None, report
    assert report["quality"]["quality_pass"] and report["quality"]["switch_count"] == 1
    switches = np.flatnonzero(path.directions[2:] != path.directions[1:-1]) + 1
    assert abs(path.states[switches[0], 3]) > 0.01
    result = {"checks_passed": 6, "loss_with_clearance": metrics, "optimizer": report}
    output_path = Path("runs/verification.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("6组关键检查通过，报告：", output_path)


if __name__ == "__main__":
    main()
