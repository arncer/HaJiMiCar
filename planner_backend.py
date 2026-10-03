"""复用已有教师的RK4、完整车体检查和CasADi/IPOPT，不改动教师仓库。"""
import sys
import time
import math
import numpy as np
import torch
from config import teacher_dir, vehicle_parameters, integration_dt
from trajectory_data import resample_states

if not (teacher_dir / "pipelines/teacher/casadi_route.py").is_file():
    raise ImportError("找不到教师代码，请修改config.py中的teacher_dir：" + str(teacher_dir))
if str(teacher_dir) not in sys.path:
    sys.path.insert(0, str(teacher_dir))
from data.maps import OccupancyGrid
from models.kinematics import ArticulatedVehicleModel
from pipelines.safety import ConservativeRasterSafetyOracle
from pipelines.teacher.fast_safety import HierarchicalFootprintSafety
from pipelines.teacher.fast_dynamics import NumpyArticulatedRollout
from pipelines.teacher.casadi_route import CasadiRouteTeacher, CasadiRouteConfig


def make_environment(map_data, parameters=None):
    # 第1步：用相同车辆尺寸和地图坐标建立教师环境。
    # 成功后碰撞检查覆盖铲斗、前车体和后车体，采样0.2米、安全裕度0.1米。
    parameters = dict(vehicle_parameters if parameters is None else parameters)
    grid = OccupancyGrid(
        occupancy=torch.tensor(map_data["map_features"][0:1]),
        esdf_m=torch.tensor(map_data["map_features"][1:2]),
        resolution_m=float(map_data["resolution"]),
        origin_xy=tuple(float(x) for x in map_data["origin"]),
        origin_yaw=float(map_data["origin_yaw"]),
    )
    vehicle = ArticulatedVehicleModel(
        front_axle_to_hitch_m=parameters["front_axle_to_hitch_m"],
        hitch_to_rear_axle_m=parameters["hitch_to_rear_axle_m"],
        articulation_limit_rad=parameters["articulation_limit_rad"],
        max_speed_mps=parameters["max_speed_mps"],
        max_articulation_rate_rps=parameters["max_articulation_rate_rps"],
    )
    oracle = ConservativeRasterSafetyOracle(
        front_axle_to_hitch_m=parameters["front_axle_to_hitch_m"],
        hitch_to_rear_axle_m=parameters["hitch_to_rear_axle_m"],
        bucket_front_m=parameters["bucket_front_m"],
        rear_axle_to_tail_m=parameters["rear_axle_to_tail_m"],
        front_body_width_m=parameters["front_body_width_m"],
        bucket_width_m=parameters["bucket_width_m"],
        rear_body_width_m=parameters["rear_body_width_m"],
        footprint_sample_spacing_m=0.2, safety_margin_m=0.1,
        articulation_limit_rad=parameters["articulation_limit_rad"],
    )
    safety = HierarchicalFootprintSafety(grid=grid, oracle=oracle, cover_cell_size_m=1.0)
    return grid, vehicle, oracle, safety


def validate_path(states, directions, start, goal, map_data, controls=None, environment=None):
    # 第2步：分别记录几何检查和控制回放检查；没有控制的网络输出不能标记为可执行。
    result = {"quality_pass": False, "replay_ok": False, "safe": False, "reason": "invalid_structure"}
    states = np.asarray(states, dtype=np.float64)
    directions = np.asarray(directions)
    if states.ndim != 2 or states.shape[1] != 4 or len(states) < 2:
        return result
    if directions.shape != (len(states) - 1,) or not np.isin(directions, [-1, 1]).all() or not np.isfinite(states).all():
        return result
    if environment is None:
        environment = make_environment(map_data)
    grid, vehicle, oracle, safety = environment
    windows = np.stack([states[:-1], states[1:]], axis=1)
    minimum_clearance = float("inf")
    minimum_margin = float("inf")
    safe = True
    for begin in range(0, len(windows), 64):
        checked = safety.check(windows[begin:begin + 64], exact_clearance=True)
        safe = safe and bool(checked.safe_mask.all())
        minimum_clearance = min(minimum_clearance, float(checked.minimum_clearance_m.min()))
        minimum_margin = min(minimum_margin, float((checked.minimum_clearance_m - checked.required_clearance_m).min()))
    position_error = float(np.linalg.norm(states[-1, :2] - goal[:2]))
    heading_error = abs(math.atan2(math.sin(states[-1, 2] - goal[2]), math.cos(states[-1, 2] - goal[2])))
    articulation_error = abs(float(states[-1, 3] - goal[3]))
    start_difference = states[0] - start
    start_difference[2] = math.atan2(math.sin(start_difference[2]), math.cos(start_difference[2]))
    start_ok = np.max(np.abs(start_difference)) < 1e-4
    terminal_ok = position_error <= 0.30 and heading_error <= math.radians(5) and articulation_error <= math.radians(3)
    articulation_ok = bool(np.max(np.abs(states[:, 3])) <= vehicle.articulation_limit_rad + 1e-6)
    result.update({"safe": safe, "terminal_ok": terminal_ok, "start_ok": bool(start_ok),
                   "articulation_ok": articulation_ok, "position_error_m": position_error,
                   "heading_error_deg": math.degrees(heading_error), "articulation_error_deg": math.degrees(articulation_error),
                   "minimum_clearance_m": minimum_clearance, "minimum_margin_m": minimum_margin,
                   "path_length_m": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()),
                   "switch_count": int(np.count_nonzero(directions[1:] != directions[:-1]))})
    # 额外诊断：比较车头朝向与当前挡位下的路径切线，帮助定位“位置看着对，但姿态不对”。
    # 这里只作几何诊断，不把没有时间和控制量的网络输出当成已经通过运动学验证。
    delta_xy = np.diff(states[:, :2], axis=0)
    segment_lengths = np.linalg.norm(delta_xy, axis=1)
    moving = segment_lengths > 1e-5
    heading_change = np.arctan2(np.sin(np.diff(states[:, 2])), np.cos(np.diff(states[:, 2])))
    middle_heading = states[:-1, 2] + heading_change * 0.5
    path_heading = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
    path_heading = path_heading + np.where(directions < 0, math.pi, 0.0)
    heading_difference = path_heading - middle_heading
    heading_difference = np.abs(np.arctan2(np.sin(heading_difference), np.cos(heading_difference)))
    if moving.any():
        result["path_heading_error_median_deg"] = float(np.degrees(np.median(heading_difference[moving])))
        result["path_heading_error_p95_deg"] = float(np.degrees(np.percentile(heading_difference[moving], 95)))
    result["reason"] = "geometric_prediction_without_controls"
    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.shape != (len(states) - 1, 2) or not np.isfinite(controls).all():
            result["reason"] = "invalid_controls"
            return result
        rollout = NumpyArticulatedRollout.from_vehicle(vehicle)
        replay = [np.asarray(start, dtype=np.float64).copy()]
        bound_error = 0.0
        for control in controls:
            state, applied = rollout.step_rk4(replay[-1], control, integration_dt)
            replay.append(state)
            bound_error = max(bound_error, float(np.max(np.abs(applied - control))))
        differences = np.asarray(replay) - states
        differences[:, 2] = np.arctan2(np.sin(differences[:, 2]), np.cos(differences[:, 2]))
        replay_error = float(np.abs(differences).max())
        direction_ok = bool(np.array_equal(np.sign(controls[:, 0]), directions))
        replay_ok = replay_error < 1e-3 and bound_error < 1e-6 and direction_ok
        result.update({"replay_ok": replay_ok, "replay_max_error": replay_error,
                       "control_bound_error": bound_error, "direction_ok": direction_ok})
        result["quality_pass"] = bool(safe and terminal_ok and start_ok and articulation_ok and replay_ok)
        result["reason"] = "passed" if result["quality_pass"] else "physical_postcheck_failed"
    return result


class NetworkInitializedTeacher(CasadiRouteTeacher):
    def __init__(self, predicted_states, predicted_directions, **kwargs):
        super().__init__(**kwargs)
        self.predicted_states = predicted_states
        self.predicted_directions = predicted_directions

    def _solve(self, *, start, goal, reference, gates, time_limit_s,
               enforce_collision_constraints=None, warm_start=None,
               repulsion_weight_multiplier=1.0, repulsion_buffer_multiplier=1.0):
        # 第3步：把网络状态按阶段重采样到教师的实际积分网格，作为IPOPT初值。
        # 这里只估计初始控制；最终控制由约束优化求解并用0.1秒RK4重新回放。
        if warm_start is None:
            boundaries = np.flatnonzero(self.predicted_directions[1:] != self.predicted_directions[:-1]) + 1
            input_stops = [0] + boundaries.tolist() + [len(self.predicted_states) - 1]
            output_stops = [0] + list(reference.phase_stop_control_indices)
            parts = []
            for phase in range(len(output_stops) - 1):
                phase_states = self.predicted_states[input_stops[phase]:input_stops[phase + 1] + 1]
                count = output_stops[phase + 1] - output_stops[phase] + 1
                part = resample_states(phase_states, count).astype(np.float64)
                parts.append(part if phase == 0 else part[1:])
            initial_states = np.concatenate(parts)
            initial_states[0] = start
            initial_states[-1] = goal
            initial_states[:, 2] = np.unwrap(initial_states[:, 2])
            speed = np.linalg.norm(np.diff(initial_states[:, :2], axis=0), axis=1) / self.config.macro_dt
            speed = np.clip(speed, self.config.minimum_speed_mps, self.vehicle.max_speed_mps)
            speed *= reference.control_directions
            rate = np.diff(initial_states[:, 3]) / self.config.macro_dt
            rate = np.clip(rate, -self.vehicle.max_articulation_rate_rps, self.vehicle.max_articulation_rate_rps)
            warm_start = (initial_states.T, np.stack([speed, rate]))
        return super()._solve(
            start=start, goal=goal, reference=reference, gates=gates, time_limit_s=time_limit_s,
            enforce_collision_constraints=enforce_collision_constraints, warm_start=warm_start,
            repulsion_weight_multiplier=repulsion_weight_multiplier,
            repulsion_buffer_multiplier=repulsion_buffer_multiplier,
        )


def optimize_path(states, directions, start, goal, map_data, time_limit=10.0, use_network_initialization=True):
    started = time.perf_counter()
    environment = make_environment(map_data)
    grid, vehicle, oracle, safety = environment
    # 第4步：先检查起终车辆轮廓，避免中心在空地但车尾已穿墙的任务继续求解。
    endpoint_check = safety.check(np.asarray([start, goal])[:, None, :], exact_clearance=True)
    if not endpoint_check.safe_mask.all():
        return None, {"reason": "endpoint_body_collision", "optimizer_seconds": time.perf_counter() - started, "iterations": 0}
    config = CasadiRouteConfig(enforce_collision_constraints=True, use_topology_cusp_hint=True,
                               articulation_limit_buffer_rad=0.0, ipopt_max_iterations=400)
    arguments = {"grid": grid, "vehicle": vehicle, "safety_oracle": oracle, "config": config}
    if use_network_initialization:
        teacher = NetworkInitializedTeacher(states, directions, **arguments)
    else:
        teacher = CasadiRouteTeacher(**arguments)
    boundaries = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    word = [int(directions[0])]
    for boundary in boundaries:
        word.append(int(directions[boundary]))
    cusp = states[int(boundaries[0]), :2] if len(boundaries) else None
    try:
        path = teacher.plan(start, goal, start_direction=word[0], required_direction_word=tuple(word),
                            guide_route_world_xy=states[:, :2], cusp_hint_xy=cusp,
                            time_limit_s=time_limit)
        reason = teacher.last_termination_reason
    except (ValueError, RuntimeError) as error:
        path = None
        reason = str(error)
    # 教师的总耗时包含建模，失败分类优先采用IPOPT实际状态，避免把所有慢失败都叫超时。
    if path is None:
        if teacher.last_solver_status == "Maximum_CpuTime_Exceeded":
            reason = "timeout"
        elif teacher.last_solver_status == "Infeasible_Problem_Detected":
            reason = "infeasible"
        elif teacher.last_solver_status == "Maximum_Iterations_Exceeded":
            reason = "iteration_limit"
    iterations = teacher.last_expanded_nodes
    if path is None and teacher.last_solver_status != "not_started":
        # 现有教师没有暴露失败求解的迭代数，写null表示未知，不能伪装成零次迭代。
        iterations = None
    report = {"reason": reason, "solver_status": teacher.last_solver_status,
              "iterations": iterations,
              "optimizer_seconds": time.perf_counter() - started,
              "initialization": "network" if use_network_initialization else "original"}
    if path is not None:
        quality = validate_path(path.states, path.directions[1:], start, goal, map_data, path.controls, environment)
        report["quality"] = quality
        if not quality["quality_pass"]:
            path = None
            report["reason"] = "independent_postcheck_failed"
    return path, report
