"""事件编辑的可微批量推演、车体失败证据和独立验收。"""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import vehicle_parameters, integration_dt


BODY_KEYS = tuple(vehicle_parameters)
KNOTS_PER_PHASE = 8
KNOT_COUNT = 2 * KNOTS_PER_PHASE
TOKEN_COUNT = 64
TOKEN_DIM = 29
CONTEXT_DIM = 13 + len(BODY_KEYS)
EDIT_SCALES = (0.5, 0.12)  # 米/秒、弧度/秒；每轮编辑均在此尺度内。
FAMILY_NAMES = ("early_steering", "early_speed", "late_steering", "late_speed",
                "local_steering", "local_speed", "joint")


def body_variant(name):
    """已知单因素用于训练；组合变化只用于留出评估。"""
    result = dict(vehicle_parameters)
    if name in ("wide", "wide_slow"):
        for key in ("front_body_width_m", "bucket_width_m", "rear_body_width_m"):
            result[key] *= 1.15
    if name in ("long", "long_slow"):
        result["rear_axle_to_tail_m"] *= 1.2
    if name in ("slow", "wide_slow", "long_slow"):
        result["max_articulation_rate_rps"] *= 0.8
    if name not in ("nominal", "wide", "long", "slow", "wide_slow", "long_slow"):
        raise ValueError("未知身体条件：" + name)
    return result


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def rollout_controls(start, requested, parameters, dt=integration_dt):
    """[B,T,2]→[B,T+1,4]。利用本模型的三角结构并行计算RK4。

    theta仅依赖控制；航向增量仅依赖theta与控制，因此能先累加theta和
    航向，再按原RK4的四个阶段累加位置。它不是把Euler当作RK4。
    先构造有界theta序列，再取差得到实际施加的角速度，最终独立重放
    使用这些实际控制；不会把控制裁剪造成的状态误差隐藏起来。
    """
    if requested.ndim != 3 or requested.shape[-1] != 2:
        raise ValueError("控制应为[B,T,2]")
    start = start.expand(requested.shape[0], -1)
    speed = requested[..., 0].clamp(-parameters["max_speed_mps"], parameters["max_speed_mps"])
    rate = requested[..., 1].clamp(-parameters["max_articulation_rate_rps"],
                                  parameters["max_articulation_rate_rps"])
    theta_end = (start[:, 3:4] + torch.cumsum(rate * dt, dim=1)).clamp(
        -parameters["articulation_limit_rad"], parameters["articulation_limit_rad"])
    theta = torch.cat([start[:, 3:4], theta_end], dim=1)
    rate = torch.diff(theta, dim=1) / dt
    lf, lr = parameters["front_axle_to_hitch_m"], parameters["hitch_to_rear_axle_m"]

    def heading_rate(angle):
        return (speed * torch.sin(angle) + lr * rate) / (lf * torch.cos(angle) + lr)

    h1 = heading_rate(theta[:, :-1])
    h2 = heading_rate(theta[:, :-1] + 0.5 * dt * rate)
    h4 = heading_rate(theta[:, 1:])
    heading_delta = dt / 6 * (h1 + 4 * h2 + h4)
    heading = torch.cat([start[:, 2:3], start[:, 2:3] + torch.cumsum(heading_delta, 1)], 1)
    psi = heading[:, :-1]
    psi2, psi3, psi4 = psi + dt * h1 / 2, psi + dt * h2 / 2, psi + dt * h2
    dx = speed * dt / 6 * (torch.cos(psi) + 2 * torch.cos(psi2) + 2 * torch.cos(psi3) + torch.cos(psi4))
    dy = speed * dt / 6 * (torch.sin(psi) + 2 * torch.sin(psi2) + 2 * torch.sin(psi3) + torch.sin(psi4))
    x = torch.cat([start[:, 0:1], start[:, 0:1] + dx.cumsum(1)], 1)
    y = torch.cat([start[:, 1:2], start[:, 1:2] + dy.cumsum(1)], 1)
    return torch.stack([x, y, wrap_angle(heading), theta], -1), torch.stack([speed, rate], -1)


def phase_basis(directions, device):
    directions = np.asarray(directions)
    boundaries = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    if len(boundaries) > 1 or not np.isin(directions, [-1, 1]).all():
        raise ValueError("事件编辑目前支持零次或一次换挡")
    switch = int(boundaries[0]) if len(boundaries) else len(directions)
    basis = np.zeros((len(directions), KNOT_COUNT), dtype=np.float32)
    for phase, (low, high) in enumerate(((0, switch), (switch, len(directions)))):
        if high <= low:
            continue
        position = np.linspace(0, KNOTS_PER_PHASE - 1, high - low)
        for knot in range(KNOTS_PER_PHASE):
            basis[low:high, phase * KNOTS_PER_PHASE + knot] = np.maximum(1 - np.abs(position - knot), 0)
    return torch.tensor(basis, device=device), switch


@dataclass
class RepairProblem:
    map_data: dict
    start: np.ndarray
    goal: np.ndarray
    directions: np.ndarray
    parameters: dict
    device: torch.device

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=np.float64)
        self.goal = np.asarray(self.goal, dtype=np.float64)
        self.directions = np.asarray(self.directions, dtype=np.int64)
        self.parameters = dict(self.parameters)
        self.device = torch.device(self.device)
        if self.start.shape != (4,) or self.goal.shape != (4,) or len(self.directions) < 2:
            raise ValueError("起终点或控制长度无效")
        if not np.isfinite(self.start).all() or not np.isfinite(self.goal).all():
            raise ValueError("起终点包含非有限值")
        if min(self.parameters.values()) <= 0 or self.parameters["articulation_limit_rad"] >= math.pi / 2:
            raise ValueError("车辆尺寸、速度界必须为正，铰接界小于90度")
        self.start_tensor = torch.tensor(self.start, dtype=torch.float32, device=self.device)
        self.goal_tensor = torch.tensor(self.goal, dtype=torch.float32, device=self.device)
        self.direction_tensor = torch.tensor(self.directions, dtype=torch.float32, device=self.device)
        self.basis, self.switch = phase_basis(self.directions, self.device)
        self.height, self.width = self.map_data["map_features"].shape[1:]
        self.resolution = float(self.map_data["resolution"])
        self.origin = torch.tensor(self.map_data["origin"], dtype=torch.float32, device=self.device)
        self.yaw = float(self.map_data["origin_yaw"])
        self.distance_map = torch.tensor(self.map_data["map_features"][1], device=self.device).float()[None, None]
        raw_map = torch.tensor(self.map_data["map_features"][:2], device=self.device).float()[None]
        self.map_image = F.interpolate(raw_map, (64, 64), mode="bilinear", align_corners=True)[0]
        self.map_image[1] = (self.map_image[1] / 10).clamp(-1, 1)
        self.sample_indices = self._indices()
        self.environment = None

    def _indices(self):
        if self.switch < len(self.directions):
            indices = np.r_[np.linspace(0, self.switch, TOKEN_COUNT // 2, dtype=int),
                            np.linspace(self.switch, len(self.directions), TOKEN_COUNT // 2, dtype=int)]
        else:
            indices = np.linspace(0, len(self.directions), TOKEN_COUNT, dtype=int)
        return torch.tensor(indices, device=self.device)

    def project_controls(self, controls):
        # 每个阶段保持原方向词，不能用高频前后切换伪造修复。
        magnitude = (controls[..., 0] * self.direction_tensor).clamp(0.01, self.parameters["max_speed_mps"])
        rate = controls[..., 1].clamp(-self.parameters["max_articulation_rate_rps"],
                                     self.parameters["max_articulation_rate_rps"])
        return torch.stack([magnitude * self.direction_tensor, rate], -1)

    def apply_edit(self, controls, delta):
        if controls.ndim == 2:
            controls = controls[None]
        if delta.ndim == 2:
            delta = delta[None]
        if delta.shape[1:] != (KNOT_COUNT, 2):
            raise ValueError("编辑系数形状必须为[B,16,2]")
        change = torch.einsum("tk,bkc->btc", self.basis, delta) * delta.new_tensor(EDIT_SCALES)
        return self.project_controls(controls + change)

    def _map_xy(self, points):
        difference = points - self.origin
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return torch.stack([cosine * difference[..., 0] + sine * difference[..., 1],
                            -sine * difference[..., 0] + cosine * difference[..., 1]], -1) / self.resolution

    def _distance(self, points):
        shape = points.shape[:-1]
        xy = self._map_xy(points)
        normalizer = xy.new_tensor([self.width - 1, self.height - 1])
        grid = 2 * xy / normalizer - 1
        # 所有候选查询同一张图，不复制B份地图。
        values = F.grid_sample(self.distance_map, grid.reshape(1, 1, -1, 2),
                               align_corners=True, padding_mode="zeros").reshape(shape)
        outside = torch.maximum(-xy, xy - normalizer).clamp_min(0).amax(-1) * self.resolution
        return torch.where(outside > 0, -outside - 0.1, values)

    def body_evidence(self, states):
        # 固定网格覆盖三个完整矩形的内部，最终仍交由0.2米独立检查器验收。
        p = self.parameters
        fractions_x = torch.linspace(0, 1, 11, device=self.device)
        fractions_y = torch.linspace(-0.5, 0.5, 9, device=self.device)
        fx, fy = torch.meshgrid(fractions_x, fractions_y, indexing="ij")
        parts = ((-p["front_axle_to_hitch_m"], 0., p["front_body_width_m"], False),
                 (0., p["bucket_front_m"], p["bucket_width_m"], False),
                 (-p["hitch_to_rear_axle_m"] - p["rear_axle_to_tail_m"], 0., p["rear_body_width_m"], True))
        clearances, gradients, margins = [], [], []
        for low, high, width, rear in parts:
            angle = states[..., 2] - (states[..., 3] if rear else 0)
            center = states[..., :2]
            if rear:
                center = center - p["front_axle_to_hitch_m"] * torch.stack(
                    [torch.cos(states[..., 2]), torch.sin(states[..., 2])], -1)
            x, y = low + (high - low) * fx.flatten(), width * fy.flatten()
            cosine, sine = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
            points = center[..., None, :] + torch.stack([cosine * x - sine * y, sine * x + cosine * y], -1)
            distance = self._distance(points)
            minimum, index = distance.min(-1)
            worst_point = points.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2)
            grad = []
            for axis in range(2):
                offset = points.new_zeros(2)
                offset[axis] = self.resolution
                grad.append((self._distance(worst_point + offset) - self._distance(worst_point - offset)) / (2 * self.resolution))
            clearances.append(minimum)
            gradients.append(torch.stack(grad, -1))
            margins.append(0.1 + math.hypot((high - low) / 20, width / 16) + self.resolution / math.sqrt(2))
        return torch.stack(clearances, -1), torch.stack(gradients, -2), states.new_tensor(margins)

    def evaluate(self, controls, features=False):
        controls = torch.as_tensor(controls, device=self.device, dtype=torch.float32)
        if controls.ndim == 2:
            controls = controls[None]
        projected = self.project_controls(controls)
        states, applied = rollout_controls(self.start_tensor, projected, self.parameters)
        sampled = states[:, self.sample_indices]
        clearance, gradient, margins = self.body_evidence(sampled)
        violation = F.relu(margins - clearance)
        terminal = states[:, -1] - self.goal_tensor
        terminal = torch.cat([terminal[:, :2], wrap_angle(terminal[:, 2:3]), terminal[:, 3:4]], -1)
        terminal_cost = 4 * terminal[:, :2].square().sum(-1) + 10 * terminal[:, 2].square() + 10 * terminal[:, 3].square()
        collision_cost = 100 * violation.square().mean((1, 2)) + 20 * violation.square().amax((1, 2))
        same_gear = (self.direction_tensor[1:] == self.direction_tensor[:-1]).float()
        smoothness = (torch.diff(applied, dim=1).square().sum(-1) * same_gear).mean(1)
        score = terminal_cost + collision_cost + 0.02 * smoothness
        result = {"score": score, "states": states, "controls": applied,
                  "clearance": clearance, "violation": violation, "terminal": terminal,
                  "terminal_cost": terminal_cost, "collision_cost": collision_cost,
                  "switch_state": states[:, self.switch], "smoothness": smoothness}
        if features:
            result.update(self._features(sampled, applied, clearance, gradient, violation))
        return result

    def _features(self, sampled, applied, clearance, gradient, violation):
        p = self.parameters
        cosine, sine = math.cos(self.start[2]), math.sin(self.start[2])

        def local(vector):
            return torch.stack([cosine * vector[..., 0] + sine * vector[..., 1],
                                -sine * vector[..., 0] + cosine * vector[..., 1]], -1)

        xy = local(sampled[..., :2] - self.start_tensor[:2]) / 20
        angle = sampled[..., 2] - self.start[2]
        index = self.sample_indices.clamp_max(len(self.directions) - 1)
        control = applied[:, index] / applied.new_tensor([p["max_speed_mps"], p["max_articulation_rate_rps"]])
        progress = self.sample_indices.float() / len(self.directions)
        phase = (self.sample_indices >= self.switch).float()
        count = len(sampled)
        one = sampled.new_ones(count, TOKEN_COUNT, 1)
        heading_goal = self.goal[2] - sampled[..., 2]
        tokens = torch.cat([
            xy, torch.sin(angle)[..., None], torch.cos(angle)[..., None],
            sampled[..., 3:4] / p["articulation_limit_rad"], one * progress[None, :, None],
            control, one * self.direction_tensor[index][None, :, None], one * phase[None, :, None],
            clearance.clamp(-5, 10) / 5, violation.clamp_max(5) / 5,
            local(gradient).flatten(-2).clamp(-2, 2),
            one * ((self.sample_indices - self.switch).float() / len(self.directions))[None, :, None],
            local(self.goal_tensor[:2] - sampled[..., :2]) / 20,
            torch.sin(heading_goal)[..., None], torch.cos(heading_goal)[..., None],
            self._map_xy(sampled[..., :2]) / sampled.new_tensor([self.width, self.height]),
        ], -1)
        goal_xy = local(self.goal_tensor[:2] - self.start_tensor[:2]) / 20
        heading = self.goal[2] - self.start[2]
        values = [*goal_xy.tolist(), math.sin(heading), math.cos(heading),
                  self.start[3] / p["articulation_limit_rad"], self.goal[3] / p["articulation_limit_rad"],
                  len(self.directions) * integration_dt / 100, self.switch / len(self.directions),
                  float(self.switch < len(self.directions))]
        values.extend(p[key] / vehicle_parameters[key] for key in BODY_KEYS)
        map_ends = self._map_xy(torch.stack([self.start_tensor[:2], self.goal_tensor[:2]]))
        values.extend((map_ends / map_ends.new_tensor([self.width, self.height])).flatten().tolist())
        context = sampled.new_tensor(values).expand(count, -1)
        if tokens.shape[-1] != TOKEN_DIM or context.shape[-1] != CONTEXT_DIM:
            raise ValueError("失败证据特征维度不一致")
        return {"tokens": tokens, "context": context, "map_image": self.map_image.expand(count, -1, -1, -1)}

    def family_masks(self, evaluation):
        # 控制基函数定位在时间上；早期编辑会真实改变后续整条轨迹。
        masks = self.basis.new_zeros(len(FAMILY_NAMES), KNOT_COUNT, 2)
        masks[0, :KNOTS_PER_PHASE, 1] = 1
        masks[1, :KNOTS_PER_PHASE, 0] = 1
        masks[2, KNOTS_PER_PHASE:, 1] = 1
        masks[3, KNOTS_PER_PHASE:, 0] = 1
        severity = evaluation["violation"][0].amax(-1)
        failure_index = int(self.sample_indices[severity.argmax()]) if bool(severity.max() > 0) else len(self.directions) - 1
        active = self.basis[min(failure_index, len(self.directions) - 1)] > 0
        masks[4, active, 1] = 1
        masks[5, active, 0] = 1
        masks[6] = 1
        unused = self.basis.sum(0) == 0
        masks[:, unused] = 0
        return masks

    def independent_check(self, controls):
        # 最终状态用教师NumPy RK4重新生成，完整车体检查也使用同一身体参数。
        from planner_backend import make_environment, validate_path, NumpyArticulatedRollout
        if self.environment is None:
            self.environment = make_environment(self.map_data, self.parameters)
        model = NumpyArticulatedRollout.from_vehicle(self.environment[1])
        requested = np.asarray(controls, dtype=np.float64)
        state = self.start.copy()
        states, actual = [state], []
        for control in requested:
            state, applied = model.step_rk4(state, control, integration_dt)
            states.append(state)
            actual.append(applied)
        states, actual = np.asarray(states), np.asarray(actual)
        quality = validate_path(states, self.directions, self.start, self.goal, self.map_data,
                                actual, self.environment)
        quality["input_control_projection_error"] = float(np.abs(actual - requested).max())
        quality["quality_pass"] = bool(quality["quality_pass"] and quality["input_control_projection_error"] < 1e-5)
        return states, actual, quality
