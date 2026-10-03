"""几何路径监督与可选车体间隙损失。没有用求解耗时冒充轨迹时间。"""
import numpy as np
import torch
import torch.nn.functional as F
from config import articulation_limit, vehicle_parameters


def state_loss(predicted, target, map_size):
    # 第1步：位置先转换成米，再用5米尺度归一化，避免大位置误差淹没挡位和姿态监督。
    # 航向用正余弦内积，铰接角使用归一化误差；各项现在处于可比较的数量级。
    position = ((predicted[..., :2] - target[..., :2]) * map_size / 5.0).square().mean()
    heading = (1 - (predicted[..., 2:4] * target[..., 2:4]).sum(-1)).mean()
    articulation = F.mse_loss(predicted[..., 4], target[..., 4])
    return position + heading + articulation


def body_clearance_loss(states, map_data_list):
    # 第2步：把前车体、铲斗和后车体内部网格投影到距离图，保留梯度。
    # 成功后，车体靠墙或越界会产生正损失；最终安全判定仍使用教师完整检查。
    total = states.sum() * 0
    parts = [
        (-vehicle_parameters["front_axle_to_hitch_m"], 0.0, vehicle_parameters["front_body_width_m"], False),
        (0.0, vehicle_parameters["bucket_front_m"], vehicle_parameters["bucket_width_m"], False),
        (-vehicle_parameters["hitch_to_rear_axle_m"] - vehicle_parameters["rear_axle_to_tail_m"],
         0.0, vehicle_parameters["rear_body_width_m"], True),
    ]
    for index, map_data in enumerate(map_data_list):
        height, width = map_data["map_features"].shape[1:]
        resolution = float(map_data["resolution"])
        size = states.new_tensor([width * resolution, height * resolution])
        xy = states[index, :, :2] * size
        heading = torch.atan2(states[index, :, 2], states[index, :, 3])
        theta = states[index, :, 4] * articulation_limit
        distance_map = torch.tensor(map_data["map_features"][1], device=states.device).reshape(1, 1, height, width)
        for low, high, body_width, rear in parts:
            local_x = torch.linspace(low, high, int(np.ceil((high - low) / 0.5)) + 1, device=states.device)
            local_y = torch.linspace(-body_width / 2, body_width / 2, int(np.ceil(body_width / 0.5)) + 1, device=states.device)
            x_grid, y_grid = torch.meshgrid(local_x, local_y, indexing="ij")
            body_heading = heading
            origin = xy
            if rear:
                origin = xy - vehicle_parameters["front_axle_to_hitch_m"] * torch.stack([torch.cos(heading), torch.sin(heading)], -1)
                body_heading = heading - theta
            cosine = torch.cos(body_heading).unsqueeze(1)
            sine = torch.sin(body_heading).unsqueeze(1)
            x = origin[:, :1] + cosine * x_grid.flatten() - sine * y_grid.flatten()
            y = origin[:, 1:2] + sine * x_grid.flatten() + cosine * y_grid.flatten()
            # align_corners=True 对应地图索引0到width-1；矩形地图分别归一化。
            query = torch.stack([2 * x / (resolution * (width - 1)) - 1,
                                 2 * y / (resolution * (height - 1)) - 1], -1)
            clearance = F.grid_sample(distance_map, query.unsqueeze(0), align_corners=True, padding_mode="zeros")
            # 越界项提供朝地图内部的梯度，防止全零采样无法纠正远处预测。
            outside = F.relu(query.abs() - 1).square().mean()
            total = total + F.relu(0.5 - clearance).square().mean() + outside
    return total / (len(map_data_list) * len(parts))


def trajectory_loss(output, batch, clearance_weight=0.0, couple_switch=True):
    target = batch["target_states"]
    size = batch["map_size_m"].unsqueeze(1)
    geometry = state_loss(output["raw_states"], target, size)
    direction = F.binary_cross_entropy_with_logits(output["direction_logits"], batch["target_directions"])
    mode = F.cross_entropy(output["mode_logits"], batch["target_mode"])
    switch_loss = geometry * 0
    switched = batch["target_mode"] >= 2
    if couple_switch and switched.any():
        switch_loss = state_loss(output["switch_state"][switched], target[switched, target.shape[1] // 2], batch["map_size_m"][switched])
    clearance = geometry * 0
    if clearance_weight > 0:
        clearance = body_clearance_loss(output["states"], batch["map_data"])
    loss = geometry + direction + mode + switch_loss + clearance_weight * clearance
    return loss, {"loss": float(loss.detach()), "geometry": float(geometry.detach()),
                  "direction": float(direction.detach()), "mode": float(mode.detach()),
                  "switch": float(switch_loss.detach()), "clearance": float(clearance.detach())}
