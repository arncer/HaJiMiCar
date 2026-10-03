"""任务级轨迹监督：每个任务只使用它自己的参考轨迹。"""
import json
from pathlib import Path
import numpy as np
import torch
from config import articulation_limit
from coarse_route import CoarseRoutePlanner, world_to_grid, grid_to_world
from data_preparation import (
    prepare_state_features, prepare_task_batch, prepare_route_batch,
    prepare_map_patch_batch, prepare_padded_route_batch,
    prepare_padded_map_patch_batch, prepare_position_encoding_batch,
    read_index_records, read_map_data,
)


def read_trajectory_data(data_dir, record):
    # 第1步：读取任务参考轨迹。不会把同一轨迹复制给多个候选。
    sample_data = {}
    with np.load(Path(data_dir) / record["sample_path"], allow_pickle=False) as sample:
        if str(sample["map_id"]) != record["map_id"]:
            raise ValueError("任务内的地图编号与索引不一致：" + record["sample_path"])
        for key in ["start_state", "goal_state", "trajectory_states", "trajectory_controls",
                    "trajectory_directions", "switch_states", "phase_directions"]:
            sample_data[key] = sample[key]
    return sample_data


def resample_states(states, point_count):
    # 第2步：在一个固定挡位阶段内按弧长重采样，航向先展开再插值。
    # 成功后，跨越 -pi/pi 的航向不会错误地绕回0，阶段首尾状态会被保留。
    states = np.asarray(states, dtype=np.float64).copy()
    if len(states) < 2 or point_count < 2 or not np.isfinite(states).all():
        raise ValueError("每个轨迹阶段至少需要两个有限状态")
    lengths = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
    # 原地改变姿态也保留进度，避免零长度段导致插值丢失事件。
    lengths = np.maximum(lengths, 1e-6)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    positions = np.linspace(0.0, cumulative[-1], point_count)
    states[:, 2] = np.unwrap(states[:, 2])
    result = np.zeros((point_count, 4), dtype=np.float32)
    for column in range(4):
        result[:, column] = np.interp(positions, cumulative, states[:, column])
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result


def resample_reference(sample_data, point_count=64):
    states = sample_data["trajectory_states"]
    controls = sample_data["trajectory_controls"]
    directions = sample_data["trajectory_directions"]
    if states.ndim != 2 or states.shape[1] != 4 or len(states) < 2:
        raise ValueError("trajectory_states 应为 [T>=2,4]")
    if controls.shape != (len(states) - 1, 2) or directions.shape != (len(states),):
        raise ValueError("状态、控制和方向长度不对应")
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        raise ValueError("参考轨迹包含非有限数")
    if not np.isin(directions, [-1, 1]).all():
        raise ValueError("方向必须是 -1 或 +1")
    # 教师的 directions[1:] 对应每个控制区间；换挡发生在新区间起点。
    edge_directions = directions[1:]
    if not np.array_equal(np.sign(controls[:, 0]), edge_directions):
        raise ValueError("控制速度符号与轨迹区间方向不一致")
    boundaries = np.flatnonzero(edge_directions[1:] != edge_directions[:-1]) + 1
    phase_directions = np.concatenate([edge_directions[:1], edge_directions[boundaries]])
    if not np.array_equal(phase_directions, sample_data["phase_directions"]):
        raise ValueError("phase_directions 与真实轨迹换挡不一致")
    if len(boundaries) != len(sample_data["switch_states"]) or len(boundaries) > 1:
        raise ValueError("本版支持零次或一次换挡，请先核对数据中的换挡数量")
    if point_count < 4:
        raise ValueError("输出点数至少为4")
    midpoint = point_count // 2
    if len(boundaries) == 0:
        result = resample_states(states, point_count)
        result_directions = np.full(point_count - 1, edge_directions[0], dtype=np.int64)
        mode = 0 if edge_directions[0] == 1 else 1
    else:
        boundary = int(boundaries[0])
        switch = sample_data["switch_states"][0]
        difference = states[boundary].astype(np.float64) - switch
        difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
        if np.max(np.abs(difference)) > 1e-3:
            raise ValueError("switch_states 与控制换挡位置不一致")
        # 两段共用同一个换挡状态，只保存一次，绝不平均前进和倒车标签。
        first = resample_states(states[:boundary + 1], midpoint + 1)
        second = resample_states(states[boundary:], point_count - midpoint)
        result = np.concatenate([first, second[1:]], axis=0)
        result_directions = np.full(point_count - 1, edge_directions[-1], dtype=np.int64)
        result_directions[:midpoint] = edge_directions[0]
        mode = 2 if edge_directions[0] == 1 else 3
    return result, result_directions, mode


def normalize_states(states, map_data):
    result = []
    for state in states:
        result.append(prepare_state_features(state, map_data))
    return torch.stack(result)


def restore_states(normalized_states, map_data):
    # 第3步：把 [x,y,sin(psi),cos(psi),theta] 恢复成世界坐标车辆状态。
    values = np.asarray(normalized_states)
    height, width = map_data["map_features"].shape[1:]
    points = values[:, :2] * np.array([width, height])
    result = np.zeros((len(values), 4), dtype=np.float64)
    result[:, :2] = grid_to_world(points, map_data)
    result[:, 2] = np.arctan2(values[:, 2], values[:, 3]) + float(map_data["origin_yaw"])
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    result[:, 3] = values[:, 4] * articulation_limit
    return result


def prepare_trajectory_input(map_data, start_state, goal_state, route):
    # 第4步：训练和手动规划都调用这里，输入仅由地图、起终点、在线粗路线构成。
    route_grid = world_to_grid(route, map_data)
    sample_data = {"start_state": start_state, "goal_state": goal_state}
    return {
        "candidate_map_patch_batch": prepare_map_patch_batch(route_grid[:, 0], route_grid[:, 1], map_data),
        "normalized_route_batch": prepare_route_batch(route_grid[:, 0], route_grid[:, 1], map_data),
        "normalized_task_batch": prepare_task_batch(sample_data, map_data),
    }


def collate_trajectories(prepared_candidates):
    # 第5步：复用已有的路线补齐和局部地图补齐；输出轨迹有自己的固定长度。
    route, mask = prepare_padded_route_batch(prepared_candidates)
    result = {
        "normalized_route_batch": route,
        "route_padding_mask": mask,
        "candidate_map_patch_batch": prepare_padded_map_patch_batch(prepared_candidates, route.shape[1]),
        "position_encoding_batch": prepare_position_encoding_batch(route.shape[1], batch_size=len(prepared_candidates)),
    }
    tasks = []
    for candidate in prepared_candidates:
        tasks.append(candidate["normalized_task_batch"])
    result["normalized_task_batch"] = torch.cat(tasks)
    if "target_states" in prepared_candidates[0]:
        for key in ["target_states", "target_directions", "target_mode", "map_size_m"]:
            result[key] = torch.stack([candidate[key] for candidate in prepared_candidates])
        # 地图尺寸允许不同，因此保留列表，几何损失逐图查询。
        result["map_data"] = [candidate["map_data"] for candidate in prepared_candidates]
    return result


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_name="train", point_count=64, limit=0,
                 cache_dir=None, feedback_dir=None):
        self.data_dir = Path(data_dir)
        self.records = read_index_records(self.data_dir, split_name)
        if limit > 0:
            # 均匀取样，避免只取到排在文件前面的某一种场景。
            indices = np.linspace(0, len(self.records) - 1, min(limit, len(self.records)), dtype=int)
            self.records = [self.records[i] for i in indices]
        self.point_count = point_count
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.maps = {}
        self.planners = {}
        self.feedback_files = []
        if feedback_dir and split_name == "train":
            self.feedback_files = sorted(Path(feedback_dir).glob("*.npz"))
            train_maps = {r["map_id"] for r in read_index_records(self.data_dir, "train")}
            for path in self.feedback_files:
                with np.load(path, allow_pickle=False) as sample:
                    if str(sample["map_id"]) not in train_maps or not bool(sample["quality_pass"]):
                        raise ValueError("反馈数据必须通过质量检查，且只能来自训练地图：" + str(path))

    def __len__(self):
        return len(self.records) + len(self.feedback_files)

    def __getitem__(self, index):
        if index < len(self.records):
            record = self.records[index]
            sample_data = read_trajectory_data(self.data_dir, record)
        else:
            path = self.feedback_files[index - len(self.records)]
            with np.load(path, allow_pickle=False) as sample:
                sample_data = {key: sample[key] for key in sample.files}
            record = {"map_id": str(sample_data["map_id"]), "map_path": str(sample_data["map_path"])}
        map_id = record["map_id"]
        if map_id not in self.maps:
            # 小缓存控制主机内存；DataLoader 的各个进程分别维护自己的缓存。
            if len(self.maps) >= 8:
                self.maps.clear()
                self.planners.clear()
            self.maps[map_id] = read_map_data(self.data_dir, record)
        map_data = self.maps[map_id]
        cache_path = None
        if self.cache_dir and "sample_id" in record:
            cache_path = self.cache_dir / (record["sample_id"] + ".npz")
        if cache_path and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["data_dir"]) != str(self.data_dir.resolve()):
                    raise ValueError("粗路线缓存属于另一个数据目录，请重新准备数据")
                route = cached["route"]
        else:
            if map_id not in self.planners:
                self.planners[map_id] = CoarseRoutePlanner(map_data)
            route = self.planners[map_id].plan(sample_data["start_state"], sample_data["goal_state"])
        target_states, target_directions, mode = resample_reference(sample_data, self.point_count)
        prepared = prepare_trajectory_input(map_data, sample_data["start_state"], sample_data["goal_state"], route)
        prepared["target_states"] = normalize_states(target_states, map_data)
        prepared["target_directions"] = torch.tensor(target_directions > 0, dtype=torch.float32)
        prepared["target_mode"] = torch.tensor(mode, dtype=torch.long)
        height, width = map_data["map_features"].shape[1:]
        prepared["map_size_m"] = torch.tensor([width, height], dtype=torch.float32) * float(map_data["resolution"])
        prepared["map_data"] = map_data
        return prepared


def check_map_splits(data_dir):
    # 第6步：训练前检查地图划分；成功后 train/val/test 两两交集应为空。
    groups = {}
    for split in ["train", "val", "test"]:
        groups[split] = {r["map_id"] for r in read_index_records(Path(data_dir), split)}
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        if groups[first] & groups[second]:
            raise ValueError(first + " 与 " + second + " 存在地图泄漏")
    return {key: len(value) for key, value in groups.items()}
