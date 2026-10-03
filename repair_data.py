"""任务输入、网络控制初值，以及配对干预数据集读取。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config import integration_dt
from coarse_route import CoarseRoutePlanner
from data_preparation import read_index_records, read_map_data
from loader_model import predict_batch
from train import move_batch
from trajectory_data import prepare_trajectory_input, collate_trajectories, restore_states, resample_states
from trajectory_decoder import decode_directions


def selected_records(data_dir, split, limit):
    records = read_index_records(data_dir, split)
    if limit:
        records = [records[index] for index in np.linspace(0, len(records) - 1, min(limit, len(records)), dtype=int)]
    return records


def read_task(data_dir, record, reference=False):
    with np.load(Path(data_dir) / record["sample_path"], allow_pickle=False) as sample:
        task = {"start": sample["start_state"].astype(float), "goal": sample["goal_state"].astype(float)}
        if reference:
            task["reference_controls"] = sample["trajectory_controls"].astype(float)
    task["map_data"] = read_map_data(data_dir, record)
    return task


def controls_from_states(states, directions, max_steps=1600):
    """几何初值转为控制初值；是否可行由物理推演判断，不复用教师答案。"""
    boundaries = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    stops = [0] + boundaries.tolist() + [len(states) - 1]
    controls, words = [], []
    total_length = float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum())
    if total_length / integration_dt > max_steps:
        raise ValueError("初始路径过长，超出事件编辑步数上限")
    for low, high in zip(stops[:-1], stops[1:]):
        segment = states[low:high + 1]
        length = np.linalg.norm(np.diff(segment[:, :2], axis=0), axis=1).sum()
        steps = max(8, int(np.ceil(length / integration_dt)))
        sampled = resample_states(segment, steps + 1)
        speed = np.full(steps, max(0.05, length / (steps * integration_dt))) * directions[low]
        rate = np.diff(sampled[:, 3]) / integration_dt
        controls.append(np.stack([speed, rate], 1))
        words.append(np.full(steps, directions[low]))
    return np.concatenate(controls).astype(np.float32), np.concatenate(words).astype(np.int64)


def network_seed(generator, task, device):
    route = CoarseRoutePlanner(task["map_data"]).plan(task["start"], task["goal"])
    prepared = prepare_trajectory_input(task["map_data"], task["start"], task["goal"], route)
    batch = move_batch(collate_trajectories([prepared]), device)
    with torch.no_grad():
        output = predict_batch(generator, batch)
        states = restore_states(output["states"][0].cpu().numpy(), task["map_data"])
        directions = decode_directions(output)[0].cpu().numpy()
    controls, control_directions = controls_from_states(states, directions)
    return controls, control_directions, route


class RepairDataset(torch.utils.data.Dataset):
    FIELDS = ("tokens", "context", "map_image", "delta", "family", "improved")

    def __init__(self, directory, split):
        directory = Path(directory)
        metadata = json.loads((directory / "manifest.json").read_text())
        if not metadata.get("complete"):
            raise ValueError("配对干预数据尚未完成")
        if split not in ("train", "val"):
            raise ValueError("训练读取仅允许train和val")
        path = directory / (split + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["splits"][split]["sha256"]:
            raise ValueError("配对数据指纹发生变化")
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {key: torch.from_numpy(data[key].copy()) for key in self.FIELDS}
        self.metadata = metadata

    def __len__(self):
        return len(self.arrays["family"])

    def __getitem__(self, index):
        return {key: value[index].float() if value.dtype.is_floating_point else value[index]
                for key, value in self.arrays.items()}
