"""用最长粗路线实测GPU批量上限，探测时不覆盖训练模型。"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from config import data_dir
from trajectory_data import TrajectoryDataset, collate_trajectories
from train import load_model, run_epoch


def find_longest_sample(args, point_count):
    # 第1步：遍历训练和验证缓存，找到最长路线。
    # 批量会补齐到最长路线，使用这个样本探测可避免只用短路线高估容量。
    longest = None
    longest_length = 0
    sample_info = {}
    for split in ["train", "val"]:
        dataset = TrajectoryDataset(args.data_dir, split, point_count, cache_dir=args.cache_dir,
                                    feedback_dir=args.feedback_dir)
        for index, record in enumerate(dataset.records):
            path = args.cache_dir / (record["sample_id"] + ".npz")
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["data_dir"]) != str(args.data_dir.resolve()):
                    raise ValueError("缓存目录与数据集不一致，请重新准备缓存")
                length = len(cached["route"])
            if length > longest_length:
                longest_length = length
                longest = dataset[index]
                sample_info = {"split": split, "sample_id": record["sample_id"]}
        for index in range(len(dataset.records), len(dataset)):
            sample = dataset[index]
            length = sample["candidate_map_patch_batch"].shape[0]
            if length > longest_length:
                longest_length = length
                longest = sample
                sample_info = {"split": split, "feedback_index": index - len(dataset.records)}
    if longest is None:
        raise ValueError("数据集中没有可用于探测的样本")
    sample_info["route_points"] = longest_length
    return longest, sample_info


def measure_batch(args, sample, batch_size):
    # 第2步：每次重新加载同一checkpoint，并执行真实前向、反向和Adam更新。
    # 连续三个批次覆盖优化器状态及上批输出仍在内存中的情况；不保存这些临时权重。
    device = torch.device("cuda")
    model, checkpoint = load_model(args.checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    batch = collate_trajectories([sample] * batch_size)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_epoch(model, [batch] * 3, device, optimizer,
              checkpoint["clearance_weight"], checkpoint["couple_switch"])
    torch.cuda.synchronize()
    return {
        "batch_size": batch_size, "passed": True,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "three_steps_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description="从256开始寻找最长路线可用的GPU批量")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--cache-dir", type=Path, default=Path("prepared_data/full"))
    parser.add_argument("--feedback-dir", type=Path)
    parser.add_argument("--start", type=int, default=256)
    parser.add_argument("--step", type=int, default=32, help="批量搜索粒度，默认32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/batch_capacity.json"))
    args = parser.parse_args()
    if args.step < 1 or args.start < args.step or args.start % args.step or args.threads < 1:
        parser.error("起始批量必须为正且是搜索粒度的整数倍，线程数必须为正")
    if not torch.cuda.is_available():
        parser.error("CUDA不可用，请在允许访问GPU的环境中运行")
    torch.set_num_threads(args.threads)
    _, checkpoint = load_model(args.checkpoint, torch.device("cpu"))
    sample, sample_info = find_longest_sample(args, checkpoint["point_count"])
    report = {
        "gpu": torch.cuda.get_device_name(),
        "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "longest_sample": sample_info, "step": args.step, "attempts": [],
    }
    print("最长路线探测样本：", sample_info, flush=True)
    low = 0
    high = None
    candidate = args.start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        print("正在探测 batch_size =", candidate, flush=True)
        try:
            result = measure_batch(args, sample, candidate)
        except torch.cuda.OutOfMemoryError:
            # 第3步：显存不足时只缩小探测范围，不修改正式checkpoint。
            result = {"batch_size": candidate, "passed": False, "reason": "CUDA out of memory"}
        gc.collect()
        torch.cuda.empty_cache()
        report["attempts"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["passed"]:
            low = candidate
        else:
            high = candidate
        report["recommended_batch_size"] = low
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if high is None:
            candidate *= 2
        elif high - low <= args.step:
            break
        else:
            candidate = ((low + high) // (2 * args.step)) * args.step
    if low == 0:
        raise ValueError("当前搜索粒度内没有可用批量，请减小--step再运行")
    print("探测完成。当前模型在最长路线下可用的最大批量（按粒度搜索）：", low, flush=True)


if __name__ == "__main__":
    main()
