"""训练轨迹生成器。先运行小样本命令，再增加训练样本和轮数。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from config import data_dir, trajectory_point_count
from trajectory_data import TrajectoryDataset, collate_trajectories, check_map_splits
from loader_model import build_trajectory_model, predict_batch
from trajectory_loss import trajectory_loss


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA不可用，请检查驱动或使用 --device cpu")
    return torch.device(name)


def move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def run_epoch(model, loader, device, optimizer, clearance_weight, couple_switch):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            output = predict_batch(model, batch)
            loss, metrics = trajectory_loss(output, batch, clearance_weight, couple_switch)
            if not torch.isfinite(loss):
                raise ValueError("损失出现非有限数，已停止更新，请检查数据和学习率")
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = batch["target_states"].shape[0]
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            print_every = 10 if len(loader) < 100 else 100
            if training and (batch_index + 1) % print_every == 0:
                print("训练批次", batch_index + 1, "/", len(loader), "当前平均损失", round(totals["loss"] / count, 5), flush=True)
    if count == 0:
        raise ValueError("当前数据集没有可训练或验证的样本")
    return {key: value / count for key, value in totals.items()}


def load_model(checkpoint_path, device):
    # 第1步：只接受本流程的模型，防止把旧候选评分参数误当成轨迹生成参数。
    # 用同一份文件内容加载参数并计算指纹，训练同时更新best.pt也不会让记录和实际权重错位。
    checkpoint_bytes = Path(checkpoint_path).read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    checkpoint["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint.get("format") != "loader_trajectory_v2":
        raise ValueError("权重格式与当前轨迹结构不匹配，请使用本版train.py产生的best.pt或last.pt")
    model = build_trajectory_model(checkpoint["point_count"], checkpoint["couple_switch"],
                                   checkpoint.get("use_sequence_directions", False))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def main():
    parser = argparse.ArgumentParser(description="训练、验证并保存装载机轨迹生成器")
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/trajectory"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--feedback-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20, help="本次增加的训练轮数")
    parser.add_argument("--batch-size", type=int, default=512, help="按本机P104-100训练设置默认512，可用calibrate_batch_size.py重新测量")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--point-count", type=int, default=trajectory_point_count)
    parser.add_argument("--train-limit", type=int, default=0, help="0表示全部训练任务")
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-switch-coupling", action="store_true", help="基础生成器消融")
    parser.add_argument("--clearance-weight", type=float, default=0.0, help="车体间隙损失权重，默认关闭")
    parser.add_argument("--use-sequence-directions", action="store_true", help="由区间方向预测合成连续方向词，并用于轨迹分段")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.point_count < 4 or args.clearance_weight < 0:
        parser.error("轮数、批量和输出点数必须为正，间隙权重不能为负")
    if args.threads < 1 or args.workers < 0 or args.learning_rate <= 0 or min(args.train_limit, args.val_limit) < 0:
        parser.error("线程数、学习率必须为正，进程数与样本限制不能为负")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print("设备：", device, "地图划分：", check_map_splits(args.data_dir), flush=True)
    start_epoch = 0
    best_loss = float("inf")
    checkpoint = None
    couple_switch = not args.no_switch_coupling
    if args.resume:
        model, checkpoint = load_model(args.resume, device)
        args.point_count = checkpoint["point_count"]
        couple_switch = checkpoint["couple_switch"]
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        if checkpoint["data_dir"] != str(args.data_dir.resolve()):
            raise ValueError("续训数据目录不同，请明确使用原来的训练/验证划分")
        # 验证目标变更后，不能继续沿用旧目标下的最优值。
        if checkpoint["clearance_weight"] != args.clearance_weight or checkpoint["val_limit"] != args.val_limit:
            best_loss = float("inf")
        if args.use_sequence_directions and not checkpoint.get("use_sequence_directions", False):
            model.trajectory_decoder.use_sequence_directions = True
            best_loss = float("inf")
    else:
        model = build_trajectory_model(args.point_count, couple_switch, args.use_sequence_directions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    train_data = TrajectoryDataset(args.data_dir, "train", args.point_count, args.train_limit, args.cache_dir, args.feedback_dir)
    val_data = TrajectoryDataset(args.data_dir, "val", args.point_count, args.val_limit, args.cache_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, collate_fn=collate_trajectories)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, collate_fn=collate_trajectories)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output_dir / "last.pt").exists():
        raise ValueError("输出目录已有模型，请使用--resume继续训练，或选择新的输出目录")
    if not (args.output_dir / "best.pt").is_file():
        # 续训写到新目录时，本目录的第一轮也应产生可用于规划的best.pt。
        best_loss = float("inf")
    print("训练任务：", len(train_data), "验证任务：", len(val_data), "参数量：", sum(p.numel() for p in model.parameters()), flush=True)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # 第2步：训练集更新参数，验证集只统计损失；成功后终端应每轮打印两种损失。
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.clearance_weight, couple_switch)
        val_metrics = run_epoch(model, val_loader, device, None, args.clearance_weight, couple_switch)
        improved = val_metrics["loss"] < best_loss
        best_loss = min(best_loss, val_metrics["loss"])
        saved = {
            "format": "loader_trajectory_v2", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "best_loss": best_loss,
            "point_count": args.point_count, "couple_switch": couple_switch,
            "use_sequence_directions": model.trajectory_decoder.use_sequence_directions,
            "clearance_weight": args.clearance_weight, "val_limit": args.val_limit,
            "data_dir": str(args.data_dir.resolve()), "train_count": len(train_data),
            "batch_size": args.batch_size,
            "val_count": len(val_data), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "state_features": ["x_map_norm", "y_map_norm", "sin_heading_map", "cos_heading_map", "theta_norm"],
            "route_source": "map_only_skeleton_weighted_grid_v1",
        }
        # 第3步：先写临时文件再替换，避免中断留下半个checkpoint。
        torch.save(saved, args.output_dir / "last.tmp")
        (args.output_dir / "last.tmp").replace(args.output_dir / "last.pt")
        if improved:
            torch.save(saved, args.output_dir / "best.tmp")
            (args.output_dir / "best.tmp").replace(args.output_dir / "best.pt")
        metrics = {"epoch": epoch, "batch_size": args.batch_size, "train": train_metrics, "val": val_metrics,
                   "epoch_seconds": time.perf_counter() - epoch_started}
        if device.type == "cuda":
            # 记录本轮张量实际占用峰值，用户可据此调整8GB显卡上的批量大小。
            metrics["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            metrics["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print("轮次", epoch, "训练损失", round(train_metrics["loss"], 5), "验证损失", round(val_metrics["loss"], 5),
              "本轮秒数", round(metrics["epoch_seconds"], 1), flush=True)
    print("训练完成。继续训练使用last.pt，手动规划建议使用best.pt：", args.output_dir)


if __name__ == "__main__":
    main()
