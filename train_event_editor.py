"""训练物理证据驱动的事件编辑网络，保留独立地图验证和数据指纹。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from event_editor import EventEditor, editor_loss, load_editor
from repair_data import RepairDataset
from train import choose_device, move_batch


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["tokens"], batch["context"], batch["map_image"])
            loss, metrics = editor_loss(output, batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError("事件编辑损失出现非有限值")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            number = len(batch["tokens"])
            count += number
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + number * value
    return {key: value / count for key, value in total.items()}


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/event_editor"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden) < 1 or args.hidden % 4 or args.learning_rate <= 0:
        parser.error("轮数和批量必须为正，隐藏维度为4的倍数")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    datasets = {split: RepairDataset(args.data_dir, split) for split in ("train", "val")}
    manifest_sha = hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest()
    code_sha = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in ("config.py", "event_editor.py", "repair_physics.py", "train_event_editor.py")}
    loaders = {split: torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=split == "train", num_workers=0)
               for split, dataset in datasets.items()}
    model = EventEditor(args.hidden, not args.no_evidence).to(device)
    start_epoch, best_loss = 0, float("inf")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.resume:
        model, checkpoint = load_editor(args.resume, device)
        if checkpoint["manifest_sha256"] != manifest_sha or checkpoint["use_evidence"] == args.no_evidence:
            raise ValueError("续训数据或证据消融配置发生变化")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
    elif (args.output_dir / "last.pt").exists():
        raise ValueError("输出目录已有模型，请续训或使用新目录")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        for current in range(start_epoch + 1, start_epoch + args.epochs + 1):
            begin = time.perf_counter()
            train_metrics = epoch(model, loaders["train"], device, optimizer)
            val_metrics = epoch(model, loaders["val"], device)
            improved = val_metrics["loss"] < best_loss
            best_loss = min(best_loss, val_metrics["loss"])
            checkpoint = {"format": "loader_event_editor_v1", "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "epoch": current,
                          "hidden": model.hidden, "use_evidence": model.use_evidence,
                          "best_loss": best_loss, "batch_size": args.batch_size, "seed": args.seed,
                          "manifest_sha256": manifest_sha, "data_dir": str(args.data_dir.resolve()),
                          "code_sha256": code_sha,
                          "base_checkpoint_sha256": datasets["train"].metadata["base_checkpoint_sha256"],
                          "torch_rng": torch.get_rng_state()}
            if device.type == "cuda":
                checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
            atomic_save(checkpoint, args.output_dir / "last.pt")
            if improved:
                atomic_save(checkpoint, args.output_dir / "best.pt")
            row = {"epoch": current, "train": train_metrics, "val": val_metrics,
                   "seconds": time.perf_counter() - begin, "best": improved}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(current, "训练", round(train_metrics["loss"], 6), "验证", round(val_metrics["loss"], 6),
                  "最佳" if improved else "", flush=True)
    summary = {"completed_epochs": start_epoch + args.epochs, "elapsed_seconds": time.perf_counter() - started,
               "train_count": len(datasets["train"]), "val_count": len(datasets["val"]),
               "batch_size": args.batch_size, "device": str(device), "use_evidence": model.use_evidence,
               "best_val_loss": best_loss, "manifest_sha256": manifest_sha,
               "code_sha256": code_sha, "parameters": sum(p.numel() for p in model.parameters()),
               "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0,
               "note": "验证损失只用于选模型；规划改善必须查看独立测试评估"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
