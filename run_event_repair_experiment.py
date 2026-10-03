"""等待配对数据完成后，顺序运行核验、训练、消融和固定测试协议。"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/event_repair_v1"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument("--wait-hours", type=float, default=6)
    parser.add_argument("--resume", action="store_true", help="恢复已中断的流水线，保留已完成阶段")
    parser.add_argument("--resume-data", action="store_true", help="数据未完成时先从逐轮记录恢复生成")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.test_limit, args.wait_hours) <= 0:
        parser.error("实验参数必须为正")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "pipeline.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.output_dir / "pipeline.json"
    if protocol_path.exists() and not args.resume:
        parser.error("流水线目录已存在，请显式检查记录后另选目录")
    stages = [
        ("verification", ["verify_event_editor.py", "--data-dir", str(args.data_dir), "--device", "cuda",
                          "--witness-limit", "12", "--output", str(args.output_dir / "verification.json")]),
        ("train_evidence", ["train_event_editor.py", "--data-dir", str(args.data_dir),
                            "--output-dir", str(args.output_dir / "evidence"), "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size), "--device", "cuda"]),
        ("train_no_evidence", ["train_event_editor.py", "--data-dir", str(args.data_dir),
                               "--output-dir", str(args.output_dir / "no_evidence"), "--epochs", str(args.epochs),
                               "--batch-size", str(args.batch_size), "--device", "cuda", "--no-evidence"]),
        ("test_network_seed", ["evaluate_event_editor.py", "--editor-checkpoint", str(args.output_dir / "evidence/best.pt"),
                               "--no-evidence-checkpoint", str(args.output_dir / "no_evidence/best.pt"),
                               "--split", "test", "--limit", str(args.test_limit), "--device", "cuda",
                               "--evaluations", "97", "--rounds", "12", "--time-limit", "10",
                               "--visualize", "--output-dir", str(args.output_dir / "test_network")]),
        ("val_reference_diagnostic", ["evaluate_event_editor.py", "--editor-checkpoint", str(args.output_dir / "evidence/best.pt"),
                                     "--no-evidence-checkpoint", str(args.output_dir / "no_evidence/best.pt"),
                                     "--split", "val", "--limit", "32", "--source", "perturbed_reference",
                                     "--variants", "nominal", "--device", "cuda", "--evaluations", "97",
                                     "--rounds", "12", "--time-limit", "10", "--visualize",
                                     "--output-dir", str(args.output_dir / "val_reference_diagnostic")]),
    ]
    state = {"complete": False, "waiting_for": str(args.data_dir / "manifest.json"), "stages": [],
             "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
             "commands": [{"name": name, "argv": [sys.executable, "-u", *command]} for name, command in stages]}
    if args.resume:
        previous = json.loads(protocol_path.read_text())
        for key in ("data_dir", "output_dir", "epochs", "batch_size", "test_limit"):
            if previous["protocol"][key] != state["protocol"][key]:
                raise ValueError("恢复时实验协议发生变化：" + key)
        if previous["commands"] != state["commands"]:
            raise ValueError("恢复时实验阶段命令发生变化")
        state = previous
        if state["complete"]:
            print("流水线已完成，无需恢复", flush=True)
            return
        state.setdefault("resume_times", []).append(time.time())

    def save():
        temporary = protocol_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(protocol_path)

    save()
    if args.resume_data and not json.loads(Path(state["waiting_for"]).read_text()).get("complete", False):
        print("开始恢复配对数据生成", flush=True)
        command = [sys.executable, "-u", "resume_repair_data.py", "--data-dir", str(args.data_dir), "--device", "cuda"]
        recovery = {"argv": command, "complete": False}
        state.setdefault("data_recovery", []).append(recovery)
        save()
        log_path = args.output_dir / ("data_recovery_" + str(len(state["data_recovery"])) + ".log")
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        recovery.update({"exit_code": result.returncode, "complete": result.returncode == 0, "log": str(log_path)})
        save()
        if result.returncode:
            raise SystemExit("数据恢复失败，请查看：" + str(log_path))
    waiting = time.monotonic()
    while time.monotonic() - waiting < args.wait_hours * 3600:
        try:
            ready = json.loads(Path(state["waiting_for"]).read_text()).get("complete", False)
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready:
            break
        time.sleep(10)
    else:
        state["error"] = "等待数据完成超时，未开始训练"
        save()
        raise SystemExit(state["error"])
    state["wait_seconds"] = time.monotonic() - waiting
    for name, command in stages:
        existing = next((stage for stage in state["stages"] if stage["name"] == name), None)
        if existing and existing["complete"]:
            print("保留已完成阶段：", name, flush=True)
            continue
        if name in ("train_evidence", "train_no_evidence"):
            folder = args.output_dir / ("evidence" if name == "train_evidence" else "no_evidence")
            if (folder / "last.pt").exists():
                import torch
                checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
                remaining = args.epochs - checkpoint["epoch"]
                if remaining <= 0:
                    if not (folder / "summary.json").exists():
                        raise ValueError("训练checkpoint已完成但summary缺失，需要核对训练日志")
                    if existing is None:
                        existing = {"name": name}
                        state["stages"].append(existing)
                    existing.update({"complete": True, "exit_code": 0, "restored_from_checkpoint": True})
                    save()
                    continue
                command = list(command)
                command[command.index("--epochs") + 1] = str(remaining)
                command += ["--resume", str(folder / "last.pt")]
        if name in ("test_network_seed", "val_reference_diagnostic"):
            folder = args.output_dir / ("test_network" if name == "test_network_seed" else "val_reference_diagnostic")
            if (folder / "tasks.jsonl").exists():
                command = list(command) + ["--resume"]
        print("开始阶段：", name, flush=True)
        stage = {"name": name, "complete": False}
        if existing is not None:
            stage["previous_attempts"] = existing.get("previous_attempts", []) + [dict(existing)]
            state["stages"][state["stages"].index(existing)] = stage
        else:
            state["stages"].append(stage)
        save()
        begin = time.perf_counter()
        log_path = args.output_dir / (name + ".log")
        if log_path.exists():
            log_path.rename(log_path.with_name(log_path.name + ".previous_" + str(time.time_ns())))
        with log_path.open("w") as log:
            result = subprocess.run([sys.executable, "-u", *command], stdout=log, stderr=subprocess.STDOUT)
        stage.update({"exit_code": result.returncode, "seconds": time.perf_counter() - begin,
                      "complete": result.returncode == 0})
        save()
        if result.returncode:
            raise SystemExit("阶段失败，请查看日志：" + str(args.output_dir / (name + ".log")))
        print("完成阶段：", name, "耗时", round(stage["seconds"], 2), flush=True)
    state["complete"] = True
    save()
    print("实验流水线完成：", protocol_path, flush=True)


if __name__ == "__main__":
    main()
