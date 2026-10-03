"""把手动任务的合格优化轨迹加入训练反馈，失败结果继续保留供排查。"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from config import data_dir, integration_dt
from data_preparation import read_index_records, read_map_data
from planner_backend import validate_path
from trajectory_data import resample_reference


def add_feedback(result_path, dataset_dir, output_dir):
    with np.load(result_path, allow_pickle=False) as saved:
        sample = {key: saved[key] for key in saved.files}
    # 第1步：确认地图来自训练集合，验证/测试地图的交互任务不能回流进训练。
    map_id = str(sample["map_id"])
    allowed = {}
    for record in read_index_records(dataset_dir, "train"):
        allowed[record["map_id"]] = record["map_path"]
    if map_id not in allowed:
        raise ValueError("反馈只能来自已有训练地图；验证、测试和未知地图不允许加入")
    if str(sample["data_dir"]) != str(Path(dataset_dir).resolve()) or str(sample["map_path"]) != allowed[map_id]:
        raise ValueError("反馈记录的数据集或地图路径与训练索引不一致")
    if not bool(sample["quality_pass"]) or "trajectory_controls" not in sample:
        raise ValueError("此结果没有通过优化与质量检查，不能作为成功轨迹监督")
    if abs(float(sample["integration_dt"]) - integration_dt) > 1e-9:
        raise ValueError("反馈积分步长与教师不一致")
    map_data = read_map_data(dataset_dir, {"map_path": allowed[map_id]})
    directions = sample["trajectory_directions"][1:]
    # 第2步：重新执行独立检查，不单凭文件中的成功标记决定是否采纳。
    quality = validate_path(sample["trajectory_states"], directions, sample["start_state"],
                            sample["goal_state"], map_data, sample["trajectory_controls"])
    if not quality["quality_pass"]:
        raise ValueError("反馈复查未通过：" + quality["reason"])
    boundaries = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    sample["switch_states"] = sample["trajectory_states"][boundaries]
    sample["phase_directions"] = np.concatenate([directions[:1], directions[boundaries]])
    resample_reference(sample)
    # 第3步：使用内容指纹去重，重复导入同一条轨迹不会增加训练权重。
    digest = hashlib.sha256()
    digest.update(map_id.encode())
    digest.update(np.asarray(sample["trajectory_states"], dtype=np.float32).tobytes())
    digest.update(np.asarray(sample["trajectory_controls"], dtype=np.float32).tobytes())
    path = Path(output_dir) / ("feedback_" + digest.hexdigest()[:20] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.savez_compressed(path, **sample)
    return path


def main():
    parser = argparse.ArgumentParser(description="把规划通过的训练地图任务加入反馈数据")
    parser.add_argument("result", type=Path, help="plan.py保存的trajectory.npz")
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data/feedback"))
    args = parser.parse_args()
    path = add_feedback(args.result, args.data_dir, args.output_dir)
    print("反馈已保存：", path)
    print("继续训练时增加 --feedback-dir", args.output_dir)


if __name__ == "__main__":
    main()
