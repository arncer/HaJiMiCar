"""第一个里程碑：画出真实参考轨迹、换挡点和装载机完整姿态。"""
import argparse
from pathlib import Path
from config import data_dir
from data_preparation import read_index_records, read_map_data
from trajectory_data import read_trajectory_data, resample_reference
from coarse_route import CoarseRoutePlanner
from visualization import save_visualization


def main():
    parser = argparse.ArgumentParser(description="查看参考轨迹与按阶段保留的64个监督状态")
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("planning_results/reference"))
    args = parser.parse_args()
    records = read_index_records(args.data_dir, args.split)
    if args.sample_index < 0 or args.sample_index >= len(records):
        parser.error("sample-index超出范围")
    record = records[args.sample_index]
    sample = read_trajectory_data(args.data_dir, record)
    map_data = read_map_data(args.data_dir, record)
    states, directions, mode = resample_reference(sample)
    route = CoarseRoutePlanner(map_data).plan(sample["start_state"], sample["goal_state"])
    # 此处只是复用绘图入口：绿色/橙色曲线为重采样参考，紫色曲线为原始参考。
    save_visualization(args.output_dir, map_data, sample["start_state"], sample["goal_state"],
                       route, states, directions, reference=sample["trajectory_states"],
                       prediction_label="Resampled reference")
    print("样本：", record["sample_id"], "地图：", record["map_id"], "方向模式：", mode)
    print("原始轨迹：", sample["trajectory_states"].shape, "重采样轨迹：", states.shape)
    print("换挡状态：", sample["switch_states"])
    print("查看：", args.output_dir / "trajectory.html")


if __name__ == "__main__":
    main()
