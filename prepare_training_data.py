"""审计参考轨迹、地图划分，并缓存仅从地图生成的粗路线。"""
import argparse
import json
from pathlib import Path
import numpy as np
from config import data_dir
from data_preparation import read_index_records, read_map_data
from coarse_route import CoarseRoutePlanner
from trajectory_data import read_trajectory_data, resample_reference, check_map_splits


def main():
    parser = argparse.ArgumentParser(description="为训练缓存无参考答案泄漏的粗路线")
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--limit", type=int, default=0, help="每个集合最多检查多少条；0表示全部")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit不能为负")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"map_splits": check_map_splits(args.data_dir), "splits": {}, "failures": []}
    for split in ["train", "val", "test"]:
        records = read_index_records(args.data_dir, split)
        if args.limit:
            records = records[:args.limit]
        records.sort(key=lambda record: record["map_id"])
        last_map = None
        counts = [0, 0, 0, 0]
        for index, record in enumerate(records):
            # 第1步：按地图排序后复用图结构；成功后每个sample产生一份粗路线缓存。
            if record["map_id"] != last_map:
                map_data = read_map_data(args.data_dir, record)
                planner = CoarseRoutePlanner(map_data)
                last_map = record["map_id"]
            try:
                sample = read_trajectory_data(args.data_dir, record)
                states, directions, mode = resample_reference(sample)
                route = planner.plan(sample["start_state"], sample["goal_state"])
                np.savez_compressed(args.output_dir / (record["sample_id"] + ".npz"),
                                    route=route, data_dir=str(args.data_dir.resolve()))
                counts[mode] += 1
            except ValueError as error:
                report["failures"].append({"sample_id": record["sample_id"], "reason": str(error)})
            if (index + 1) % 500 == 0:
                print(split, index + 1, "/", len(records), flush=True)
        report["splits"][split] = {"requested": len(records), "valid": sum(counts), "F_R_FR_RF": counts}
        print(split, report["splits"][split], flush=True)
    # 第2步：留下可核对的审计文件；失败不能静默当成通过。
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["failures"]:
        raise SystemExit("存在数据问题，请检查audit.json中的failures")
    print("准备完成，训练可增加 --cache-dir", args.output_dir)


if __name__ == "__main__":
    main()
