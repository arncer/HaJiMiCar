"""校正早期离线日志漏计的一次初值前向；保留原日志，不修改训练数组或manifest。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    args = parser.parse_args()
    directory = args.data_dir
    manifest_content = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_content)
    if not manifest["complete"] or manifest["format"] != "loader_repair_pairs_v1":
        raise ValueError("只校正已完成的v1配对数据")
    output = directory / "query_count_correction.json"
    if output.exists():
        print("已存在计数校正记录：", output)
        return
    report = {"manifest_sha256_unchanged": sha(manifest_content), "splits": {},
              "scope": "paired_interventions内部：证据前向1次、梯度教师初值前向1次、每步前后各1次、每候选1次；独立CPU验收另计",
              "excluded": "外层每轮控制投影时的额外前向及初值生成，不包含在该函数字段内"}
    for split in ("train", "val"):
        path = directory / (split + "_interventions.jsonl")
        original = path.read_bytes()
        rows = [json.loads(line) for line in original.splitlines() if line.strip()]
        changed = 0
        for row in rows:
            expected = 2 + 2 * row["gradient_steps"] + len(row["candidate_scores"])
            if row["proxy_evaluations"] not in (expected - 1, expected):
                raise ValueError("遇到未知计数格式，拒绝推断：" + row["sample_id"])
            if row["proxy_evaluations"] == expected - 1:
                row["proxy_evaluations_original"] = row["proxy_evaluations"]
                row["proxy_evaluations"] = expected
                row["proxy_evaluations_scope"] = "paired_interventions_only"
                changed += 1
        if changed:
            backup = path.with_name(path.stem + ".before_count_correction.jsonl")
            if backup.exists() and backup.read_bytes() != original:
                raise ValueError("原始日志备份已经存在且内容不同")
            backup.write_bytes(original)
            corrected = ("\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows) + "\n").encode()
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(corrected)
            temporary.replace(path)
        report["splits"][split] = {"rows": len(rows), "corrected_rows": changed,
                                     "original_sha256": sha(original), "corrected_sha256": sha(path.read_bytes())}
    if sha((directory / "manifest.json").read_bytes()) != sha(manifest_content):
        raise ValueError("计数校正期间manifest发生变化")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("离线查询计数已校正：", output)


if __name__ == "__main__":
    main()
