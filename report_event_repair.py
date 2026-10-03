"""从正式实验记录生成可阅读的结果表、学习曲线和成功率/耗时图。"""
import argparse
import json
from pathlib import Path

import numpy as np


LABELS = {"neural": "Neural", "no_evidence": "No body evidence", "random": "Random",
          "gradient": "Gradient", "local": "Local gradient"}


def load_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(item, key, statistic="median"):
    value = item.get(key, {}).get(statistic)
    return "—" if value is None else f"{value:.3f}"


def result_table(summary):
    lines = ["| 身体条件 | 方法 | 时限内通过 | 物理通过 | 初值通过 | 总时长中位数/P95（秒） | 超时运行数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for variant, methods in summary["by_variant_method"].items():
        for method, item in methods.items():
            lines.append(f"| {variant} | {method} | {item['passed_within_time_budget']}/{item['count']} | "
                         f"{item['passed']}/{item['count']} | {item['initial_passed']}/{item['count']} | "
                         f"{metric(item, 'total_seconds')}/{metric(item, 'total_seconds', 'p95')} | {item['time_budget_overruns']} |")
    return "\n".join(lines)


def plots(root, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = test["protocol"]["methods"]
    variants = test["protocol"]["variants"]
    colors = ["#176aa5", "#79a7ca", "#7f8c8d", "#d67b30", "#bfaa72"][:len(methods)]
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        values = test["by_variant_method"][variant]
        rates = [100 * values[method]["success_rate_within_time_budget"] for method in methods]
        bars = ax.bar(np.arange(len(methods)), rates, color=colors)
        for bar, method in zip(bars, methods):
            item = values[method]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{item['passed_within_time_budget']}/{item['count']}", ha="center", va="bottom", fontsize=9)
        ax.set(ylim=(0, 108), title=variant, ylabel="Accepted within repair budget (%)")
        ax.set_xticks(np.arange(len(methods)), [LABELS[method] for method in methods], rotation=30, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out maps and body combinations | one training seed")
    fig.tight_layout()
    fig.savefig(root / "success_rates.png", dpi=180)
    fig.savefig(root / "success_rates.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        values = test["by_variant_method"][variant]
        medians = [values[method]["total_seconds"]["median"] for method in methods]
        p95 = [values[method]["total_seconds"]["p95"] for method in methods]
        x = np.arange(len(methods))
        ax.scatter(x, medians, c=colors, label="Median", s=45)
        ax.scatter(x, p95, c=colors, marker="x", label="P95", s=45)
        ax.vlines(x, medians, p95, colors=colors, alpha=.65)
        ax.set(title=variant, ylabel="Total seconds (failures included)", ylim=(0, None))
        ax.set_xticks(x, [LABELS[method] for method in methods], rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Shared seed generation + per-method setup + repair and independent validation")
    fig.tight_layout()
    fig.savefig(root / "planning_times.png", dpi=180)
    fig.savefig(root / "planning_times.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in (("evidence", "#176aa5"), ("no_evidence", "#d67b30")):
        history = rows(root / label / "history.jsonl")
        ax.plot([r["epoch"] for r in history], [r["train"]["loss"] for r in history], "--", color=color, alpha=.6,
                label=label + " train")
        ax.plot([r["epoch"] for r in history], [r["val"]["loss"] for r in history], color=color, label=label + " validation")
    ax.set(xlabel="Epoch", ylabel="Supervised editor loss", title="Validation loss selects the checkpoint")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=180)
    fig.savefig(root / "training_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/event_repair_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data/event_repair_v1"))
    args = parser.parse_args()
    root = args.output_dir
    pipeline = load_json(root / "pipeline.json")
    if not pipeline["complete"]:
        raise ValueError("正式实验尚未完成，不生成完整结果报告")
    audit = load_json(root / "completion_audit.json")
    if not audit["passed"]:
        raise ValueError("实验审计未通过")
    manifest = load_json(args.data_dir / "manifest.json")
    test = load_json(root / "test_network/summary.json")
    diagnostic = load_json(root / "val_reference_diagnostic/summary.json")
    plots(root, test)
    lines = ["# 失败证据驱动的神经事件编辑：本机实验记录", "",
             "本报告由实际完成的训练、评估和审计记录生成。它描述当前实现的效果，不构成论文首创性或实车可用性的证明。", "",
             "## 数据与训练", "",
             f"配对数据来自训练集 {manifest['splits']['train']['task_count']} 项任务、验证集 {manifest['splits']['val']['task_count']} 项任务；"
             f"分别包含 {manifest['splits']['train']['count']} 和 {manifest['splits']['val']['count']} 条编辑监督。",
             f"训练身体条件：{', '.join(manifest['variants'])}；留出组合：{', '.join(manifest['heldout_variants'])}。", "",
             "| 模型 | 完成轮数 | 最佳轮次 | 最佳验证损失 | 批量 |", "|---|---:|---:|---:|---:|"]
    for label, item in audit["training"]["models"].items():
        lines.append(f"| {label} | {item['last_epoch']} | {item['best_epoch']} | {item['validation_loss']:.6f} | {item['batch_size']} |")
    lines += ["", "验证损失只用于模型选择；规划成功取决于独立运动学重放、完整车体安全和终点检查。", "",
              "## 留出地图：原生成器初值", "",
              f"每个身体条件使用相同的 {len(test['protocol']['sample_ids'])} 项测试任务。每种方法最多 "
              f"{test['protocol']['evaluations']} 次轨迹代理前向、{test['protocol']['rounds']} 轮、"
              f"{test['protocol']['time_limit']} 秒修复软停止时限。所有失败保留在分母中。", "",
              result_table(test), "", "![Success rates](success_rates.png)", "",
              "![Planning times](planning_times.png)", "",
              "总时长包括共享初值生成的实测耗时、各方法问题构建、修复及独立验收；模型加载、绘图和存盘另计。"
              "批次计算和最终验收可能超过软停止时限，超时后的合格轨迹不计入时限内通过。"
              "失败很多时较短耗时不能单独证明有效规划更快。", "",
              "## 验证集：参考轨迹扰动诊断", "",
              "此项使用参考控制构造失败初值，只衡量指定扰动下的修复能力，不能作为端到端部署结果。", "",
              result_table(diagnostic), "", "## 范围与限制", "",
              "- 当前只编辑固定方向词和固定换挡时刻下的控制；换挡构型由积分自然改变。",
              "- 代理检查采样64个时间点；最终通过依赖完整车体检查与独立RK4重放。",
              "- 无证据消融只去掉网络输入的车体间隙、违反量和距离梯度；物理排序和验收仍共享。",
              "- 连续编辑采用单目标回归，尚未实现多模态编辑、显式因果图或学习回溯。",
              "- 只运行一个训练随机种子，身体组合的范围也有限；本轮结果不足以支持普遍泛化结论。",
              "- 当前为低速运动学模型，未包含轮胎滑移、真实变速箱停留和执行器动态。", "",
              "## 可复核证据", "",
              "- [完成审计](completion_audit.json)：模型身份、划分、逐条数据对应关系、预算及独立重放。",
              "- [GPU组件和配对重放检查](verification.json)。",
              "- [测试协议](test_network/protocol.json)及[test_network/cases](test_network/cases)：逐任务控制、状态和编辑轨迹。",
              "- [成功/失败回放](test_network/visualizations)：按每种方法、身体条件保存首个成功和失败。",
              "- [训练曲线](training_curves.png)。", ""]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("结果报告：", root / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
