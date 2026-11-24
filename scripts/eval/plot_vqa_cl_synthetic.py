"""Generate explicitly synthetic forgetting predictions and a reproducible figure.

This illustrates an assumed pattern; it is NOT an experiment on a trained model.
Run: python scripts/eval/plot_vqa_cl_synthetic.py
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.vqa_cl.calibration import Observation, calibration_metrics, paired_forgetting
from evaluation.vqa_cl.protocol import TASKS, ABBREVIATIONS
from scripts.eval.vqa_cl_panel_a import export_panel_a

INITIAL_ACCURACY = (.86, .84, .88, .82, .85, .83, .87, .84, .82, .80)
FINAL_ACCURACY = (.482, .520, .614, .538, .645, .650, .753, .752, .774, .800)


def trajectories(seed):
    """Uneven synthetic forgetting and independently varying confidence.

    Endpoints preserve the previous illustration's scale, not paper results.
    Gamma weights distribute the net loss unevenly across task transitions;
    correlated perturbations permit pauses and recovery without alternating
    signs by construction. No point is digitized from a published experiment.
    """
    accuracy_rng = random.Random(seed + 735)
    confidence_rng = random.Random(seed + 1207)
    accuracy = [[None] * len(TASKS) for _ in TASKS]
    confidence = [[None] * len(TASKS) for _ in TASKS]
    for j in range(len(TASKS)):
        steps = len(TASKS) - 1 - j
        accuracy[j][j], confidence[j][j] = INITIAL_ACCURACY[j], 0.
        weights = [accuracy_rng.gammavariate(.65, 1.) for _ in range(steps)]
        cumulative = perturbation = shift = 0.
        for age, weight in enumerate(weights, 1):
            cumulative += weight
            perturbation = .65 * perturbation + accuracy_rng.gauss(0., .006)
            net_loss = INITIAL_ACCURACY[j] - FINAL_ACCURACY[j]
            accuracy[j][j + age] = FINAL_ACCURACY[j] if age == steps else max(.01, min(.99,
                INITIAL_ACCURACY[j] - net_loss * cumulative / sum(weights) + perturbation))
            shift = .75 * shift + confidence_rng.gauss(0., .0025)
            confidence[j][j + age] = shift
    return accuracy, confidence


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate(output, samples, seed):
    rng = random.Random(seed)
    accuracy_targets, confidence_offsets = trajectories(seed)
    templates = []
    for task in TASKS:
        kind = "noul" if task == "judge" else "score" if task == "count" else "choice"
        keys = ["true", "false"] if kind == "noul" else [str(k) for k in range(21 if kind == "score" else 8)]
        priority = list(range(samples))
        rng.shuffle(priority)
        rank = dict(zip(priority, range(samples)))
        questions = []
        for index in range(samples):
            gold = rng.choice(keys)
            wrong = rng.choice([key for key in keys if key != gold])
            questions.append({"gold": gold, "wrong": wrong, "rank": rank[index],
                "confidence": rng.uniform(.80, .99), "phase": rng.uniform(0, 2 * math.pi),
                "weights": {key: rng.uniform(.1, 1.) for key in keys}})
        templates.append((kind, keys, questions))

    cells = [[None] * len(TASKS) for _ in TASKS]
    baseline, previous = {}, {}
    summaries = []
    prediction_count = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for stage in range(len(TASKS)):
            for task_index in range(stage + 1):
                task = TASKS[task_index]
                age = stage - task_index
                kind, keys, questions = templates[task_index]
                confidence_offset = confidence_offsets[task_index][stage]
                correct_count = round(samples * accuracy_targets[task_index][stage])
                observations = {}
                for index, item in enumerate(questions):
                    correct = item["rank"] < correct_count
                    predicted = item["gold"] if correct else item["wrong"]
                    confidence = max(.51, min(.995, item["confidence"] + confidence_offset
                                     + .002 * math.sin(item["phase"] + age * .7)))
                    remaining = sum(value for key, value in item["weights"].items() if key != predicted)
                    probabilities = {key: confidence if key == predicted else
                        (1 - confidence) * item["weights"][key] / remaining for key in keys}
                    target = {key: float(key == item["gold"]) for key in keys}
                    record_id = f"synthetic:{task}:{index:04d}"
                    observation = Observation(index, "q0", record_id, kind, len(keys), predicted,
                        item["gold"], probabilities[predicted], probabilities[item["gold"]], correct)
                    observations[index, "q0"] = observation
                    row = {"synthetic": True, "stage_index": stage, "stage_task": TASKS[stage],
                        "eval_task": task, "record_index": index, "record_id": record_id, "qid": "q0",
                        "task": kind, "target": target, "probabilities": probabilities,
                        "predicted": predicted, "gold": item["gold"], "correct": correct,
                        "confidence": confidence}
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    prediction_count += 1
                metrics = calibration_metrics(observations.values(), bins=15, threshold=.8)
                if age == 0:
                    baseline[task_index] = observations
                forgetting = paired_forgetting(baseline[task_index], observations, .8)
                recent = paired_forgetting(previous.get(task_index), observations, .8)
                previous[task_index] = observations
                cells[stage][task_index] = {"synthetic": True, "stage_index": stage,
                    "stage_task": TASKS[stage], "task": task, "overall": metrics,
                    "forgetting_from_learning": forgetting, "forgetting_from_previous_stage": recent}
                lost = forgetting["forgotten"]
                summaries.append({"synthetic": True, "stage_index": stage, "stage_task": TASKS[stage],
                    "eval_task": task, "count": samples,
                    **{key: metrics[key] for key in ("accuracy", "mean_confidence", "ece", "overconfidence_error",
                        "underconfidence_error", "signed_confidence_gap", "mean_wrong_confidence", "high_confidence_error_rate")},
                    "forgotten_count": lost["count"], "forgotten_mean_confidence": lost["mean_current_confidence"],
                    "forgotten_mean_confidence_change": lost["mean_confidence_change"],
                    "forgotten_mean_gold_probability_change": lost["mean_gold_probability_change"]})
    with (output / "task_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    result = {"synthetic": True, "description": "Designed illustration, not measured model performance",
        "seed": seed, "samples_per_task": samples, "prediction_rows": prediction_count,
        "tasks": list(TASKS), "ece_bins": 15, "high_confidence_threshold": .8,
        "generator_version": "irregular_shared_v1",
        "initial_accuracy": list(INITIAL_ACCURACY), "final_accuracy_targets": list(FINAL_ACCURACY),
        "accuracy_targets_by_task_then_stage": accuracy_targets,
        "confidence_offsets_by_task_then_stage": confidence_offsets,
        "trajectory_parameters": {"loss_weight_gamma_shape": .65, "accuracy_ar_coefficient": .65,
            "accuracy_innovation_sd": .006, "confidence_ar_coefficient": .75, "confidence_innovation_sd": .0025},
        "confidence_distribution": "Uniform(0.80, 0.99) per question plus independent correlated stage shifts and small individual oscillations; clipped to [0.51, 0.995]",
        "forgetting_rule": "Fixed question ranks with irregular stage thresholds; gamma-weighted net drops plus correlated perturbations allow recoveries",
        "provenance": {"synthetic": True, "literature_context": "https://arxiv.org/html/2503.00413v1",
            "note": "All values are synthetic. The paper motivates task-dependent forgetting but does not supply these stage curves, confidences, or ECE values. Endpoints preserve the original illustration, not the paper's reported accuracy."},
        "ece_matrix": [[cell["overall"]["ece"] if cell else None for cell in row] for row in cells],
        "cells": cells}
    write_json(output / "calibration_matrix.json", result)
    write_json(output / "reference_provenance.json", result["provenance"])
    return result


def draw(output, result, panel_a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    teal, orange, blue, ink, muted = "#008A83", "#D96C39", "#4267A9", "#182B3A", "#5D6B78"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.titleweight": "bold", "axes.titlesize": 12, "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 10.4))
    fig.subplots_adjust(left=.075, right=.94, bottom=.13, top=.81, wspace=.30, hspace=.40)
    stages = np.arange(1, 11)
    recognition = [row["overall"] for row in panel_a["stages"]]
    accuracy = np.array([row["accuracy"] for row in recognition])
    confidence = np.array([row["mean_confidence"] for row in recognition])

    def timeline(ax, title):
        ax.set_title(title, loc="left", pad=15)
        ax.set_xticks(stages, ABBREVIATIONS, fontsize=9)
        ax.set_xlabel("Latest task learned", labelpad=8)
        ax.set_xlim(.8, 10.35)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        for spine in ax.spines.values():
            spine.set_color("#D6DEE3")

    ax = axes[0]
    timeline(ax, "A   Old task: Recognition")
    ax.plot(stages, confidence, color=orange, marker="o", markersize=4, linewidth=2.3, label="Mean confidence")
    ax.plot(stages, accuracy, color=teal, marker="o", markersize=4, linewidth=2.3, label="Accuracy")
    ax.fill_between(stages, accuracy, confidence, color=orange, alpha=.10)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Accuracy / confidence")
    ax.legend(loc="lower left", frameon=False)
    for values, color, offset in ((confidence, orange, 10), (accuracy, teal, -18)):
        for index in (0, 9):
            ax.annotate(f"{values[index]:.1%}", (stages[index], values[index]), xytext=(0, offset),
                        textcoords="offset points", ha="center", color=color, fontsize=10, weight="bold")

    # ax = axes[0, 1]
    # timeline(ax, "B   Recognition: calibration error grows")
    # ece = [row["ece"] for row in recognition]
    # ax.plot(stages, ece, color=blue, marker="o", markersize=4, linewidth=2.3, label="Task ECE")
    # ax.plot(stages, [row["signed_confidence_gap"] for row in recognition], color=orange,
    #         linestyle="--", linewidth=1.7, label="Mean confidence − accuracy")
    # ax.set_ylim(0, .50)
    # ax.set_ylabel("Calibration error / gap")
    # ax.legend(loc="upper left", frameon=False)
    # for index in (0, 9):
    #     ax.annotate(f"{ece[index]:.1%}", (stages[index], ece[index]), xytext=(0, 10),
    #                 textcoords="offset points", ha="center", color=blue, weight="bold")

    ax = axes[1]
    ax.set_title("B   ECE for every stage × evaluation task", loc="left", pad=15)
    matrix = np.array([[value if value is not None else np.nan for value in row] for row in result["ece_matrix"]])
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F5F7")
    heatmap = ax.imshow(matrix * 100, cmap=cmap, vmin=0, vmax=45, aspect="auto")
    ax.set_xticks(range(10), ABBREVIATIONS, fontsize=9)
    ax.set_yticks(range(10), ABBREVIATIONS, fontsize=9)
    ax.set_xlabel("Evaluation task", labelpad=8)
    ax.set_ylabel("Latest task learned")
    for i in range(10):
        for j in range(i + 1):
            value = matrix[i, j] * 100
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 29 else ink)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=.035, pad=.025)
    colorbar.set_label("ECE (%)")
    colorbar.outline.set_visible(False)

    # ax = axes[1, 1]
    # ax.set_title("D   Recognition: before vs. after forgetting", loc="left", pad=15)
    # ax.plot([0, 1], [0, 1], color="#9DAAB3", linestyle="--", linewidth=1.3, label="Perfect calibration")
    # for index, color, label in ((0, teal, "After Recognition"), (9, orange, "After Causal")):
    #     bins = [row for row in recognition[index]["reliability_bins"] if row["count"]]
    #     x = [row["mean_confidence"] for row in bins]
    #     y = [row["accuracy"] for row in bins]
    #     ax.plot(x, y, color=color, linewidth=1.8, label=label)
    #     ax.scatter(x, y, s=[25 + 160 * row["count"] / result["samples_per_task"] for row in bins], color=color, zorder=3)
    # ax.set_xlim(0, 1.02)
    # ax.set_ylim(0, 1.02)
    # ax.set_xlabel("Mean confidence within bin", labelpad=8)
    # ax.set_ylabel("Accuracy within bin")
    # ax.xaxis.set_major_formatter(PercentFormatter(1))
    # ax.yaxis.set_major_formatter(PercentFormatter(1))
    # ax.grid(alpha=.18)
    # ax.legend(loc="upper left", frameon=False, fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"synthetic_forgetting_calibration.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/vqa-cl-synthetic-calibration")
    parser.add_argument("--samples-per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the shared predictions and BOTH panels")
    args = parser.parse_args()
    if args.samples_per_task < 100:
        parser.error("Use at least 100 samples per task")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output / "calibration_matrix.json"
    # Both panels read the SAME canonical dataset. A cannot change independently
    # of B's Recognition column while preserving statistical consistency.
    result = json.loads(matrix_path.read_text()) if matrix_path.exists() and not args.regenerate else generate(args.output, args.samples_per_task, args.seed)
    panel_a = export_panel_a(args.output, result)
    draw(args.output, result, panel_a)
    first, last = result["cells"][0][0]["overall"], result["cells"][-1][0]["overall"]
    forgotten = result["cells"][-1][0]["forgetting_from_learning"]["forgotten"]
    (args.output / "README.md").write_text(f"""# VQA-CL 遗忘与过度自信：合成示例

**这是人为构造的模拟预测，不是真实 Valen / CL-MoE 实验结果，也不能证明研究假设。**

为了展示“旧任务准确率下降，置信度维持高位”，固定随机种子 {result['seed']}，为十个任务各构造 {result['samples_per_task']} 道虚拟题。
沿用任务顺序和原语候选数量（judge 两类、count 21 类、其余八类），没有真实图片或问题文本，不可作为训练数据。
正确样本由固定题目排名与各阶段阈值决定。十个任务的阶段阈值均采用 Gamma 随机权重分配总体下降幅度，再加入相关扰动。
这样可以出现连续缓慢下降、少数较大变化、平台与局部恢复，不强制一高一低交替。
置信度采用独立随机种子的 AR(1) 阶段过程，不跟随准确率的涨跌方向。
每道题的基础最大预测概率位于 0.80–0.99，叠加阶段偏移和微小个体波动后截断到 [0.51, 0.995]，与遗忘排名独立。
答错时将最大概率分配给固定错误候选；完整概率向量归一化，题目 ID 和标准答案跨阶段不变。

## 生成的具体数值

两张图读取同一批模拟预测，B 的 Recognition 第一列就是 A 对应的任务 ECE。
`panel_a.json`、`panel_a.csv` 和 `panel_a_predictions.jsonl` 只是原始数据中 Recognition 的导出子集，不另造曲线。
每次绘图都会从逐题概率重算 A 的准确率、置信度与 ECE，校验与 B 一致。
A 的准确率为 {[round(row['overall']['accuracy'] * 100, 1) for row in panel_a['stages']]}%。

严格关系是 `ECE = 平均置信度 − 准确率 + 2 × underconfidence_error`。
如果每个非空区间都过度自信，最后一项为零，ECE 就等于 A 两条曲线之差。
少量低估自信的区间会使 ECE 高于整体差值，精确数值可查看 `panel_a.csv` 的 `underconfidence_error` 列。
本次统一重建十任务逐题预测，B 的全部 55 个单元由同一数据重算；A 不使用独立的展示曲线。
保留原合成示例的起止准确率量级，并没有把这些数值当成 CL-MoE 论文结果。
论文只作为任务相关遗忘的背景，不能提供这里的逐阶段波形或置信度；来源边界记录在 `reference_provenance.json`。

Recognition 从第一个阶段到最后一个阶段的数值：

| 指标 | 刚学完 Recognition | 学完 Causal |
| --- | ---: | ---: |
| 准确率 | {first['accuracy']:.2%} | {last['accuracy']:.2%} |
| 平均置信度 | {first['mean_confidence']:.2%} | {last['mean_confidence']:.2%} |
| ECE | {first['ece']:.2%} | {last['ece']:.2%} |
| 答错样本平均置信度 | {first['mean_wrong_confidence']:.2%} | {last['mean_wrong_confidence']:.2%} |

最终有 {forgotten['count']} 道 Recognition 题从正确变为错误，这些题当前平均置信度为 {forgotten['mean_current_confidence']:.2%}。

## 文件

- `predictions.jsonl`：共 {result['prediction_rows']} 条逐阶段、逐题概率记录，每条都有 `synthetic: true`。用 `(eval_task, record_index, qid)` 跨阶段对齐。
- `task_metrics.csv`：55 行阶段 × 任务指标，数值采用 0–1 单位。
- `calibration_matrix.json`：生成参数、15 个等宽区间的 ECE、reliability bins 和两种遗忘基准的统计。
- `panel_a.json` / `panel_a.csv` / `panel_a_predictions.jsonl`：同一数据中 Recognition 的统计和逐题预测子集。
- `synthetic_forgetting_calibration.png`：A 曲线与 B 热力图的双面板预览图，使用当前绘图布局。
- `synthetic_forgetting_calibration.svg` / `.pdf`：同一合成数据图的可编辑矢量图与导出版本。

所有指标由现有 `evaluation.vqa_cl.calibration` 的函数从模拟预测计算，没有手填 ECE。
这里没有真实运行的完成标记，不能把本目录当作 `evaluation.vqa_cl.calibration --run` 的真实训练输出。
图 A 固定评测 Recognition，横轴表示最新学完的任务；B 为原始情景的十任务 ECE 矩阵。
生成规则刻意保持高置信度，因此不能从这些数据估计真实模型发生该现象的概率。

## 复现

在仓库根目录运行（需要 matplotlib）：

```bash
python scripts/eval/plot_vqa_cl_synthetic.py --seed {result['seed']} --samples-per-task {result['samples_per_task']} --regenerate
```

输出默认写入 `data/vqa-cl-synthetic-calibration/`。项目现有 `.gitignore` 忽略此类生成数据，生成脚本单独保留。
不加 `--regenerate` 时复用原始数据；加上该参数会按指定 seed 和样本数同步重建 A/B 数据和图，不能独立重造 A。
""", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "synthetic": True, "prediction_rows": result["prediction_rows"],
        "recognition_initial": {key: first[key] for key in ("accuracy", "mean_confidence", "ece")},
        "recognition_final": {key: last[key] for key in ("accuracy", "mean_confidence", "ece")}}, indent=2))


if __name__ == "__main__":
    main()
