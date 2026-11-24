# Bench-C → Valen

将 Bench-C TSV 转为 Valen 原生 `request / targets / assets / meta` JSONL，并评测
候选概率质量、置信度校准及 clean/corrupted 配对变化。所有题按原始多选一任务处理，
无需生成文本答案，也不改变训练权重。

## 当前提供的数据

本地 `data/Bench-C/difficulty_bench.tsv` 实际有 **868 道题、915 张图像引用**：
831 道单图、35 道双图、2 道七图。它与当前公开文档描述的 849 题版本不同，
脚本保留用户提供 TSV 的全部行，不按目录名删题，也不声称已验证官方划分。

TSV 内嵌图像完整，优先解码使用；`849_pictures` 目录缺少部分引用文件，不影响转换。
原始图像编码字节和多图顺序保持不变，写入相对路径与 SHA256 校验值。
当前附件只有 clean 图像，没有提供 19×5 退化图像；默认不会生成或声称评测了退化集。

## 从本地 TSV 构造完整数据（新增）

`construct.py` 加载官方 `src/corruption/base.py` 的 19 个 processor，保留 severity 参数，
并修复一个会导致 snow 无法运行的尺寸错误（见下文）。固定官方 commit `9ff3ea6279bf8576ad9720944176202134e914d7`，
改写外围图片枚举和 TSV 路径映射以支持本地多图、嵌套目录与 PNG 原图。
不从候选池重新选题，也不删减本地 868 行。当前该 commit 的 selection CSV 中
`is_selected=1` 只有 812 行，不能作为 README 所称 849 行的可靠替代清单。

以下命令均在 **Valen 仓库根目录**执行。完整生成不需要 GPU。
生成环境独立于模型推理环境，避免官方代码受可选 Wand 后端影响。

```bash
git clone https://github.com/xiangjieSui/Bench-C.git data/Bench-C-official
git -C data/Bench-C-official checkout 9ff3ea6279bf8576ad9720944176202134e914d7

python3.12 -m venv .venv-benchc-build
.venv-benchc-build/bin/python -m pip install -r evaluation/benchc/requirements-generation.txt

.venv-benchc-build/bin/python -m evaluation.benchc.construct \
  --data data/Bench-C \
  --official-repo data/Bench-C-official \
  --output data/Bench-C-constructed \
  --workers 4 --seed 0
```

生成环境请使用 Python 3.10–3.13（上例为 3.12），与官方 NumPy 2.2 版本范围兼容。
当前这台机器也可用
`/Users/wjd/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`
替换上述 `python3.12`；它是本次测试使用的 Python 3.12。

默认生成完整 19×5 条件。中断后用同一命令加 `--resume`，会核对源数据、代码、
依赖版本与种子，并核验已有图片哈希；缺失或损坏图片重新生成。每张图片使用
由 seed/退化类型/强度/图片身份导出的独立随机种子，结果不依赖 worker 数或续跑顺序。
输出 `construction.json` 在完整生成后才标记 `status=complete`。

```text
data/Bench-C-constructed/
  construction.json                  # 官方 revision、代码哈希、依赖、种子、数量、完成状态
  clean/
    bench_c.tsv                      # 全部原始问题与标签；image 置空，image_path 指向所有图片
    images/<question-id>/image_0.png  # 原始编码字节不变；后缀取真实图片格式
  corrupted/
    image_manifest.json              # clean 文件与退化 JPEG 的一一映射
    gaussian_noise/1/
      bench_c.tsv                    # 保留全部问题，多图列表对应这一退化条件
      images.jsonl                   # 每张退化图片的 SHA256 和随机种子
      <question-id>/image_0.jpg       # 与官方 pipeline 相同，RGB 输入、JPEG quality=95
    ...                              # 19 种退化 × 5 个强度
```

在本地 868 题版本上：**82,460 个退化题目实例 + 868 个 clean 实例 = 83,328 个评测实例**；
因为包含多图题，需要生成 **86,925 张退化图片**（915×95）。这不是官方 849 题划分的榜单结果。
每种退化独立作用于 clean 图片，强度 2 不会叠加在强度 1 上。

官方此版本没有附带 frost 纹理，默认执行其内置纯色叠加 fallback；motion_blur 和 snow
在无 Wand 环境分别执行官方 OpenCV/SciPy fallback。这些后端选择写入 construction.json。
snow 的 SciPy 分支遗漏了放大后雪层的尺寸恢复，会产生广播错误；适配器在内存加载时
补上 `cv2.resize(..., (w,h), interpolation=cv2.INTER_LINEAR)`，与官方 Wand 分支的
恢复步骤一致，不修改 checkout 文件或 severity 参数。原始源码哈希、修复描述和实际
加载源码哈希均有记录。因此这是带有明确兼容修复的公开代码生成结果。
有 Wand 时，该 commit 的 motion_blur 分支缺少返回值，因此构造器会在生成前报错，
不悄悄切换后端。不宣称生成结果与论文私有生成产物逐字节一致。

将构造完成的原始数据转为 Valen 原生格式：

```bash
.venv-benchc-build/bin/python -m evaluation.benchc.prepare \
  --data data/Bench-C-constructed \
  --output data/Bench-C-valen-full
```

转换器自动读取 clean、全部退化条件及图片映射，逐张核对构造阶段的哈希，
完整保留多图顺序。未完成的构造目录不能被转换。

切回安装好 Valen 推理依赖的 Python 环境进行完整评测：

```bash
python -m evaluation.benchc.evaluate \
  --data data/Bench-C-valen-full --validate-only --require-full-grid

python -m evaluation.benchc.evaluate \
  --model Valen-Team/Valen-2B \
  --data data/Bench-C-valen-full \
  --output output/benchc-full \
  --require-full-grid
```

`--require-full-grid` 验证每个 clean 题目在全部 19×5 条件中都有且只有一个对应实例；
它验证条件完整性，不代表验证了官方 849 题划分。可用
`--corruptions gaussian_noise gaussian_blur --severities 1 3 5` 做部分评测，
此时不传 `--require-full-grid`；clean 对照总会保留，无需再次指定。
只想试跑构造时，construct 也支持同名的 `--corruptions` / `--severities` 参数，
请使用另一个输出目录，不与完整生成混用。

## 预处理

在 Valen 仓库根目录、已安装 Pillow 的 Python 环境执行：

```bash
python -m evaluation.benchc.prepare \
  --data data/Bench-C \
  --output data/Bench-C-valen
```

输出目录必须不存在。支持 `--data /path/to/difficulty_bench.tsv`；如果 TSV 没有
内嵌图像，可用 `--image-root /path/to/images`。`image_path` 的 `['/5.jpg']`
按图像根目录下的 `5.jpg` 处理；多图列表用安全的字面值解析，不执行字符串代码。

输出结构：

```text
data/Bench-C-valen/
  eval.jsonl          # 所有导入条件；默认只有 clean
  clean.jsonl        # clean 基线
  preparation.json   # TSV SHA256、实际题数、图像数、分类、条件和文件哈希
  assets/clean/0/...  # 本地图片
```

`question` 原样作为 `instructions`，有效的 A–E 按顺序作为 `criteria`；空候选剔除。
`hint` 放入 state 文本。答案字母只进入 one-hot `targets`，类别与退化信息只进入
`meta`，不会作为模型输入。多图以 Image 1、Image 2 等顺序放入同一 state。

## 校验与评测

无需模型或 GPU 即可验证全部题目、配对身份、图片路径和 SHA256：

```bash
python -m evaluation.benchc.evaluate \
  --data data/Bench-C-valen --validate-only
```

安装项目正常的推理依赖后：

```bash
python -m evaluation.benchc.evaluate \
  --model Valen-Team/Valen-2B \
  --data data/Bench-C-valen \
  --output output/benchc
```

支持与 VisualDecisionBench evaluator 相同的 `--revision`、`--lora`、
`--base-model`、`--device`、`--execution question|shared_state`、
`--attn-implementation eager|sdpa|flash_attention_2`。结果目录须为新目录或空目录；
中断留下的预测文件不会自动覆盖，也不支持续跑。

结果为 `predictions.jsonl`（逐题标签、原始候选概率、预测、confidence、指标及
base_id/退化元信息）、`paired_predictions.jsonl`（每个 clean/退化配对的 dAcc、dS、dC、RAS）
和 `metrics.json`。正常推理与数据校验都不会静默丢题。

## 导入已有退化图像

[Bench-C 官方项目](https://github.com/xiangjieSui/Bench-C) 提供退化工具。
此预处理器负责导入已有图片，**不实现或替代官方退化生成协议**。
请先生成图片，并按下列布局放置；后半段路径与 TSV 的 `image_path` 一致，去掉开头 `/`：

```text
corrupted_images/
  gaussian_noise/1/5.jpg
  gaussian_noise/1/rotation/1823_1.jpg
  gaussian_noise/1/rotation/1823_2.jpg
  gaussian_noise/2/...
  jpeg_compression/1/...
```

```bash
python -m evaluation.benchc.prepare \
  --data data/Bench-C/difficulty_bench.tsv \
  --corrupted-root /path/to/corrupted_images \
  --corruptions gaussian_noise jpeg_compression \
  --severities 1 2 3 4 5 \
  --output data/Bench-C-valen-corrupted

python -m evaluation.benchc.evaluate \
  --model Valen-Team/Valen-2B \
  --data data/Bench-C-valen-corrupted \
  --output output/benchc-corrupted
```

省略 `--corruptions` / `--severities` 时发现目录中已有的条件。每个选中条件都必须
覆盖全部 TSV 题的全部图片，缺图即报错，不允许用 clean 补齐。生成的 `eval.jsonl`
包含 clean 与全部选中条件，另外保存如 `gaussian_noise_1.jsonl` 的分项文件。
多图题的每张图都使用相应退化版本。所有条件共用 `base_id`、题干、选项及标签。

单独传入某个退化 JSONL 也能评测，但缺少 clean 时只报告配对缺失数，不计算变化。

## 指标字段

`metrics` 按以下键分组：

- `overall`：当前文件内的全部条件/题目。
- `subset/clean` / `subset/corrupted`：分别统计 clean 和退化条件，避免混淆基线与退化性能。
- `condition/clean/0`、`condition/gaussian_noise/1`：特定条件。
- `corruption/<name>`、`severity/<level>`：退化类型或强度聚合。
- `category/<name>`、`l2_category/<name>`：原始细粒度与粗粒度类别。

每组指标：

| 字段 | 含义 |
|---|---|
| `questions` / `correct` / `accuracy` | 题数、正确数、准确率 |
| `nll` | 从 logits 的 log-softmax 稳定计算的负对数似然，越低越好 |
| `brier` | 各候选概率与 one-hot 标签平方误差之和的平均，越低越好；沿用 Valen 定义，官方 Bench-C 的同名函数还会逐题除以选项数 K |
| `mean_confidence` | 每题最大候选概率的平均，不是 Valen `answer.confidence` |
| `confidence_minus_accuracy` | 平均把握减准确率；正值表示平均过度自信，不同组可能抵消 |
| `ece` / `mce` | 分桶校准误差的加权平均 / 最大值；大小不区分过度或不足自信 |
| `reliability_bins` | 分桶边界、题数、平均把握、实际准确率；空桶均值为 null |
| `normalized_entropy` | 候选分布熵除以 log(K)，用于不同选项数的比较 |
| `high_confidence_questions` / `high_confidence_errors` | 达到阈值的题数及其中错误数 |
| `high_confidence_error_rate` | 上述错误数 / 高置信题数；没有高置信题时为 null |
| `mean_confidence_on_errors` | 错题平均把握；没有错题时为 null |

ECE 默认 10 个等宽桶，可用 `--bins 15` 修改；高置信阈值默认 0.9，
可用 `--high-confidence 0.95` 修改。没有使用测试标签拟合温度或其他校准参数。

`paired_changes["gaussian_noise/1"]` 使用匹配的 clean 问题计算：

- `pairs`、`missing_clean`：配对数量与缺少基线数量。
- `delta_accuracy` / `delta_confidence` / `delta_normalized_entropy` / `delta_nll` / `delta_brier`：退化值减 clean 值。
- `delta_confidence_minus_accuracy`：上述有方向校准偏差的变化。
- `correct_to_wrong` / `wrong_to_correct`：正确变错误、错误变正确的数量。
- `top1_consistency`：预测选项保持一致的比例。

此外包含官方 `src/metrics.py` 定义的四个配对指标，均先逐题计算再平均：

- `dAcc`：退化正确性减 clean 正确性。
- `dS`：退化归一化熵减 clean 归一化熵。
- `dC`：`abs(conf_corrupt - correct_corrupt) - abs(conf_clean - correct_clean)`。
  **它不是分桶 ECE 的变化**，也不是 `delta_confidence_minus_accuracy`。
- `RAS`：`-dC - max(0,dC)*max(0,-dS) - max(0,dS)*max(0,-dC)`。
  先对每对样本算 RAS，再平均，不能把平均 dS/dC 代入公式。

四项指标遵循固定官方 commit，包括对概率的 `1e-12` 截断及归一化；已用官方函数
对照测试。`paired_overall` 是全部有效配对的加权平均，`paired_changes` 是按条件的汇总。
分桶 ECE 保持原有定义，单独在 `metrics` 中报告。
`overall` 混合多个条件时会重复统计同一原题，clean 基线应读取
`metrics["condition/clean/0"]`。868 题的本地版本不应直接冒充官方 849 题榜单成绩。

## 测试

```bash
python -m pytest tests/integration/test_benchc.py -q
```

覆盖多图顺序、embedded/file 图像、标签、路径与哈希、退化配对、软标签/重复拒绝、
ECE 边界与高置信错误、极端 logits 的 NLL 稳定性，以及两种执行模式下的标签隔离。
测试使用小型 fake model，不下载模型权重。

完整测试（还会在合成图片上实际调用官方全部 19×5 processor，并对照官方指标函数）：

```bash
BENCHC_OFFICIAL_REPO="$PWD/data/Bench-C-official" \
  python -m pytest tests/integration/test_benchc.py -q
```

需要同时安装 pytest、torch 和图像生成依赖；未设置上述变量时，三个需要官方 checkout
的测试会跳过。不会为本地真实数据执行全量生成或模型推理。
