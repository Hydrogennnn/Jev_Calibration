# decision-v7 按 source 持续学习

将 `data/v7/decision-v7` 的记录转换为 Valen 原生 JSONL，按 **`_meta.source`** 划分任务。
同一条 state 的多个问题保留在一起；问题中的 `src` 仅保存在元数据中，不用它拆分任务。
预处理只依赖 Python 标准库和 `valen.data.schema`，无需 PyTorch、模型或 GPU。

## 预处理

从仓库根目录运行：

```bash
python3 -m evaluation.decision_v7.prepare \
  --source data/v7/decision-v7 \
  --output data/decision-v7-cl
```

默认按输入 `manifest.json` 中的 `trainable_sources` 排序；若训练文件出现未列出的来源，
追加到末尾并按名称排序。`--tasks` 可指定任务子集和顺序，例如 `--tasks imdb amazon yelp`。
所有原始来源仍会转换，未选中的训练来源记入 `unselected_train_sources`。
输出目录必须不存在；源文件不修改，失败时自动清理临时输出。

当前数据的默认顺序和记录数如下（计数单位是 state，不是 QA）：

| 任务 | train | development | calibration | 原始 test |
| --- | ---: | ---: | ---: | ---: |
| agnews | 1000 | 116 | 60 | 116 |
| amazon | 1000 | 80 | 60 | 80 |
| banking77 | 1000 | 116 | 60 | 116 |
| boolq | 1000 | 80 | 60 | 80 |
| dbpedia14 | 1000 | 116 | 60 | 116 |
| imdb | 1000 | 80 | 60 | 80 |
| mnli | 1000 | 116 | 60 | 116 |
| sst5 | 1000 | 80 | 60 | 80 |
| trec | 1000 | 116 | 60 | 116 |
| yelp | 1000 | 80 | 60 | 80 |
| legacy_policy | 896 | 96 | 128 | — |
| compositional | 1680 | 128 | 240 | — |
| contrastive（仅评测） | — | — | — | 196 |

完整输入共 15,924 条 state、19,632 条 QA；其中训练集有 12,576 条 state、15,576 条 QA。
默认全部训练来源作为独立任务，包括两个合成来源；这是新的 source 持续学习协议，
并非原 manifest 中“公共来源加一个合成 arm”的实验选择协议。

## 划分与输出协议

默认 `--eval-split development`，所有任务统一使用原始 development 做逐阶段评测。
`legacy_policy`、`compositional` 没有原始 test，因此不能对完整十二任务序列直接选择 test。
可显式选取具有 test 的任务，例如：

```bash
python3 -m evaluation.decision_v7.prepare \
  --source data/v7/decision-v7 --output data/decision-v7-cl-test \
  --eval-split test --tasks agnews amazon banking77 boolq dbpedia14 imdb mnli sst5 trec yelp
```

不会将 development、calibration 或 test 混入训练，不会自动回退评测划分。
保留原始 `_meta.group_id`，所有来源合并检查四个划分间的 group 交集，发现交集直接报错。
这项检查不等于语义去重或预训练污染检查。

```text
data/decision-v7-cl/
  sequence.json
  source_manifest.json
  agnews/{train,development,calibration,test}.jsonl
  ...
  legacy_policy/{train,development,calibration}.jsonl
  compositional/{train,development,calibration}.jsonl
  contrastive/test.jsonl
```

`sequence.json` 复用现有 `coin_decision_v1` 容器协议，benchmark 为 `decision_v7_source_cl_v1`。
每个任务的 `train` 指向训练文件，**`test` 指向所选评测文件**：默认是 `任务/development.jsonl`，
对应 `test_sha256` 和 `counts.test` 也是 development 的摘要和计数。
原始 test 仍保留为 `任务/test.jsonl`；完整路径、SHA256、state/QA/原语计数在 `files` 中。
仅评测来源不进入 `tasks`，需要时单独调用 `valen.evaluate` 评测。

## 输入转换规则

- 字符串 state 原样保留；字典、列表 state 作为结构化文本序列化为 JSON，避免被误解为 messages。
- 字典或列表形式的 instructions 也序列化为 JSON 文本。
- Choice 保留全部候选及顺序（包括 Banking77 的 77 类），不重新采样负例。
  `null` 描述使用原候选 key；对象、数组等描述转成 JSON 文本。
- Noul 的布尔标签映射为 `true/false`；自定义 true/false 判断准则补入 instructions，
  因为 Valen 的 Noul 候选描述是固定的。
- Score 的整数标签是 **从 0 开始的档位索引**，保留原档位顺序。
- 将 `label` 转为 `targets[qid].probabilities` 中的 one-hot 分布。
  `request` 仅包含 state、题目类型、instructions 和适用的 criteria。
- 来源、问题 `src`、原 ID、证书等信息放在 `meta`，原 `_meta` 完整保存在 `meta.original_meta`。
  来源和标签不作为额外提示输入模型。

每条输出通过 `validate_record` 校验。原 manifest 声明的输入文件摘要、state/QA 数也会核对。
不生成新的答案，不丢弃无效标签；格式错误会报告源文件及行号。

## 持续训练

配套配置为 `configs/train/qwen/decision_v7_cl_moe.json`：纯文本阶段、shared_state、12 个任务容量。
模型路径需指向本地完整 Valen 模型；学习率和预算是实验起点，按硬件及实验设计调整。

```bash
python -m valen.continual \
  --config configs/train/qwen/decision_v7_cl_moe.json \
  --sequence data/decision-v7-cl/sequence.json \
  --output output/decision-v7-cl-moe \
  --gpus 2
```

该入口依次训练任务、使用上一任务 consolidated checkpoint 初始化下一任务，
并在每阶段结束后评估所有已见任务。`--dry-run` 可检查计划；同一输出继续执行时加 `--resume`。
使用 `valen.continual`，不要使用限制 CL4VQA 十个任务名的 `evaluation.vqa_cl.run`。
需要原始 test 最终结果时，另用训练完成的 checkpoint 评测相应 `任务/test.jsonl`。

### 读取进度不动时定位

`read` 进度条在 pack 编译后刷新，随后进行前向、反向、梯度同步和参数更新。
它不代表这些记录已完成训练。通过环境变量启用逐 rank 阶段日志：

```bash
VALEN_TRACE_STEPS=1 VALEN_TRACE_TIMEOUT=60 \
  bash scripts/train/decisionv7_cl_sft.sh 2>&1 | tee decision-v7-debug.log
```

也可以将相同环境变量加在自己的 `python -m valen.continual ...` 命令前。
`VALEN_TRACE_STEPS=1` 跟踪各任务全局 step 1；恢复到更晚的 step 时需提高此上限。
日志包括 rank、PID、step、阶段、该阶段调用序号，以及 pack 的 state 数和序列长度。
`begin/end` 表示阶段开始/完成；`body` 表示执行阶段本体；`wait_before/wait_after`
表示尚在等待阶段前后的 CUDA 同步。例如 `forward/body` 后无 `end`，需查看前向调用栈；
`sync/body` 表示执行梯度同步，需同时查看另一 rank 是否完成反向。

单个阶段超过 `VALEN_TRACE_TIMEOUT` 秒时，faulthandler 重复打印当前 Python 调用栈，
不会终止进程，也不额外发起跨卡通信。这不是 CUDA 原生调用栈，不能单靠 GPU 利用率
判定计算与通信阻塞。诊断使用进程级 faulthandler 定时器，不与其他使用该定时器的工具同时启用。
不设置 `VALEN_TRACE_STEPS` 时关闭诊断；代码和环境变量需在启动训练进程前生效。

## 验证

```bash
python3 -m unittest discover -s tests/integration -p test_decision_v7.py -v
```

覆盖三类标签、结构化输入、空描述候选、Noul 准则、标签隔离、确定性、全部划分保留、
自定义任务/评测选择、清单摘要、重复 ID、跨来源 group 泄漏、失败清理与拒绝覆盖。




# 持续学习离线评测

`eval_continual.py` 读取原生 `sequence.json` 和已完成的持续训练目录。
外层按训练阶段遍历：加载阶段 i 的模型，连续评测任务 0…i，释放模型，再加载下一阶段。
完整 t 任务序列进行 t(t+1)/2 次数据集评测，但每个 rank 仅构建和加载模型 t 次。
Processor/Tokenizer 也在阶段内复用，每个数据集的媒体根目录单独更新。
多卡是数据并行，每个进程各加载 t 次，不是所有 GPU 总计 t 个副本。

## 命令

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python -m valen.evaluation.eval_continual \
  --sequence data/decision-v7-cl/sequence.json \
  --run output/decision-v7-cl-sft \
  --output output/decision-v7-cl-sft-evaluation
```

双卡使用一次 torchrun 启动，整个序列共用进程组：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m valen.evaluation.eval_continual \
  --sequence data/decision-v7-cl/sequence.json \
  --run output/decision-v7-cl-sft \
  --output output/decision-v7-cl-sft-evaluation
```

CL-MoE 将 `--run` 换成对应训练目录即可。LoRA/full 使用各阶段的 `latest/`，
CL-MoE 使用 `consolidated/`，不需要额外传训练配置。
省略 `--output` 时写到 `RUN/continual_evaluation/`，保留原训练计划、checkpoint 和评测结果。
`--device cpu` 支持 CPU；`--attn-implementation sdpa` 可覆盖 checkpoint 的注意力后端。

评测前会核对训练计划中的 sequence 摘要和任务顺序、所有数据摘要、阶段完成标记、
checkpoint 摘要及配置。只需要 sequence 中的 `test` 文件，不要求训练 JSONL 仍在。
数据路径始终采用 sequence 的 `test` 字段，decision-v7 默认实际指向 development。
只遍历 `tasks`，不会自动评测 `files` 中的 contrastive 等 eval-only 来源。

## 输出与指标

```text
OUTPUT/
  evaluation_run.json
  sequence_run.json                 # 评测来源信息，非原训练计划的修改
  continual_metrics.json
  calibration_matrix.json
  calibration_tasks.csv
  00_agnews/
    stage_complete.json             # 原训练完成 checkpoint 的来源摘要
    eval/agnews/
      metrics.json
      predictions.jsonl
      predictions.rank0.jsonl
      evaluation_complete.json
      calibration.json
      forgotten_samples.jsonl
  ...
```

- `continual_metrics.json`：accuracy/NLL/Brier 矩阵、各单元 overall 指标、每阶段平均表现、
  最终 AP、BWT、AF、历史最高表现相对最终表现的遗忘量。矩阵行为训练阶段，列为评测任务，
  未见任务为 null。AP 按任务等权；BWT = mean(final[i] - diagonal[i])（不含最后任务），
  AF = -BWT，与 `valen.continual` 口径相同。未评测未来任务，不计算 FWT。
- 每单元 `metrics.json`：沿用单数据集评测的 accuracy、NLL、Brier；Noul 分组包含 macro-F1，
  Score 包含期望分值 MAE 和 RPS；并有题型、模态、语言、领域分组及耗时。
- `calibration_matrix.json` / `calibration_tasks.csv`：top-label ECE、平均置信度、过度自信、
  错误样本置信度、高置信错误率以及逐样本遗忘分析。复用 VQA-CL 分析逻辑，支持原生任意任务名。
  默认 15 个等宽 bin、高置信阈值 0.8，可用 `--bins`、`--high-confidence-threshold` 修改。
  Score 的校准正确性按 argmax 档位定义，不按期望分值四舍五入。

校准分析要求硬 one-hot 标签；含软标签时加 `--skip-calibration`，保留原生 NLL/Brier 等评测。
没有硬标签 accuracy 的单元相应为 null，不伪造 AP/AF/BWT。

## 中断恢复

相同命令加 `--resume`。清单一致时复用有完成标记且预测/指标摘要一致的单元，
重新计算缺失或损坏的单元，并重建所有汇总矩阵。一个阶段全部单元已完成时不加载模型；
部分未完成时也只加载一次。因此恢复运行的加载次数可以小于 t。
输出目录非空且未指定 `--resume` 会报错；输入或评测选项变化时使用新输出目录。

日志 `continual_model_loaded` 显示每个 rank 的 stage 和本次累计模型加载次数。
数据集报告的 `model_reused` 表明模型是否已评测过其他数据集；加载耗时仅计入阶段内首个
实际执行单元的 `load_and_evaluation_wall_seconds`，后续单元不重复计入。

## 测试

```bash
python3 -m unittest discover -s tests/common -p test_eval_continual.py -v
python3 -m unittest discover -s tests/common -p test_evaluation_session.py -v
```

这些测试使用模拟模型验证加载次数、释放、预测覆盖和校准、数据/模型摘要、CL-MoE checkpoint
选择及恢复流程；真实 CUDA 推理仍需在安装项目依赖并有模型的训练机器上执行。
