# CL4VQA 十任务持续学习：Valen 适配

本入口读取 CL-MoE 作者发布的 CL4VQA 划分，在 Valen 上执行十任务持续训练、任务结束后的路由统计/动量融合，以及每阶段对所有已见任务的评测。它与现有 CoIN 八数据集入口独立。

**保留作者的任务划分与持续学习策略，输出适配为 Valen 的 typed decision；不是 LLaVA 生成式 VQA 的逐位复现，分数不能直接与论文表格比较。**

## 1. 参考与任务顺序

- [CL-MoE 论文](https://arxiv.org/html/2503.00413v1)，尤其 §4、§5。
- [作者代码](https://github.com/ECNU-ICALK/CL-MoE/tree/a9eff1c35a0cea13aa48e7233f5bbc8dd7b09471)，核对 revision `a9eff1c35a0cea13aa48e7233f5bbc8dd7b09471`。
- [作者 CL4VQA 标注](https://drive.google.com/drive/folders/1mcAjzmCU1UVW0TKvsHAy9Sr1hmwsJudo)，读取已有 `train/train_q_TASK.json`、`test/test_q_TASK.json`，不根据关键词重划 VQA-v2，也不读取混合全部任务的 `train_q_all.json`。

| 序号 | 缩写 | 任务目录 | 含义 | 本次发布数据展开后的 train / test QA 数 |
| --- | --- | --- | --- | --- |
| 0 | Rec. | recognition | 识别 | 146425 / 6265 |
| 1 | Loc. | location | 位置 | 14921 / 702 |
| 2 | Jud. | judge | 判断 | 178990 / 7970 |
| 3 | Com. | commonsense | 常识 | 28312 / 1227 |
| 4 | Cou. | count | 计数 | 68564 / 2905 |
| 5 | Act. | action | 动作 | 35185 / 1448 |
| 6 | Col. | color | 颜色 | 57303 / 2451 |
| 7 | Typ. | type | 类型 | 26563 / 1211 |
| 8 | Sub. | subcategory | 子类别 | 35880 / 1604 |
| 9 | Cau. | causal | 因果 | 6376 / 217 |

这些是对公开下载文件的实际统计，不是从论文表格硬编码的截断数。公开文件可能更新，每次下载和转换都会记录 SHA256。训练原始文件共 365871 行，其中一行可能包含同图多轮问答；转换后共 598519 QA。每个 QA 成为独立 state，不将其他轮的标准答案放入请求。重复的 image ID / source ID 不会被覆盖，输出 ID 加入源行索引和轮次。测试保留全部 26000 条，不从训练集另行抽样。图片 group 哈希与主机根目录无关；本次检查 train/test 图片交集为 0。

## 2. 下载数据

只为下载标注安装 `gdown`；训练仍使用项目现有依赖。下面命令会下载 20 个任务文件，约 164 MiB，已有合法文件会复用：

```bash
python -m pip install 'gdown>=5,<6'
python -m evaluation.vqa_cl.download --output data/CL4VQA
```

如果 Google Drive 在服务器不可访问，可在能访问的机器下载后复制整个目录。目录应为：

```text
data/CL4VQA/
  train/train_q_recognition.json
  train/train_q_location.json
  ...
  test/test_q_recognition.json
  ...
  download_manifest.json
```

图片只需要 COCO `train2014` 与 `val2014`。如果已运行 CoIN 的下载脚本，可以直接复用 `data/coin/images/COCO/`，无需再下载。没有图片时可用：

```bash
python -m evaluation.vqa_cl.download \
  --output data/CL4VQA --images data/COCO2014
```

这会额外下载约 19 GB 压缩包并解压，需要同时容纳压缩包和图片的磁盘空间。使用官方 HTTPS 地址、断点下载和成功解压标记；不关闭 TLS 校验。

## 3. 转换成 Valen 输入

```bash
python -m evaluation.vqa_cl.prepare \
  --source data/CL4VQA \
  --image-root data/coin/images/COCO \
  --output data/vqa-cl-decision
```

如果图片下载到 `data/COCO2014`，相应改 `--image-root`。脚本兼容作者的 `./COCO2014/train2014/...` 路径和 CoIN 的 `COCO/train2014/...` 布局。默认检查图片存在；`--allow-missing-images` 仅供转换检查，实际训练必须有图片。

默认 `--primitives typed`，按任务语义构造原语：

- Judge 中标准答案为 yes/no 的题目使用 `noul`，target 的键为 `true/false`。该划分确实包含其他答案，其他情况回退为 `choice`，不丢弃。
- Count 中规范的非负整数字符串、且位于 `0..20` 的答案使用 `score`。criteria 固定为 `0,1,...,20`，标签索引就是真实计数；`--count-max N` 可显式修改量程，不使用测试集推断量程。`two`、`many`、`02` 或超量程答案回退为 `choice`，避免擅自改写标准答案。
- 其他任务使用 `choice`，默认最多 8 个候选：一个标准答案加来自**当前任务训练集**的干扰项。测试标签和未来任务标签不进入干扰池；候选不足时保留实际数量，少于 2 个则报错。标准答案会包含在本题候选里，因此这是 gold-included 候选适配。

`--primitives choice` 可运行统一选择题对照。`--choices` 只控制 choice 候选数，不改变 noul 的两类或 score 的固定量程。

答案去重使用去除首尾空白后的大小写不敏感匹配，展示保留原始拼写与标点，包括 `.`。作者的生成式评测脚本也是大小写不敏感的单答案精确匹配；没有在这里伪造十个人类标注或 VQA 共识分数。全部 target 是单答案 one-hot，使通用评测器能产生 accuracy/AP/AF。

输出结构：

```text
data/vqa-cl-decision/
  sequence.json
  recognition/{train.jsonl,test.jsonl,answer_pool.json}
  location/{train.jsonl,test.jsonl,answer_pool.json}
  ...
```

manifest 包含原始行数、展开问答数、原语数量、回退原因数量、输入/输出哈希、图片检查标志和 smoke 标志。兼容现有驱动器，容器 `format` 沿用 `coin_decision_v1`，另以 `benchmark=cl4vqa_valen_v1` 明确区分数据协议。新运行入口会拒绝误传 CoIN 的序列。

## 4. 训练设置与启动

修改 [`vqa_cl_moe.json`](../../configs/train/qwen/vqa_cl_moe.json) 中的 `model.model_path` 为服务器上的完整 Valen 模型目录，建议使用绝对路径。当前模型/Processor 加载代码要求本地文件；配置的 `models/Valen-2B` 是示例目录。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.vqa_cl.run \
  --config configs/train/qwen/vqa_cl_moe.json \
  --sequence data/vqa-cl-decision/sequence.json \
  --output output/vqa-cl-moe \
  --gpus 4
```

默认设置：8 experts、top-2、总 rank 64、alpha 128、beta 0.5、gamma 0.7、最多 10 tasks；每任务 1 epoch，无历史样本回放；AdamW、weight decay 0、LoRA/router LR `2e-4`、merger LR `2e-5`、head LR `2e-4`、cosine decay、3% warmup。关闭额外 Brier/RPS 损失，仅用决策交叉熵。

`global_batch_size=16` 表示跨卡合计 16 个 QA/state，每卡累积 `16 / gpus` 个，必须整除。例如 4 卡时每卡累积 4 个，`microbatch_size=1` 控制每次前向/反向的显存占用。数据尾部不补齐、不丢弃，最后一个 global batch 可以更小；空闲 rank 仍同步梯度。`tokens_per_step` 作为这一包的安全上限，超限会明确报错，不静默减少固定 batch。

每任务重新启动 cosine 日程；首次 warmup 更新学习率为 0，与 step=0 的线性 warmup 一致。日程基于完整任务的计划 optimizer steps；如有 `max_steps` 则按该预算截断。中断恢复从 checkpoint 的 optimizer_steps 和 initial_lr 重建，不受预取影响。要做严格恢复，不改变 global batch、epoch 或 max_steps 等日程参数。

每阶段从前一阶段 `consolidated` 初始化，创建新优化器；训练后对当前任务完整训练集做路由/原型统计，再融合专家。模型推理通过已有的任务原型选择路由，不把任务 ID 当作推理标签输入。默认评测每个阶段的所有已见任务，完整十阶段共 55 次评测。

```text
output/vqa-cl-moe/
  sequence_run.json
  00_recognition/{stage_config.json,latest/,consolidated/,consolidation.json,eval/...}
  ...
  09_causal/...
  continual_metrics.json
  calibration_matrix.json
  calibration_tasks.csv
```

`continual_metrics.json` 给出矩阵、最终 AP、AF、BWT；数值为 0–1，乘 100 才是百分数。AF 为各旧任务训练后对角线准确率减最终准确率的平均，允许负值；不是历史最大值遗忘（后者单独报告）。`--skip-eval` 关闭评测后不生成该矩阵。

`--dry-run` 只验证文件并生成阶段配置/命令。之后同一目录正式执行加 `--resume`；中断恢复也加 `--resume`。新实验用新目录，配置哈希变化时不覆盖旧结果。Ctrl+C 的进程树清理复用 `valen.processes`。

### 任务级 ECE 与遗忘后的过度自信

`evaluation.vqa_cl.run` 默认在每次评测完成后统计校准指标，完整十任务产生 55 个单元。默认 **15 个等宽置信度区间**，高置信度阈值为 **0.8**；启动命令可加 `--ece-bins 15 --high-confidence-threshold 0.8`。`--skip-eval` 或 `--dry-run` 不计算这些指标。恢复运行会从已有完整预测重建统计，不重复执行已完成的模型评测。

采用 top-label ECE：每题置信度 `c = max(probabilities)`，预测为概率最大的候选，正确性 `a` 是预测与 one-hot 标准答案相等（0 或 1）。**不使用 `answer.confidence`**，因为那个字段在部分原语中表示经变换的分布集中度。`noul` 使用 true/false 两类中较大的概率；`score` 使用概率最大的计数类别，与已有 accuracy 一致，不使用期望分数四舍五入。

对任务的固定测试集，设第 b 个区间包含 `n_b` 个样本，总样本数 N：

```text
ECE = Σ_b (n_b / N) × |mean_accuracy_b − mean_confidence_b|
```

区间为 `[b/B, (b+1)/B)`，最后一区间包含 1；空区间不计入 ECE。仅归一化保存的 float32 概率的求和舍入误差，不拟合温度。所有结果必须使用相同的评测温度。定义参考 [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)。数值均为 0–1；乘 100 后为百分数/百分点。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `calibration_matrix.json` | 行为训练阶段，列为评测任务；含 `ece_matrix`、accuracy、平均置信度、过度自信等矩阵，以及各单元完整统计 |
| `calibration_tasks.csv` | 每行一个阶段 × 任务，便于画曲线或做表；包含遗忘样本数量与置信度变化 |
| `NN_STAGE/eval/TASK/calibration.json` | 本任务总体、按原语/候选数分组的指标和 reliability bins；两种基准下的遗忘统计 |
| `NN_STAGE/eval/TASK/forgotten_samples.jsonl` | 对齐后的遗忘样本 ID、当前预测、真实答案、前后置信度及真实答案概率，便于复查原样本 |

`macro_task_ece_by_stage` 是该阶段所有已见任务 ECE 的等权平均，不是把所有任务样本混合后重新分箱的 ECE。未评测单元为 `null`；如果该阶段有缺失单元，其宏平均也为 `null`。没有错误/没有遗忘样本时，相应平均置信度为 `null`，而不是 0。

为了检验“被遗忘时可能对错误答案过度自信”，除了 ECE 还记录：

- `overconfidence_error` / `underconfidence_error`：分别累加区间内置信度高于/低于准确率的加权差；两者之和为 ECE。
- `signed_confidence_gap`：整体平均置信度减准确率，正值表示总体偏向过度自信；正负区间可能相互抵消。
- `mean_wrong_confidence`：答错样本的平均置信度。
- `high_confidence_error_rate`：高置信度预测中答错的比例，分母是高置信度样本数。
- `forgetting_from_learning`：与该任务刚训练完时的对角线评测比较，基准答对、当前答错的样本记为 `forgotten`。同时记录持续答对、持续答错、由错变对三组作参照。
- `forgetting_from_previous_stage`：与紧邻的上一训练阶段比较，识别本阶段新发生的遗忘；若上一阶段评测缺失则不可用，不跨过缺口冒充相邻比较。

遗忘组包含 `mean_current_confidence`、`mean_confidence_change`（当前最大类别概率减基准最大类别概率）、`mean_gold_probability_change`、高置信度比例、占基准答对样本的比例等。最大概率可能在前后属于不同答案，因此同时保留标准答案概率变化。两种遗忘定义都通过 `(record_index, qid)` 对齐同一测试题，并核对题目身份、候选和标准答案，不按预测文件行位置对齐。

**ECE 增大本身不能证明过度自信**，也可能是低估自信。建议同时查看某旧任务随阶段变化的 accuracy/ECE、遗忘组当前置信度和其变化，并与持续答错组比较。不同任务的候选数不同，跨任务比较时可查看 `by_primitive` 和 `by_candidate_count`。这些统计对应当前候选适配后的 Valen 任务，不是生成式 VQA 的校准结论；smoke 数据只验证流程。

已有运行可以直接离线补算或调整分箱，无需重新训练、加载 checkpoint 或执行模型推理：

```bash
python -m evaluation.vqa_cl.calibration \
  --sequence data/vqa-cl-decision/sequence.json \
  --run output/vqa-cl-moe \
  --bins 15 \
  --high-confidence-threshold 0.8
```

需要原始转换后的测试 JSONL，以及运行目录内的 `sequence_run.json`、阶段完成标记、各次评测的 `metrics.json`、`predictions.jsonl` 和 `evaluation_complete.json`。补算核验序列、测试集和预测文件的哈希及样本覆盖；不改原有评测文件和完成标记。未完成评测的单元跳过，已标记完成却被修改的结果会报错。重算会覆盖校准统计文件，新的分箱数与阈值会写入结果；用于比较的实验应固定这两个设置。

## 5. Smoke test

先用四个任务覆盖三种原语和多个持续阶段：

```bash
python -m evaluation.vqa_cl.prepare \
  --source data/CL4VQA --image-root data/coin/images/COCO \
  --output data/vqa-cl-smoke \
  --tasks Rec Jud Cou Col --limit 20

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.vqa_cl.run \
  --config configs/train/qwen/vqa_cl_moe_smoke.json \
  --sequence data/vqa-cl-smoke/sequence.json \
  --output output/vqa-cl-smoke --gpus 4
```

也要修改 smoke 配置里的模型路径。它设置每任务最多 2 steps、global batch 4、warmup 0、最多 8 条路由校准样本，实际执行模型训练/保存/融合/评测。此配置不是正式实验；恢复正式训练应使用完整数据、完整配置和新输出目录。

## 6. 与论文/发布脚本的边界

| 项目 | 参考 | 此实现 |
| --- | --- | --- |
| 数据划分和顺序 | CL4VQA 十任务 | 直接读取作者文件，支持显式 `--tasks` 重排/子集 |
| 专家配置 | 论文 8/rank64/alpha128；发布脚本 4/rank32/alpha64 | 默认选论文配置，记录差异 |
| batch | 论文写 batch16；发布脚本 per-device16，示例用2卡 | 明确使用 global batch16；QA 展开为独立 state，不宣称与多轮生成训练 batch 完全等价 |
| 基座和损失 | LLaVA-7B，答案 token 自回归 CE | Valen/Qwen，typed decision CE |
| 序列长度 | 发布脚本 2048 | 默认8192，容纳 Qwen 图像 token 和候选描述 |
| 优化 | AdamW + cosine，发布脚本 warmup .03 | 新增可选固定 batch/cosine，原有 token 模式不变 |
| 模型状态 | 作者 LLaVA projector 和 PEFT fork | 原生 Valen Mixer、merger 和已有 CL-MoE 层 |
| 路由与动量 | 论文公式与脚本存在实现差异 | 复用项目既有实现；详见[现有算法说明](../../docs/continual_coin_zh.md#4-算法实现与参考代码的差异) |
| 评测 | 自由生成答案的单答案精确匹配 | 给定候选/量程的决策 accuracy；不能直接复现论文 AP/AF 数值 |

尤其：Qwen 任务原型用冻结文本输入 embedding 的均值，并非论文描述的 CLS 特征；现有 MoE 为所有专家加权求和，top-K 用于任务专家集合及动量，不是运行时只算 K 个专家。图像分辨率/视觉 token 策略沿用模型 Processor。这些适配没有在本次任务中悄悄改成另一套实现。

新增功能集中在 `evaluation/vqa_cl/`、两份配置、`valen/training/schedule.py` 和新测试；训练器增加组批/调度接入，`valen/continual.py` 增加可选的评测完成回调以运行 VQA-CL 校准统计，不改 CoIN 预处理、原有配置、分布式梯度逻辑或 CL-MoE 核心算法。

## 7. 验证

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m pytest \
  tests/integration/test_vqa_cl.py tests/integration/test_vqa_cl_calibration.py \
  tests/common/test_training_schedule.py \
  tests/qwen/test_vqa_cl_training.py tests/common/test_prefetch.py \
  tests/qwen/test_cl_moe.py tests/integration/test_distributed.py -q
```

测试包括十阶段任务传递和缓存、训练池无测试标签、重复 image ID/多轮展开、缺失图片/坏标注失败、三个原语和回退、固定 batch 的不均匀尾部、日程及 optimizer 恢复，以及小型真实 Qwen 的带图三阶段训练/融合/评测与恢复。校准测试覆盖 ECE 手算值/分箱边界、正负偏差、乱序样本配对、两种遗忘基准、缺失评测、无遗忘样本、坏预测/来源变更拒绝，以及在线与离线补算一致性。公开完整标注另行完成全量转换检查；这些验证不等于服务器上的完整多 GPU 实验结果。
