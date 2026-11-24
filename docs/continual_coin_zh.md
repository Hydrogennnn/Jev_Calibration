# CL-MoE 持续微调与 CoIN-Decision

本实现将 CL-MoE 的 MoE-LoRA、双路由和任务结束后的动态动量融合接入 Valen Qwen 后端。训练目标仍为原生候选概率的 SFT；支持发布的完整 Valen 模型，不需要重新初始化其 Mixer。提供 CoIN 八任务转换、顺序训练、每阶段历史任务复测和 AP/AF/BWT 汇总。

这是 **CL-MoE 在 Valen 上的适配**与 **CoIN 的固定候选改编协议**，不能直接与原论文的生成式 CoIN 分数比较。参考实现原来使用 CL4VQA，不是 CoIN。

## 1. 准备模型与原始数据

安装仓库依赖，在训练机器上准备完整的 `models/Valen-2B`，或者修改配置中的 `model.model_path`。下载 [CoIN 官方指令](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main/Instructions_Original)，图像按 [原始仓库](https://github.com/zackschen/CoIN) 的来源分别准备。

```text
data/coin/Instructions_Original/
  ScienceQA/{train,test}.json
  TextVQA/{train,val,test}.json
  ImageNet/{train,test}.json
  GQA/{train,test}.json
  VizWiz/{train,val,test}.json
  Grounding/{train,test}.json
  VQAv2/{train,val,test}.json
  OCRVQA/{train,test}.json
data/coin/images/
  ScienceQA/images/...
  COCO2014/{train2014,val2014,test2015}/...
  TextVQA/{train_images,test_images}/...
  ImageNet_withlabel/{train,val}/...
  GQA/images/...
  VizWiz/{train,val,test}/...
  OCR-VQA/images/...
```

指令目录是 `OCRVQA`，图像目录是 `OCR-VQA`。转换器支持 train 中的 `conversations` 和评测中的 `text/answer` 两种结构；将多轮训练问答拆为同图、同 group 的独立决策，不把标注回答作为下一题的输入。ScienceQA 中无图的题仍作为文本题处理。修复发布包 VizWiz val 中的 `.jpg.jpg` 重复后缀。

## 2. 转换八任务

在仓库根目录执行：

```bash
python -m evaluation.coin.prepare \
  --source data/coin/Instructions_Original \
  --image-root data/coin/images \
  --output data/coin-decision \
  --eval-source official \
  --choices 8 --seed 42
```

输出每任务的 `train.jsonl`、`test.jsonl`、适用时的 `answer_pool.json`，以及描述任务顺序、样本数、来源与转换文件 SHA256 的 `sequence.json`。已存在的输出目录会报错，失败转换不会留下半成品目录。图像存为绝对路径；搬运数据后应重新转换。仅做格式检查可加 `--allow-missing-images`，训练时仍必须准备图像。

默认顺序：`ScienceQA → TextVQA → ImageNet → GQA → VizWiz → Grounding → VQAv2 → OCRVQA`。用 `--tasks` 显式指定子集或其他顺序，实际顺序记录在 manifest 中。

`official` 使用 VQAv2、TextVQA、VizWiz 的 **val.json**，其余任务使用 **test.json**。这三个数据集的公开 test 通常只有问题，没有标准答案；不会伪造答案或静默丢弃。可用 `--annotations TASK=/path/annotations.json` 补充缺失标签，支持官方 VQAv2 annotations、TextVQA data、VizWiz 列表和 `{id: answer}` 映射。已有答案不被侧文件覆盖。

可选 `--eval-source holdout --holdout 0.1` 从训练数据按图像 group 切出评测集。同一图像在不同任务中归入同一切分，划分不依赖图像根目录。它不等价于原始 CoIN 切分。官方切分模式保持原划分，并报告跨任务 train/eval 图像 group 的交集数；同图不同题不能仅凭这个交集就判定为答案泄漏，但需披露。

`--limit 20` 是显式 smoke 子集：每任务每 split 最多 20 个展开后的问答。样本数量和 smoke 标记写入 manifest，不应报告为完整基准成绩。

### 候选协议

| 任务 | 转换行为 |
| --- | --- |
| ScienceQA | 解析原有 A/B/... 选项，保留原候选语义，字母映射到概率标签 |
| GQA / VQAv2 / VizWiz / TextVQA / OCRVQA | 正确答案纳入候选；干扰项仅来自该任务保留训练集的答案池，优先匹配 boolean/color/number/text 类型 |
| ImageNet | 同上，形成抽样 K-way 分类；不是完整 1000 类 ImageNet |
| Grounding | 正确 normalized xyxy 框加随机、低 IoU 的负框，形成 **oracle box selection** 诊断任务；不是开放定位或检测器端到端结果 |

候选构造不调用 LLM。每题用固定 seed、任务、题号、split 生成选项，并同步打乱候选与标签；测试候选在所有阶段保持不变。训练 compiler 仍可以独立打乱候选顺序。候选 ID 不编码类别或正误。

开放答案做大小写、空白和句末句点归一化；这不是完整官方 VQA normalization。训练池不足时保留可构造的实际候选数，少于两个时报错；yes/no 使用两个选项。**粗粒度匹配不保证所有干扰项语义互斥或足够困难**，正式论文实验应审核同义答案，并增加更强干扰项协议。Grounding 的随机负框比较容易，候选构造依赖 gold，结果只反映给定候选的判断能力。

已有 `answers` 列表会转成标注票数分布，保存在 targets 和 meta，不能把它解释为官方 VQA 共识得分。只有一个答案的 CoIN 原始行不会被伪造成人类投票。当前通用评测器对软标签报告 NLL/Brier，不报告硬标签 accuracy；这种矩阵的 AP/AF/BWT 会相应为空。所有 meta/targets 均不进入模型请求。

## 3. 持续训练并逐阶段复测

配置：[`configs/train/qwen/coin_cl_moe.json`](../configs/train/qwen/coin_cl_moe.json)。确认模型路径、设备、训练预算和学习率；示例学习率是起点，不是调优后的结果。

```bash
python -m valen.continual \
  --config configs/train/qwen/coin_cl_moe.json \
  --sequence data/coin-decision/sequence.json \
  --output output/coin-cl-moe \
  --gpus 4
```

单卡设置 `--gpus 1`。该命令在外层启动各阶段的 torchrun，**不要再用 torchrun 包裹 `valen.continual`**。`--dry-run` 只写 stage config 并显示命令；之后用同样命令加 `--resume` 执行。默认每阶段评测所有已见任务；`--skip-eval` 可关闭，但不会产生持续评测矩阵。

中断后使用相同命令加 `--resume`。驱动器检查配置、序列和 JSONL 哈希，跳过已完成且 checkpoint 哈希一致的阶段；未完成阶段从 `latest` 恢复。每次评测进程成功退出后写入 `evaluation_complete.json`，记录权重、数据、metrics 与 predictions 的哈希；缺少标记或输出不完整的评测会重跑。不要修改已完成阶段的权重后复用其完成标记。

在 Linux/macOS 上，驱动器跟踪训练和评测的子孙进程（包括 torchrun 创建的独立会话）。按一次 `Ctrl+C` 会转发中断，等待最多 10 秒，再发送 `SIGTERM` 等待最多 5 秒，最后用 `SIGKILL` 清理仍存活的进程；再次按 `Ctrl+C` 会跳过温和退出等待。不会给中断阶段写入完成标记。此功能依赖 `psutil>=5.9,<8`。它不保证额外保存 checkpoint；恢复使用已有的 `latest`。父进程被 `kill -9`、系统崩溃或子进程在被发现前脱离进程树时，Python 清理逻辑无法保证执行；需要更强隔离时使用作业调度器或 cgroup。

```text
output/coin-cl-moe/
  sequence_run.json
  00_ScienceQA/
    stage_config.json
    latest/                 # 原始训练状态，精确 resume 用
    consolidated/           # 完成统计和动量融合，下一阶段与推理用
    consolidation.json      # 每层路由均值、top-K、历史集合、融合权重
    stage_complete.json
    eval/ScienceQA/...
  01_TextVQA/
    eval/{ScienceQA,TextVQA}/...
  continual_metrics.json    # accuracy/NLL/Brier 矩阵，AP、AF、BWT
```

也可以手动使用 `python -m valen.train --config ...`：首任务 `cl_moe_task=0`，下一任务递增并用 `--initialize previous/consolidated`。任务编号只供训练流程校验，不进入推理或作为 oracle task label。

`latest` 保存融合前优化器、RNG、数据进度和旧专家锚点。`consolidated` 保存融合后权重、历史路由、任务原型和专家集合；旧优化器状态不再匹配融合参数，因此禁止对它 `--resume`。下一任务 `--initialize` 会新建优化器。达到 `max_steps` 也视为当前配置预算结束并生成 consolidated；若要增加同一任务预算，从 raw latest 恢复，重新统计和融合，避免重复融合。

## 4. 算法实现与参考代码的差异

参考：[CL-MoE 论文](https://arxiv.org/abs/2503.00413)、[作者代码](https://github.com/ECNU-ICALK/CL-MoE)。对照了提供的 `CLMoE/peft/tuners/clitmoelora.py`、`statistic.py`、`params.py` 和训练脚本。这里是独立的原生 PyTorch 实现，不依赖作者仓库里修改过的旧 PEFT，也不需要 TensorFlow。

- **MoE-LoRA**：替换语言主干 Linear，冻结原参数；总 rank 在专家间均分，缩放保持 `lora_alpha / lora_rank`，不是 alpha / 每专家 rank。默认 4 专家、总 rank 32、top-K 2；论文报告 8 专家、总 rank 64、alpha 128，可改配置对照。top-K 用于选择任务专家集合，前向仍是 dense soft mixture，不能理解为稀疏计算加速。
- **训练路由**：使用 token hidden state 的可学习 softmax router。每一步前向无文件 IO、无计时采样、无统计副作用，兼容梯度检查点重算。
- **任务路由**：训练完成后在当前训练数据上做固定顺序、无梯度统计。每序列先对非 padding token 的路由概率平均，再跨序列平均，各层独立统计；共享 state 一条记录对应一序列。多卡汇总计数和概率和，不平均各卡的均值。默认完整当前训练集；`calibration_limit>0` 是显式前 N 条近似，报告实际覆盖。
- **任务索引**：Qwen 没有论文所述的 CLS encoder readout，使用冻结输入 embedding 的文本 token 均值构造原型，排除 padding 和视觉占位 token。输入包括 state、指令和候选文字；推理选择最近欧氏距离原型，不读 meta/task ID。这个特征提取器是迁移选择，**不是论文 CLS 实现的逐位复现**，应作为后续消融因素。
- **双路由推理**：`beta × instance_route + (1-beta) × inferred_task_route`。历史 task route 保留在 checkpoint 中；评测不更新原型或路由。
- **动态动量**：每层累积所有历史任务 top-K 专家的并集。当前 top-K 中历史未出现的专家使用旧权重 `1-gamma`，其余使用 `gamma`；更新为 `old_weight × old + (1-old_weight) × new`。首次任务不与随机初始化混合。融合发生在 LoRA 的 A/B 因子上，与参考代码的因子融合方向一致；不声称它等价于融合完整 BA 矩阵。
- **纠正参考脚本差异**：原 `statistic.py` 统计 top-2 命中次数、`params.py` 比较前一个任务并按字符串前缀匹配参数，且分支方向与论文“当前任务专属专家”的定义不一致。本实现采用论文公式 6、10–12 的均值、完整历史集合及新任务专属定义，并做数值测试。
- **Valen 特有参数**：Mixer 与视觉 merger 仍按现有 stage 训练，不进行专家动量融合。router 使用当前任务训练后的参数。原型和 task route 属于持久 buffer，原生 checkpoint 会显式保存。

当前支持 Qwen `text/joint/vision_top` 阶段的 SFT，不开放 RLCD、warmup 或 dual_encoder 的 CL-MoE；会在配置阶段明确报错。输入相关的 MoE 无法静态合并成普通 LoRA 权重，现有 HF 合并/标准 PEFT adapter 导出会拒绝 CL-MoE。使用原生 checkpoint 的 `valen.evaluate` / `valen.inference` 路径。

## 5. 对照与验证

顺序 LoRA 基线可复制配置，把 `finetuning_type` 改成 `lora`，移除 `cl_moe`/`cl_moe_task`，保持总 rank、训练数据和训练预算一致，用独立 output 运行同一 `valen.continual`。LoRA-MoE 有额外 router 参数和训练后统计成本，比较时应同时报告。

矩阵 (R_{t,j}) 的数值是 0–1。AP 为最后一阶段各任务平均 accuracy；BWT 为历史任务最终 accuracy 减其刚学完时 accuracy 的平均；AF=-BWT（沿用 CL-MoE 的定义），另报相对历史最高分的 max_past_forgetting。不计算没有训练前评测支持的 FWT。每格保留 NLL、Brier 和样本计数。

```bash
python -m pytest tests/qwen/test_cl_moe.py \
  tests/qwen/test_published_finetuning.py \
  tests/integration/test_coin.py tests/common/test_checkpoint.py -q
```

这些测试覆盖：真实小型 Qwen3.5 的两阶段训练/重载、原始断点恢复、图像推理、专家融合数值、双路由与 padding、多轮 CoIN 转换、八任务确定性、标签缺失失败、跨任务图像 holdout 与序列配置/哈希校验。小模型验证不替代完整 CoIN GPU 训练或论文指标复现。
