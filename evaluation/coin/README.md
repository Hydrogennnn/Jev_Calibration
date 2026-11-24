# CoIN-Decision

Preprocess CoIN into Valen's typed decision JSONL and train sequentially with
CL-MoE LoRA experts. See [the complete setup, protocols and commands](../../docs/continual_coin_zh.md).

```bash
python -m evaluation.coin.prepare --source data/coin/Instructions_Original \
  --image-root data/coin/images --output data/coin-decision
python -m valen.continual --config configs/train/qwen/coin_cl_moe.json \
  --sequence data/coin-decision/sequence.json --output output/coin-cl-moe --gpus 1
```

This is a **fixed-choice adaptation**, not the original generative CoIN protocol.
ScienceQA keeps native options; open answers use gold-included candidates and
training-only distractor pools. ImageNet becomes sampled K-way classification;
Grounding uses oracle boxes with synthetic negative boxes, not open localization.
VQAv2/TextVQA/VizWiz use labeled validation files instead of unlabeled public test
questions. All eight tasks, actual split choices and hashes are recorded.

No data, images, checkpoints or GPU training results are bundled.
