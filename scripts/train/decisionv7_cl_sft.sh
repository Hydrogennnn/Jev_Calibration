CUDA_VISIBLE_DEVICES=0,1 \
python -m valen.continual \
  --config configs/train/qwen/decision_v7_sft_cl.json \
  --sequence data/decision-v7-cl/sequence.json \
  --output output/decision-v7-cl-sft \
  --gpus 2 \
  --resume \
  --skip-eval
