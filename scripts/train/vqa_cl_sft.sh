CUDA_VISIBLE_DEVICES=0,1 \
python -m evaluation.vqa_cl.run \
    --config configs/train/qwen/vqa_cl_moe.json \
    --sequence data/vqa-cl-decision/sequence.json \
    --output output/vqa-cl-moe \
    --gpus 2 \
    --resume