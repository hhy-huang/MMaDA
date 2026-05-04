# Evaluation Demo

This is an evaluation demo for MMaDA.

## 1. VLM Evaluation

We use `VLMEvalKit` to evaluate MMaDA's VLM capabilities.

### 1.1 Install Dependencies
```bash
cd evaluation_demo/VLMEvalKit
pip install -e
```

### 1.2 Configure Model Paths
In `VLMEvalKit/vlmeval/config.py`, set the following paths:

```python
mmada = {
    "MMaDA-MixCoT": partial(
        MMaDA, 
        model_path="Gen-Verse/MMaDA-8B-MixCoT",
        tokenizer_path="/Gen-Verse/MMaDA-8B-MixCoT",
        vq_model_path="showlab/magvitv2",
        vq_model_type="magvitv2",
        resolution=512,
    ),
}
```

### 1.3 Configure Dataset Configs
In `VLMEvalKit/vlmeval/vlm/mmada/dataset_configs.py`, you can set the max_new_tokens, steps, block_length for each dataset. For example: 

```python
DATASET_CONFIGS = {
    "MathVista_MINI": {
        "max_new_tokens": 96,
        "steps": 96,
        "block_length": 48,
    },
    
    "MathVerse_MINI_Vision_Only": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 32,
    },
}
```

### 1.4 Run VLM Evaluation
```bash
# Put VLMEvalKit datasets under /data/haoyuhuang/data/MMaDA
export LMUData=/data/haoyuhuang/data/MMaDA/LMUData

# Single GPU
CUDA_VISIBLE_DEVICES=0 python run.py --data {dataset_name} --model MMaDA-MixCoT

# Multi-GPU
torchrun --nproc-per-node=8 --master-port=54321 run.py --data {dataset_name} --model MMaDA-MixCoT

# USE COT 
USE_COT=1 torchrun --nproc-per-node=8 --master-port=54321 run.py --data MathVista_MINI --model MMaDA-MixCoT
```

### 1.5 WISE benchmark support (PKU-YuanGroup/WISE)
WISE is a text-to-image benchmark and should be run in the T2I pipeline (not `VLMEvalKit/run.py`).
Repo: [PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE)

## 2. LLM Evaluation

We directly adopt LLaDA and Fast-dLLM's evaluation scripts. Please note we have not yet implemented and tuned the reasoning process and currently only implemented the non-thinking version, and the results are not yet aligned with our internal results. 
Configuring `lm_eval_harness` in the future may resolve this issue.

### 2.1 Install Dependencies
```bash
cd evaluation_demo/lm
pip install lm-eval 
```

### 2.2 Run LLM Evaluation
```bash
# Using lm-eval-harness
bash eval.sh
```

## 3. Text to image generation

We use [GenEval](https://github.com/djghosh13/geneval) to evaluate the text to image generation capabilities of MMaDA. Please refer to the [GenEval](https://github.com/djghosh13/geneval) for specific instructions.

## 3. Text to image generation (hhy implemented)

### Sample 50 prompts from COCO-VAL.

```shell
LMUData=/data/haoyuhuang/data/MMaDA/LMUData \
python evaluation/sample_prompts_from_coco_val.py \
  --out-jsonl /data/haoyuhuang/mmada_cot_t2i_50_entropy/prompts_50.jsonl \
  --num-prompts 50 \
  --seed 42
```


Image generation based on the sampled prompts.

```shell
CUDA_VISIBLE_DEVICES=7 python evaluation/run_geneval_mmada.py \
  --prompts-jsonl /data/haoyuhuang/mmada_cot_t2i_50_entropy/prompts_50.jsonl \
  --outdir /data/haoyuhuang/mmada_cot_t2i_50_entropy/images \
  --model-path /data/haoyuhuang/model/models--Gen-Verse--MMaDA-8B-MixCoT/snapshots/3ee0085f0c42541f1134aae30482954451952406 \
  --vq-model-path showlab/magvitv2 \
  --num-images-per-prompt 1 \
  --steps 30 \
  --guidance-scale 6 \
  --scheduler cosine \
  --remasking entropy \
  --cfg-schedule static 
```

CLIP score.

```shell
python evaluation/clip_score.py \
  --images-dir /data/haoyuhuang/mmada_cot_t2i_50_entropy/images \
  --pairs-jsonl /data/haoyuhuang/mmada_cot_t2i_50_entropy/pairs.jsonl \
  --clip-model ViT-L-14 \
  --clip-pretrained openai \
  --batch-size 64 \
  --score-scale 100
```

### Sample 50 prompts from WISE.

```shell
# 1) Clone WISE benchmark
cd /data/haoyuhuang
git clone https://github.com/PKU-YuanGroup/WISE.git

# 2) Build a 50-prompt jsonl from a WISE category file
#    (change --wise-json to cultural_common_sense.json / spatio-temporal_reasoning.json / natural_science.json)
cd /home/haoyuhuang/www/baselines/MMaDA
python evaluation/build_prompts_from_wise.py \
  --wise-json /data/haoyuhuang/WISE/data/cultural_common_sense.json \
  --out-jsonl /data/haoyuhuang/mmada_cot_wise_50_entropy/prompts_wise_50.jsonl \
  --max-prompts 50

# 3) Generate images with MMaDA
#    Also export flat files 1.png ... 50.png required by WISE gpt_eval.py
CUDA_VISIBLE_DEVICES=5 python evaluation/run_geneval_mmada.py \
  --prompts-jsonl /data/haoyuhuang/mmada_cot_wise_50_entropy/prompts_wise_50.jsonl \
  --outdir /data/haoyuhuang/mmada_cot_wise_50_entropy/images_structured \
  --flat-output-dir /data/haoyuhuang/mmada_cot_wise_50_entropy/images_flat \
  --flat-index-base 1 \
  --model-path /data/haoyuhuang/model/models--Gen-Verse--MMaDA-8B-MixCoT/snapshots/3ee0085f0c42541f1134aae30482954451952406 \
  --vq-model-path showlab/magvitv2 \
  --num-images-per-prompt 1 \
  --steps 30 \
  --guidance-scale 3.5 \
  --scheduler cosine \
  --remasking entropy \
  --cfg-schedule static

# 4) Run WISE official scoring
cd /data/haoyuhuang/WISE
IMAGE_DIR=/data/haoyuhuang/mmada_cot_wise_50_entropy/images_flat
python gpt_eval.py \
  --json_path data/cultural_common_sense.json \
  --output_dir ${IMAGE_DIR}/Results/cultural_common_sense \
  --image_dir ${IMAGE_DIR} \
  --api_key "sk-1mHc4Aj2s54jNRvJH9VlqGlN3ogqsgj4gHrrg0XkmVI77kEJ" \
  --model "gpt-4o-mini" \
  --result_full ${IMAGE_DIR}/Results/cultural_common_sense_full_results.json \
  --result_scores ${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl \
  --max_workers 32

# 5) Compute mean WiScore for the evaluated subset (e.g., first 50)
python /home/haoyuhuang/www/baselines/MMaDA/evaluation/wise_subset_wiscore.py \
  --scores-jsonl ${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl \
  --max-id 50
```

### Result:

CLIP scores on **COCO-VAL** and **WISE (Cultural)** summary table (CLIP=`ViT-L-14/openai`, score_scale=`100`, device=`cuda`, num_pairs=`50`):

| Remasking | CFG schedule | CLIP score | WISE (Cultural) |
|---|---|---|---|
| low_confidence | fixed (static) | 24.99 | 0.3900 |
| entropy (high entropy) | fixed (static) | 24.92 |  |
| margin (low margin) | fixed (static) | **25.30** | 0.3940 |
| low_confidence | linear_decay | 24.60 |  |
