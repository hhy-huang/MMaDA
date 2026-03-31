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

Sample 50 prompts from COCO-VAL.

```shell
LMUData=/data/haoyuhuang/data/MMaDA/LMUData \
python evaluation/sample_prompts_from_coco_val.py \
  --out-jsonl /data/haoyuhuang/mmada_t2i_50/prompts_50.jsonl \
  --num-prompts 50 \
  --seed 42
```


Image generation based on the sampled prompts.

```shell
CUDA_VISIBLE_DEVICES=7 python evaluation/run_geneval_mmada.py \
  --prompts-jsonl /data/haoyuhuang/mmada_t2i_50_linear_CFG/prompts_50.jsonl \
  --outdir /data/haoyuhuang/mmada_t2i_50_linear_CFG/images \
  --model-path /data/haoyuhuang/model/models--Gen-Verse--MMaDA-8B-Base/snapshots/065b30692dd6a2d0560d280d264e5e0092c05bc4 \
  --vq-model-path showlab/magvitv2 \
  --num-images-per-prompt 1 \
  --steps 30 \
  --guidance-scale 6 \
  --scheduler cosine \
  --remasking low_confidence \
  --cfg-schedule linear_decay 
```

CLIP score.

```shell
python evaluation/clip_score.py \
  --images-dir /data/haoyuhuang/mmada_t2i_50_entropy/images \
  --pairs-jsonl /data/haoyuhuang/mmada_t2i_50_entropy/pairs.jsonl \
  --clip-model ViT-L-14 \
  --clip-pretrained openai \
  --batch-size 64 \
  --score-scale 100
```

Result:

CLIP scores on **COCO-VAL** summary table (CLIP=`ViT-L-14/openai`, score_scale=`100`, device=`cuda`, num_pairs=`50`):

| Remasking | CFG schedule | CLIP score |
|---|---|---|
| low_confidence | fixed (static) | 24.99 |
| entropy (high entropy) | fixed (static) | 24.92 |
| margin (low margin) | fixed (static) | **25.30** |
| low_confidence | linear_decay | 24.60 |
