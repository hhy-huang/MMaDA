import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import models` works when running from evaluation/.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from training.prompting_utils import UniversalPrompting


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GenEval-format images with MMaDA.")
    parser.add_argument("--prompts-jsonl", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Gen-Verse/MMaDA-8B-MixCoT")
    parser.add_argument("--vq-model-path", type=str, default="showlab/magvitv2")
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "sigmoid", "linear"])
    parser.add_argument(
        "--cfg-schedule",
        type=str,
        default="static",
        choices=["static", "linear_decay", "cosine_decay"],
    )
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random", "entropy", "margin"],
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument(
        "--flat-output-dir",
        type=str,
        default=None,
        help="Optional directory to save flat indexed images (e.g., 1.png..N.png) for WISE eval",
    )
    parser.add_argument(
        "--flat-index-base",
        type=int,
        default=1,
        help="Start index for flat image names, default 1 => 1.png",
    )
    return parser.parse_args()


def load_prompts(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_runtime(model_path: str, vq_model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = MMadaModelLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    vq_model = MAGVITv2().from_pretrained(vq_model_path).to(device)

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=512,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True,
    )
    return tokenizer, model, vq_model, uni_prompting


@torch.no_grad()
def generate_one_image(
    prompt_text,
    model,
    vq_model,
    uni_prompting,
    mask_id,
    steps,
    guidance_scale,
    cfg_schedule,
    scheduler,
    remasking,
    device,
):
    image_tokens = torch.ones((1, 1024), dtype=torch.long, device=device) * mask_id
    input_ids, attention_mask = uni_prompting(([prompt_text], image_tokens), "t2i_gen")

    if guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([""], image_tokens), "t2i_gen")
    else:
        uncond_input_ids, uncond_attention_mask = None, None

    noise_schedule = get_mask_schedule(scheduler)
    final_img = None
    for image_step, _ in model.t2i_generate_decoding_stepwise(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        uncond_attention_mask=uncond_attention_mask,
        temperature=1.0,
        timesteps=steps,
        guidance_scale=guidance_scale,
        cfg_schedule=cfg_schedule,
        remasking=remasking,
        noise_schedule=noise_schedule,
        noise_type="mask",
        seq_len=1024,
        vq_model=vq_model,
        uni_prompting=uni_prompting,
    ):
        final_img = image_step

    if final_img is None:
        raise RuntimeError("No image generated.")
    return final_img


def main():
    args = parse_args()
    run_start = time.perf_counter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    prompts = load_prompts(Path(args.prompts_jsonl))
    end_index = len(prompts) - 1 if args.end_index < 0 else min(args.end_index, len(prompts) - 1)
    start_index = max(0, args.start_index)
    if start_index > end_index:
        raise ValueError(f"Invalid range: start={start_index}, end={end_index}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    flat_output_dir = Path(args.flat_output_dir) if args.flat_output_dir else None
    if flat_output_dir is not None:
        flat_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, vq_model, uni_prompting = build_runtime(args.model_path, args.vq_model_path, device)
    mask_id = tokenizer.mask_token_id if getattr(tokenizer, "mask_token_id", None) is not None else 126336

    total = end_index - start_index + 1
    print(f"Generating {total} prompts on {device}")
    print(f"Remasking strategy: {args.remasking}")
    print(f"CFG schedule: {args.cfg_schedule}")
    generated_images = 0

    for i, idx in enumerate(range(start_index, end_index + 1), 1):
        prompt_start = time.perf_counter()
        meta = prompts[idx]
        prompt = (meta.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Missing prompt at index {idx}")

        item_dir = outdir / str(idx)
        sample_dir = item_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with (item_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        for sample_id in range(args.num_images_per_prompt):
            seed = args.seed + idx * 100000 + sample_id
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            img = generate_one_image(
                prompt,
                model,
                vq_model,
                uni_prompting,
                mask_id,
                args.steps,
                args.guidance_scale,
                args.cfg_schedule,
                args.scheduler,
                args.remasking,
                device,
            )
            img.save(sample_dir / f"{sample_id}.png")
            if flat_output_dir is not None and sample_id == 0:
                flat_idx = args.flat_index_base + (idx - start_index)
                img.save(flat_output_dir / f"{flat_idx}.png")
            generated_images += 1

        prompt_elapsed = time.perf_counter() - prompt_start
        print(f"[{i}/{total}] idx={idx} | prompt_time={prompt_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - run_start
    avg_prompt_time = total_elapsed / total if total > 0 else 0.0
    avg_image_time = total_elapsed / generated_images if generated_images > 0 else 0.0
    print(f"Done: {outdir}")
    print(
        f"Timing summary | total={total_elapsed:.2f}s | "
        f"avg_per_prompt={avg_prompt_time:.2f}s | avg_per_image={avg_image_time:.2f}s"
    )


if __name__ == "__main__":
    main()

