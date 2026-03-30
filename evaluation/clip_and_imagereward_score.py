import argparse
import json
import gc
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute CLIPScore-style cosine and ImageReward for (image, prompt) pairs."
    )
    p.add_argument("--pairs-jsonl", type=str, required=True, help="JSONL with fields: prompt, image.")
    p.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L-14",
        help="open_clip model name (e.g. ViT-L-14, ViT-B-32).",
    )
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default="openai",
        help="open_clip pretrained tag (e.g. openai).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument(
        "--clip-device",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"],
        help="Device used for CLIP stage. Defaults to --device.",
    )
    p.add_argument(
        "--imagereward-device",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"],
        help="Device used for ImageReward stage. Defaults to --device.",
    )
    p.add_argument(
        "--score-scale",
        type=float,
        default=100.0,
        help="Multiply mean cosine by this value (for CLIPScore-style reporting).",
    )
    return p.parse_args()


def load_pairs(path: Path) -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            prompt = (j.get("prompt") or j.get("caption") or "").strip()
            image = j.get("image") or j.get("image_path") or j.get("filepath")
            if not prompt or not image:
                raise ValueError("pairs-jsonl requires fields: prompt and image (or image_path).")
            img_path = Path(image)
            if not img_path.is_absolute():
                img_path = img_path.resolve()
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))
            pairs.append((img_path, prompt))
    return pairs


def build_open_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def encode_clip_batch(
    model,
    preprocess,
    tokenizer,
    images: List[Path],
    prompts: List[str],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = []
    for p in images:
        im = Image.open(p).convert("RGB")
        imgs.append(preprocess(im))
    image_tensor = torch.stack(imgs, dim=0).to(device)
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_feat = model.encode_image(image_tensor)
        text_feat = model.encode_text(text_tokens)
    return image_feat, text_feat


def main():
    args = parse_args()
    default_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_device = args.clip_device or default_device
    ir_device = args.imagereward_device or default_device

    pairs = load_pairs(Path(args.pairs_jsonl))
    print(f"Loaded {len(pairs)} pairs.")

    # -----------------------
    # Stage 1: CLIP (only)
    # -----------------------
    print(f"[Stage 1/2] CLIP on {clip_device}")
    clip_model, clip_preprocess, clip_tokenizer = build_open_clip(
        args.clip_model, args.clip_pretrained, clip_device
    )

    cos_sims: List[float] = []
    bs = int(args.batch_size)

    for i in range(0, len(pairs), bs):
        batch = pairs[i : i + bs]
        image_paths = [p[0] for p in batch]
        prompts = [p[1] for p in batch]

        img_feat, txt_feat = encode_clip_batch(
            clip_model, clip_preprocess, clip_tokenizer, image_paths, prompts, clip_device
        )
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cosine = (img_feat * txt_feat).sum(dim=-1)
        cos_sims.extend(cosine.detach().float().cpu().tolist())

        if (i // bs) % 10 == 0:
            print(f"[CLIP] Processed {min(i+bs, len(pairs))}/{len(pairs)}")

    mean_cos = sum(cos_sims) / max(1, len(cos_sims))

    # Release CLIP GPU memory before loading ImageReward.
    del clip_model
    gc.collect()
    if clip_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------
    # Stage 2: ImageReward (only)
    # -----------------------------
    print(f"[Stage 2/2] ImageReward on {ir_device}")
    try:
        import ImageReward as RM
        ir_model = RM.load("ImageReward-v1.0", device=ir_device)
        score_fn = lambda prompts, image_paths: ir_model.score(prompts, image_paths)
    except ImportError:
        # Fallback for alternative package layouts.
        from imagereward import ImageReward as ImageRewardScorer
        ir_model = ImageRewardScorer(device=ir_device)
        score_fn = lambda prompts, image_paths: ir_model.score(prompts, image_paths)

    ir_scores: List[float] = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i : i + bs]
        image_paths = [p[0] for p in batch]
        prompts = [p[1] for p in batch]
        ir_batch_scores = score_fn(prompts, [str(p) for p in image_paths])
        ir_scores.extend([float(s) for s in ir_batch_scores])
        if (i // bs) % 10 == 0:
            print(f"[ImageReward] Processed {min(i+bs, len(pairs))}/{len(pairs)}")

    mean_ir = sum(ir_scores) / max(1, len(ir_scores))

    print(
        json.dumps(
            {
                "num_pairs": len(pairs),
                "mean_cosine": mean_cos,
                "clip_score": mean_cos * float(args.score_scale),
                "image_reward": mean_ir,
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "clip_device": clip_device,
                "imagereward_device": ir_device,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

