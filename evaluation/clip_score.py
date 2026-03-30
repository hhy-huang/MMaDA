import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from PIL import Image


@dataclass
class Pair:
    image_path: Path
    prompt: str


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute CLIPScore-style text-image similarity (mean cosine) for a set of (image, prompt) pairs."
    )
    p.add_argument("--images-dir", type=str, default=None, help="Folder containing images.")
    p.add_argument(
        "--pairs-jsonl",
        type=str,
        default=None,
        help="JSONL file with items containing at least {prompt, image}. image can be a relative path under --images-dir.",
    )
    p.add_argument(
        "--pairs-tsv",
        type=str,
        default=None,
        help="TSV file with header containing at least columns: prompt, image (or image_path).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.png",
        help="When no pairs file is provided, glob images under --images-dir and use --prompt for all.",
    )
    p.add_argument("--prompt", type=str, default=None, help="A single prompt used for all images (glob mode).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
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
    p.add_argument(
        "--score-scale",
        type=float,
        default=100.0,
        help="Multiply mean cosine by this value (CLIPScore papers often report *100).",
    )
    p.add_argument(
        "--clamp-min",
        type=float,
        default=None,
        help="Optional clamp for cosine similarity lower bound (e.g. 0.0).",
    )
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_pairs(args) -> List[Pair]:
    images_dir = Path(args.images_dir) if args.images_dir else None

    if args.pairs_jsonl:
        pairs = []
        for item in _iter_jsonl(Path(args.pairs_jsonl)):
            prompt = (item.get("prompt") or item.get("caption") or "").strip()
            image = item.get("image") or item.get("image_path") or item.get("filepath")
            if not prompt or not image:
                raise ValueError("pairs_jsonl requires fields: prompt and image (or image_path).")
            image_path = Path(image)
            if images_dir and not image_path.is_absolute():
                image_path = images_dir / image_path
            pairs.append(Pair(image_path=image_path, prompt=prompt))
        return pairs

    if args.pairs_tsv:
        pairs = []
        with open(args.pairs_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                prompt = (row.get("prompt") or row.get("caption") or "").strip()
                image = row.get("image") or row.get("image_path") or row.get("filepath")
                if not prompt or not image:
                    raise ValueError("pairs_tsv requires columns: prompt and image (or image_path).")
                image_path = Path(image)
                if images_dir and not image_path.is_absolute():
                    image_path = images_dir / image_path
                pairs.append(Pair(image_path=image_path, prompt=prompt))
        return pairs

    if not images_dir:
        raise ValueError("Provide --images-dir with --pairs-jsonl/--pairs-tsv or for glob mode.")
    if not args.prompt:
        raise ValueError("Glob mode requires --prompt.")

    pairs = []
    for img in sorted(images_dir.glob(args.glob)):
        pairs.append(Pair(image_path=img, prompt=args.prompt))
    return pairs


def _try_load_open_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _encode_batch_open_clip(
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
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(args)
    for p in pairs:
        if not p.image_path.exists():
            raise FileNotFoundError(str(p.image_path))

    # Use open_clip by default (matches GenEval's evaluate_images.py usage).
    model, preprocess, tokenizer = _try_load_open_clip(args.clip_model, args.clip_pretrained, device)

    sims: List[float] = []
    bs = int(args.batch_size)

    for i in range(0, len(pairs), bs):
        batch = pairs[i : i + bs]
        image_paths = [x.image_path for x in batch]
        prompts = [x.prompt for x in batch]
        img_feat, txt_feat = _encode_batch_open_clip(
            model, preprocess, tokenizer, image_paths, prompts, device
        )
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cosine = (img_feat * txt_feat).sum(dim=-1)

        if args.clamp_min is not None:
            cosine = torch.clamp(cosine, min=float(args.clamp_min))

        sims.extend(cosine.detach().float().cpu().tolist())
        if (i // bs) % 10 == 0:
            print(f"Processed {min(i+bs, len(pairs))}/{len(pairs)}")

    mean_cos = sum(sims) / max(1, len(sims))
    print(
        json.dumps(
            {
                "num_pairs": len(sims),
                "mean_cosine": mean_cos,
                "score_scale": args.score_scale,
                "clip_score": mean_cos * float(args.score_scale),
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "device": device,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

