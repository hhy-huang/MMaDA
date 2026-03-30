import argparse
import ast
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample prompts from VLMEvalKit COCO_VAL.tsv and write prompts_*.jsonl."
    )
    p.add_argument(
        "--coco-tsv",
        type=str,
        default=None,
        help="Path to COCO_VAL.tsv. If omitted, will use $LMUData/COCO_VAL.tsv or ~/LMUData/COCO_VAL.tsv.",
    )
    p.add_argument("--out-jsonl", type=str, required=True, help="Output prompts JSONL path.")
    p.add_argument("--num-prompts", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=10)
    p.add_argument("--max-chars", type=int, default=200)
    p.add_argument("--dedup", action="store_true", default=True)
    return p.parse_args()


def resolve_default_coco_path() -> Path:
    import os

    if "LMUData" in os.environ and os.path.exists(os.environ["LMUData"]):
        return Path(os.environ["LMUData"]) / "COCO_VAL.tsv"
    return Path.home() / "LMUData" / "COCO_VAL.tsv"


def _tsv_rows(tsv_path: Path) -> Iterable[dict]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    for _, row in df.iterrows():
        yield {k: row[k] for k in df.columns}


def _pick_caption(row: dict) -> Optional[str]:
    # Prefer explicit caption fields if present.
    for key in ["caption", "captions", "text", "prompt"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()

    # COCO_VAL in VLMEvalKit usually stores GT captions in `answer` as a python list string.
    if "answer" in row and isinstance(row["answer"], str) and row["answer"].strip():
        s = row["answer"].strip()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and obj:
                # pick the first non-empty caption
                for x in obj:
                    if isinstance(x, str) and x.strip():
                        return x.strip()
        except Exception:
            pass

    # Fallback: some TSVs may include a question field; it's not ideal for T2I but better than nothing.
    if "question" in row and isinstance(row["question"], str) and row["question"].strip():
        return row["question"].strip()

    return None


def main():
    args = parse_args()
    coco_path = Path(args.coco_tsv) if args.coco_tsv else resolve_default_coco_path()
    if not coco_path.exists():
        raise FileNotFoundError(
            f"COCO TSV not found at {coco_path}. "
            "Please download via VLMEvalKit first or pass --coco-tsv explicitly."
        )

    prompts: List[str] = []
    for row in _tsv_rows(coco_path):
        cap = _pick_caption(row)
        if not cap:
            continue
        cap = " ".join(cap.split())
        if len(cap) < args.min_chars or len(cap) > args.max_chars:
            continue
        prompts.append(cap)

    if args.dedup:
        # Preserve order while deduping
        seen = set()
        deduped = []
        for x in prompts:
            if x in seen:
                continue
            seen.add(x)
            deduped.append(x)
        prompts = deduped

    if len(prompts) < args.num_prompts:
        raise ValueError(
            f"Not enough prompts after filtering: {len(prompts)} < {args.num_prompts}. "
            "Try relaxing min/max chars or disabling dedup."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(prompts, k=args.num_prompts)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(sampled):
            f.write(json.dumps({"id": i, "prompt": prompt}, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "coco_tsv": str(coco_path),
                "total_candidates": len(prompts),
                "num_prompts": len(sampled),
                "seed": args.seed,
                "out_jsonl": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

