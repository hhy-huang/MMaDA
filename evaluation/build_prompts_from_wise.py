import argparse
import json
from pathlib import Path


PROMPT_KEYS = ("prompt", "Prompt", "text", "question", "caption")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert WISE JSON prompts to jsonl for run_geneval_mmada.py")
    parser.add_argument("--wise-json", type=str, required=True, help="Path to WISE json file")
    parser.add_argument("--out-jsonl", type=str, required=True, help="Output jsonl path")
    parser.add_argument("--max-prompts", type=int, default=-1, help="Keep only first N prompts")
    return parser.parse_args()


def try_get_prompt(item):
    if isinstance(item, dict):
        # Direct key match first.
        for key in PROMPT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Fallback: case-insensitive key match.
        lowered = {str(k).lower(): v for k, v in item.items()}
        for key in ("prompt", "text", "question", "caption"):
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def extract_prompts(obj):
    prompts = []
    if isinstance(obj, list):
        for item in obj:
            p = try_get_prompt(item)
            if p is not None:
                prompts.append(p)
    elif isinstance(obj, dict):
        # First try dict values if they are sample records.
        for value in obj.values():
            if isinstance(value, dict):
                p = try_get_prompt(value)
                if p is not None:
                    prompts.append(p)
            elif isinstance(value, list):
                for item in value:
                    p = try_get_prompt(item)
                    if p is not None:
                        prompts.append(p)
        # Fallback: current node itself might be a sample record.
        if not prompts:
            p = try_get_prompt(obj)
            if p is not None:
                prompts.append(p)
    return prompts


def main():
    args = parse_args()
    wise_json = Path(args.wise_json)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with wise_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = extract_prompts(data)
    if not prompts:
        raise RuntimeError(
            "No prompts found. Please inspect WISE json format and adapt keys in PROMPT_KEYS."
        )

    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            row = {"id": idx, "prompt": prompt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} prompts to {out_jsonl}")


if __name__ == "__main__":
    main()
