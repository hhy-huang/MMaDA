import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser(description="Compute mean WiScore for a WISE JSONL score file (supports subsets).")
    p.add_argument("--scores-jsonl", type=str, required=True, help="WISE *_scores_results.jsonl")
    p.add_argument("--max-id", type=int, default=-1, help="Only include prompt_id <= max-id (default: all)")
    p.add_argument("--min-id", type=int, default=-1, help="Only include prompt_id >= min-id (default: all)")
    return p.parse_args()


def wiscore(c: float, r: float, a: float) -> float:
    return (0.7 * c + 0.2 * r + 0.1 * a) / 2.0


def main():
    args = parse_args()
    path = Path(args.scores_jsonl)
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("prompt_id")
            if isinstance(pid, int):
                if args.max_id > 0 and pid > args.max_id:
                    continue
                if args.min_id > 0 and pid < args.min_id:
                    continue
            c = obj.get("consistency")
            r = obj.get("realism")
            a = obj.get("aesthetic_quality")
            if not all(isinstance(x, (int, float)) for x in (c, r, a)):
                continue
            rows.append((pid, float(c), float(r), float(a)))

    if not rows:
        raise SystemExit("No valid rows found. Check file path and filters.")

    wis = [wiscore(c, r, a) for _, c, r, a in rows]
    pids = [pid for pid, *_ in rows if isinstance(pid, int)]
    pid_min = min(pids) if pids else None
    pid_max = max(pids) if pids else None

    out = {
        "num_samples": len(wis),
        "mean_wiscore": mean(wis),
        "prompt_id_min": pid_min,
        "prompt_id_max": pid_max,
        "filters": {"min_id": args.min_id, "max_id": args.max_id},
        "scores_jsonl": str(path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

