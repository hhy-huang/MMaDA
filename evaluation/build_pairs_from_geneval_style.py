import json
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert GenEval-style folders (<idx>/metadata.jsonl + samples/*.png) to pairs.jsonl."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing <idx>/metadata.jsonl and samples/*.png (e.g. GenEval-style output).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output JSONL path; each line has {'prompt','image'}.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            meta_path = folder / "metadata.jsonl"
            if not meta_path.exists():
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            prompt = metadata.get("prompt", "").strip()
            if not prompt:
                continue
            samples_dir = folder / "samples"
            if not samples_dir.exists():
                continue
            for img_path in sorted(samples_dir.glob("*.png")):
                rec = {
                    "prompt": prompt,
                    "image": str(img_path.resolve()),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} pairs to {out_path}")


if __name__ == "__main__":
    main()

