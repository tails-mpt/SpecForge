#!/usr/bin/env python3
"""A8 prompt corpus builder — Eagle3-paper-scale prompt mix for MiniMax-M2.5 regen.

Builds a ≥100K diverse prompt corpus for A8 regen at --temperature 0.
At T=0 the model is deterministic so per-prompt diversity = 0; effective
training-set size equals the prompt count. This script targets ≥100K
deduplicated first-user-turn prompts sampled in the Eagle3 paper's ratio
(ShareGPT 12.8%, UltraChat 87.2% of the combined ShareGPT+UltraChat pool,
per the paper's 68K/532K ≈ 12.8/87.2 split), plus PerfectBlend as a
coding/tool-use coverage booster.

Final mix target: 120K prompts (can be reduced with --total-samples).
  ShareGPT      :  15K  (12.5%)
  UltraChat     : 100K  (83.3%)
  PerfectBlend  :   5K  ( 4.2%)

Why these sources:
  - ShareGPT: real-user multi-turn chat; strongest signal for reasoning/creativity
  - UltraChat: instruction-following at scale; fills long-tail coverage
  - PerfectBlend: code + tool use; critical for HumanEval/Terminal-Bench perf

Usage:
    python scripts/prepare_a8_prompts.py [--total-samples N] [--cache-dir DIR] [--output PATH]

Requires (produced by prepare_data.py):
    <cache-dir>/sharegpt_train.jsonl
    <cache-dir>/ultrachat_train.jsonl
    <cache-dir>/perfectblend_train.jsonl

Deduplication: SHA-256 of first user turn, deduplicated across ALL prior
mixed_train and regen corpora so none of the 120K prompts overlap with A1.1/A1.2.

Output: <output>.jsonl  (same schema as prepare_squeeze_a11_prompts.py)
        {"id": "a8-<source>-<i>", "conversations": [{"role": "user", "content": "..."}]}

CLAUDE.md rule #34: T=0 regen needs ≥100K prompts. Default here is 120K.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def first_user_turn(record: dict) -> str | None:
    for turn in record.get("conversations", []):
        role = turn.get("role", "")
        if role in ("user", "human"):
            return turn.get("content") or turn.get("value") or None
    return None


def load_existing_hashes(*paths: Path) -> set[str]:
    """Load SHA-256 hashes of first user turns from existing corpora for dedup."""
    seen: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"  [dedup] skip {p} (not found)")
            continue
        n = 0
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = first_user_turn(rec)
            if t:
                seen.add(hash_prompt(t))
                n += 1
        print(f"  [dedup] loaded {n:,} hashes from {p.name}")
    return seen


def sample_source(
    records: list[dict],
    target_n: int,
    used: set[str],
    source_name: str,
    prefix: str,
) -> list[dict]:
    eligible = []
    for r in records:
        t = first_user_turn(r)
        if t and hash_prompt(t) not in used:
            eligible.append(t)

    if len(eligible) < target_n:
        print(
            f"  WARNING: {source_name} has only {len(eligible):,} eligible prompts; "
            f"wanted {target_n:,}. Using all eligible."
        )
        target_n = len(eligible)

    picked_texts = random.sample(eligible, target_n)
    # mark as used so later sources don't overlap
    for t in picked_texts:
        used.add(hash_prompt(t))

    out = []
    for i, t in enumerate(picked_texts):
        out.append({
            "id": f"{prefix}-{source_name}-{i}",
            "conversations": [{"role": "user", "content": t}],
        })
    print(f"  {source_name}: sampled {len(out):,} from {len(eligible):,} eligible")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/mnt/persistent/crucible/specforge/cache/dataset"),
                    help="Directory containing <source>_train.jsonl files")
    ap.add_argument("--output", type=Path,
                    default=Path("/mnt/persistent/datasets/a8-prompts/a8_prompts.jsonl"),
                    help="Output path for the prompt JSONL")
    ap.add_argument("--total-samples", type=int, default=120_000,
                    help="Total prompt count (default: 120000 per CLAUDE.md rule #34)")
    ap.add_argument("--ratios", type=str, default="0.125,0.833,0.042",
                    help="Comma-separated ratios for sharegpt,ultrachat,perfectblend")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup-against", type=str, nargs="*",
                    default=[
                        "/mnt/persistent/datasets/minimax-regen-v2/mixed_train.jsonl",
                        "/mnt/persistent/datasets/minimax-regen-v2/mixed_train_regen_v2.jsonl",
                        "/mnt/persistent/datasets/minimax-regen-v2-b32/mixed_train_regen_v2_b32.jsonl",
                    ],
                    help="Existing corpora to dedup against (first-user-turn SHA-256)")
    args = ap.parse_args()

    random.seed(args.seed)

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 3, "Need exactly 3 ratios: sharegpt,ultrachat,perfectblend"
    total = sum(ratios)
    ratios = [r / total for r in ratios]  # normalize

    targets = {
        "sharegpt":    int(args.total_samples * ratios[0]),
        "ultrachat":   int(args.total_samples * ratios[1]),
        "perfectblend": args.total_samples - int(args.total_samples * ratios[0]) - int(args.total_samples * ratios[1]),
    }

    print(f"A8 prompt corpus builder")
    print(f"  Total target : {args.total_samples:,}")
    print(f"  ShareGPT     : {targets['sharegpt']:,} ({ratios[0]*100:.1f}%)")
    print(f"  UltraChat    : {targets['ultrachat']:,} ({ratios[1]*100:.1f}%)")
    print(f"  PerfectBlend : {targets['perfectblend']:,} ({ratios[2]*100:.1f}%)")
    print(f"  Cache dir    : {args.cache_dir}")
    print(f"  Output       : {args.output}")
    print(f"  Seed         : {args.seed}")
    print()

    # Load dedup hashes from all prior corpora
    print("Loading dedup hashes from prior corpora...")
    used: set[str] = load_existing_hashes(*[Path(p) for p in args.dedup_against])
    print(f"  Total prior hashes: {len(used):,}")
    print()

    # Load source datasets
    sources = {}
    for name in ("sharegpt", "ultrachat", "perfectblend"):
        path = args.cache_dir / f"{name}_train.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/prepare_data.py --dataset {name}"
            )
        print(f"Loading {name} from {path}...")
        sources[name] = load_jsonl(path)
        print(f"  {len(sources[name]):,} records")

    print()
    print("Sampling...")
    all_prompts: list[dict] = []
    for name in ("sharegpt", "ultrachat", "perfectblend"):
        sampled = sample_source(sources[name], targets[name], used, name, "a8")
        all_prompts.extend(sampled)

    random.shuffle(all_prompts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in all_prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(all_prompts):,} prompts → {args.output}")
    print()

    # Validation summary
    print("Validation:")
    print(f"  Total written     : {len(all_prompts):,}")
    print(f"  Target was        : {args.total_samples:,}")
    shortfall = args.total_samples - len(all_prompts)
    if shortfall > 0:
        print(f"  ⚠  Shortfall: {shortfall:,} prompts. Source pools may be exhausted.")
        print(f"     Consider adding more source datasets or reducing --total-samples.")
    else:
        print(f"  ✓  Target met. Ready for A8 regen at --temperature 0.")
    print()
    print("Next step:")
    print(f"  python scripts/regenerate_train_data.py \\")
    print(f"    --model /mnt/persistent/models/MiniMax-M2.5 \\")
    print(f"    --server-address localhost:30000 \\")
    print(f"    --temperature 0 \\")
    print(f"    --max-tokens 4096 \\")
    print(f"    --concurrency 64 \\")
    print(f"    --input-file-path {args.output} \\")
    print(f"    --output-file-path /mnt/persistent/datasets/a8-regen/a8_regen.jsonl \\")
    print(f"    --resume \\")
    print(f"    --regen-validation-strict")


if __name__ == "__main__":
    main()
