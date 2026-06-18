"""Merge multiple HF DatasetDicts into a single DatasetDict.

Used for multi-env SFT training (PvP tournament 2026-05-25+ where each task
spans 2-6 envs per `dataset_type.environment_names: list`). Per-env datasets
are generated separately via `generate_trajectories.py`, then merged here
into one combined dataset that the SFT trainer consumes.

Discord 2026-05-23: "all the envs are passed in, and all will be evaluated
PvP. You can't optimise for a single env training since you'll be pipped on
all the others." → SFT data must cover ALL envs in the task payload.

Each input DatasetDict expected to have schema {messages: list[dict]} per row.
Splits (train, validation) are concatenated across all inputs.

After merge, train split is shuffled (seed=42) so env examples interleave —
this prevents the trainer from seeing all-of-env-A then all-of-env-B in
sequence, which would cause catastrophic forgetting on env A during env B
training.

Usage:
  python -m envs.merge_datasets \
    --inputs /workspace/scripts/datasets/sft_env_<task>_part0_liars_dice \
             /workspace/scripts/datasets/sft_env_<task>_part1_gin_rummy \
    --output /workspace/scripts/datasets/sft_env_<task>
"""

import argparse
import sys
from collections import Counter

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to HF DatasetDicts to merge (will be concatenated per split).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination path for the merged DatasetDict (save_to_disk).",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling the merged train split. Set to 0 to disable shuffle.",
    )
    args = p.parse_args()

    print(f"[merge_datasets] Merging {len(args.inputs)} datasets → {args.output}",
          flush=True)

    splits_data: dict[str, list] = {}
    per_input_stats = []
    for inp_path in args.inputs:
        try:
            dd = load_from_disk(inp_path)
        except Exception as exc:
            print(f"[merge_datasets] WARN failed to load {inp_path}: {exc}",
                  flush=True)
            per_input_stats.append((inp_path, "FAILED", 0))
            continue

        train_rows = len(dd.get("train", []))
        for split_name in dd.keys():
            splits_data.setdefault(split_name, []).append(dd[split_name])
        per_input_stats.append((inp_path, "OK", train_rows))
        print(f"  loaded {inp_path}: splits={list(dd.keys())} "
              f"train_rows={train_rows}", flush=True)

    if not splits_data:
        print("[merge_datasets] FATAL: no datasets loaded successfully.",
              file=sys.stderr, flush=True)
        sys.exit(1)

    merged_splits = {
        split: concatenate_datasets(datasets)
        for split, datasets in splits_data.items()
    }

    # Shuffle train so env examples interleave. Without shuffle the trainer sees
    # all liars examples → then all gin examples → catastrophic forgetting on
    # liars during gin batches.
    if args.shuffle_seed and "train" in merged_splits:
        merged_splits["train"] = merged_splits["train"].shuffle(seed=args.shuffle_seed)
        print(f"[merge_datasets] Shuffled train split (seed={args.shuffle_seed})",
              flush=True)

    merged = DatasetDict(merged_splits)

    print(f"[merge_datasets] Final splits:", flush=True)
    for split, ds in merged.items():
        print(f"  {split}: {len(ds)} rows", flush=True)

    merged.save_to_disk(args.output)
    print(f"[merge_datasets] Saved to {args.output}", flush=True)

    # Final summary table
    print(f"\n[merge_datasets] Input summary:")
    print(f"  {'path':<60s}  {'status':<6s}  {'rows':>8s}")
    for path, status, rows in per_input_stats:
        path_short = path[-58:] if len(path) > 58 else path
        print(f"  {path_short:<60s}  {status:<6s}  {rows:>8d}")


if __name__ == "__main__":
    main()
