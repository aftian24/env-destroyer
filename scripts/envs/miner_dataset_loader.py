"""
Loader for validator-provided MINER_DATASETS into the HF DatasetDict shape
that ``train_sft_env.py`` expects (i.e., ``{"messages": [...]}`` per row).

Wires the feature/miner-dataset-whitelist contract:
- Validator mounts whitelisted datasets at ``MINER_DATASETS_DIR``.
- The list of directory names is in ``MINER_DATASETS`` (comma-separated).
- Each subdirectory is a ``datasets`` package (load_from_disk-compatible) OR
  raw JSONL/Parquet files.

Schema normalisation: every supported dataset has its own column names; we
best-effort convert to ``{"messages": [{"role": ..., "content": ...}, ...]}``.

The loader is OPTIONAL — call it explicitly when you want to incorporate miner
datasets into SFT. The default sft_env_config flow (generate_trajectories.py)
still works when no miner datasets are present.

Usage::

    from envs.miner_dataset_loader import build_miner_sft_dataset

    dd = build_miner_sft_dataset(env_name="liars_dice")  # DatasetDict or None
    if dd is not None:
        dd.save_to_disk(out_path)

Whitelist reference: ``core/whitelisted_sft_datasets.json`` upstream main.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_from_disk


# ---------------------------------------------------------------------------
# Schema normalisers per whitelisted dataset
# ---------------------------------------------------------------------------

def _row_matches_game(row: dict[str, Any], env_name: str) -> bool:
    """Return True if row's 'game' field matches env_name (or no filter applied).

    Used to filter ``gradients-io-tournaments/env_training_gradients`` which
    contains multi-game traces (gin_rummy + liars_dice + leduc_poker + others).
    """
    if not env_name:
        return True
    game = row.get("game") or row.get("env") or row.get("environment_name")
    if not isinstance(game, str):
        return True  # row has no game field — keep it (other datasets)
    return game.lower().strip() == env_name.lower().strip()


def _row_to_messages_generic(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Try several common schemas; return list-of-messages or None."""
    # Schema 1: already "messages" formatted
    if isinstance(row.get("messages"), list):
        msgs = [m for m in row["messages"]
                if isinstance(m, dict) and "role" in m and "content" in m]
        return msgs or None
    # Schema 2: "conversations" / "dialog" with role/content or from/value
    for key in ("conversations", "dialog", "chat", "turns"):
        seq = row.get(key)
        if isinstance(seq, list) and seq:
            converted: list[dict[str, str]] = []
            for m in seq:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from") or m.get("speaker")
                content = m.get("content") or m.get("value") or m.get("text")
                if role and content is not None:
                    role = "assistant" if role.lower() in ("gpt", "model", "ai", "assistant") else (
                        "user" if role.lower() in ("human", "user") else (
                        "system" if role.lower() == "system" else role.lower()
                    ))
                    converted.append({"role": role, "content": str(content)})
            if converted:
                return converted
    # Schema 3: instruction / output single-turn
    instruction = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if instruction and output:
        msgs: list[dict[str, str]] = []
        system = row.get("system") or row.get("system_prompt")
        if system:
            msgs.append({"role": "system", "content": str(system)})
        msgs.append({"role": "user", "content": str(instruction)})
        msgs.append({"role": "assistant", "content": str(output)})
        return msgs
    # Schema 4: text + label
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return [{"role": "user", "content": text}, {"role": "assistant", "content": ""}]
    return None


def _normalize_rows(
    rows: Iterable[dict[str, Any]],
    env_name: str = "",
) -> list[dict[str, Any]]:
    """Apply ``_row_to_messages_generic`` to each row; drop failures + game-filter.

    Args:
        rows: iterable of raw rows.
        env_name: if non-empty, only rows whose 'game' field matches are kept
            (rows without a 'game' field are NOT filtered out).
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _row_matches_game(row, env_name):
            continue
        msgs = _row_to_messages_generic(row)
        if msgs:
            out.append({"messages": msgs})
    return out


# ---------------------------------------------------------------------------
# Loaders for different on-disk layouts
# ---------------------------------------------------------------------------

def _load_one_dataset(path: Path) -> Dataset | None:
    """Best-effort load: HF save_to_disk → parquet → jsonl → json."""
    # 1. HF datasets save_to_disk layout (has dataset_info.json)
    try:
        if (path / "dataset_info.json").exists() or any(path.glob("*.arrow")):
            ds = load_from_disk(str(path))
            if isinstance(ds, DatasetDict):
                # flatten train+validation+test into one
                parts = []
                for split in ds:
                    parts.extend(ds[split])
                return Dataset.from_list(list(parts))
            return ds
    except Exception:
        pass
    # 2. Parquet files
    try:
        parquets = list(path.rglob("*.parquet"))
        if parquets:
            from datasets import load_dataset
            return load_dataset("parquet", data_files=[str(p) for p in parquets], split="train")
    except Exception:
        pass
    # 3. JSONL files
    try:
        jsonls = list(path.rglob("*.jsonl")) + list(path.rglob("*.json"))
        rows: list[dict[str, Any]] = []
        for jl in jsonls:
            with open(jl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Try parsing the whole file as a JSON array
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list):
                            rows.extend(d for d in data if isinstance(d, dict))
                        break
        if rows:
            return Dataset.from_list(rows)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_miner_datasets_inventory() -> list[tuple[str, Path]]:
    """Mirror of ``tournament_env_utils.get_miner_datasets`` to avoid cross-import.

    Returns ``[(hf_repo_name, local_dir), ...]``. Empty when env vars are unset.
    """
    parent_raw = os.environ.get("MINER_DATASETS_DIR")
    if not parent_raw:
        return []
    parent = Path(parent_raw)
    if not parent.exists():
        return []
    names = os.environ.get("MINER_DATASETS", "")
    if not names.strip():
        return []
    out: list[tuple[str, Path]] = []
    for entry in names.split(","):
        entry = entry.strip()
        if not entry:
            continue
        local = parent / entry
        if local.exists():
            out.append((entry.replace("--", "/", 1), local))
    return out


def build_miner_sft_dataset(
    env_name: str = "",
    *,
    validation_ratio: float = 0.01,
    seed: int = 42,
    cap_rows: int | None = None,
) -> DatasetDict | None:
    """Build a single DatasetDict from validator-mounted miner datasets, or None.

    Args:
        env_name: optional environment name (for logging only — datasets are
            requested by miner ahead of time, not per-env).
        validation_ratio: fraction held out for the ``validation`` split.
        seed: split seed (reproducibility).
        cap_rows: optional cap on total rows (post-merge, pre-split).

    Returns:
        DatasetDict({"train": ..., "validation": ...}) or None if no datasets.
    """
    inventory = _get_miner_datasets_inventory()
    if not inventory:
        return None

    all_rows: list[dict[str, Any]] = []
    print(f"[MINER_DATASETS] env_name={env_name!r} loading {len(inventory)} dataset(s)", flush=True)
    for hf_name, local in inventory:
        ds = _load_one_dataset(local)
        if ds is None:
            print(f"[MINER_DATASETS] skipped {hf_name}: cannot load from {local}", flush=True)
            continue
        normalised = _normalize_rows(ds, env_name=env_name)
        if not normalised:
            print(f"[MINER_DATASETS] skipped {hf_name}: no rows match expected schemas/game", flush=True)
            continue
        all_rows.extend(normalised)
        print(
            f"[MINER_DATASETS] loaded {hf_name}: {len(normalised)} rows (raw {len(ds)})"
            + (f" filtered to game={env_name!r}" if env_name else ""),
            flush=True,
        )

    if not all_rows:
        print("[MINER_DATASETS] no usable rows after normalisation", flush=True)
        return None

    if cap_rows is not None and len(all_rows) > cap_rows:
        # deterministic prefix cap (validator data is already curated)
        all_rows = all_rows[:cap_rows]

    base = Dataset.from_list(all_rows)
    splits = base.train_test_split(test_size=validation_ratio, seed=seed)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def merge_with_synthetic(
    miner_dd: DatasetDict | None,
    synthetic_path: str | None,
) -> DatasetDict | None:
    """Optionally merge miner-provided data with synthetic trajectories.

    If both are present, concatenate train+train and validation+validation.
    If only one is present, return it. If neither, return None.

    Args:
        miner_dd: from ``build_miner_sft_dataset()``.
        synthetic_path: path to a ``generate_trajectories.py`` output dir.
    """
    synth_dd: DatasetDict | None = None
    if synthetic_path and Path(synthetic_path).exists():
        try:
            loaded = load_from_disk(synthetic_path)
            if isinstance(loaded, DatasetDict):
                synth_dd = loaded
        except Exception:
            synth_dd = None

    if miner_dd is None and synth_dd is None:
        return None
    if miner_dd is None:
        return synth_dd
    if synth_dd is None:
        return miner_dd

    from datasets import concatenate_datasets
    return DatasetDict({
        "train":      concatenate_datasets([miner_dd["train"],      synth_dd["train"]]),
        "validation": concatenate_datasets([miner_dd["validation"], synth_dd["validation"]]),
    })


if __name__ == "__main__":
    # Smoke test: print stats of available miner datasets.
    inv = _get_miner_datasets_inventory()
    print(f"miner datasets inventory ({len(inv)}):")
    for name, path in inv:
        print(f"  - {name}  ({path})")
    dd = build_miner_sft_dataset(env_name=os.environ.get("ENV_NAME", ""))
    if dd is None:
        print("no miner datasets present; falling back to synthetic generation")
    else:
        print(f"built DatasetDict: train={len(dd['train'])} validation={len(dd['validation'])}")
