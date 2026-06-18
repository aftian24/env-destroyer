"""
Tournament environment helpers — graceful parsing of validator-injected env vars.

Handles upstream G.O.D changes:
1. feature/miner-dataset-whitelist (merged main): MINER_DATASETS_DIR + MINER_DATASETS
2. feature/model-prep-container (Discord: May 13 merge): BASELINE_STATS
3. Anonymous model repos: model.config._name_or_path may be scrubbed

All functions are defensive — return None / [] when env vars are unset so the
training pipeline keeps working pre-merge.

References:
- core/whitelisted_sft_datasets.json (gradients-ai/G.O.D main)
- core/models/model_prep_models.py (gradients-ai/G.O.D feature/model-prep-container)
- docs/tourn_miner.md "Miner-Requested Datasets" section
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# MINER_DATASETS handling
# ---------------------------------------------------------------------------

# Validator-imposed cap (core/whitelisted_sft_datasets.py:MAX_REQUESTED_DATASETS)
MAX_REQUESTED_DATASETS = 2

# Whitelisted dataset names — those allowed per
# core/whitelisted_sft_datasets.json (gradients-ai/G.O.D main, 2026-05-01).
# Used for sanity logging only; the validator enforces the real whitelist.
WHITELISTED_SFT_DATASETS = (
    "SoelMgd/Poker_Dataset",
    "RZ412/PokerBench",
    "albarji/ballmatro",
    "Mahesh111000/Hanabi_dataset",
    "hkust-nlp/agentboard",
    "tasksource/Boardgame-QA",
    "GoodStartLabs/gin-rummy-trajectories-32k",
    "gradients-io-tournaments/ArkadiumGinrummy",
    "the-acorn-ai/textarena-player-game-traces",
    "gradients-io-tournaments/env_training_gradients",
)


def get_miner_datasets_dir() -> Path | None:
    """Return the on-disk parent directory of miner-requested datasets, or None.

    Validator passes ``MINER_DATASETS_DIR=/cache/miner_datasets`` (read-only mount)
    when whitelisted datasets are requested in the miner's TrainingRepoResponse.
    """
    raw = os.environ.get("MINER_DATASETS_DIR")
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def get_miner_datasets() -> list[tuple[str, Path]]:
    """Return list of ``(hf_repo_name, local_path)`` for available miner datasets.

    Validator passes ``MINER_DATASETS`` as a comma-separated list of directory
    names with ``--`` replacing ``/`` (e.g., ``SoelMgd--Poker_Dataset,RZ412--PokerBench``).
    Each entry is a subdirectory of ``MINER_DATASETS_DIR``.

    Returns empty list when env vars are unset OR the directories are missing.
    """
    parent = get_miner_datasets_dir()
    if parent is None:
        return None or []
    raw_list = os.environ.get("MINER_DATASETS", "").strip()
    if not raw_list:
        return []
    out: list[tuple[str, Path]] = []
    for entry in raw_list.split(","):
        entry = entry.strip()
        if not entry:
            continue
        local = parent / entry
        if not local.exists():
            continue
        hf_name = entry.replace("--", "/", 1)  # only first -- (org/name) reversed
        out.append((hf_name, local))
    return out


# ---------------------------------------------------------------------------
# BASELINE_STATS handling
# ---------------------------------------------------------------------------

@dataclass
class EnvStats:
    """Per-environment baseline score statistics (one game)."""
    num_episodes: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    median_score: float = 0.0


@dataclass
class WeightGroupStats:
    """Per-layer-group weight statistics from the base/augmented model."""
    weight_rms: float = 0.0
    weight_norm: float = 0.0
    max_abs: float = 0.0


@dataclass
class WeightStats:
    """Aggregated weight statistics."""
    by_group: dict[str, WeightGroupStats] = field(default_factory=dict)


@dataclass
class EnvBaselineStats:
    """``BaselineStats`` variant produced for EnvTask by trainer/model_prep/env_stats.py.

    The discriminator field ``task_type`` is "env". The shape mirrors
    ``core/models/model_prep_models.py:EnvBaselineStats`` in gradients-ai/G.O.D
    feature/model-prep-container.
    """
    task_type: str = "env"
    weights: WeightStats = field(default_factory=WeightStats)
    env_stats: dict[str, EnvStats] = field(default_factory=dict)


def parse_baseline_stats() -> EnvBaselineStats | None:
    """Parse validator-injected baseline stats into a typed object. Returns None on absence/error.

    Validator writes a JSON file to the cache mount and exposes its path via
    ``BASELINE_STATS_PATH`` (see ``trainer/image_manager.py`` upstream).
    The inline ``BASELINE_STATS`` env var was deprecated; we still fall back to
    it for backward compatibility during local testing.
    """
    raw: str | None = None
    path = os.environ.get("BASELINE_STATS_PATH")
    if path:
        p = Path(path)
        if p.exists():
            try:
                raw = p.read_text()
            except OSError:
                raw = None
    if raw is None:
        # Deprecated inline JSON env var — kept for local-test backward compat.
        raw = os.environ.get("BASELINE_STATS")
    if not raw:
        return None
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None

    # Only env baseline stats are relevant for env tasks; ignore other task types.
    if data.get("task_type") != "env":
        return None

    env_stats_raw = data.get("env_stats", {}) or {}
    env_stats: dict[str, EnvStats] = {}
    for name, fields_ in env_stats_raw.items():
        if not isinstance(fields_, dict):
            continue
        env_stats[name] = EnvStats(
            num_episodes=int(fields_.get("num_episodes", 0)),
            mean_score=float(fields_.get("mean_score", 0.0)),
            std_score=float(fields_.get("std_score", 0.0)),
            min_score=float(fields_.get("min_score", 0.0)),
            max_score=float(fields_.get("max_score", 0.0)),
            median_score=float(fields_.get("median_score", 0.0)),
        )

    weights_raw = (data.get("weights") or {}).get("by_group", {}) or {}
    weights = WeightStats(by_group={
        group: WeightGroupStats(
            weight_rms=float(stats.get("weight_rms", 0.0)),
            weight_norm=float(stats.get("weight_norm", 0.0)),
            max_abs=float(stats.get("max_abs", 0.0)),
        )
        for group, stats in weights_raw.items()
        if isinstance(stats, dict)
    })

    return EnvBaselineStats(task_type="env", weights=weights, env_stats=env_stats)


# ---------------------------------------------------------------------------
# Anonymous-model robustness
# ---------------------------------------------------------------------------

def is_anonymous_model_path(path: str | None) -> bool:
    """True if the model path looks like a validator-anonymized repo.

    Anon repos follow ``gradients-io/augmented-<sha256[:16]>`` per
    trainer/model_prep/entrypoint.py:generate_anonymous_repo_name.
    """
    if not path:
        return False
    return "/augmented-" in path or path.startswith("gradients-io/augmented-")


def model_architecture_from_config(model) -> str:
    """Robust architecture detection that does NOT rely on ``model.config._name_or_path``.

    Anon model configs have ``_name_or_path`` SCRUBBED — so prefer
    ``config.architectures[0]`` which validator preserves.
    """
    try:
        arch = (getattr(model.config, "architectures", None) or [""])[0]
    except Exception:
        arch = ""
    return arch.strip().lower()


# ---------------------------------------------------------------------------
# Unified logging
# ---------------------------------------------------------------------------

def log_tournament_environment(env_name: str = "", logger=None) -> dict[str, Any]:
    """Single entry point to discover + log validator-injected tournament context.

    Call this once at trainer startup. Returns a dict with the parsed info:
        {
            "env_name": "<game>",
            "miner_datasets": [(hf_name, local_path), ...],
            "baseline_stats": EnvBaselineStats | None,
            "is_anon_model": bool,
            "miner_datasets_dir": str | None,
            "baseline_stats_path": str | None,
        }

    Safe to call when no env vars are set (returns benign defaults + logs).
    """
    def _log(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    miner_datasets = get_miner_datasets()
    baseline = parse_baseline_stats()
    miner_datasets_dir = os.environ.get("MINER_DATASETS_DIR")
    baseline_stats_path = os.environ.get("BASELINE_STATS_PATH")
    is_anon = bool(os.environ.get("AUGMENTED_MODEL", "")) or False

    _log("=" * 72)
    _log("[TOURNAMENT_ENV] Validator-injected context (Bittensor SN56 G.O.D)")
    _log("-" * 72)
    _log(f"[TOURNAMENT_ENV] env_name             = {env_name or '<unset>'}")
    _log(f"[TOURNAMENT_ENV] BASELINE_STATS_PATH  = {baseline_stats_path or '<unset>'}")
    _log(f"[TOURNAMENT_ENV] MINER_DATASETS_DIR   = {miner_datasets_dir or '<unset>'}")
    if miner_datasets:
        _log(f"[TOURNAMENT_ENV] miner datasets ({len(miner_datasets)}):")
        for name, path in miner_datasets:
            _log(f"[TOURNAMENT_ENV]   - {name}  ({path})")
    else:
        _log("[TOURNAMENT_ENV] miner datasets       = <none>")

    if baseline is not None:
        _log(f"[TOURNAMENT_ENV] BASELINE_STATS task  = {baseline.task_type}")
        if env_name and env_name in baseline.env_stats:
            es = baseline.env_stats[env_name]
            _log(
                f"[TOURNAMENT_ENV] baseline {env_name}: mean={es.mean_score:.4f} "
                f"std={es.std_score:.4f} median={es.median_score:.4f} "
                f"min={es.min_score:.4f} max={es.max_score:.4f} n={es.num_episodes}"
            )
        if baseline.weights.by_group:
            _log(f"[TOURNAMENT_ENV] weight groups stats: {len(baseline.weights.by_group)} groups")
    else:
        _log("[TOURNAMENT_ENV] BASELINE_STATS       = <unset>")
    _log("=" * 72)

    return {
        "env_name": env_name,
        "miner_datasets": miner_datasets,
        "baseline_stats": baseline,
        "is_anon_model": is_anon,
        "miner_datasets_dir": miner_datasets_dir,
        "baseline_stats_path": baseline_stats_path,
    }


def adjust_curriculum_from_baseline(
    args, baseline: EnvBaselineStats | None, env_name: str
) -> bool:
    """If baseline shows the base model already scores reasonably, skip early exploration.

    Heuristic (conservative): if ``mean_score >= 0.5`` (env tasks score higher = better),
    bump ``initial_max_turn`` halfway to ``final_max_turn``. Returns True if adjusted.

    Args:
        args: argparse-style namespace with ``initial_max_turn`` attribute.
        baseline: parsed baseline stats (or None).
        env_name: environment name to look up in ``baseline.env_stats``.
    """
    if baseline is None or not env_name:
        return False
    es = baseline.env_stats.get(env_name)
    if es is None or es.num_episodes < 5:
        return False
    if es.mean_score < 0.5:
        return False
    if not hasattr(args, "initial_max_turn"):
        return False
    cur = int(getattr(args, "initial_max_turn") or 0)
    # Final max turn per env (mirrors curriculum factories):
    final_map = {"gin_rummy": 30, "leduc_poker": 10, "liars_dice": 15}
    final_max = final_map.get(env_name, max(cur * 2, 4))
    new_val = max(cur, (cur + final_max) // 2)
    if new_val == cur:
        return False
    args.initial_max_turn = new_val
    print(
        f"[TOURNAMENT_ENV] curriculum boost: initial_max_turn {cur} -> {new_val} "
        f"(baseline mean={es.mean_score:.4f} for {env_name})",
        flush=True,
    )
    return True
