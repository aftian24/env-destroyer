"""
Orchestrator config for EnvTask SFT training.
Counterpart of instruct_config.py for the SFT environment mode.
No GRPO content; no vLLM/num_generations/beta.
"""

import os
from copy import deepcopy

from lrs_lookup import get_instruct_lr
from model_utility import (
    disable_flash_attention,
    get_gpu_count,
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
)

SFT_ENV_SIZE_CONFIG: dict[str, dict] = {
    # PvP tournament 2026-05-25 config — Loki-validated truth (NOT winner repo file values):
    #
    # Validation source: Loki training logs at 185.141.218.122:3001, queried 2026-05-23
    # for hotkey 5EgpWgYvVyxYKtHuZnYYX95VFExbk2BE97y6VfosvRC68KNf (tournament 20260518 rank-1).
    # Tasks audited:
    # - 242d3d26 (R1 liars Qwen3-4B): "Using lr from architecture config: 3.571e-5 (arch=qwen3forcausallm)"
    # - c606369f (R4a gin Qwen2.5-7B WIN 0.6516): "Using lr from architecture config: 4.302e-5 (arch=qwen2forcausallm)"
    # - 29ef1316 (R4c liars Qwen3-4B WIN 0.7927): same as 242d3d26
    # - All tasks: rank0-3 in torchrun → gpu_count=4 (NOT 2 as winner repo file claims)
    #
    # IMPORTANT: Winner's repo bucket values are DEAD FALLBACKS overridden by
    # `get_lr_from_ar_instruct(arch, size)` lookup. We use OUR `get_instruct_lr(name)`
    # hash-based lookup, so bucket values DO matter when hash not in instruct.json.
    #
    # Bucket LR set to match Loki-validated values so behavior is consistent
    # whether or not the hash lookup hits.
    "0_1_b":  {"lr": 1e-4,    "distributed": "ddp", "gpu_count": 1, "batch_size": 140, "use_lora": False},
    "1_2_b":  {"lr": 1e-4,    "distributed": "ddp", "gpu_count": 1, "batch_size": 100, "use_lora": False},
    "2_4_b":  {"lr": 7.5e-5,  "distributed": "ddp", "gpu_count": 1, "batch_size": 48,  "use_lora": True},
    # 4_5_b = Qwen3-4B (Loki winner R1+R4c: lr=3.571e-5, gpu=4, LoRA)
    "4_5_b":  {"lr": 3.571e-5,"distributed": "ddp", "gpu_count": 4, "batch_size": 20,  "use_lora": True},
    # 5_9_b = Qwen2-7B / Qwen2.5-7B (Loki winner R4a: lr=4.302e-5, gpu=4, LoRA)
    "5_9_b":  {"lr": 4.302e-5,"distributed": "ddp", "gpu_count": 4, "batch_size": 16,  "use_lora": True},
    "9_12_b": {"lr": 1e-4,    "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "use_lora": True},
    "12_15_b":{"lr": 1e-4,    "distributed": "ds",  "gpu_count": 4, "batch_size": 30,  "use_lora": True},
    "15_40_b":{"lr": 8e-5,    "distributed": "ds",  "gpu_count": 4, "batch_size": 18,  "use_lora": True},
    "40_80_b":{"lr": 8e-5,    "distributed": "ds",  "gpu_count": 8, "batch_size": 8,   "use_lora": True},
}


def get_adaptive_epochs(hours_to_complete: float) -> int:
    """Adaptive training epochs based on available time budget.

    Empirical data from tournament 20260518 (Loki-validated):
    - **1 epoch LoRA r=128 in 1.5h** (5EgpWgYv R2/R4a/R4c): scores 0.6424/0.6516/0.7927
      → THIS IS WHAT WON 3 OF 6 FINALS. Winner pattern.
    - 3 epochs Full FT in 1.5h (5GNP9XWd R1): score 0.7989 (rank 1, +0.003 over LoRA)
      → Full FT runs faster per step (no LoRA overhead) so 3 epochs fits.
      → But Full FT is EXCLUDED from PvP group eval (memory: pvp:41-42)
      → For PvP path, LoRA + fewer epochs is the move.
    - Empirical 2-epoch attempt (this session 2026-05-23): could only complete ~30%
      of 2nd epoch in 1.5h budget at observed 4s/step throughput. Wastes the
      `save_before_remaining_time` checkpoint moment.

    Validator-controlled training hours (`validator/tournament/constants.py`):
      ENV_TRAINING_HOURS = 1.5                          # Round 1, 2, 3 default
      ENV_TRAINING_HOURS_BOSS_ROUND_FROM_SCRATCH = 3.0  # Boss/Final Round 4 task #2 only

    Validator dispatch (`validator/tasks/synthetic_scheduler.py`):
      _get_training_hours_for_environment_task(round_number=1) -> float
        return ENV_TRAINING_HOURS  # always 1.5 unless override

    Boss Round 4 special handling (`validator/tournament/task_creator.py`):
      - Task 1 (CONTINUATION):    1.5h, same base as Round 1
      - Task 2 (FROM_SCRATCH):    3.0h, random base model
      - Task 3 (PREVIOUS_WINNER): 1.5h, prev tournament winner checkpoint
        (see [[reference_pvp_boss_training]] — only 1 epoch desired for this variant
         to avoid overfitting on already-trained checkpoint. Detection happens
         outside this function via `_detect_previous_winner_model`.)

    Pattern (matches winner 5EgpWgYv 5-of-6 finals):
      < 1.5h:   1 epoch (emergency, minimal training)
      1.5-2.5h: **1 epoch** (5EgpWgYv pattern — winner-validated)
      2.5-3.5h: 2 epochs (extended budget; partial-3rd-epoch risk acceptable)
      ≥ 3.5h:   3 epochs (max useful for LoRA; Full FT could do 4 but we use LoRA)
    """
    if hours_to_complete >= 3.5:
        return 3
    if hours_to_complete >= 2.5:
        return 2
    # 1.5-2.5h → 1 epoch (winner pattern, fits budget cleanly with data gen + tokenize)
    # < 1.5h   → 1 epoch (emergency)
    return 1

for _key in SFT_ENV_SIZE_CONFIG:
    SFT_ENV_SIZE_CONFIG[_key]["label"] = _key


# Per-model override map — when validator sends one of these specific models,
# use this exact config INSTEAD OF the size-based bucket. Useful when the bucket
# default is too generic and we have Loki-validated per-model truth.
#
# Lookup order in get_sft_env_config:
#   1. PER_MODEL_OVERRIDE[model_name] if present (exact match)
#   2. SFT_ENV_SIZE_CONFIG[size_bucket] fallback (range-based)
#
# Validation: Loki training logs at 185.141.218.122:3001 for tournament 20260518
# winner 5EgpWgYv tasks 242d3d26 (R1 liars), c606369f (R4a gin WIN), 29ef1316
# (R4c liars WIN). All used gpu_count=4 (rank0-3 in torchrun).
PER_MODEL_OVERRIDE: dict[str, dict] = {
    "Qwen/Qwen3-4B-Instruct-2507": {
        "lr": 3.571428571428571e-05,   # Loki: arch=qwen3forcausallm 4.022B → 3.571e-5
        "distributed": "ddp",
        "gpu_count": 4,                # Loki: rank0-3 visible
        "batch_size": 20,
        "use_lora": True,
        "label": "Qwen3-4B-Instruct-2507_pvp",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "lr": 4.302500000000001e-05,   # Loki: arch=qwen2forcausallm 7.615B → 4.302e-5 (task c606369f R4a WIN 0.6516)
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 16,
        "use_lora": True,
        "label": "Qwen2.5-7B-Instruct_pvp",
    },
    "Qwen/Qwen2-7B-Instruct": {
        "lr": 4.302500000000001e-05,   # Loki: same arch (qwen2forcausallm) as 2.5-7B
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 16,
        "use_lora": True,
        "label": "Qwen2-7B-Instruct_pvp",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        # NOT Loki-validated (winner used this for R3 leduc which they LOST to
        # 5GU4Xkd3 GRPO). Conservative LR; 2_4_b bucket default is 7.5e-5.
        "lr": 6e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 24,
        "use_lora": True,
        "label": "Qwen2.5-3B-Instruct_pvp",
    },
}


def get_sft_env_config(param_nums: int, model_name: "str | None" = None) -> dict:
    """Return SFT training config for the given model.

    Lookup priority:
    1. PER_MODEL_OVERRIDE[model_name] — exact match (Loki-validated truth)
    2. Size-based SFT_ENV_SIZE_CONFIG bucket — range fallback

    Args:
        param_nums: model parameter count (from safetensors / config.json)
        model_name: HF repo id, e.g. "Qwen/Qwen3-4B-Instruct-2507". When provided,
            takes precedence over size-based bucket if name matches PER_MODEL_OVERRIDE.
    """
    if model_name and model_name in PER_MODEL_OVERRIDE:
        print(f"[sft_env_config] PER_MODEL_OVERRIDE hit: {model_name} "
              f"→ lr={PER_MODEL_OVERRIDE[model_name]['lr']:.4e}, "
              f"gpu={PER_MODEL_OVERRIDE[model_name]['gpu_count']}", flush=True)
        return deepcopy(PER_MODEL_OVERRIDE[model_name])

    result = {"lr": 4e-5, "distributed": "ds", "gpu_count": 8, "batch_size": 6, "use_lora": True}
    if param_nums < 1_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["12_15_b"]
    elif param_nums < 40_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported, using 40_80_b")
    return deepcopy(result)


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger_kernel",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    # Cap nproc_per_node to config's recommended gpu_count.
    # Rationale: SFT 4-rank on 7B (5_9_b config gpu_count=2) fails with
    # bitsandbytes/NCCL errors on H100 PCIe (no NVLink P2P). The config's
    # gpu_count reflects winner-validated configurations — exceeding it leads
    # to library compatibility issues that GRPO (which uses LoRA + vLLM-isolated
    # processes) doesn't hit. If validator allocates more GPUs than config says,
    # we still respect config to avoid known failure modes.
    available_gpus = get_gpu_count()
    gpu_nums = min(available_gpus, config.get("gpu_count", available_gpus))
    if available_gpus > gpu_nums:
        print(f"[sft_env_config] Capping nproc_per_node from {available_gpus} (visible) "
              f"to {gpu_nums} (config-recommended) to avoid multi-rank SFT issues.", flush=True)
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = "deepspeed"
    else:
        start_cmd = "python"

    template = (
        start_cmd
        + """ train_sft_env.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger_kernel {use_liger_kernel} \
    --packing {packing} \
    --disable_fa {disable_fa} \
    --max_length 4096"""
    )

    if run_type == "ds":
        template += " --deepspeed ds_config/zero3.json"

    if config.get("use_lora", False):
        template += " --use_lora True"

    if not config.get("disable_fa", False):
        template += " --padding_free True"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


# ───────────────────────────────────────────────────────────────────────────
# Per-env trajectory generation args (single source of truth)
# ───────────────────────────────────────────────────────────────────────────
# Tuned 2026-05-24 — caps + multi_env_min floor per Discord PvP guidance:
# - liars_dice : 100K games × max_turn=30 (winner default; expert dataset dense enough)
# - gin_rummy  : 15K games × max_turn=200 (bumped from 8K — 9 hand/knock variants
#                need coverage; longer episodes naturally produce more turn samples)
# - leduc_poker: 100K games × max_turn=10 + sample_by_score=True (capped from 200K
#                to fit time budget; soft filter score^3 preserved per user request
#                — winner-proven, dataset shrink ~40% via filter).
#
# `multi_env_min` per-env floor: prevents tiny per-env datasets when n_envs is large
# (e.g. R3 with 6 envs would naively give gin 8K/6≈1.3K — too small for variant cov).
_ENV_GENERATE_ARGS: dict[str, dict] = {
    "liars_dice":  {"num_games": 100_000, "max_turn":  30,
                    "multi_env_min": 25_000},
    "gin_rummy":   {"num_games":  15_000, "max_turn": 200,
                    "multi_env_min":  5_000},
    "leduc_poker": {"num_games": 100_000, "max_turn":  10,
                    "sample_by_score": True, "score_power": 3.0,
                    "multi_env_min": 25_000},
}

# Global fallback floor for envs without per-env multi_env_min
_MIN_GAMES_PER_ENV_MULTI = 2_000

# Time budget for trajectory generation (seconds). Predictable upper bound so
# data gen never overruns the validator's training-hours allocation. Distributed
# across multiple envs in multi-env path: per_env_budget = total / n_envs.
# Set via env var TRAJECTORY_GEN_MAX_SECONDS (default 2700 = 45 min).
_DEFAULT_TRAJ_GEN_BUDGET_SECONDS = 2_700


def _wins_only_enabled() -> bool:
    """User can flip leduc filter from soft (score^3) to hard (wins-only) via env var.

    Default OFF — winner 5EgpWgYv committed `--sample-by-score --score-power 3.0`
    which is the empirically-proven setting. Wins-only is stricter (drop all
    losses), useful if dataset noise is high but risks overfitting to lucky wins.

    Toggle: `WINS_ONLY=1` env var.
    """
    return os.environ.get("WINS_ONLY", "0").strip() in ("1", "true", "True")


def _build_generate_cmd(env_name: str, dataset_path: str, env_args: dict,
                         num_games: int | None = None,
                         max_time_seconds: int = 0) -> str:
    """Build the `python -m envs.generate_trajectories` cmd for one env.

    Args:
        env_name: liars_dice / gin_rummy / leduc_poker
        dataset_path: where to save HF DatasetDict
        env_args: per-env config dict (from _ENV_GENERATE_ARGS)
        num_games: if None, uses env_args["num_games"]. Multi-env caller scales.
        max_time_seconds: wall-time cap (0 = no cap). Passed via --max-time-seconds.
    """
    if num_games is None:
        num_games = env_args["num_games"]
    cmd = (
        f"python -m envs.generate_trajectories"
        f" --environment_name {env_name}"
        f" --output_path {dataset_path}"
        f" --num_games {num_games}"
        f" --max_turn {env_args['max_turn']}"
    )
    # Time-bounded mode (Patch A) — caps total wall time for this env's traj gen
    if max_time_seconds and max_time_seconds > 0:
        cmd += f" --max-time-seconds {max_time_seconds}"
    # Score-based filter (leduc): soft (score^3, default) OR wins-only (env var toggle)
    if env_args.get("sample_by_score"):
        if _wins_only_enabled():
            # User opted into wins-only via WINS_ONLY=1 — overrides soft filter
            cmd += " --wins-only"
        else:
            cmd += f" --sample-by-score --score-power {env_args['score_power']}"
    elif env_args.get("wins_only"):
        cmd += " --wins-only"
    return cmd


def _scale_games_for_multi_env(env_name: str, n_envs: int) -> tuple[int, dict]:
    """Return (scaled_num_games, base_env_args) for multi-env training.

    Strategy:
      base[num_games] // n_envs, but never below max(global floor, per_env_min)

    Per-env floor (`multi_env_min` in `_ENV_GENERATE_ARGS`) prevents tiny datasets
    that don't generalize. Gin needs 5K min to cover 9 hand/knock variants;
    liars/leduc need 25K min for reasonable distribution.
    """
    base = _ENV_GENERATE_ARGS.get(env_name, {"num_games": 100_000, "max_turn": 30})
    per_env_floor = max(_MIN_GAMES_PER_ENV_MULTI, base.get("multi_env_min", 0))
    scaled = max(per_env_floor, base["num_games"] // n_envs)
    return scaled, base


def _get_traj_gen_budget(n_envs: int = 1) -> int:
    """Total trajectory-gen wall-time budget (seconds), distributed per-env.

    Env var override: `TRAJECTORY_GEN_MAX_SECONDS` (default 2700 = 45 min).
    Returns per-env budget; multi-env divides total by n_envs so all envs
    finish within the same total window.

    For 1.5h budget (5400s): 45 min data gen + 5 min tokenize + 40 min train
    leaves clean schedule. Adjust via env var if env_server slow.
    """
    total = int(os.environ.get("TRAJECTORY_GEN_MAX_SECONDS",
                                _DEFAULT_TRAJ_GEN_BUDGET_SECONDS))
    return max(60, total // max(1, n_envs))


def _detect_previous_winner_model(model_name: str) -> bool:
    """Heuristic: validator's Boss task #3 (PREVIOUS_WINNER) sends the prev
    tournament champion's HF repo as model_id. Pattern:
        `gradients-io-tournaments/tournament-*`
    See [[reference_pvp_boss_training]] memory file.
    """
    return model_name.startswith("gradients-io-tournaments/tournament-")


def get_training_json(train_info: dict) -> dict:
    """Single-env SFT training path. Used when validator sends ONE env, OR
    when `environment_names: list` has length 1.

    For multi-env tasks (PvP tournament 2026-05-25+) use
    `get_training_json_multi_env(train_info, env_names)` instead.
    """
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_sft_env_config(param_nums, model_name=model_name)

    task_id = train_info["task_id"]
    # PvP tournament (2026-05-25+) may send `environment_names: list` for multi-env
    # interleave tasks. Backward-compat: also support legacy `environment_name: str`.
    dataset_type = train_info.get("dataset_type", {})
    env_names = dataset_type.get("environment_names")
    if env_names and isinstance(env_names, list) and len(env_names) > 0:
        env_name = env_names[0]
    else:
        env_name = dataset_type.get("environment_name", "liars_dice")
    dataset_path = f"/workspace/scripts/datasets/sft_env_{task_id}"

    _env_args = _ENV_GENERATE_ARGS.get(env_name, {"num_games": 100000, "max_turn": 30})
    num_games = _env_args["num_games"]
    max_turn  = _env_args["max_turn"]

    # Adaptive epochs based on validator's time budget (PvP rounds: 1.5h/3.0h)
    hours = train_info.get("hours_to_complete", 1.5)
    adaptive_epochs = get_adaptive_epochs(hours)

    # Boss Round 4 task #3 (PREVIOUS_WINNER): validator sends prev tournament
    # winner's HF repo as model_id. That checkpoint is already-trained — cap
    # to 1 epoch to avoid overfitting on top. See [[reference_pvp_boss_training]].
    if _detect_previous_winner_model(model_name):
        print(f"[sft_env_config] PREVIOUS_WINNER model detected ({model_name}) — "
              f"capping epoch_num=1 (override adaptive={adaptive_epochs})", flush=True)
        adaptive_epochs = 1

    run_config = {
        "epoch_num": adaptive_epochs,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger_kernel": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": "False",  # pre-tokenised dataset; TRL packing not used
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
    }
    print(f"[sft_env_config] PvP adaptive epoch_num={adaptive_epochs} "
          f"(hours_to_complete={hours}h, env={env_name})", flush=True)

    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    run_config["gradient_accumulation_steps"] = (
        1 if data_per_step >= 64 else int(64 / data_per_step)
    )

    if train_info.get("find_lk_lr"):
        lr = get_instruct_lr(model_name)
        if lr is not None:
            print(f"Using lr from lk: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            print(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["dataset_path"] = dataset_path
    train_request["save_before_remaining_time"] = 3
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 70
    train_request["min_steps"] = max(
        int(train_info["hours_to_complete"] * 70),
        train_info.get("min_steps", 100),
    )

    # Use shared helper so both single- and multi-env paths produce identical
    # `generate_trajectories` invocations (with score-based sampling kalau leduc).
    # Single-env gets full traj-gen budget (default 45 min, override via env var).
    traj_budget = _get_traj_gen_budget(n_envs=1)
    generate_cmd = _build_generate_cmd(
        env_name, dataset_path, _env_args, num_games,
        max_time_seconds=traj_budget,
    )

    return {
        "train_request": train_request,
        "run_cmd": run_cmd,
        "generate_cmd": generate_cmd,
    }


def get_training_json_multi_env(train_info: dict, env_names: list[str]) -> dict:
    """Multi-env SFT training path for PvP tournament 2026-05-25+.

    Validator sends `dataset_type.environment_names: list[str]` (R1=2 envs,
    R2=4 envs, R3=6 envs per Discord 2026-05-23). ALL envs are PvP-evaluated
    head-to-head ⇒ SFT data must cover ALL envs in the payload.

    Strategy:
    1. Generate per-env dataset at SCALED `num_games` (linear /n_envs, floor 2000)
       so total trajectory-gen time stays bounded regardless of env count.
    2. Chain per-env `generate_trajectories` cmds with `&&` so they run sequentially.
    3. Merge all per-env DatasetDicts into one combined dataset via
       `envs.merge_datasets` (with shuffle so env examples interleave — prevents
       catastrophic forgetting of env_A during env_B batches).
    4. Single SFT run on combined dataset (same trainer config as single-env).

    All other knobs (LR, batch, gpu_count, LoRA, epochs) identical to single-env
    path — multi-env affects only data preparation, not the training loop.

    PREVIOUS_WINNER boss task: if model is `gradients-io-tournaments/tournament-*`,
    cap epoch_num at 1 to avoid overfitting on already-trained checkpoint.
    """
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_sft_env_config(param_nums, model_name=model_name)

    task_id = train_info["task_id"]
    n_envs  = len(env_names)
    if n_envs < 1:
        raise ValueError(f"get_training_json_multi_env: env_names must be non-empty list")

    # Per-env trajectory-gen budget: total time / n_envs (so all envs finish
    # within the same window regardless of n_envs).
    per_env_traj_budget = _get_traj_gen_budget(n_envs=n_envs)

    # Build per-env trajectory-gen cmds at scaled num_games + collect output paths
    per_env_paths: list[str] = []
    per_env_cmds:  list[str] = []
    per_env_scaled_counts: list[tuple[str, int]] = []
    for idx, env_name in enumerate(env_names):
        per_env_path = f"/workspace/scripts/datasets/sft_env_{task_id}_part{idx}_{env_name}"
        scaled_games, env_args = _scale_games_for_multi_env(env_name, n_envs)
        per_env_paths.append(per_env_path)
        per_env_cmds.append(_build_generate_cmd(
            env_name, per_env_path, env_args, scaled_games,
            max_time_seconds=per_env_traj_budget,
        ))
        per_env_scaled_counts.append((env_name, scaled_games))

    # Final merged dataset path (consumed by the SFT trainer)
    final_dataset_path = f"/workspace/scripts/datasets/sft_env_{task_id}"
    merge_cmd = (
        f"python -m envs.merge_datasets"
        f" --inputs {' '.join(per_env_paths)}"
        f" --output {final_dataset_path}"
    )

    # Chain: gen env1 && gen env2 && ... && merge
    full_generate_cmd = " && ".join(per_env_cmds + [merge_cmd])

    # ── Training config (same as single-env) ────────────────────────────
    hours = train_info.get("hours_to_complete", 1.5)
    adaptive_epochs = get_adaptive_epochs(hours)

    # Boss Round 4 task #3 (PREVIOUS_WINNER): validator sends prev tournament
    # winner's HF repo as model_id. That checkpoint is already trained — 1 epoch
    # is plenty; more epochs risk overfitting. See [[reference_pvp_boss_training]].
    if _detect_previous_winner_model(model_name):
        print(f"[sft_env_config] PREVIOUS_WINNER model detected ({model_name}) — "
              f"capping epoch_num=1 (override adaptive={adaptive_epochs})", flush=True)
        adaptive_epochs = 1

    run_config = {
        "epoch_num": adaptive_epochs,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger_kernel": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": "False",
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
    }
    print(f"[sft_env_config] Multi-env training: {env_names} "
          f"(n={n_envs}, scaled_counts={dict(per_env_scaled_counts)}, "
          f"epoch_num={adaptive_epochs}, hours={hours}h)", flush=True)

    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    run_config["gradient_accumulation_steps"] = (
        1 if data_per_step >= 64 else int(64 / data_per_step)
    )

    if train_info.get("find_lk_lr"):
        lr = get_instruct_lr(model_name)
        if lr is not None:
            print(f"Using lr from lk: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            print(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["dataset_path"] = final_dataset_path
    train_request["save_before_remaining_time"] = 3
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 70
    train_request["min_steps"] = max(
        int(train_info["hours_to_complete"] * 70),
        train_info.get("min_steps", 100),
    )

    return {
        "train_request": train_request,
        "run_cmd": run_cmd,
        "generate_cmd": full_generate_cmd,
    }
