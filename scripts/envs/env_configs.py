"""
Environment training configuration registry.

Hyperparameter resolution order for (env, mode, size_label):
  1. mode_cfg.per_size[size_label]  → all 5 SizeHyperparams fields (explicit)
  2. DEFAULT_HYPERPARAMS[size_label] + mode_cfg.num_generations if set
  Then mode_cfg scalars (temperature, top_k, initial_max_turn, …) on top.
"""

from dataclasses import dataclass, field, fields
from typing import Callable

from envs.alf_world_env import (
    alfworld_rollout_first_prompt_and_completion_parallelized as _alf_rollout_last,
    alfworld_rollout_full_prompt_and_completion_parallelized  as _alf_rollout_full,
    alfworld_rollout_reward_func                              as _alf_reward,
)
from envs.gin_rummy_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_rollout_last,
    rollout_reward_func                                        as _gin_reward,
    _curriculum_factory                                        as _gin_curriculum,
)
from envs.gin_rummy_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_last,
    rollout_reward_func                                        as _gin_opp_reward,
    _curriculum_factory                                        as _gin_opp_curriculum,
)
from envs.gin_rummy_refined import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_ref_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_ref_rollout_last,
    rollout_reward_func                                        as _gin_ref_reward,
    _curriculum_factory                                        as _gin_ref_curriculum,
)
from envs.gin_rummy_mmd import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_mmd_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_mmd_rollout_last,
    rollout_reward_func                                        as _gin_mmd_reward,
    _curriculum_factory                                        as _gin_mmd_curriculum,
)
from envs.goof_spiel_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _goof_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _goof_rollout_last,
    rollout_reward_func                                        as _goof_reward,
    _curriculum_factory                                        as _goof_curriculum,
)
from envs.leduc_poker_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_rollout_last,
    rollout_reward_func                                        as _leduc_reward,
    _curriculum_factory                                        as _leduc_curriculum,
)
from envs.leduc_poker_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_last,
    rollout_reward_func                                        as _leduc_opp_reward,
    _curriculum_factory                                        as _leduc_opp_curriculum,
)
from envs.liar_dice_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _liar_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _liar_rollout_last,
    rollout_reward_func                                        as _liar_reward,
    _curriculum_factory                                        as _liar_curriculum,
)
from envs.liar_dice_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _liar_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _liar_opp_rollout_last,
    rollout_reward_func                                        as _liar_opp_reward,
    _curriculum_factory                                        as _liar_opp_curriculum,
)


@dataclass
class SizeHyperparams:
    """Co-tuned VRAM/dynamics params for one (env, mode, size). All fields required."""
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations:             int
    vllm_gpu_memory_utilization: float
    beta:                        float

    def apply(self, args) -> None:
        for f in fields(self):
            setattr(args, f.name, getattr(self, f.name))


@dataclass
class ModeConfig:
    """Per-mode overrides for one environment. None means use the default."""
    initial_max_turn:     int | None   = None
    rollouts_per_stage:   int | None   = None
    trainer_class:        "type | None" = None  # default: GRPOTrainer or ActionMaskedGRPOTrainer
    max_completion_length: int | None  = None   # default: 2048 (reasoning) or 16
    num_generations:      int | None   = None
    temperature:          float | None = None
    top_k:                int | None   = None
    per_size: dict[str, SizeHyperparams] = field(default_factory=dict)

    def apply_scalars(self, args) -> None:
        for attr in ("initial_max_turn", "rollouts_per_stage", "temperature", "top_k"):
            val = getattr(self, attr)
            if val is not None:
                setattr(args, attr, val)


@dataclass
class EnvTrainingConfig:
    rollout_full: Callable
    rollout_last: Callable
    reward_func:  Callable
    curriculum_factory:   Callable | None = None
    vllm_max_model_length: int = 5248  # reasoning mode adds 2048 on top at runtime
    num_iterations:       int = 2
    reasoning:   ModeConfig = field(default_factory=ModeConfig)
    no_mask:     ModeConfig = field(default_factory=ModeConfig)
    full_prompt: ModeConfig = field(default_factory=ModeConfig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EnvTrainingConfig] = {
    "goof_spiel": EnvTrainingConfig(
        rollout_full=_goof_rollout_full,
        rollout_last=_goof_rollout_last,
        reward_func=_goof_reward,
        curriculum_factory=_goof_curriculum,
        reasoning=ModeConfig(initial_max_turn=1, num_generations=4, temperature=1.0, top_k=0),
        no_mask=ModeConfig(initial_max_turn=1, num_generations=4, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(num_generations=4, temperature=1.0, top_k=0),
    ),
    "gin_rummy": EnvTrainingConfig(
        rollout_full=_gin_rollout_full,
        rollout_last=_gin_rollout_last,
        reward_func=_gin_reward,
        curriculum_factory=_gin_curriculum,
        num_iterations=3,
        # num_generations=8 matches gin_rummy winner (5GU4Xkd3) Loki trace + leduc/liars convention.
        # Denser GRPO advantage estimate per rollout group. VRAM impact: ~2x peak during
        # rollout (8 vs 4 completions in flight); fits H100 80GB but tight at >7B base.
        reasoning=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=8, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
    ),
    "gin_rummy_opponent_modeling": EnvTrainingConfig(
        rollout_full=_gin_opp_rollout_full,
        rollout_last=_gin_opp_rollout_last,
        reward_func=_gin_opp_reward,
        curriculum_factory=_gin_opp_curriculum,
        # num_generations=8 to match winner pattern (see comment above).
        reasoning=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=8, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
    ),
    "gin_rummy_refined": EnvTrainingConfig(
        rollout_full=_gin_ref_rollout_full,
        rollout_last=_gin_ref_rollout_last,
        reward_func=_gin_ref_reward,
        curriculum_factory=_gin_ref_curriculum,
        # num_generations=8 for consistency with other gin_rummy variants and winner.
        reasoning=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=8, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
    ),
    "gin_rummy_mmd": EnvTrainingConfig(
        # MMD (Myopic Meld Distance) + SK (Stubborn Knocking) GRPO shaping
        # Adopted verbatim from tournament 20260518 winner 5EgpWgYv repo.
        # GRPO-only variant; for SFT path stay on default "gin_rummy" registry key.
        # Activate via _VARIANT_OVERRIDES["gin_rummy"] = "gin_rummy_mmd" below.
        rollout_full=_gin_mmd_rollout_full,
        rollout_last=_gin_mmd_rollout_last,
        reward_func=_gin_mmd_reward,
        curriculum_factory=_gin_mmd_curriculum,
        reasoning=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=8, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=8, temperature=1.0, top_k=0),
    ),
    "liars_dice": EnvTrainingConfig(
        rollout_full=_liar_rollout_full,
        rollout_last=_liar_rollout_last,
        reward_func=_liar_reward,
        curriculum_factory=_liar_curriculum,
        reasoning=ModeConfig(rollouts_per_stage=1024, initial_max_turn=1, num_generations=4, temperature=2.0, top_k=5),
        no_mask=ModeConfig(rollouts_per_stage=1280, initial_max_turn=1, num_generations=4, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(rollouts_per_stage=1024, initial_max_turn=2, num_generations=4, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "leduc_poker": EnvTrainingConfig(
        rollout_full=_leduc_rollout_full,
        rollout_last=_leduc_rollout_last,
        reward_func=_leduc_reward,
        curriculum_factory=_leduc_curriculum,
        reasoning=ModeConfig(num_generations=8, temperature=2.0, top_k=5),
        # Winner-match per Loki trace of 5GKSa6y1 (tournament 20260511):
        # gradient_accumulation_steps=4 (was 8), rollouts_per_stage=1024 (was None=1280).
        no_mask=ModeConfig(num_generations=8, temperature=2.0, top_k=5, rollouts_per_stage=1024, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(num_generations=8, temperature=2.0, top_k=5, rollouts_per_stage=1024, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "leduc_poker_opponent_modeling": EnvTrainingConfig(
        rollout_full=_leduc_opp_rollout_full,
        rollout_last=_leduc_opp_rollout_last,
        reward_func=_leduc_opp_reward,
        curriculum_factory=_leduc_opp_curriculum,
        reasoning=ModeConfig(num_generations=8, temperature=2.0, top_k=5),
        # Winner-match per Loki trace (variant route from "leduc_poker"):
        no_mask=ModeConfig(num_generations=8, temperature=2.0, top_k=5, rollouts_per_stage=1024, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(num_generations=8, temperature=2.0, top_k=5, rollouts_per_stage=1024, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "liars_dice_opponent_modeling": EnvTrainingConfig(
        rollout_full=_liar_opp_rollout_full,
        rollout_last=_liar_opp_rollout_last,
        reward_func=_liar_opp_reward,
        curriculum_factory=_liar_opp_curriculum,
        reasoning=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1, num_generations=8, temperature=2.0, top_k=5),
        no_mask=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1, num_generations=8, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(rollouts_per_stage=2048, initial_max_turn=1, num_generations=8, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "alfworld": EnvTrainingConfig(
        rollout_full=_alf_rollout_full,
        rollout_last=_alf_rollout_last,
        reward_func=_alf_reward,
        reasoning=ModeConfig(num_generations=4, temperature=1.0, top_k=0),
        no_mask=ModeConfig(num_generations=4, temperature=1.0, top_k=0),
        full_prompt=ModeConfig(num_generations=4, temperature=1.0, top_k=0),
    ),
}


# ---------------------------------------------------------------------------
# Variant routing
# ---------------------------------------------------------------------------

# Change this to select a non-default variant for a base environment name.
# Activated to match the tournament winner's setup (gradients-opensource position-1):
# all three games auto-route to their opponent_modeling variants.
_VARIANT_OVERRIDES: dict[str, str] = {
    "gin_rummy":   "gin_rummy_opponent_modeling",
    "liars_dice":  "liars_dice_opponent_modeling",
    "leduc_poker": "leduc_poker_opponent_modeling",
}


def get_env_config(name: str) -> EnvTrainingConfig:
    """Look up the training config for a named environment.

    If ``name`` has an entry in ``_VARIANT_OVERRIDES``, that registry key is
    used instead — allowing a single code-level switch between implementations
    without changing the caller's environment name.

    Raises ``ValueError`` with a helpful message if the name is unknown.
    """
    resolved = _VARIANT_OVERRIDES.get(name, name)
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown environment: {name!r} (resolved to {resolved!r}). "
            f"Known environments: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[resolved]
