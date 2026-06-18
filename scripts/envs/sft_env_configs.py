"""Lightweight registry mapping env names to their SFT expert trajectory generator.

All 3 games registered for PvP tournament prep (2026-05-25). Empirical evidence:
- liars_dice + Qwen3-4B: SFT+LoRA r=128 wins (5EF2nASX 0.7967, Darcy 0.7922)
- gin_rummy: SFT+LoRA r=128 wins (5EgpWgYv 0.6516)
- leduc_poker: SFT+LoRA r=128 wins (5EgpWgYv 0.5390 vs GRPO 0.5446 — same ceiling
  because game tree tiny; SFT preferred for consistency across 3 games)

Tournament 20260518 winner 5EgpWgYv used UNIVERSAL SFT+LoRA recipe across all
3 games. We adopt same pattern: route all 3 to SFT path.

For PvP eval, model needs to handle:
- Pure integer output (no CoT, max_tokens=20)
- 5-second per-turn timeout
- Zero parsing retries (bad output = random fallback)
- Both player_id 0+1 perspectives (position swap)

See reference_pvp_tournament_2026_05_25.md for full PvP eval mechanism.
"""

from typing import Callable

from envs.liar_dice_trajectories   import generate_expert_episode as _liar_gen
from envs.leduc_poker_trajectories import generate_expert_episode as _leduc_gen
from envs.gin_rummy_trajectories   import generate_expert_episode as _gin_gen

_SFT_REGISTRY: dict[str, Callable] = {
    "liars_dice":  _liar_gen,
    "leduc_poker": _leduc_gen,
    "gin_rummy":   _gin_gen,
}


def supports_sft(env_name: str) -> bool:
    return env_name in _SFT_REGISTRY


def get_sft_trajectory_generator(env_name: str) -> Callable:
    if env_name not in _SFT_REGISTRY:
        raise ValueError(f"No SFT trajectory generator for env: {env_name!r}")
    return _SFT_REGISTRY[env_name]
