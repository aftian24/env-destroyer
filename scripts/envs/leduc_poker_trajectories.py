"""Random trajectory generator for Leduc Poker SFT training.

Synced to tournament winner 5EgpWgYv (2026-05-18) strategy: random action
selection + score-based filtering via generate_trajectories.py's
`--sample-by-score --score-power 3.0` flags.

Why random instead of expert: Leduc Poker's tiny state space lets MCTS play
near-optimally, so even a strong heuristic player rarely wins. Generating
random games and filtering/sampling by final reward is more practical than
hand-crafting an expert that beats MCTS. The winner's empirical evidence:
this approach matches or beats GRPO at the same score ceiling (~0.5446).

Output format: PvP-eval-compatible (matches `validator/evaluation/pvp/
agents.py:LeducPokerAgent.format_state` from G.O.D feature/pvp-eval-container).
Each user message gets reformatted via `build_pvp_user_prompt_leduc_poker`.

Return type: ``(messages, final_reward)`` tuple so generate_trajectories.py
can apply score-based sampling. final_reward is the raw env reward from the
terminal step (positive = win, negative = loss). The trajectory generator
clamps this to [0, 1] before using it as a sampling probability.
"""

import random
import re

import requests

from envs.leduc_poker_env import _format_observation
from envs.pvp_format import (
    SYSTEM_PROMPT_LEDUC_POKER,
    build_pvp_user_prompt_leduc_poker,
)

_TIMEOUT = 2400

# System prompt sourced from PvP canonical (matches validator eval format exactly)
_SYSTEM_PROMPT = SYSTEM_PROMPT_LEDUC_POKER


def _random_action(obs: str) -> str:
    """Pick a uniformly random legal action ID from the observation text.

    Falls back to '1' (Call) if no actions parseable (defensive).
    """
    actions = re.findall(r"^[ \t]*(\d+)\s*->", obs, re.MULTILINE)
    return random.choice(actions) if actions else "1"


def generate_random_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 10,
) -> "tuple[list[dict], float] | None":
    """Run one Leduc Poker game with random actions vs MCTS opponent.

    Returns ``(messages, final_reward)`` on success, or ``None`` on failure.
    User messages are PvP-formatted to match validator's eval prompt exactly,
    so SFT data trains the distribution the model sees at PvP eval time.

    PvP eval covers both player perspectives via position swap, so we alternate
    ``player_id = game_id % 2`` to cover both in the SFT dataset.
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,  # mirrors training opponent in leduc_poker_opponent_modeling
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        # Raw env observation (used by `_random_action` for legal-action parsing
        # AND by `build_pvp_user_prompt_leduc_poker` to reformat into PvP eval format)
        raw_observation = _format_observation(block.get("observation", ""))
    except Exception as exc:
        print(f"[leduc_poker_trajectories] Reset failed (game {game_id}): {exc}")
        return None

    # PvP eval alternates player perspectives via position swap
    player_id = game_id % 2

    user_prompt = build_pvp_user_prompt_leduc_poker(raw_observation, player_id=player_id)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    final_reward = 0.0

    for _ in range(max_turn):
        # Random action from RAW observation (legal action IDs are in both
        # raw and PvP-formatted observations — the IDs themselves are unchanged)
        action = _random_action(raw_observation)
        messages.append({"role": "assistant", "content": action})

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            raw_observation = _format_observation(step_block.get("observation", ""))
            done = step_block.get("done", False)
            if done:
                final_reward = float(step_block.get("reward", 0.0))
        except Exception as exc:
            print(f"[leduc_poker_trajectories] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        user_prompt = build_pvp_user_prompt_leduc_poker(raw_observation, player_id=player_id)
        messages.append({"role": "user", "content": user_prompt})
    else:
        print(f"[leduc_poker_trajectories] max_turn={max_turn} reached (game {game_id})")

    return messages, final_reward


# Back-compat alias — `generate_expert_episode` was the old name used by
# `sft_env_configs.py`. Keep it pointing to the new random+score generator so
# existing imports don't break.
generate_expert_episode = generate_random_episode
