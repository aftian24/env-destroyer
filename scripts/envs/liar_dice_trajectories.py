"""Expert trajectory generator for Liar's Dice SFT training.

Produces SFT data in PvP-eval-compatible format (matches what validator feeds
the model during PvP tournament eval starting 2026-05-25).

Output trajectory:
- system: PvP system prompt (`pvp_game_prompts.yml:system_prompt_template`)
- user:   Current State + Player ID + Legal Actions + 'Your choice (ID only):'
- assistant: pure integer action ID (no CoT)

Format source: G.O.D feature/pvp-eval-container
- validator/evaluation/pvp/agents.py:LiarsDiceAgent.format_state
- core/config/pvp_game_prompts.yml
"""

import math
import random
import re

import requests

from envs.liar_dice_env import parse_game_state
from envs.pvp_format import (
    SYSTEM_PROMPT_LIARS_DICE,
    build_pvp_user_prompt_liars_dice,
)

_TIMEOUT = 2400
_SAMPLING_TEMPERATURE = 0.01

# System prompt now sourced from PvP canonical (matches validator eval format exactly)
_SYSTEM_PROMPT = SYSTEM_PROMPT_LIARS_DICE


def _softmax_weights(probs: list[float], temperature: float) -> list[float]:
    """Convert raw probs to softmax weights."""
    if temperature <= 0:
        best = max(range(len(probs)), key=lambda i: probs[i])
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    scaled = [p / temperature for p in probs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def get_expert_action(messages: list[dict]) -> str:
    """Probability-based action selection."""
    try:
        gs = parse_game_state(messages)
    except Exception:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"

    if not gs.actions:
        return "0"

    probs = [a.prob for a in gs.actions]
    weights = _softmax_weights(probs, _SAMPLING_TEMPERATURE)
    chosen = random.choices(gs.actions, weights=weights, k=1)[0]
    return str(chosen.action_id)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "list[dict] | None":
    """
    Run one Liar's Dice game against the env server using the expert policy.
    Returns the messages list (system/user/assistant) or None on failure.

    User messages are reformatted to **PvP eval format** (Current State + Player ID
    + 'Your choice (ID only):' suffix) so SFT trains the exact distribution that
    PvP tournament will feed at eval time.

    Player perspective alternates between 0 and 1 per game_id (parity) so SFT
    dataset covers both perspectives (PvP eval plays each seed twice with
    positions swapped).
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    # PvP eval covers both player perspectives via position swap (PVP_NUM_GAMES_PER_ENV
    # × 2). Alternate per game_id parity so SFT dataset includes both.
    player_id = game_id % 2

    user_prompt = build_pvp_user_prompt_liars_dice(observation, player_id=player_id)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    for _ in range(max_turn):
        action = get_expert_action(messages)
        messages.append({"role": "assistant", "content": action})

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = step_block.get("observation", "")
            done = step_block.get("done", False)
        except Exception as exc:
            print(f"[env] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        user_prompt = build_pvp_user_prompt_liars_dice(observation, player_id=player_id)
        messages.append({"role": "user", "content": user_prompt})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages
