"""
Gin Rummy environment with MMD (Myopic Meld Distance) + SK (Stubborn Knocking)
shaping, ported from the old ``gin_rummy_environment_function_mmd.py`` (commit
7ae2e8a8).

Reuses card utilities, parsing, and the rollout loop structure from
``gin_rummy_env`` and swaps in a dedicated ``RewardCalculator`` that adds:
  - Per-step MMD delta (skipped during Layoff phase) clipped to ±MMD_PER_STEP_CLIP.
  - Terminal SK shaping: doubled gin bonus + non-gin-knock penalty.
  - Asymmetric *1.5 scaling when the deadwood improvement is negative.
"""

import functools
import json
import os
import random
from collections import Counter
from concurrent.futures import as_completed
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)
from envs.gin_rummy_env import (
    GameState,
    extract_and_format_observation,
    find_potential_runs,
    get_rank,
    parse_game_state,
    remove_reasoning_tags,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "gin_rummy"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 16384 - 256
_TIMEOUT = 2400

# MMD per-step shaping (SIGRA-inspired)
MMD_WEIGHT          = 0.02
MMD_PER_STEP_CLIP   = 0.03
SHAPING_REWARD_CLIP = 0.30

# Terminal SK (Stubborn Knocking) shaping
GIN_BONUS             = 0.50
NON_GIN_KNOCK_PENALTY = 0.15


# ---------------------------------------------------------------------------
# MMD score
# ---------------------------------------------------------------------------

def mmd_score(hand: list[str]) -> float:
    """SIGRA r_m proxy: weighted near-meld count.

    Completed melds contribute 2.0 each (locked-in value); almost-melds
    (pairs and 2-card potential runs) contribute 1.0 each (optionality).
    """
    if not hand:
        return 0.0
    rank_counts = Counter(get_rank(c) for c in hand)
    pairs       = sum(1 for n in rank_counts.values() if n == 2)
    triples     = sum(1 for n in rank_counts.values() if n >= 3)
    runs        = find_potential_runs(hand)
    potential_runs = sum(1 for r in runs if len(r) == 2)
    complete_runs  = sum(1 for r in runs if len(r) >= 3)
    return 2.0 * (triples + complete_runs) + 1.0 * (pairs + potential_runs)


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    MMD per-step shaping + clipped invalid penalty + terminal SK shaping.

    Mirrors the old (commit 7ae2e8a8) gin_rummy_environment_function_mmd.py
    reward topology.
    """

    INVALID_PENALTY = -0.1

    def __init__(self):
        self.invalid_penalty = self.INVALID_PENALTY
        self.prev_mmd: "float | None" = None

    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float,
        is_invalid: bool = False,
        current_state: "GameState | None" = None,
    ) -> float:
        """Per-step reward: invalid penalty OR MMD delta.

        MMD delta is ``MMD_WEIGHT * (curr_mmd - prev_mmd)`` clipped symmetrically.
        Layoff phase is skipped (hand can shrink mechanically when laying off
        onto opponent's melds — those deltas are not skill-related).
        """
        if is_invalid:
            return self.invalid_penalty

        if current_state is None:
            return 0.0

        curr_mmd = mmd_score(current_state.hand)
        prev_mmd = self.prev_mmd
        self.prev_mmd = curr_mmd

        if current_state.phase == "Layoff":
            return 0.0
        if prev_mmd is None:
            return 0.0

        raw = MMD_WEIGHT * (curr_mmd - prev_mmd)
        return max(-MMD_PER_STEP_CLIP, min(MMD_PER_STEP_CLIP, raw))

    def calculate_episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        initial_state: "GameState | None",
        final_state: "GameState | None",
    ) -> float:
        """Combine deadwood improvement + terminal SK + clipped shaping."""
        # 1. Deadwood improvement ratio (always available)
        if initial_state and final_state and initial_state.deadwood > 0:
            deadwood_component = (initial_state.deadwood - final_state.deadwood) / initial_state.deadwood
        else:
            deadwood_component = 0.0

        if deadwood_component < 0.0:
            deadwood_component *= 1.5

        # 2. Terminal bonus / truncation penalty with SK shaping
        if done:
            if env_reward > 0.5:
                terminal = 1.0
                if final_state is not None:
                    if final_state.deadwood == 0:
                        terminal += GIN_BONUS
                    elif final_state.deadwood > 0:
                        terminal -= NON_GIN_KNOCK_PENALTY
            else:
                terminal = -0.5
        elif final_state:
            terminal = -final_state.deadwood / 100.0
        else:
            terminal = 0.0

        # 3. Accumulated shaping (per-step MMD deltas + invalid penalties),
        # clipped symmetrically.
        accumulated = sum(step_rewards)
        shaping     = max(-SHAPING_REWARD_CLIP, min(SHAPING_REWARD_CLIP, accumulated))

        return deadwood_component + terminal + shaping


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=30,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        warmup_rollouts=128,
    )


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 25,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    _log_rank = os.environ.get("LOG_RANK", "0")
    if _log_rank == "all" or str(rank) == _log_rank:
        print(
            f"[CURRICULUM/MMD] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
            f"final_max_turn=30, rollouts_per_stage={trainer.args.rollouts_per_stage}"
        )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
    )


# ---------------------------------------------------------------------------
# Core episode runner (parallels gin_rummy_env._run_episode but passes
# ``current_state`` into the MMD step calculator)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)\n"
    "Examples:\n- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥\n"
    "- Invalid: K♠-A♠-2♠ (Ace is LOW only)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(♠), h(♥), d(♦), c(♣)\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood ≤ knock_card\n\n"
    "KNOCKING:\n- Gin: 0 deadwood = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY.\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n**Think short and act quickly!**\n\n# Strategy Tips\n"
    "- Early game: Draw from deck to see more cards\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have ≤10 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points"
)


def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt: bool,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, "dict | None"]:
    game_id = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    invalid_count = 0
    done          = False
    train_reward  = 0.0
    final_reward  = 0.0
    turn_number   = 0
    game_state_history: list[GameState] = []
    rewards: list[float] = []
    calculator = RewardCalculator()
    use_hints = random.random() < current_hint_prob

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id, "seed": random.randint(0, 2**31 - 1),
        "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id = result_block.get("episode_id", "")
        raw_observation = result_block.get("observation", "")
        formatted_observation = extract_and_format_observation(raw_observation)
        game_state_history.append(parse_game_state(formatted_observation))
    except Exception as exc:
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": formatted_observation},
    ]

    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    print(f"Warning: token shift at turn {turn_number}. Skipping delta mask.")
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- Parse action ---
        action_to_send = remove_reasoning_tags(completion_text)
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        is_invalid = False
        try:
            formatted_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            raw_observation       = step_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1
            is_invalid = True

        if done:
            final_reward = step_reward
        messages.append({"role": "user", "content": formatted_observation})

        # --- Reward bookkeeping (MMD per-step delta, invalid penalty, or 0) ---
        parse_failed = False
        current_state: "GameState | None" = None
        if not is_invalid and not done:
            try:
                current_state = parse_game_state(formatted_observation)
            except Exception as exc:
                print(f"Failed to parse game state: {exc}")
                parse_failed = True
            else:
                game_state_history.append(current_state)

        rewards.append(calculator.calculate_step_reward(
            game_state_history, action_to_send, 0.0,
            is_invalid=is_invalid or parse_failed,
            current_state=current_state,
        ))
        turn_number += 1

    initial_state = game_state_history[0] if game_state_history else None
    final_state   = game_state_history[-1] if game_state_history else None
    train_reward  = calculator.calculate_episode_reward(
        rewards, final_reward, done, initial_state, final_state,
    )
    initial_dw = initial_state.deadwood if initial_state else 0
    final_dw   = final_state.deadwood if final_state else 0
    _metric_line = (
        f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
        f"Ret:{train_reward:6.2f} EnvR:{final_reward:5.1f} "
        f"DW:{initial_dw:2d}→{final_dw:2d} Inv:{invalid_count}"
    )

    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
            "done":           done,
            "metric_line":    _metric_line,
            "messages":       messages,
        }
    else:
        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
            "done":           done,
            "metric_line":    _metric_line,
            "messages":       messages,
        }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    _log_rank = os.environ.get("LOG_RANK", "0")
    _should_log = _log_rank == "all" or str(_state["rank"]) == _log_rank
    if _should_log:
        print(f"[CURRICULUM/MMD] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "done": False, "metric_line": "[FALLBACK]", "messages": []}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "done": False, "metric_line": "[FALLBACK]", "messages": []}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished  = sum(1 for r in list_results if r.get("done", False))
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0

    _log_trajectories = bool(os.environ.get("LOG_TRAJECTORIES"))
    _batch_lines = [f"[BATCH/MMD] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}"]
    for r in list_results:
        line = r.get("metric_line", "")
        if _log_trajectories:
            line += "\n" + json.dumps(r.get("messages", []))
        _batch_lines.append(line)
    print("\n".join(_batch_lines), flush=True)

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised MMD rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised MMD rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
