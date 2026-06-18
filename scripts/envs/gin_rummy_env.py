import functools
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import as_completed
from dataclasses import dataclass
from threading import Semaphore
from typing import Optional

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "gin_rummy"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 5000
_TIMEOUT = 2400

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]


# ---------------------------------------------------------------------------
# Card utilities
# ---------------------------------------------------------------------------

def get_rank(card: str) -> str:
    return card[0]

def get_suit(card: str) -> str:
    return card[1]

def get_value(card: str) -> int:
    return CARD_VALUES[get_rank(card)]


def find_potential_runs(hand: list[str], additional_card: Optional[str] = None) -> list[list[str]]:
    test_hand = hand.copy()
    if additional_card:
        test_hand.append(additional_card)
    suit_groups: dict[str, list[str]] = {}
    for card in test_hand:
        suit_groups.setdefault(get_suit(card), []).append(card)
    runs = []
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_ORDER.index(get_rank(sorted_cards[j])) == RANK_ORDER.index(get_rank(run[-1])) + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            if len(run) >= 2:
                runs.append(run)
            i = j if len(run) > 1 else i + 1
    return runs


def count_complete_runs(hand: list[str]) -> int:
    return sum(1 for r in find_potential_runs(hand) if len(r) >= 3)


def would_complete_run(hand: list[str], card: str) -> bool:
    current = sum(1 for r in find_potential_runs(hand) if len(r) >= 3)
    return sum(1 for r in find_potential_runs(hand, card) if len(r) >= 3) > current


def would_improve_run(hand: list[str], card: str) -> bool:
    rank_idx = RANK_ORDER.index(get_rank(card))
    suit = get_suit(card)
    for existing in hand:
        if get_suit(existing) != suit:
            continue
        if abs(rank_idx - RANK_ORDER.index(get_rank(existing))) == 1:
            for run in find_potential_runs(hand + [card]):
                if card in run and len(run) >= 3:
                    return True
    return False


def would_complete_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) >= 2


def would_improve_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) == 1


def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(rf"<{tag_name}>.*?</{close_name}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hand: list[str]
    deadwood: int
    phase: str
    knock_card: int
    upcard: str
    stock_size: int
    discard_pile: list[str]
    player_id: int

    def total_hand_value(self) -> int:
        return sum(get_value(c) for c in self.hand)

    def num_high_cards(self) -> int:
        return sum(1 for c in self.hand if get_value(c) == 10)

    def can_knock(self) -> bool:
        return self.deadwood <= self.knock_card

    def count_pairs(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 2)

    def count_sets(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 3)

    def count_runs(self) -> int:
        return count_complete_runs(self.hand)

    def count_potential_runs(self) -> int:
        return sum(1 for r in find_potential_runs(self.hand) if len(r) == 2)


def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n(.*)', obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    current_state_text, legal_action_text = state_text.split('Legal Actions:')
    return current_state_text + f"You are Player {player_id}.\nLegal Actions:" + legal_action_text


def parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    section = re.search(
        rf'Player{player_id}: Deadwood=\d+\n\+-+\+\n(.*?)\n\+-+\+', observation, re.DOTALL
    )
    hand = []
    if section:
        for row in section.group(1).strip().split('\n'):
            hand.extend(re.findall(r'([A2-9TJQK][shdc])', row))
    return hand


def parse_discard_pile(observation: str) -> list[str]:
    m = re.search(r'Discard pile: (.*?)\n', observation)
    if not m:
        return []
    pile_str = m.group(1).strip()
    if not pile_str:
        return []
    if ' ' in pile_str:
        return pile_str.split()
    return [pile_str[i:i+2] for i in range(0, len(pile_str), 2)]


def parse_game_state(observation: str) -> GameState:
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response — not a game state")
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    hand = parse_hand_from_observation(observation)
    dw_match = re.search(r'Deadwood=(\d+)', observation)
    deadwood = int(dw_match.group(1)) if dw_match else 0
    phase_match = re.search(r'Phase: (\w+)', observation)
    phase = phase_match.group(1) if phase_match else 'Draw'
    knock_match = re.search(r'Knock card: (\d+)', observation)
    knock_card = int(knock_match.group(1)) if knock_match else 10
    upcard_match = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard = upcard_match.group(1) if upcard_match else 'XX'
    stock_match = re.search(r'Stock size: (\d+)', observation)
    stock_size = int(stock_match.group(1)) if stock_match else 0
    return GameState(
        hand=hand, deadwood=deadwood, phase=phase, knock_card=knock_card,
        upcard=upcard, stock_size=stock_size,
        discard_pile=parse_discard_pile(observation), player_id=player_id,
    )


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.deadwood_weight = 0.5
        self.high_card_penalty = -0.2
        self.pair_bonus = 2.0
        self.set_bonus = 10.0
        self.potential_run_bonus = 3.0
        self.run_bonus = 12.0
        self.break_pair_penalty = -2.0
        self.break_set_penalty = -10.0
        self.break_run_penalty = -12.0
        self.knock_ready_bonus = 20.0
        self.discard_useful_penalty = -2.0
        self.missed_opportunity_penalty = -3.0
        self.picked_up_useless_upcard_penalty = -3.0

    def calculate_step_reward(self, states: list[GameState], action: str, env_reward: float) -> float:
        if len(states) < 2:
            return 0.0
        prev, curr = states[-2], states[-1]
        reward = 0.0
        reward += self.deadwood_weight * (prev.deadwood - curr.deadwood)
        reward += self.high_card_penalty * curr.num_high_cards()
        pair_change = curr.count_pairs() - prev.count_pairs()
        reward += (self.pair_bonus if pair_change > 0 else self.break_pair_penalty) * abs(pair_change) if pair_change != 0 else 0
        set_change = curr.count_sets() - prev.count_sets()
        reward += (self.set_bonus if set_change > 0 else self.break_set_penalty) * abs(set_change) if set_change != 0 else 0
        run_change = curr.count_runs() - prev.count_runs()
        reward += (self.run_bonus if run_change > 0 else self.break_run_penalty) * abs(run_change) if run_change != 0 else 0
        potential_run_change = curr.count_potential_runs() - prev.count_potential_runs()
        if potential_run_change > 0:
            reward += self.potential_run_bonus * potential_run_change
        if curr.can_knock() and not prev.can_knock():
            reward += self.knock_ready_bonus
        if prev.phase == 'Discard' and len(curr.discard_pile) > len(prev.discard_pile):
            newly_discarded = [c for c in curr.discard_pile if c not in prev.discard_pile]
            if newly_discarded:
                dc = newly_discarded[0]
                if sum(1 for c in prev.hand if get_rank(c) == get_rank(dc)) >= 2:
                    reward += self.discard_useful_penalty
                if would_improve_run(prev.hand, dc):
                    reward += self.discard_useful_penalty
        if prev.phase == 'Draw' and prev.upcard != 'XX':
            upcard = prev.upcard
            if action == '53':
                if would_complete_set(prev.hand, upcard) or would_complete_run(prev.hand, upcard):
                    reward += self.missed_opportunity_penalty
            else:
                if not (would_complete_set(prev.hand, upcard) or would_complete_run(prev.hand, upcard)):
                    reward += self.picked_up_useless_upcard_penalty
        if env_reward != 0.0:
            reward += max(min(env_reward * 100.0, 50.0), -50.0)
        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))


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
        warmup_rollouts=args.rollouts_per_stage,
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
            f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
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
# Core episode runner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(\u2660), h(\u2665), d(\u2666), c(\u2663)\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood \u2264 knock_card\n\n"
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
    "- Knock when you have \u226410 deadwood points and think you're ahead\n"
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

    # Last-prompt fallback (overwritten every loop iteration in use_full_prompt=False mode)
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
        "task_id": game_id, "seed": game_id,
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

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation ---
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

        # --- Step environment ---
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

        # --- Reward calculation ---
        if not is_invalid and not done:
            try:
                game_state = parse_game_state(formatted_observation)
            except Exception as exc:
                print(f"Failed to parse game state: {exc}")
                immediate_reward = -10.0
            else:
                game_state_history.append(game_state)
                immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0)
        elif is_invalid:
            immediate_reward = -10.0
        else:
            immediate_reward = max(min((step_reward - 0.5) * 100.0, 50.0), -50.0)

        rewards.append(immediate_reward)
        turn_number += 1

    train_reward = calculator.calculate_discounted_return(rewards)
    initial_dw = game_state_history[0].deadwood if game_state_history else 0
    final_dw   = game_state_history[-1].deadwood if game_state_history else 0
    _metric_line = (
        f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
        f"Ret:{train_reward:6.2f} EnvR:{final_reward:5.1f} "
        f"DW:{initial_dw:2d}\u2192{final_dw:2d} Inv:{invalid_count}"
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
        print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

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
    _batch_lines = [f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}"]
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
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
