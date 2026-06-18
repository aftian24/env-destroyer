import functools
import json
import os
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from threading import Semaphore

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

_SELECTED_GAME = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 4096
_TIMEOUT = 2400
_INVALID_PENALTY = -0.1

# System prompts
_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits \u00d7 (num_players + 1) ranks. For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check (match current bet), "
    "Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "Round 1:\n"
    "- Hold K or Q → call a raise; raise first if unchallenged.\n"
    "- Hold J → fold against a raise; check if unchallenged.\n\n"
    "Round 2 (public card revealed):\n"
    "- You have a PAIR → raise; never fold.\n"
    "- You have K (no pair) → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is K → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is J → check; fold if opponent raises.\n"
    "- You have J (no pair) → check; fold if opponent raises.\n"
)


# ---------------------------------------------------------------------------
# Game state dataclass and parser
# ---------------------------------------------------------------------------

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


@dataclass
class GameState:
    """
    Structured representation of a Leduc Poker observation.

    All fields are parsed directly from the observation text.  Derived
    properties compute common strategy quantities so reward-shaping code
    can stay readable.

    Betting history notes
    ---------------------
    P0 always acts first in each round.  The betting lists record every
    action taken so far in that round, interleaved in turn order:
      r1_betting = ["Raise", "Call"]  → P0 raised, P1 called
      r2_betting = ["Call", "Raise"]  → P0 checked, P1 raised

    When it is our turn, the last entry in the current-round list is always
    the most recent *opponent* action (because the lists only contain
    completed actions — ours is not recorded until after we respond).
    """
    player_id:         int               # 0 or 1
    private_card:      str               # e.g. "K♠"
    private_card_rank: int               # J=1, Q=2, K=3
    public_card:       "str | None"      # None in round 1
    public_card_rank:  "int | None"
    has_pair:          bool              # True when Hand: Pair appears
    round:             int               # 1 or 2
    pot:               int
    our_chips:         int
    opp_chips:         int
    r1_betting:        list[str]         # e.g. ["Raise", "Call"]
    r2_betting:        list[str]         # empty until round 2
    legal_actions:     dict[int, str]    # {0: "Fold", 1: "Call", 2: "Raise"}

    # ---- derived properties ------------------------------------------------

    @property
    def our_invested(self) -> int:
        """Chips we have put into the pot so far (= 100 - our_chips)."""
        return 100 - self.our_chips

    @property
    def opp_invested(self) -> int:
        """Chips opponent has put into the pot so far."""
        return 100 - self.opp_chips

    @property
    def current_round_betting(self) -> list[str]:
        return self.r1_betting if self.round == 1 else self.r2_betting

    @property
    def raises_this_round(self) -> int:
        """Total raises already taken in the current round (by both players)."""
        return self.current_round_betting.count("Raise")

    @property
    def opp_last_action(self) -> "str | None":
        """
        The most recent action the opponent took in the current round, or None
        if the opponent has not yet acted this round.

        Because P0 always leads, the last entry in the current-round betting
        list is always the opponent's most recent move (since it's our turn).
        """
        betting = self.current_round_betting
        return betting[-1] if betting else None

    @property
    def opp_raised_this_round(self) -> bool:
        return "Raise" in self.current_round_betting

    @property
    def can_raise(self) -> bool:
        return 2 in self.legal_actions

    @property
    def can_fold(self) -> bool:
        return 0 in self.legal_actions

    @property
    def hand_strength(self) -> int:
        """
        Rough hand strength on a 1–4 scale.
          4 = Pair (always wins showdown)
          3 = K, no pair
          2 = Q, no pair
          1 = J, no pair (always loses showdown)
        """
        if self.has_pair:
            return 4
        return self.private_card_rank

    @property
    def is_strong(self) -> bool:
        """Pair or K — almost always wins showdown."""
        return self.hand_strength >= 3

    @property
    def is_weak(self) -> bool:
        """J without a pair — almost always loses showdown."""
        return self.hand_strength == 1


def parse_game_state(obs: str) -> "GameState | None":
    """
    Parse a formatted Leduc Poker observation into a ``GameState``.

    Returns ``None`` if the observation cannot be parsed (e.g. empty or
    "Nothing happens" / "Invalid" messages).
    """
    if not obs or "Current State:" not in obs:
        return None

    def _find(pattern, default=None):
        m = re.search(pattern, obs)
        return m.group(1) if m else default

    # Player
    pid_str = _find(r"You are Player (\d+)\.")
    if pid_str is None:
        return None
    player_id = int(pid_str)

    # Private card  ("Your card: K♠")
    private_card = _find(r"Your card:\s*(\S+)")
    if private_card is None:
        return None
    private_card_rank = _CARD_RANK.get(private_card[0], 0)

    # Public card (optional)
    pub_raw = _find(r"Public card:\s*(\S+)")
    public_card      = pub_raw
    public_card_rank = _CARD_RANK.get(pub_raw[0], 0) if pub_raw else None

    # Pair
    has_pair = "Hand: Pair" in obs

    # Round
    round_str = _find(r"Current round:\s*(\d+)/\d+", "1")
    round_ = int(round_str)

    # Chips
    pot      = int(_find(r"Pot size:\s*(\d+)", "0"))
    our_chips = int(_find(r"Your chips:\s*(\d+)", "100"))
    opp_chips = int(_find(r"Opponent chips:\s*(\d+)", "100"))

    # Betting histories
    def _parse_betting(label: str) -> list[str]:
        m = re.search(rf"{label} betting:\s*(.+)", obs)
        if not m:
            return []
        return [a.strip() for a in m.group(1).split(",")]

    r1_betting = _parse_betting("Round 1")
    r2_betting = _parse_betting("Round 2")

    # Legal actions
    legal_actions: dict[int, str] = {}
    for m in re.finditer(r"^(\d+)\s*->\s*(.+)$", obs, re.MULTILINE):
        legal_actions[int(m.group(1))] = m.group(2).strip()

    return GameState(
        player_id=player_id,
        private_card=private_card,
        private_card_rank=private_card_rank,
        public_card=public_card,
        public_card_rank=public_card_rank,
        has_pair=has_pair,
        round=round_,
        pot=pot,
        our_chips=our_chips,
        opp_chips=opp_chips,
        r1_betting=r1_betting,
        r2_betting=r2_betting,
        legal_actions=legal_actions,
    )


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Shaped reward calculator for Leduc Poker training."""

    SIGNALS = {
        "fold_pair":         -2.0,
        "fold_k":            -1.5,
        "fold_kq_r1_raise":  -1.5,
        "fold_q_pubk_raise": -0.5,
        "fold_j_r1_raise":   +0.3,
        "fold_j_r2_raise":   +0.2,
        "fold_q_pubj_raise": +0.2,
        "raise_pair_r2":     +0.3,
        "raise_k_r2":        +0.2,
        "call_kq_r1_raise":  +0.2,
    }

    def __init__(self, terminal_weight: float = 1.0, gamma: float = 0.9):
        self.terminal_weight = terminal_weight
        self.gamma = gamma

    def calculate_step_reward(self, gs: "GameState | None", action_str: str, env_reward: float) -> float:
        reward = 0.0

        if gs is not None:
            pub = gs.public_card_rank or 0

            if action_str == "Fold":
                if gs.has_pair:
                    reward += self.SIGNALS["fold_pair"]
                elif gs.private_card_rank == 3:
                    reward += self.SIGNALS["fold_k"]
                elif gs.round == 1 and gs.opp_last_action == "Raise":
                    if gs.private_card_rank >= 2:
                        reward += self.SIGNALS["fold_kq_r1_raise"]
                    else:
                        reward += self.SIGNALS["fold_j_r1_raise"]
                elif gs.round == 2 and gs.opp_last_action == "Raise":
                    if gs.private_card_rank == 1:
                        reward += self.SIGNALS["fold_j_r2_raise"]
                    elif gs.private_card_rank == 2 and pub == 1:
                        reward += self.SIGNALS["fold_q_pubj_raise"]
                    elif gs.private_card_rank == 2 and pub == 3:
                        reward += self.SIGNALS["fold_q_pubk_raise"]

            elif action_str == "Raise":
                if gs.round == 2 and gs.has_pair:
                    reward += self.SIGNALS["raise_pair_r2"]
                elif gs.round == 2 and gs.private_card_rank == 3 and not gs.has_pair:
                    reward += self.SIGNALS["raise_k_r2"]

            elif action_str in ("Call", "Check"):
                if gs.round == 1 and gs.opp_last_action == "Raise" and gs.private_card_rank >= 2:
                    reward += self.SIGNALS["call_kq_r1_raise"]

        if env_reward != 0.0:
            reward += env_reward * self.terminal_weight

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
        final_max_turn=10,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    """Set up server pool and curriculum once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    _log_rank = os.environ.get("LOG_RANK", "0")
    if _log_rank == "all" or str(rank) == _log_rank:
        print(
            f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
            f"final_max_turn=10, rollouts_per_stage={trainer.args.rollouts_per_stage}"
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

def _format_observation(raw: str) -> str:
    """
    Reformat the server observation to match the eval framework format.

    Server returns:
        # Game Rules\\n...\\n# Current Game State\\nGame: leduc_poker\\n
        You are Player X.\\n\\nCurrent State:\\n...\\nLegal Actions:\\n  N -> ...\\n
        Your choice (action ID only):

    Eval sends:
        Current State:\\n...\\nYour turn to act\\n\\nYou are Player X.\\n
        Legal Actions:\\nN -> ...\\n\\nYour choice (ID only):
    """
    # Extract player number
    player_match = re.search(r"You are Player (\d+)\.", raw)
    player_line  = f"You are Player {player_match.group(1)}." if player_match else ""

    # Find start of "Current State:"
    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw  # unexpected format, pass through

    body = raw[state_start:]

    # Find "Legal Actions:" to split state from actions
    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body

    state_block   = body[:legal_start].rstrip()   # ends with "Your turn to act" (or similar)
    actions_block = body[legal_start:]

    # Remove leading spaces from action lines ("  N -> X" → "N -> X")
    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)

    # Normalise the choice prompt to match eval
    actions_block = actions_block.replace(
        "Your choice (action ID only):", "Your choice (ID only):"
    )

    # Assemble in eval order: state, blank, player, actions
    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n\n".join(parts)


def _parse_action(completion_text: str) -> str:
    """Extract action ID from model output."""
    action = completion_text.strip()
    if action.endswith("</s>"):
        action = action[:-4].strip()
    if "Action:" in action:
        action = action.split("Action:")[-1].strip()
    return action


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
    """
    Run one Leduc Poker episode.

    Reward = final game return (e.g. +1.0 win / -1.0 loss), with a small
    penalty applied per invalid action.

    When ``use_full_prompt=True``, accumulates token IDs across all turns with
    action masking (mask=1 for LLM completions, mask=0 for environment tokens).
    When ``use_full_prompt=False``, only the final turn's token IDs are kept.
    """
    game_id = int(prompt)
    server_idx = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt state (overwritten each turn)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done          = False
    final_reward  = 0.0
    episode_reward = 0.0
    turn_number   = 0
    invalid_count = 0
    use_hints     = random.random() < current_hint_prob
    game_state_history: list[GameState] = []
    calculator    = RewardCalculator()
    rewards:        list[float] = []

    # Reset environment
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id  = result_block.get("episode_id", "")
        observation = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": observation},
    ]

    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (full-prompt mode) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE tokenisers are not context-free: re-tokenising the full
                    # conversation can shift earlier token IDs. Skip delta mask.
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask for this turn."
                    )
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

        # --- Parse and send action ---
        action_to_send = _parse_action(completion_text)
        prev_gs = game_state_history[-1] if game_state_history else None

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            step_reward = step_block.get("reward", 0)
            done        = step_block.get("done", False)
            if not done:
                gs = parse_game_state(observation)
                if gs is not None:
                    game_state_history.append(gs)
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation = ""
            step_reward = 0
            done        = False
            invalid_count += 1
            episode_reward += _INVALID_PENALTY

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count += 1
            episode_reward += _INVALID_PENALTY

        if done:
            final_reward = step_reward

        try:
            action_str = prev_gs.legal_actions.get(int(action_to_send.strip()), "") if prev_gs else ""
        except (ValueError, AttributeError):
            action_str = ""
        step_shaped = calculator.calculate_step_reward(prev_gs, action_str, step_reward if done else 0.0)
        rewards.append(step_shaped)

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    train_reward = calculator.calculate_discounted_return(rewards) + episode_reward
    _metric_line = (
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{:<2} | EnvR:{:>6.2f} | TrainR:{:>6.2f} | Inv:{:<2}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints), final_reward, train_reward, invalid_count,
        )
    )

    # --- Build result ---
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
    """Common dispatch + aggregation logic for both rollout variants."""
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
    max_turns: int = 10,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 10,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
