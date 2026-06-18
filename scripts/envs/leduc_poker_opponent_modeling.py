import functools
import math
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers  # noqa: F401
)


# CONSTANTS FOR LEDUC POKER

_SELECTED_GAME       = "leduc_poker"
_MAX_EPISODE_TOKENS  = 16384  # max tokens per full-prompt episode (16k context window)
_MAX_PROMPT_LEN      = 4096   # prompt token cap — above this end early to prevent OOM
_TIMEOUT             = 2400   # HTTP timeout (seconds) — 40 min covers slow MCTS reset
_MCTS_SIMS           = 50     # fixed MCTS simulations — no progressive ramp for Leduc
_INVALID_PENALTY     = -0.1   # base penalty per invalid/failed action
_CONSEC_INVALID_ESC  =  0.05  # escalation per consecutive invalid (2nd=-0.15, 3rd=-0.20)
_MAX_TURNS           = 10     # max turns per episode (Leduc is 4-8 actions; 10 = safe cap)

# Debug flag — set True for verbose per-step logging during development.
_DEBUG = False

# ReAct format toggle — forces model to reason before committing to action.
# _parse_action() already handles "Action:" prefix so enabling this works immediately.
_USE_REACT_FORMAT = False


# SYSTEM PROMPTS FOR LEDUC POKER

_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits \u00d7 (num_players + 1) ranks. For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check "
    "(match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n"
    "IMPORTANT: Fold is only available when there is a bet to match (opponent raised). "
    "If no one has raised yet, you CANNOT fold \u2014 you can only Check (Call) or Raise.\n\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Guide\n"
    "ROUND 1:\n"
    "- K in hand \u2192 Raise (strongest non-pair; builds pot for potential R2 pair)\n"
    "- Q in hand \u2192 Call (middle hand; wait to see public card)\n"
    "- J in hand \u2192 Call; fold if opponent raises twice (weakest hand, bad pot odds)\n\n"
    "ROUND 2 (public card now visible):\n"
    "- Public card SAME RANK as your card \u2192 PAIR \u2192 always Raise (dominant hand)\n"
    "- No pair + K \u2192 Call opponent raises (K beats Q and J without pair)\n"
    "- No pair + Q \u2192 Call if opponent only called; Fold to raises\n"
    "- No pair + J \u2192 Fold to any Raise (weakest non-pair)\n\n"
    "READING OPPONENT:\n"
    "- Opponent raised R1 then checked R2 \u2192 likely missed pair (caught bluffing)\n"
    "- Opponent raised both rounds \u2192 likely has a pair; be cautious without one\n"
    "- Opponent folded to your raise \u2192 bet was credible; note their threshold\n\n"
    "EXPLOITING THE MCTS OPPONENT (50 simulations, 1 random rollout per node):\n"
    "- Leduc Poker has only 936 total information states; at 50 sims MCTS covers < 10% per decision\n"
    "- MCTS uses random rollouts (not Nash equilibrium) \u2192 it underestimates bluffing value\n"
    "- Random rollouts from any position win ~1/3 of the time \u2192 MCTS sees all positions as similar\n"
    "- Play Nash equilibrium (the strategy guide above) \u2014 it ALWAYS outperforms MCTS pure strategy\n"
    "- Key exploit: MCTS is overly passive with J \u2014 raise with K/Q more than MCTS expects\n"
    "- Key exploit: MCTS folds too rarely vs aggressive raises \u2014 raise more with pairs in R2\n"
    "- MCTS cannot adapt its strategy based on your betting history \u2014 consistent patterns are safe\n"
)

# ReAct format instruction appended to observations when _USE_REACT_FORMAT=True.
# Forces the model to articulate reasoning before choosing an action (chain-of-thought).
_REACT_FORMAT_INSTRUCTIONS = (
    '\n\nYour output must strictly follow this format:\n'
    '"Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
)


# GAME STATE FOR LEDUC POKER

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


@dataclass
class GameState:
    """
    Structured representation of a Leduc Poker observation.

    All fields are parsed directly from the observation text.  Derived
    properties compute common strategy quantities so reward-shaping code
    can stay readable.
    """
    player_id:         int
    private_card:      str
    private_card_rank: int
    public_card:       "str | None"
    public_card_rank:  "int | None"
    has_pair:          bool
    round:             int
    pot:               int
    our_chips:         int
    opp_chips:         int
    r1_betting:        list[str]
    r2_betting:        list[str]
    legal_actions:     dict[int, str]

    @property
    def our_invested(self) -> int:
        return 100 - self.our_chips

    @property
    def opp_invested(self) -> int:
        return 100 - self.opp_chips

    @property
    def current_round_betting(self) -> list[str]:
        return self.r1_betting if self.round == 1 else self.r2_betting

    @property
    def raises_this_round(self) -> int:
        return self.current_round_betting.count("Raise")

    @property
    def opp_last_action(self) -> "str | None":
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
        """4=Pair, 3=K, 2=Q, 1=J."""
        if self.has_pair:
            return 4
        return self.private_card_rank

    @property
    def is_strong(self) -> bool:
        return self.hand_strength >= 3

    @property
    def is_weak(self) -> bool:
        return self.hand_strength == 1


def parse_game_state(obs: str) -> "GameState | None":
    """Parse a formatted Leduc Poker observation into a GameState."""
    if not obs or "Current State:" not in obs:
        return None

    def _find(pattern, default=None):
        m = re.search(pattern, obs)
        return m.group(1) if m else default

    pid_str = _find(r"You are Player (\d+)\.")
    if pid_str is None:
        return None
    player_id = int(pid_str)

    private_card = _find(r"Your card:\s*(\S+)")
    if private_card is None:
        return None
    private_card_rank = _CARD_RANK.get(private_card[0], 0)

    pub_raw          = _find(r"Public card:\s*(\S+)")
    public_card      = pub_raw
    public_card_rank = _CARD_RANK.get(pub_raw[0], 0) if pub_raw else None

    has_pair  = "Hand: Pair" in obs
    round_str = _find(r"Current round:\s*(\d+)/\d+", "1")
    round_    = int(round_str)
    pot       = int(_find(r"Pot size:\s*(\d+)", "0"))
    our_chips = int(_find(r"Your chips:\s*(\d+)", "100"))
    opp_chips = int(_find(r"Opponent chips:\s*(\d+)", "100"))

    def _parse_betting(label: str) -> list[str]:
        m = re.search(rf"{label} betting:\s*(.+)", obs)
        if not m:
            return []
        return [a.strip() for a in m.group(1).split(",")]

    r1_betting = _parse_betting("Round 1")
    r2_betting = _parse_betting("Round 2")

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


# HAND EQUITY ENGINE FOR LEDUC POKER
# Derived from DeepStack-Leduc (lifrordi/DeepStack-Leduc):
#   terminal_equity.lua: get_last_round_call_matrix  (rank comparison)
#   terminal_equity.lua: _set_call_matrix             (R1 board averaging)
#   card_tools.lua:      get_possible_hand_indexes    (board blocking)

_EQUITY_REWARD_SCALE    = 0.4   # scale for DeepStack equity bonus; tune: [0.2, 0.8]
_INTER_TURN_REWARD_SCALE = 0.15  # scale for inter-turn pot-commitment bonus; tune: [0.05, 0.25]

_LEDUC_RANKS = (1, 2, 3)          # J=1, Q=2, K=3
_LEDUC_DECK  = (1, 1, 2, 2, 3, 3) # full 6-card deck, card position → rank


def _compute_hand_equity(private_rank: int, public_rank: "int | None") -> float:
    """
    Compute win-probability (equity) of our private card vs a uniform opponent range.

    Returns float in [0.0, 1.0]:
      1.0 = always wins (pair is strongest possible hand)
      0.5 = break-even
      0.0 = always loses
    """
    if public_rank is not None:
        # Round 2: exact equity given the revealed board card
        we_have_pair = (private_rank == public_rank)
        wins = ties = total = 0
        for opp_rank in _LEDUC_RANKS:
            copies_in_deck = _LEDUC_DECK.count(opp_rank)
            removed   = (1 if opp_rank == private_rank else 0) + \
                        (1 if opp_rank == public_rank  else 0)
            opp_count = max(0, copies_in_deck - removed)
            if opp_count == 0:
                continue
            opp_has_pair = (opp_rank == public_rank)
            if we_have_pair and not opp_has_pair:
                wins += opp_count
            elif not we_have_pair and opp_has_pair:
                pass
            elif private_rank > opp_rank:
                wins += opp_count
            elif private_rank < opp_rank:
                pass
            else:
                ties += opp_count
            total += opp_count
        if total == 0:
            return 0.5
        return (wins + 0.5 * ties) / total
    else:
        # Round 1: average equity over all possible public cards
        total_equity = 0.0
        board_count  = 0
        for board_rank in _LEDUC_RANKS:
            copies_in_deck = _LEDUC_DECK.count(board_rank)
            blocked_by_us  = 1 if board_rank == private_rank else 0
            board_copies   = copies_in_deck - blocked_by_us
            if board_copies <= 0:
                continue
            total_equity += _compute_hand_equity(private_rank, board_rank) * board_copies
            board_count  += board_copies
        if board_count == 0:
            return 0.5
        return total_equity / board_count


# REWARD CALCULATOR FOR LEDUC POKER

class RewardCalculator:
    """Shaped reward calculator for Leduc Poker opponent-modeling training."""

    SIGNALS = {
        "fold_pair":          -2.0,   # folding a pair = surrendering dominant hand
        "fold_k":             -1.5,   # folding K = surrendering strongest non-pair
        "fold_kq_r1_raise":   -1.5,   # folding K or Q to R1 raise = wrong fold
        "fold_q_pubk_raise":  -0.5,   # folding Q when board=K and opp raised = OK
        "fold_j_r1_raise":    +0.3,   # folding J to R1 raise = correct (bad pot odds)
        "fold_j_r2_raise":    +0.2,   # folding J to R2 raise = correct (likely beaten)
        "fold_q_pubj_raise":  +0.2,   # folding Q when board=J and opp raised = careful
        "raise_pair_r2":      +0.3,   # raising with pair in R2 = dominant hand aggression
        "raise_k_r2":         +0.2,   # raising with K in R2 = strong non-pair aggression
        "call_kq_r1_raise":   +0.2,   # calling with K/Q to R1 raise = correct pot odds
        # Gap-fill signals
        "raise_k_r1":         +0.25,  # K in R1 has 70% equity — strongest non-pair; should raise
        "call_pair_r2_raise":  +0.50,  # Pair vs opp R2 raise: near-certain win (90-100%); must call
        "raise_q_r2_pubj":    +0.15,  # Q + board=J in R2: Q beats non-pair J (equity 62.5%)
        # Nash-balanced bluffing signal (added per Lal et al. arXiv 2509.04125):
        # CFR Nash equilibrium for Leduc bluffs ~30% with J in R1. Without any
        # positive reward for the occasional J-raise, training collapses into a
        # never-bluff policy that is exploitable. Small positive nudge keeps the
        # model "balanced" without overpowering the dominant fold-J-to-raise signal.
        "raise_j_r1_balanced": +0.05,  # J in R1 first-action raise (no opp raise yet)
    }

    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma  # 0.9 = standard discount for short LP episodes

    def calculate_step_reward(
        self, gs: "GameState | None", action_str: str, env_reward: float
    ) -> float:
        reward = 0.0

        if gs is not None:
            pub = gs.public_card_rank or 0

            if action_str == "Fold":
                if gs.has_pair:
                    reward += self.SIGNALS["fold_pair"]
                elif gs.private_card_rank == 3:
                    reward += self.SIGNALS["fold_k"]
                elif gs.round == 1 and gs.opp_last_action == "Raise":
                    reward += (
                        self.SIGNALS["fold_kq_r1_raise"]
                        if gs.private_card_rank >= 2
                        else self.SIGNALS["fold_j_r1_raise"]
                    )
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
                elif gs.round == 2 and gs.private_card_rank == 2 and pub == 1 and not gs.has_pair:
                    reward += self.SIGNALS["raise_q_r2_pubj"]
                elif gs.round == 1 and gs.private_card_rank == 3 and not gs.has_pair:
                    reward += self.SIGNALS["raise_k_r1"]
                elif (
                    gs.round == 1
                    and gs.private_card_rank == 1
                    and not gs.has_pair
                    and gs.opp_last_action != "Raise"
                ):
                    # First-action bluff with J (not in response to opp raise)
                    reward += self.SIGNALS["raise_j_r1_balanced"]
                if gs.has_pair:
                    reward += 1.5  # extra pair aggression bonus — pair always dominates

            elif action_str in ("Call", "Check"):
                if gs.round == 1 and gs.opp_last_action == "Raise" and gs.private_card_rank >= 2:
                    reward += self.SIGNALS["call_kq_r1_raise"]
                if gs.round == 2 and gs.has_pair and gs.opp_last_action == "Raise":
                    reward += self.SIGNALS["call_pair_r2_raise"]

        # DeepStack equity bonus (additive — does not replace any SIGNAL)
        if gs is not None and action_str in ("Raise", "Call", "Check", "Fold"):
            equity = _compute_hand_equity(gs.private_card_rank, gs.public_card_rank)
            if action_str == "Raise":
                reward += _EQUITY_REWARD_SCALE * (equity - 0.5)        # bonus ∝ equity above break-even
            elif action_str == "Fold":
                reward -= _EQUITY_REWARD_SCALE * (equity - 0.5)        # penalty for folding high equity
            else:
                reward += _EQUITY_REWARD_SCALE * 0.5 * (equity - 0.5) # small signal for call/check

        if env_reward != 0.0:
            reward += max(min(env_reward * 30.0, 30.0), -30.0)  # scale ×30, clip [-30, 30]

        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))  # γ^(T-1-i)

    def calculate_pot_commitment_bonus(
        self, gs: "GameState | None", action_str: str
    ) -> float:
        """
        Per-turn intermediate reward based on pot-odds awareness.

        G.O.D environment_tasks.md: 'Finding a way to give the model intermediate
        reward will greatly improve training.' (default miner gives 0 until game ends)
        """
        if gs is None or not action_str:
            return 0.0

        bonus  = 0.0
        equity = _compute_hand_equity(gs.private_card_rank, gs.public_card_rank)

        pot_proxy         = max(1, gs.pot) if gs.pot else 2
        required_fraction = 1.0 / (pot_proxy + 1)  # rough pot-odds fraction
        surplus           = equity - required_fraction

        if action_str == "Raise":
            bonus += _INTER_TURN_REWARD_SCALE * max(0.0, surplus)
        elif action_str == "Fold":
            bonus += _INTER_TURN_REWARD_SCALE * max(0.0, -surplus)
        else:
            bonus += _INTER_TURN_REWARD_SCALE * 0.3 * surplus

        # Extra R2 pair aggression nudge
        if gs.round == 2 and gs.has_pair and action_str == "Raise":
            bonus += _INTER_TURN_REWARD_SCALE * 0.5  # 0.15 * 0.5 = 0.075

        # Small bonus for producing any valid action string
        if action_str in ("Raise", "Call", "Check", "Fold"):
            bonus += _INTER_TURN_REWARD_SCALE * 0.1  # 0.15 * 0.1 = 0.015

        return bonus


# OBSERVATION FORMATTER AND ACTION PARSER FOR LEDUC POKER

def _format_observation(raw: str) -> str:
    """Reformat server observation to match eval framework format."""
    player_match = re.search(r"You are Player (\d+)\.", raw)
    player_line  = f"You are Player {player_match.group(1)}." if player_match else ""

    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw
    body = raw[state_start:]

    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body

    state_block   = body[:legal_start].rstrip()
    actions_block = body[legal_start:]

    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)
    actions_block = actions_block.replace(
        "Your choice (action ID only):", "Your choice (ID only):"
    )

    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n\n".join(parts)


def _parse_action(
    completion_text: str, legal_action_map: "dict[int, str] | None" = None
) -> str:
    """
    Extract action ID from model output.

    Tries in order:
    1. Strip Action: prefix (ReAct format)
    2. Find valid integer in output
    3. Keyword fallback: fold / call / check / raise  (reduces invalids when model
       returns words instead of numbers — adopted from user's implementation)
    """
    action = completion_text.strip()
    if action.endswith("</s>"):
        action = action[:-4].strip()
    if "Action:" in action:
        action = action.split("Action:")[-1].strip()

    # Try to find first number matching a legal action
    nums = re.findall(r"-?\d+", action)
    if nums:
        if legal_action_map is not None:
            for n in nums:
                if int(n) in legal_action_map:
                    return n
        else:
            return nums[0]

    # Keyword fallback — handles model outputting "fold", "call", "raise", "check"
    if legal_action_map is not None:
        normalized = action.strip().lower()
        keyword_map: dict[str, int] = {}
        for aid, label in legal_action_map.items():
            ll = label.lower()
            if "fold"  in ll: keyword_map["fold"]  = aid
            if "call"  in ll: keyword_map["call"]  = aid
            if "check" in ll: keyword_map["check"] = aid
            if "raise" in ll: keyword_map["raise"] = aid
        for kw, aid in keyword_map.items():
            if kw in normalized:
                return str(aid)

    return action


def _select_fallback_action(gs: "GameState | None") -> str:
    """
    Nash-based fallback when LLM output cannot be parsed.

    Used instead of sending empty/invalid string to env server.
    Strategy:
    - Pair             → Call  (never fold a dominant hand)
    - K without pair   → Call  (strong non-pair, stay in)
    - J/Q + opp raised → Fold if available  (bad pot odds)
    - Default          → Call  (conservative, minimize loss)
    """
    if gs is None or not gs.legal_actions:
        return "0"

    fold_id  = next((str(k) for k, v in gs.legal_actions.items() if "Fold"  in v), None)
    call_id  = next((str(k) for k, v in gs.legal_actions.items() if "Call"  in v or "Check" in v), None)

    default = call_id or str(min(gs.legal_actions.keys()))

    if gs.has_pair:
        return call_id or default          # pair: never fold

    if gs.private_card_rank == 3:
        return call_id or default          # K: strong non-pair, call

    if gs.private_card_rank <= 2 and gs.opp_raised_this_round and fold_id:
        return fold_id                     # J or Q vs raise: fold (bad pot odds)

    return default


def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    """
    Robust terminal reward extraction — tries multiple server response formats.

    Attempts in order:
    1. info.cumulative_reward  (preferred API field)
    2. 'Your Return:' in observation text
    3. 'Normalized Score:' + optional 'Result:' qualifier
    4. step_block['reward']    (basic fallback)
    """
    def _clamp(v: float) -> float:
        return max(-1.0, min(1.0, v))

    info       = step_block.get("info", {}) if isinstance(step_block, dict) else {}
    cumulative = info.get("cumulative_reward")
    if isinstance(cumulative, (int, float)) and not math.isnan(float(cumulative)):
        return _clamp(float(cumulative))

    m = re.search(r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    if m:
        return _clamp(float(m.group(1)))

    m_norm   = re.search(r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    m_result = re.search(r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", re.IGNORECASE)
    if m_norm:
        val = float(m_norm.group(1))
        if m_result:
            res = m_result.group(1).upper()
            val = (-abs(val) if val != 0 else -1.0) if res == "LOSS" \
                  else (abs(val) if val != 0 else 1.0) if res == "WIN" \
                  else 0.0
        return _clamp(val)

    return _clamp(float(step_block.get("reward", 0.0)))


# MODULE STATE AND INITIALIZATION FOR LEDUC POKER

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,      # from training args (env_configs default)
        final_max_turn=_MAX_TURNS,                   # 10 = full LP episode upper bound
        rollouts_per_stage=args.rollouts_per_stage,  # from training args (default: 1280)
        initial_hint_prob=0.75,  # 75% episodes start with strategy hints (LP needs more guidance)
        final_hint_prob=0.15,    # 15% retention — keeps MCTS exploit knowledge fresh post-curriculum
        warmup_rollouts=args.rollouts_per_stage,     # 1 stage warmup before curriculum progresses
    )


def _ensure_initialized(trainer) -> None:
    """Set up server pool and curriculum once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],  # first valid leduc_poker task ID
        "seed": 42,               # fixed seed for init ping (reproducible health check)
        "opponent": "mcts",
        "mcts_max_simulations": _MCTS_SIMS,  # 50 = strong but not too slow for training
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Leduc-OPP initialized: "
        f"initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={_MAX_TURNS}, "
        f"rollouts_per_stage={trainer.args.rollouts_per_stage}, "
        f"mcts_sims={_MCTS_SIMS} (fixed)"
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


# CORE EPISODE RUNNER FOR LEDUC POKER

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
) -> "tuple[int, dict | None]":
    """
    Run one Leduc Poker episode against a fixed MCTS(50, 1) opponent.

    Reward = discounted shaped return (per-step strategy signals + DeepStack
    equity bonus + inter-turn pot-commitment bonus + terminal env reward ×30)
    plus episode-level invalid-action penalty (escalating for consecutive invalids).

    When use_full_prompt=True, accumulates token IDs across all turns with
    action masking (mask=1 for LLM completions, 0 for env tokens).
    When use_full_prompt=False, only the final turn is kept.
    """
    game_id      = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation
    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: "list[int] | None"   = None

    # Last-prompt state (overwritten every turn)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done                 = False
    final_reward         = 0.0
    episode_reward       = 0.0
    turn_number          = 0
    invalid_count        = 0
    consecutive_invalids = 0  # tracks back-to-back invalids for escalating penalty
    use_hints            = random.random() < current_hint_prob  # 75%→0% via curriculum

    game_state_history: list[GameState] = []
    calculator = RewardCalculator()
    rewards:    list[float] = []

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,          # deterministic per game_id for reproducibility
        "opponent": "mcts",
        "mcts_max_simulations": _MCTS_SIMS,  # 50 fixed
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(
            f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT
        )
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id   = result_block.get("episode_id", "")
        observation  = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
    except Exception as exc:
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")

    if _USE_REACT_FORMAT:
        observation = observation + _REACT_FORMAT_INSTRUCTIONS

    if _DEBUG:
        print(f"[DEBUG] Game {game_id} reset OK. use_hints={use_hints}, "
              f"react_format={_USE_REACT_FORMAT}, observation_len={len(observation)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": observation},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            try:
                rollout_outputs = generate_rollout_completions(
                    trainer, prompts=[messages], as_chat=True
                )[0]
            except Exception as exc:
                print(
                    f"Warning: vLLM error at turn {turn_number} "
                    f"(game {game_id}): {type(exc).__name__}: {exc}"
                )
                done = True
                break

        prompt_ids      = rollout_outputs.get("prompt_ids", [])
        completion_ids  = rollout_outputs.get("completion_ids", [])
        logprobs        = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (full-prompt mode) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:  # 4096 token safety cap
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

        # --- Parse action with keyword fallback ---
        prev_gs        = game_state_history[-1] if game_state_history else None
        legal_map      = prev_gs.legal_actions if prev_gs else None
        action_to_send = _parse_action(completion_text, legal_map)

        # Fallback: if parse produced invalid/missing action, use Nash-based selection
        parse_ok = (
            action_to_send
            and legal_map is not None
            and any(action_to_send == str(k) for k in legal_map)
        )
        if not parse_ok:
            action_to_send = _select_fallback_action(prev_gs)
            consecutive_invalids += 1
            invalid_count += 1
            # Escalating penalty: -0.10, -0.15, -0.20, ... (capped at consecutive_invalids)
            penalty = _INVALID_PENALTY + _CONSEC_INVALID_ESC * max(0, consecutive_invalids - 1)
            episode_reward += penalty
        else:
            consecutive_invalids = 0  # reset on valid action

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
            step_block  = {"reward": 0.0, "done": False}
            invalid_count  += 1
            episode_reward += _INVALID_PENALTY

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count  += 1
            episode_reward += _INVALID_PENALTY

        if done:
            # Robust terminal reward extraction (tries multiple server formats)
            final_reward = _extract_terminal_reward(step_block, observation)

        try:
            action_str = (
                prev_gs.legal_actions.get(int(action_to_send.strip()), "")
                if prev_gs else ""
            )
        except (ValueError, AttributeError):
            action_str = ""

        step_shaped  = calculator.calculate_step_reward(prev_gs, action_str, step_reward if done else 0.0)
        inter_turn   = calculator.calculate_pot_commitment_bonus(prev_gs, action_str)
        rewards.append(step_shaped + inter_turn)

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    train_reward = calculator.calculate_discounted_return(rewards) + episode_reward
    print(
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{:<2} | EnvR:{:>6.2f} | "
        "TrainR:{:>6.2f} | Inv:{:<2} | MCTS:{}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints),
            final_reward, train_reward, invalid_count, _MCTS_SIMS,
        )
    )

    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:  # 16384 hard cap
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
        }
    return index, {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "logprobs":       logprobs,
        "reward":         train_reward,
        "final_score":    final_reward,
    }


# PUBLIC ROLLOUT FUNCTIONS FOR LEDUC POKER

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    """Common dispatch + aggregation logic for both rollout variants."""
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, "
        f"mcts_sims={_MCTS_SIMS} (fixed)"
    )

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
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0],
         "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0],
         "reward": 0.0, "final_score": 0.0}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        try:
            idx, res = f.result()
        except Exception as exc:
            print(f"[ERROR] Game thread threw unhandled exception: {type(exc).__name__}: {exc}")
            continue
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    wins       = sum(1 for r in list_results if r["final_score"] > 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0.0
    print(
        f"[BATCH] Finished: {finished}/{len(list_results)}, "
        f"Wins: {wins}/{len(list_results)}, "
        f"AvgReturn: {avg_return:.3f}"
    )

    # WandB metrics (best-effort — no crash if wandb not active)
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.log(
                {
                    "env/win_rate":         wins / len(list_results) if list_results else 0.0,
                    "env/avg_return":       avg_return,
                    "curriculum/max_turn":  current_max_turn,
                    "curriculum/mcts_sims": _MCTS_SIMS,
                    "curriculum/hint_prob": current_hint_prob,
                    "curriculum/rollouts":  curriculum.total_rollouts,
                },
                commit=False,
            )
    except Exception:
        pass

    out: dict = {
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
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
