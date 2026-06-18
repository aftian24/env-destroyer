import functools
import json
import os
import random
import re
from collections import Counter, defaultdict
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
_MAX_PROMPT_LEN = 16384 - 256
_TIMEOUT = 2400
_MCTS_SIMS = 50  # bumped from 25 — Affinetes eval uses MCTS-500/10 per agents/gin_rummy.py

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]

# Reward constants
TERMINAL_WIN_REWARD  = 1.0
TERMINAL_LOSS_REWARD = -1.0
GIN_BONUS            = 0.25
KNOCK_BONUS          = 0.1
DEADWOOD_WEIGHT      = 0.5
INVALID_PENALTY      = -0.1
INVALID_TOTAL_CLIP   = -0.3
TERMINAL_REWARD_CLIP = 1.0
SAFE_DISCARD_BONUS        = 0.02
DANGEROUS_DISCARD_PENALTY = 0.02
DRAW_UPCARD_BONUS   = 0.03
DRAW_UPCARD_PENALTY = 0.02


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


# ---------------------------------------------------------------------------
# DP optimal deadwood
# ---------------------------------------------------------------------------

def find_all_melds(hand: list[str]) -> list[frozenset[str]]:
    """Enumerate every valid meld (SET or RUN of 3+ cards) from the given hand."""
    melds: list[frozenset[str]] = []

    rank_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        rank_groups[get_rank(card)].append(card)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            melds.append(frozenset(cards[:3]))
        if len(cards) >= 4:
            melds.append(frozenset(cards[:4]))

    suit_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        suit_groups[get_suit(card)].append(card)
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
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def compute_optimal_deadwood(hand: list[str]) -> int:
    """Minimum deadwood via bitmask DP backtracking."""
    if not hand:
        return 0
    melds = find_all_melds(hand)
    n = len(hand)
    card_to_idx = {card: i for i, card in enumerate(hand)}
    meld_masks: list[int] = []
    for meld in melds:
        mask = 0
        valid = True
        for card in meld:
            if card not in card_to_idx:
                valid = False
                break
            mask |= (1 << card_to_idx[card])
        if valid:
            meld_masks.append(mask)

    card_values_list = [get_value(card) for card in hand]
    memo: dict[int, int] = {}

    def _dp(used_mask: int) -> int:
        if used_mask in memo:
            return memo[used_mask]
        base_dw = sum(card_values_list[i] for i in range(n) if not (used_mask >> i & 1))
        best = base_dw
        for mm in meld_masks:
            if (mm & used_mask) == 0:
                best = min(best, _dp(used_mask | mm))
        memo[used_mask] = best
        return best

    return _dp(0)


def meld_potential(upcard: str, hand: list[str]) -> int:
    """Estimate deadwood reduction from drawing the upcard."""
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    dw_with    = compute_optimal_deadwood(hand + [upcard])
    dw_without = compute_optimal_deadwood(hand)
    return max(0, dw_without - dw_with)


def draw_ucb_shaping(current_state: "GameState", chosen_action_id: str) -> float:
    """UCB-inspired draw decision shaping (last-prompt mode only)."""
    if current_state.phase not in ('Draw', 'FirstUpcard'):
        return 0.0
    if current_state.upcard == 'XX' or not current_state.hand:
        return 0.0
    if chosen_action_id not in ('52', '53'):
        return 0.0
    potential = meld_potential(current_state.upcard, current_state.hand)
    if chosen_action_id == '52':
        if potential > 0:
            scale = min(potential / 10.0, 1.0)
            return DRAW_UPCARD_BONUS * scale
        else:
            return -DRAW_UPCARD_PENALTY
    return 0.0


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hand:         list[str]
    deadwood:     int
    phase:        str
    knock_card:   int
    upcard:       str
    stock_size:   int
    discard_pile: list[str]
    player_id:    int

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


# ---------------------------------------------------------------------------
# Dead card tracker
# ---------------------------------------------------------------------------

class DeadCardTracker:
    """Tracks discarded cards and identifies layoff candidates."""

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self.seen_discards: set[str] = set()
        self.opponent_melds: list[list[str]] = []

    def update_from_discard_pile(self, discard_pile: list[str]) -> None:
        for card in discard_pile:
            if len(card) == 2:
                self.seen_discards.add(card.lower())

    def update_from_observation(self, obs: str) -> None:
        pile = parse_discard_pile(obs)
        self.update_from_discard_pile(pile)

    def get_dead_cards(self) -> list[str]:
        return sorted(self.seen_discards)

    def is_dead(self, card: str) -> bool:
        return card.lower() in self.seen_discards

    def get_layoff_candidates(self, hand: list[str], discard_pile: list[str]) -> list[str]:
        if not discard_pile or not hand:
            return []
        candidates: set[str] = set()

        suit_groups: dict[str, list[str]] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            suit = card[1].lower()
            suit_groups.setdefault(suit, []).append(card.lower())

        for suit, cards in suit_groups.items():
            sorted_cards = sorted(
                cards,
                key=lambda c: self.ALL_RANKS.index(c[0].upper()) if c[0].upper() in self.ALL_RANKS else 99,
            )
            for i in range(len(sorted_cards) - 1):
                r1 = sorted_cards[i][0].upper()
                r2 = sorted_cards[i + 1][0].upper()
                if r1 not in self.ALL_RANKS or r2 not in self.ALL_RANKS:
                    continue
                idx1 = self.ALL_RANKS.index(r1)
                idx2 = self.ALL_RANKS.index(r2)
                if abs(idx1 - idx2) == 1:
                    for adj in [idx1 - 1, idx2 + 1]:
                        if 0 <= adj < len(self.ALL_RANKS):
                            target = self.ALL_RANKS[adj] + suit
                            for hcard in hand:
                                if hcard.lower() == target:
                                    candidates.add(hcard)

        rank_groups: dict[str, int] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            rank = card[0].upper()
            rank_groups[rank] = rank_groups.get(rank, 0) + 1
        for rank, count in rank_groups.items():
            if count >= 2:
                for hcard in hand:
                    if hcard[0].upper() == rank:
                        candidates.add(hcard)

        return sorted(candidates)

    def summary(self, hand: list[str]) -> str:
        dead   = self.get_dead_cards()
        layoff = self.get_layoff_candidates(hand, list(self.seen_discards))
        lines  = []
        if dead:
            lines.append(f"Dead cards (discarded): {' '.join(dead[:15])}")
        if layoff:
            lines.append(f"Layoff candidates (extend opp melds): {' '.join(layoff)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bayesian opponent meld model
# ---------------------------------------------------------------------------

class BayesianOpponentModel:
    """Infers opponent's meld direction from discard pile deltas."""

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self.rank_heat: dict[str, float] = {r: 0.0 for r in self.ALL_RANKS}
        self.suit_heat: dict[str, float] = {s: 0.0 for s in self.ALL_SUITS}
        self.opp_draws:    list[str] = []
        self.opp_discards: list[str] = []

    def _update_heat(self, card: str, weight: float) -> None:
        if len(card) != 2:
            return
        rank = card[0].upper()
        suit = card[1].lower()
        if rank in self.rank_heat:
            self.rank_heat[rank] = max(-3.0, min(3.0, self.rank_heat[rank] + weight))
        if suit in self.suit_heat:
            self.suit_heat[suit] = max(-3.0, min(3.0, self.suit_heat[suit] + weight))

    def update_on_opponent_draw(self, drawn_card: str) -> None:
        self.opp_draws.append(drawn_card)
        self._update_heat(drawn_card, weight=1.0)
        if len(drawn_card) == 2:
            rank, suit = drawn_card[0].upper(), drawn_card[1].lower()
            if rank in self.ALL_RANKS:
                idx = self.ALL_RANKS.index(rank)
                for adj_idx in [idx - 1, idx + 1]:
                    if 0 <= adj_idx < len(self.ALL_RANKS):
                        self._update_heat(self.ALL_RANKS[adj_idx] + suit, weight=0.3)

    def update_on_opponent_discard(self, discarded_card: str) -> None:
        self.opp_discards.append(discarded_card)
        self._update_heat(discarded_card, weight=-0.5)

    def update_from_discard_pile_delta(
        self, prev_discard_pile: list[str], curr_discard_pile: list[str]
    ) -> None:
        prev_set = list(prev_discard_pile)
        curr_set = list(curr_discard_pile)
        if prev_set and (not curr_set or len(curr_set) < len(prev_set)):
            top_card = prev_set[-1] if prev_set else None
            if top_card:
                self.update_on_opponent_draw(top_card)
        elif curr_set and len(curr_set) > len(prev_set):
            self.update_on_opponent_discard(curr_set[-1])

    def is_dangerous_discard(self, card: str) -> bool:
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        return self.rank_heat.get(rank, 0.0) >= 1.0 or self.suit_heat.get(suit, 0.0) >= 1.5

    def is_safe_discard(self, card: str) -> bool:
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        return self.rank_heat.get(rank, 0.0) <= -0.5 and self.suit_heat.get(suit, 0.0) <= -0.5

    def get_danger_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_dangerous_discard(c)]

    def get_safe_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_safe_discard(c)]

    def summary(self, hand: list[str]) -> str:
        danger    = self.get_danger_cards(hand)
        safe      = self.get_safe_cards(hand)
        hot_suits = [s for s, h in self.suit_heat.items() if h >= 1.5]
        hot_ranks = [r for r, h in self.rank_heat.items() if h >= 1.0]
        lines = []
        if hot_suits:
            lines.append(f"[Bayesian] Opp likely building: {', '.join(hot_suits).upper()} suit(s)")
        if hot_ranks:
            lines.append(f"[Bayesian] Opp interested in rank(s): {' '.join(hot_ranks)}")
        if danger:
            lines.append(f"Dangerous discards (may complete opp meld): {' '.join(danger[:6])}")
        if safe:
            lines.append(f"Safer discards (opp de-prioritized): {' '.join(safe[:6])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bayesian opponent hand distribution
# ---------------------------------------------------------------------------

class BayesianOpponentHandModel:
    """Tracks P(card ∈ opponent_hand | all observations) via Bayesian updates."""

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self._prob: dict[str, float] = {}
        self._opp_hand_size: int = 10
        self._confirmed_in_hand:     set[str] = set()
        self._confirmed_not_in_hand: set[str] = set()

    def _all_cards(self) -> list[str]:
        return [r + s for r in self.ALL_RANKS for s in self.ALL_SUITS]

    def initialize(self, our_hand: list[str], discard_pile: list[str]) -> None:
        known_out     = set(c.lower() for c in our_hand) | set(c.lower() for c in discard_pile)
        unknown_cards = [c for c in self._all_cards() if c not in known_out]
        n             = len(unknown_cards)
        p_in_hand     = self._opp_hand_size / max(n, self._opp_hand_size)
        self._prob    = {c: p_in_hand for c in unknown_cards}
        for c in known_out:
            self._prob[c] = 0.0

    def update_opp_drew_upcard(self, upcard: str) -> None:
        card = upcard.lower()
        if len(card) == 2:
            self._prob[card] = 1.0
            self._confirmed_in_hand.add(card)
            self._renormalize(exclude={card})

    def update_opp_drew_stock(self) -> None:
        stock_candidates = [
            c for c, p in self._prob.items()
            if p > 0 and c not in self._confirmed_in_hand and c not in self._confirmed_not_in_hand
        ]
        if not stock_candidates:
            return
        n = len(stock_candidates)
        boost = 1.0 / n
        for c in stock_candidates:
            self._prob[c] = min(1.0, self._prob[c] + boost * 0.1)

    def update_opp_discarded(self, card: str) -> None:
        c = card.lower()
        if len(c) == 2:
            self._prob[c] = 0.0
            self._confirmed_not_in_hand.add(c)
            self._renormalize(exclude={c})

    def _renormalize(self, exclude: set[str]) -> None:
        uncertain = {
            c: p for c, p in self._prob.items()
            if c not in exclude
            and c not in self._confirmed_in_hand
            and c not in self._confirmed_not_in_hand
            and p > 0
        }
        confirmed_count  = len(self._confirmed_in_hand)
        remaining_slots  = max(self._opp_hand_size - confirmed_count, 0)
        n = len(uncertain)
        if n == 0 or remaining_slots == 0:
            return
        target_p = remaining_slots / n
        total    = sum(uncertain.values())
        scale    = (target_p / (total / n)) if total > 0 else 1.0
        scale    = min(max(scale, 0.5), 2.0)
        for c in uncertain:
            self._prob[c] = min(1.0, self._prob[c] * scale)

    def estimated_opponent_hand(self, top_n: int = 10) -> list[tuple[str, float]]:
        return sorted(
            [(c, p) for c, p in self._prob.items() if p > 0],
            key=lambda x: -x[1],
        )[:top_n]

    def knock_risk(self) -> str:
        top_hand = self.estimated_opponent_hand(10)
        if not top_hand:
            return "Unknown"
        estimated_dw  = sum(get_value(c) * p for c, p in top_hand)
        confirmed_in  = list(self._confirmed_in_hand)
        confirmed_dw  = sum(get_value(c) for c in confirmed_in if len(c) == 2)
        total_est     = estimated_dw + confirmed_dw
        if total_est <= 10:
            return f"HIGH (est. opp deadwood ~{total_est:.0f} — may knock soon)"
        elif total_est <= 25:
            return f"MEDIUM (est. opp deadwood ~{total_est:.0f})"
        else:
            return f"LOW (est. opp deadwood ~{total_est:.0f})"

    def summary(self, our_hand: list[str]) -> str:
        lines    = [f"[BayesHand] Opp knock risk: {self.knock_risk()}"]
        top_held = self.estimated_opponent_hand(5)
        if top_held:
            cards_str = " ".join(f"{c}({p:.0%})" for c, p in top_held if p >= 0.25)
            if cards_str:
                lines.append(f"[BayesHand] Likely opp cards: {cards_str}")
        confirmed = list(self._confirmed_in_hand)
        if confirmed:
            lines.append(f"[BayesHand] Confirmed in opp hand (drew upcard): {' '.join(confirmed[:4])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n', obs_text)
    if not state_match:
        return obs_text
    state_text   = obs_text[state_match.start():]
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id    = int(player_match.group(1)) if player_match else 0
    if 'Legal Actions:' in state_text:
        before_actions, after_actions = state_text.split('Legal Actions:', 1)
        return before_actions + f"You are Player {player_id}.\nLegal Actions:" + after_actions
    return state_text


def parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id    = int(player_match.group(1)) if player_match else 0
    section      = re.search(
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
    return [pile_str[i:i + 2] for i in range(0, len(pile_str), 2)]


def parse_game_state(observation: str) -> GameState:
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response — not a game state")
    parse_warnings = []
    player_match   = re.search(r'You are Player (\d+)', observation)
    player_id      = int(player_match.group(1)) if player_match else 0
    hand           = parse_hand_from_observation(observation)
    if not hand:
        parse_warnings.append("hand=[] (empty — shaping disabled)")
    dw_match       = re.search(r'Deadwood=(\d+)', observation)
    deadwood       = int(dw_match.group(1)) if dw_match else 0
    if not dw_match:
        parse_warnings.append("deadwood=0 (fallback — shaping will be 0)")
    phase_match    = re.search(r'Phase: (\w+)', observation)
    phase          = phase_match.group(1) if phase_match else 'Draw'
    if not phase_match:
        parse_warnings.append("phase='Draw' (fallback)")
    knock_match    = re.search(r'Knock card: (\d+)', observation)
    knock_card     = int(knock_match.group(1)) if knock_match else 10
    upcard_match   = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard         = upcard_match.group(1) if upcard_match else 'XX'
    stock_match    = re.search(r'Stock size: (\d+)', observation)
    stock_size     = int(stock_match.group(1)) if stock_match else 0
    if parse_warnings:
        print(f"[PARSE_WARN] parse_game_state fallbacks: {', '.join(parse_warnings)}")
    return GameState(
        hand=hand, deadwood=deadwood, phase=phase, knock_card=knock_card,
        upcard=upcard, stock_size=stock_size,
        discard_pile=parse_discard_pile(observation), player_id=player_id,
    )


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


def extract_action_id(completion_text: str) -> str:
    cleaned = remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5].strip()
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    match = re.search(r"-?\d+", cleaned)
    return match.group(0) if match else cleaned.strip()


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Episode-level reward: deadwood improvement (DP optimal) + terminal bonus + invalid penalties.
    Clipped to [-1, 1] for validator alignment.
    """

    def __init__(self):
        self.invalid_penalty = INVALID_PENALTY

    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float,
        is_invalid: bool = False,
    ) -> float:
        if is_invalid:
            return self.invalid_penalty
        return 0.0

    @staticmethod
    def compute_discard_safety(states: list[GameState]) -> float:
        if len(states) < 2:
            return 0.0
        agent_discards: list[str] = []
        unsafe_count = 0
        prev_pile = states[0].discard_pile
        for i in range(1, len(states)):
            curr_pile = states[i].discard_pile
            if len(curr_pile) == len(prev_pile) + 1:
                agent_discards.append(curr_pile[-1])
            elif len(curr_pile) < len(prev_pile) and agent_discards:
                taken = set(prev_pile) - set(curr_pile)
                for card in taken:
                    if card in agent_discards:
                        unsafe_count += 1
            prev_pile = curr_pile
        if not agent_discards:
            return 0.0
        return -0.1 * (unsafe_count / len(agent_discards))

    def calculate_episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        initial_state: "GameState | None",
        final_state:   "GameState | None",
        all_states:    "list[GameState] | None" = None,
    ) -> float:
        # 1. Deadwood improvement via DP optimal deadwood
        if initial_state and final_state and initial_state.hand and final_state.hand:
            dw_initial = compute_optimal_deadwood(initial_state.hand)
            dw_final   = compute_optimal_deadwood(final_state.hand)
            if dw_initial > 0:
                deadwood_component = ((dw_initial - dw_final) / dw_initial) * DEADWOOD_WEIGHT
            else:
                deadwood_component = 0.0
        elif initial_state and final_state and initial_state.deadwood > 0:
            raw_improvement    = (initial_state.deadwood - final_state.deadwood) / initial_state.deadwood
            deadwood_component = raw_improvement * DEADWOOD_WEIGHT
        else:
            deadwood_component = 0.0

        # 2. Terminal bonus
        terminal = 0.0
        if done:
            if env_reward > 0.5:
                terminal = TERMINAL_WIN_REWARD
                if final_state and final_state.deadwood == 0:
                    terminal += GIN_BONUS
                else:
                    terminal += KNOCK_BONUS
            else:
                terminal = TERMINAL_LOSS_REWARD
        elif final_state:
            terminal = -final_state.deadwood / 100.0

        # 3. Invalid action penalties (accumulated, clipped)
        invalid_total = max(sum(r for r in step_rewards if r < 0), INVALID_TOTAL_CLIP)

        raw = deadwood_component + terminal + invalid_total
        return max(min(raw, TERMINAL_REWARD_CLIP), -TERMINAL_REWARD_CLIP)


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
        "mcts_max_simulations": _MCTS_SIMS,
        "mcts_num_rollouts": 1,  # match validator eval (core/constants.py:79)
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
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(\u2660), h(\u2665), d(\u2666), c(\u2663)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood \u2264 knock_card\n\n"
    "EACH TURN:\n1. DRAW: stock (53) or upcard (52)\n"
    "2. DISCARD: choose a card by action ID\n\n"
    "KNOCKING:\n- Gin: 0 deadwood = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "- Early game: Draw from deck to see more cards\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have \u226410 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points\n"
    "- In Layoff phase: use 'Dead cards' hint to find extension opportunities\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
)


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

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
    game_id      = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt fallback (updated every loop iteration)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    invalid_count = 0
    done          = False
    train_reward  = 0.0
    final_reward  = 0.0
    turn_number   = 0
    game_state_history: list[GameState] = []
    rewards:            list[float]     = []
    calculator          = RewardCalculator()
    dead_card_tracker   = DeadCardTracker()
    prev_discard_pile:  list[str]       = []

    # Opponent modelling — full Bayesian models only for last-prompt mode
    bayesian_model: "BayesianOpponentModel | None"     = None
    bayes_hand:     "BayesianOpponentHandModel | None" = None
    if not use_full_prompt:
        bayesian_model = BayesianOpponentModel()
        bayes_hand     = BayesianOpponentHandModel()

    use_hints = random.random() < current_hint_prob

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed":    random.randint(0, 2 ** 31 - 1),
        "opponent": "mcts",
        "mcts_max_simulations": _MCTS_SIMS,
        "mcts_num_rollouts": 1,  # match validator eval (core/constants.py:79)
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block        = reset_res.json()["result"]
        episode_id          = result_block.get("episode_id", "")
        raw_observation     = result_block.get("observation", "")
        formatted_observation = extract_and_format_observation(raw_observation)
        initial_game_state  = parse_game_state(formatted_observation)
        game_state_history.append(initial_game_state)
        dead_card_tracker.update_from_discard_pile(initial_game_state.discard_pile)
        prev_discard_pile = list(initial_game_state.discard_pile)
        if not use_full_prompt and bayes_hand is not None:
            actual_hand_size = len(initial_game_state.hand)
            bayes_hand._opp_hand_size = actual_hand_size if actual_hand_size > 0 else 7
            bayes_hand.initialize(
                our_hand=initial_game_state.hand,
                discard_pile=initial_game_state.discard_pile,
            )
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
        action_to_send  = extract_action_id(completion_text)

        # --- Full-prompt token accumulation ---
        if use_full_prompt:
            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids      = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
                done = True
                break

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- UCB draw shaping (last-prompt only) ---
        ucb_draw_reward = 0.0
        if not use_full_prompt and game_state_history:
            ucb_draw_reward = draw_ucb_shaping(game_state_history[-1], action_to_send)

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
            step_block            = step_res.json()["result"]
            raw_observation       = step_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done        = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1
            is_invalid = True

        immediate_reward = 0.0
        if done:
            final_reward = step_reward
            messages.append({"role": "user", "content": formatted_observation})
        else:
            # --- Build augmented observation ---
            dead_card_tracker.update_from_observation(formatted_observation)
            try:
                current_hand = game_state_history[-1].hand if game_state_history else []
            except Exception:
                current_hand = []

            dead_summary = dead_card_tracker.summary(current_hand)

            if not use_full_prompt and bayesian_model is not None and bayes_hand is not None:
                bayes_summary      = bayesian_model.summary(current_hand)
                bayes_hand_summary = bayes_hand.summary(current_hand)
                context_parts      = [p for p in [dead_summary, bayes_summary, bayes_hand_summary] if p]
            else:
                context_parts = [dead_summary] if dead_summary else []

            obs_augmented = (
                formatted_observation + "\n\n" + "\n".join(context_parts)
                if context_parts else formatted_observation
            )
            messages.append({"role": "user", "content": obs_augmented})

            # --- Parse game state and update trackers ---
            if not is_invalid:
                try:
                    game_state = parse_game_state(formatted_observation)
                except Exception as exc:
                    print(f"Failed to parse game state: {exc}")
                    immediate_reward = calculator.calculate_step_reward(
                        game_state_history, action_to_send, 0.0, is_invalid=True
                    )
                else:
                    game_state_history.append(game_state)
                    dead_card_tracker.update_from_discard_pile(game_state.discard_pile)

                    if not use_full_prompt and bayesian_model is not None and bayes_hand is not None:
                        bayesian_model.update_from_discard_pile_delta(prev_discard_pile, game_state.discard_pile)
                        if len(game_state.discard_pile) < len(prev_discard_pile):
                            drawn_card = prev_discard_pile[-1] if prev_discard_pile else None
                            if drawn_card:
                                bayes_hand.update_opp_drew_upcard(drawn_card)
                        elif len(game_state.discard_pile) > len(prev_discard_pile):
                            discarded_card = game_state.discard_pile[-1] if game_state.discard_pile else None
                            if discarded_card:
                                bayes_hand.update_opp_discarded(discarded_card)
                        else:
                            bayes_hand.update_opp_drew_stock()

                    prev_discard_pile = list(game_state.discard_pile)
                    immediate_reward  = calculator.calculate_step_reward(
                        game_state_history, action_to_send, 0.0
                    )
            else:
                immediate_reward = calculator.calculate_step_reward(
                    game_state_history, action_to_send, 0.0, is_invalid=True
                )

        if not use_full_prompt and not is_invalid:
            immediate_reward += ucb_draw_reward

        rewards.append(immediate_reward)
        turn_number += 1

    # --- Episode reward ---
    initial_state = game_state_history[0] if game_state_history else None
    final_state   = game_state_history[-1] if game_state_history else None
    train_reward  = calculator.calculate_episode_reward(
        rewards, final_reward, done, initial_state, final_state, all_states=game_state_history
    )

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
    wins      = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0

    _log_trajectories = bool(os.environ.get("LOG_TRAJECTORIES"))
    _batch_lines = [f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}"]
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
    """Parallelised rollout — returns only the last turn's token IDs.

    Enables full Bayesian opponent modelling (BayesianOpponentModel +
    BayesianOpponentHandModel) and UCB draw-decision shaping.
    """
    return _dispatch(prompts, trainer, use_full_prompt=False)
