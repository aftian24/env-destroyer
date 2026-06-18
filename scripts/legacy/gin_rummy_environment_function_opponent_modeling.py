import os
import re
import random
import requests
from typing import Optional
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from trl.experimental.openenv import generate_rollout_completions


CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10
}

RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

# Reward constants — aligns with validator "higher is better" scoring
TERMINAL_WIN_REWARD = 1.0
TERMINAL_LOSS_REWARD = -1.0
GIN_BONUS = 0.25           # extra for 0-deadwood win (Gin = all melds)
KNOCK_BONUS = 0.1          # extra for winning via knock (tournament: reward timely knock)
DEADWOOD_WEIGHT = 0.5      # fraction of total reward from deadwood improvement
INVALID_PENALTY = -0.1
INVALID_TOTAL_CLIP = -0.3
TERMINAL_REWARD_CLIP = 1.0 # final clip for validator alignment

# Bayesian discard quality shaping (from AAAI Gin Rummy / Bayesian inference literature)
# Reward discarding cards that are "safe" (unlikely to benefit opponent's meld-building)
SAFE_DISCARD_BONUS = 0.02       # bonus for discarding a card unlikely to complete opp's meld
DANGEROUS_DISCARD_PENALTY = 0.02  # penalty for discarding a card that directly extends opp's run/set

# UCB Draw Decision shaping constants
DRAW_UPCARD_BONUS = 0.03    # bonus when model draws upcard that reduces optimal deadwood
DRAW_UPCARD_PENALTY = 0.02  # mild penalty when model draws upcard with no deadwood benefit


def get_rank(card: str) -> str:
    """Get rank from card (e.g., '7c' -> '7')"""
    return card[0]

def get_suit(card: str) -> str:
    """Get suit from card (e.g., '7c' -> 'c')"""
    return card[1]

def get_value(card: str) -> int:
    """Get point value of card"""
    return CARD_VALUES[get_rank(card)]


def find_potential_runs(hand: list[str], additional_card: Optional[str] = None) -> list[list[str]]:
    """
    Find potential runs (2+ consecutive cards same suit).
    
    Args:
        hand: Current hand
        additional_card: Optional card to test (e.g., upcard we're considering)
        
    Returns:
        List of potential runs (each run is a list of cards)
    """
    test_hand = hand.copy()
    if additional_card:
        test_hand.append(additional_card)
    
    # Group by suit
    suit_groups = {}
    for card in test_hand:
        suit = get_suit(card)
        if suit not in suit_groups:
            suit_groups[suit] = []
        suit_groups[suit].append(card)
    
    runs = []
    for suit, cards in suit_groups.items():
        # Sort by rank order
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        
        # Find consecutive sequences
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            
            while j < len(sorted_cards):
                curr_idx = RANK_ORDER.index(get_rank(sorted_cards[j]))
                prev_idx = RANK_ORDER.index(get_rank(run[-1]))
                
                if curr_idx == prev_idx + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            
            if len(run) >= 2:  # 2+ cards is potential
                runs.append(run)
            
            i = j if len(run) > 1 else i + 1
    
    return runs


def count_complete_runs(hand: list[str]) -> int:
    """Count runs of 3+ consecutive cards same suit"""
    runs = find_potential_runs(hand)
    return sum(1 for run in runs if len(run) >= 3)


# ---------------------------------------------------------------------------
# DP Optimal Deadwood (Phase 2 — replaces heuristic meld counting)
# ---------------------------------------------------------------------------

def find_all_melds(hand: list[str]) -> list[frozenset[str]]:
    """Enumerate every valid meld (SET or RUN of 3+ cards) from the given hand.

    Returns a list of frozensets, each representing one valid meld.
    A card may appear in multiple returned melds — the DP solver picks
    the non-overlapping combination that minimises deadwood.
    """
    melds: list[frozenset[str]] = []

    # --- SETs: 3 or 4 same-rank cards (any suits) ---
    from collections import defaultdict
    rank_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        rank_groups[get_rank(card)].append(card)
    for rank, cards in rank_groups.items():
        if len(cards) >= 3:
            melds.append(frozenset(cards[:3]))
        if len(cards) >= 4:
            melds.append(frozenset(cards[:4]))

    # --- RUNs: 3+ consecutive ranks, same suit ---
    suit_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        suit_groups[get_suit(card)].append(card)
    for suit, cards in suit_groups.items():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        # Find all maximal consecutive runs, then enumerate sub-runs of length >= 3
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                curr_idx = RANK_ORDER.index(get_rank(sorted_cards[j]))
                prev_idx = RANK_ORDER.index(get_rank(run[-1]))
                if curr_idx == prev_idx + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            # Extract all sub-runs of length >= 3 from this maximal run
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def compute_optimal_deadwood(hand: list[str]) -> int:
    """Compute minimum possible deadwood for a hand via bitmask DP backtracking.

    Algorithm:
      1. Find all valid melds from the hand.
      2. Use bitmask over hand indices; DP state = set of used card indices.
      3. Greedily try all melds, recurse, memoize.
      4. Deadwood = sum of point values of cards not in any chosen meld.

    Complexity: O(2^n * M) where n = hand size (<=11), M = number of melds.
    Practical runtime: <1ms for standard gin hands.

    Returns:
        Minimum deadwood value (int >= 0).
    """
    if not hand:
        return 0

    melds = find_all_melds(hand)
    n = len(hand)
    card_to_idx = {card: i for i, card in enumerate(hand)}
    # Convert each meld to a bitmask
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
    full_mask = (1 << n) - 1

    memo: dict[int, int] = {}

    def _dp(used_mask: int) -> int:
        """Return minimum deadwood for cards not yet in used_mask."""
        if used_mask in memo:
            return memo[used_mask]
        # Deadwood = sum of values of unused cards
        base_dw = sum(
            card_values_list[i] for i in range(n) if not (used_mask >> i & 1)
        )
        best = base_dw
        for mm in meld_masks:
            if (mm & used_mask) == 0:  # no overlap with already-used cards
                best = min(best, _dp(used_mask | mm))
        memo[used_mask] = best
        return best

    return _dp(0)


def meld_potential(upcard: str, hand: list[str]) -> int:
    """Estimate deadwood reduction from drawing the upcard.

    Returns: positive integer = how many points deadwood would decrease
             if we drew the upcard and made an optimal discard.
    """
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    # Simulate: add upcard, compute optimal DW, then subtract best card
    extended = hand + [upcard]
    dw_with = compute_optimal_deadwood(extended)
    # Also compute without upcard for reference
    dw_without = compute_optimal_deadwood(hand)
    return max(0, dw_without - dw_with)  # improvement (>0 = upcard helps)


def draw_ucb_shaping(current_state: 'GameState', chosen_action_id: str) -> float:
    """UCB-inspired draw decision shaping.

    Reward the model for drawing the upcard when the upcard provably reduces
    optimal deadwood. Mild penalty when model takes upcard with no benefit
    (information leak to opponent for no gain).

    Only applies at Draw phase (action 52 = upcard, 53 = stock).

    Returns:
        float: shaping reward for this draw decision.
    """
    if current_state.phase not in ('Draw', 'FirstUpcard'):
        return 0.0
    if current_state.upcard == 'XX' or not current_state.hand:
        return 0.0
    if chosen_action_id not in ('52', '53'):
        return 0.0

    potential = meld_potential(current_state.upcard, current_state.hand)

    if chosen_action_id == '52':  # drew upcard
        if potential > 0:
            # Upcard reduces deadwood — good draw
            # Scale bonus: larger improvement = larger bonus (capped at DRAW_UPCARD_BONUS)
            scale = min(potential / 10.0, 1.0)  # normalise (max ~10 pt improvement)
            return DRAW_UPCARD_BONUS * scale
        else:
            # Upcard doesn't help — mild penalty (gave info to opponent for free)
            return -DRAW_UPCARD_PENALTY
    else:  # chose stock (action 53)
        # Stock is always a valid "safe" choice — no shaping (exploration is fine)
        return 0.0





@dataclass
class GameState:
    """Simple game state - expand this gradually"""
    hand: list[str]           # Cards in hand
    deadwood: int             # Deadwood value
    phase: str                # 'Draw', 'Discard', 'FirstUpcard', 'Layoff'
    knock_card: int           # Knock threshold
    upcard: str               # Current upcard (or 'XX' if not visible)
    stock_size: int           # Cards left in stock
    discard_pile: list[str]    # Discard pile
    player_id: int            # Which player we are (0 or 1)
    
    def total_hand_value(self) -> int:
        """Calculate total value of all cards in hand"""
        return sum(get_value(card) for card in self.hand)
    
    def num_high_cards(self) -> int:
        """Count cards worth 10 points (T, J, Q, K)"""
        return sum(1 for card in self.hand if get_value(card) == 10)
    
    def can_knock(self) -> bool:
        """Check if deadwood is low enough to knock"""
        return self.deadwood <= self.knock_card
    
    def count_pairs(self) -> int:
        """Count pairs (2+ same rank) in hand"""
        rank_counts = Counter(get_rank(card) for card in self.hand)
        return sum(1 for count in rank_counts.values() if count >= 2)
    
    def count_sets(self) -> int:
        """Count sets (3+ same rank) in hand"""
        rank_counts = Counter(get_rank(card) for card in self.hand)
        return sum(1 for count in rank_counts.values() if count >= 3)
    
    def count_runs(self) -> int:
        """Count runs (3+ consecutive same suit) in hand"""
        return count_complete_runs(self.hand)
    
    def count_potential_runs(self) -> int:
        """Count 2-card potential runs"""
        runs = find_potential_runs(self.hand)
        return sum(1 for run in runs if len(run) == 2)


class DeadCardTracker:
    """
    Tracks cards that are 'dead' — already discarded and therefore can't form NEW melds.

    A dead card is a card that has been discarded and is no longer retrievable.
    Knowing dead cards prevents the model from bidding on melds it can't complete.

    Also tracks layoff candidates: cards in OUR hand that can extend the opponent's
    visible melds (useful when transitioning to Layoff phase).
    """

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self.seen_discards: set[str] = set()   # ever-discarded cards
        self.opponent_melds: list[list[str]] = []  # detected from obs when available

    def update_from_discard_pile(self, discard_pile: list[str]) -> None:
        """Add all known discard pile cards to seen set."""
        for card in discard_pile:
            if len(card) == 2:
                self.seen_discards.add(card.lower())

    def update_from_observation(self, obs: str) -> None:
        """Parse current discard pile from observation and update tracker."""
        pile = parse_discard_pile(obs)
        self.update_from_discard_pile(pile)

    def get_dead_cards(self) -> list[str]:
        """Return sorted list of seen-discarded cards."""
        return sorted(self.seen_discards)

    def is_dead(self, card: str) -> bool:
        """Check if a specific card has been discarded."""
        return card.lower() in self.seen_discards

    def get_layoff_candidates(
        self, hand: list[str], discard_pile: list[str]
    ) -> list[str]:
        """
        Identify cards in hand that might be layable on a hypothetical opponent meld.

        Heuristic: if discard_pile shows 2+ consecutive cards of a suit, or 2+ same-rank
        cards, the opponent likely has a meld of that type. Cards that extend those are
        layoff candidates.
        """
        if not discard_pile or not hand:
            return []

        candidates: set[str] = set()

        # --- Detect potential opponent run seeds from discards ---
        suit_groups: dict[str, list[str]] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            suit = card[1].lower()
            suit_groups.setdefault(suit, []).append(card.lower())

        for suit, cards in suit_groups.items():
            sorted_cards = sorted(
                cards,
                key=lambda c: self.ALL_RANKS.index(c[0].upper()) if c[0].upper() in self.ALL_RANKS else 99
            )
            for i in range(len(sorted_cards) - 1):
                r1 = sorted_cards[i][0].upper()
                r2 = sorted_cards[i + 1][0].upper()
                if r1 not in self.ALL_RANKS or r2 not in self.ALL_RANKS:
                    continue
                idx1 = self.ALL_RANKS.index(r1)
                idx2 = self.ALL_RANKS.index(r2)
                if abs(idx1 - idx2) == 1:
                    # consecutive pair → opponent may have a run
                    for adj in [idx1 - 1, idx2 + 1]:
                        if 0 <= adj < len(self.ALL_RANKS):
                            target = self.ALL_RANKS[adj] + suit
                            for hcard in hand:
                                if hcard.lower() == target:
                                    candidates.add(hcard)

        # --- Detect potential opponent set seeds from discards ---
        rank_groups: dict[str, int] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            rank = card[0].upper()
            rank_groups[rank] = rank_groups.get(rank, 0) + 1

        for rank, count in rank_groups.items():
            if count >= 2:
                # opponent discarded 2+ of same rank → they have/had a set
                for hcard in hand:
                    if hcard[0].upper() == rank:
                        candidates.add(hcard)

        return sorted(candidates)

    def summary(self, hand: list[str]) -> str:
        """Return a compact 1-2 line dead-card insight string for injection into obs."""
        dead = self.get_dead_cards()
        layoff = self.get_layoff_candidates(hand, list(self.seen_discards))
        lines = []
        if dead:
            lines.append(f"Dead cards (discarded): {' '.join(dead[:15])}")
        if layoff:
            lines.append(f"Layoff candidates (extend opp melds): {' '.join(layoff)}")
        return "\n".join(lines)


class BayesianOpponentModel:
    """Lightweight Bayesian model for inferring the opponent's likely meld direction.

    Based on AAAI Gin Rummy papers and Bayesian inference literature:
    - Opponent drawing from discard pile = strong signal they want that card (or adjacent rank/suit).
    - Opponent discarding a card = signal they are de-prioritizing that suit/rank.
    - We track "danger suits" and "danger ranks" — players consistently drawing these are
      likely building runs/sets in them.
    - Safe discards: cards in suits/ranks where the opponent has shown disinterest.
    - Dangerous discards: cards adjacent in rank/suit to what the opponent has been drawing.

    Evidence update rules (simplified Bayesian):
    - draw(card)   → heat[rank] += 1, heat[suit] += 1
    - discard(card) → heat[rank] -= 0.5, heat[suit] -= 0.5 (discard = lower priority)
    """

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        # "heat" = probability proxy for opponent interest in each suit/rank (prior = 0)
        self.rank_heat: dict[str, float] = {r: 0.0 for r in self.ALL_RANKS}
        self.suit_heat: dict[str, float] = {s: 0.0 for s in self.ALL_SUITS}
        self.opp_draws: list[str] = []    # cards opponent drew from discard pile
        self.opp_discards: list[str] = [] # cards opponent discarded

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
        """Opponent drew this card from discard pile → they want it → increase heat."""
        self.opp_draws.append(drawn_card)
        self._update_heat(drawn_card, weight=1.0)
        # Adjacent ranks in same suit (neighbours in a run) also become more likely
        if len(drawn_card) == 2:
            rank, suit = drawn_card[0].upper(), drawn_card[1].lower()
            if rank in self.ALL_RANKS:
                idx = self.ALL_RANKS.index(rank)
                for adj_idx in [idx - 1, idx + 1]:
                    if 0 <= adj_idx < len(self.ALL_RANKS):
                        adj_card = self.ALL_RANKS[adj_idx] + suit
                        self._update_heat(adj_card, weight=0.3)

    def update_on_opponent_discard(self, discarded_card: str) -> None:
        """Opponent discarded this card → they don't want it → decrease heat."""
        self.opp_discards.append(discarded_card)
        self._update_heat(discarded_card, weight=-0.5)

    def update_from_discard_pile_delta(
        self, prev_discard_pile: list[str], curr_discard_pile: list[str]
    ) -> None:
        """Infer opponent action from changes in discard pile between turns.

        If discard pile got shorter: opponent drew from it → update_on_draw.
        If discard pile got longer: opponent discarded → update_on_discard.
        """
        prev_set = list(prev_discard_pile)
        curr_set = list(curr_discard_pile)
        # Card removed from top → opponent drew it
        if prev_set and (not curr_set or len(curr_set) < len(prev_set)):
            top_card = prev_set[-1] if prev_set else None
            if top_card:
                self.update_on_opponent_draw(top_card)
        # Card added to top → opponent discarded it
        elif curr_set and len(curr_set) > len(prev_set):
            new_card = curr_set[-1]
            self.update_on_opponent_discard(new_card)

    def is_dangerous_discard(self, card: str) -> bool:
        """True if discarding this card is likely to help opponent complete a meld."""
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        rank_h = self.rank_heat.get(rank, 0.0)
        suit_h = self.suit_heat.get(suit, 0.0)
        # Dangerous if opponent has shown interest in this rank (set) or suit (run)
        return rank_h >= 1.0 or suit_h >= 1.5

    def is_safe_discard(self, card: str) -> bool:
        """True if discarding this card is unlikely to benefit opponent."""
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        rank_h = self.rank_heat.get(rank, 0.0)
        suit_h = self.suit_heat.get(suit, 0.0)
        # Safe if opponent has shown disinterest (discarded same rank/suit before)
        return rank_h <= -0.5 and suit_h <= -0.5

    def get_danger_cards(self, hand: list[str]) -> list[str]:
        """Cards in our hand that would benefit the opponent most if discarded."""
        return [c for c in hand if self.is_dangerous_discard(c)]

    def get_safe_cards(self, hand: list[str]) -> list[str]:
        """Cards in our hand safest to discard (opponent doesn't want them)."""
        return [c for c in hand if self.is_safe_discard(c)]

    def summary(self, hand: list[str]) -> str:
        """Compact Bayesian context string for injection into observation."""
        danger = self.get_danger_cards(hand)
        safe = self.get_safe_cards(hand)
        lines = []
        hot_suits = [s for s, h in self.suit_heat.items() if h >= 1.5]
        hot_ranks = [r for r, h in self.rank_heat.items() if h >= 1.0]
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
# Fase 3: Bayesian Opponent Hand Distribution (Gin Rummy)
# ---------------------------------------------------------------------------

class BayesianOpponentHandModel:
    """Track P(card ∈ opponent_hand | all observations) for Gin Rummy.

    The standard Bayesian Gin approach (cf. AAAI Gin Rummy papers):
    - Initially, each unseen card (not in our hand, not discarded) has equal
      probability of being in the opponent's hand.
    - When the opponent draws the upcard (known card), that card certainty → 1.0.
    - When the opponent draws from stock (unknown card), we update the posterior
      of stock cards.
    - When opponent discards (visible discard), that card certainty → 0.0 in hand.
    - Over turns, this gives a crude-but-useful posterior over opponent's hand.

    Key output:
      knock_risk(hand)  → estimated opponent deadwood; if low, they may knock soon
      summary(hand)     → text for observation context injection
    """

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        # Prior: probability card is in opponent's hand; updated each turn
        # {card: probability}
        self._prob: dict[str, float] = {}
        self._opp_hand_size: int = 10  # standard Gin Rummy hand size
        self._confirmed_in_hand: set[str] = set()   # 100% certain cards (upcard draw)
        self._confirmed_not_in_hand: set[str] = set()  # 100% certain not in hand (discard)

    def _all_cards(self) -> list[str]:
        return [r + s for r in self.ALL_RANKS for s in self.ALL_SUITS]

    def initialize(self, our_hand: list[str], discard_pile: list[str]) -> None:
        """Set up uniform prior over all cards not in our hand and not discarded."""
        known_out = set(c.lower() for c in our_hand) | set(c.lower() for c in discard_pile)
        unknown_cards = [c for c in self._all_cards() if c not in known_out]
        n = len(unknown_cards)
        # Prior: opponent holds exactly opp_hand_size cards from unknown pool
        p_in_hand = self._opp_hand_size / max(n, self._opp_hand_size)
        self._prob = {c: p_in_hand for c in unknown_cards}
        for c in known_out:
            self._prob[c] = 0.0

    def update_opp_drew_upcard(self, upcard: str) -> None:
        """Opponent drew the visible upcard → certainty 1.0."""
        card = upcard.lower()
        if len(card) == 2:
            self._prob[card] = 1.0
            self._confirmed_in_hand.add(card)
            # Condition: one fewer slot from stock pool — renormalize stock cards slightly down
            self._renormalize(exclude={card})

    def update_opp_drew_stock(self) -> None:
        """Opponent drew an unknown stock card → raise probability of each stock card slightly."""
        # All non-confirmed, non-zero-prob cards are candidates
        stock_candidates = [
            c for c, p in self._prob.items()
            if p > 0 and c not in self._confirmed_in_hand and c not in self._confirmed_not_in_hand
        ]
        if not stock_candidates:
            return
        # Bayesian update: we know one of them is now in opp's hand — increase all slightly
        n = len(stock_candidates)
        boost = 1.0 / n  # probability mass to spread across candidates
        for c in stock_candidates:
            self._prob[c] = min(1.0, self._prob[c] + boost * 0.1)

    def update_opp_discarded(self, card: str) -> None:
        """Opponent discarded this card → it's no longer in their hand."""
        c = card.lower()
        if len(c) == 2:
            self._prob[c] = 0.0
            self._confirmed_not_in_hand.add(c)
            self._renormalize(exclude={c})

    def _renormalize(self, exclude: set[str]) -> None:
        """Soft renormalisation after a certainty update."""
        # Scale uncertain cards so estimated hand count stays plausible
        uncertain = {c: p for c, p in self._prob.items()
                     if c not in exclude and c not in self._confirmed_in_hand
                     and c not in self._confirmed_not_in_hand and p > 0}
        confirmed_count = len(self._confirmed_in_hand)
        remaining_slots = max(self._opp_hand_size - confirmed_count, 0)
        n = len(uncertain)
        if n == 0 or remaining_slots == 0:
            return
        target_p = remaining_slots / n
        scale = target_p / (sum(uncertain.values()) / n) if sum(uncertain.values()) > 0 else 1.0
        scale = min(max(scale, 0.5), 2.0)  # cap scale to avoid explosion
        for c in uncertain:
            self._prob[c] = min(1.0, self._prob[c] * scale)

    def estimated_opponent_hand(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the top_n cards most likely to be in the opponent's hand."""
        return sorted(
            [(c, p) for c, p in self._prob.items() if p > 0],
            key=lambda x: -x[1]
        )[:top_n]

    def knock_risk(self) -> str:
        """Estimate opponent's knock risk based on hand composition.

        High-probability held high-value cards = high deadwood = low risk.
        High-probability held low-value cards + known draws = escalating risk.
        """
        top_hand = self.estimated_opponent_hand(10)
        if not top_hand:
            return "Unknown"
        estimated_dw = sum(
            get_value(c) * p for c, p in top_hand
        )
        confirmed_in = list(self._confirmed_in_hand)
        confirmed_dw = sum(get_value(c) for c in confirmed_in if len(c) == 2)
        total_est = estimated_dw + confirmed_dw

        if total_est <= 10:
            return f"HIGH (est. opp deadwood ~{total_est:.0f} — may knock soon)"
        elif total_est <= 25:
            return f"MEDIUM (est. opp deadwood ~{total_est:.0f})"
        else:
            return f"LOW (est. opp deadwood ~{total_est:.0f})"

    def likely_meld_cards(self) -> list[str]:
        """Cards in the uncertain pool that could form melds (runs/sets) for the opponent."""
        top = [c for c, p in self.estimated_opponent_hand(15) if p >= 0.3]
        # Filter to low-value cards (likely deadwood) or cards forming potential melds
        return top[:8]

    def summary(self, our_hand: list[str]) -> str:
        """Brief Bayesian context for injection into observation prompt."""
        lines = []
        knock = self.knock_risk()
        lines.append(f"[BayesHand] Opp knock risk: {knock}")
        top_held = self.estimated_opponent_hand(5)
        if top_held:
            cards_str = " ".join(f"{c}({p:.0%})" for c, p in top_held if p >= 0.25)
            if cards_str:
                lines.append(f"[BayesHand] Likely opp cards: {cards_str}")
        confirmed = list(self._confirmed_in_hand)
        if confirmed:
            lines.append(f"[BayesHand] Confirmed in opp hand (drew upcard): {' '.join(confirmed[:4])}")
        return "\n".join(lines)


def extract_and_format_observation(obs_text: str) -> str:
    """
    Extract observation from server response.

    Handles three cases:
    1. Invalid-action obs  → pass through unchanged (already has Legal Actions)
    2. Reset obs           → has a '# Game Rules' preamble; strip it, keep from
                             'Current State:' onward with player-id injected before
                             Legal Actions (same layout as step obs)
    3. Step obs            → already clean; just locate 'Current State:' and inject
                             player-id before Legal Actions for parser compatibility

    Args:
        obs_text: Raw observation string from /reset or /step result block.

    Returns:
        Clean observation string starting at 'Current State:'.
    """
    # Case 1: invalid action — server already formatted it with Legal Actions
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text

    # Find 'Current State:' — present in both reset and step obs
    state_match = re.search(r'Current State:\n', obs_text)
    if not state_match:
        # Unexpected format — return as-is so downstream can log a PARSE_WARN
        return obs_text

    # Everything from 'Current State:' onward
    state_text = obs_text[state_match.start():]

    # Extract player ID from anywhere in the full obs (present before Current State in reset)
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0

    # Inject "You are Player N." just before Legal Actions for parser compatibility
    if 'Legal Actions:' in state_text:
        before_actions, after_actions = state_text.split('Legal Actions:', 1)
        return before_actions + f"You are Player {player_id}.\nLegal Actions:" + after_actions

    return state_text


def parse_hand_from_observation(observation: str) -> list[str]:
    """
    Extract just the player's hand from observation.
    
    Args:
        observation: Full game state string
        
    Returns:
        List of cards in hand, e.g., ['3s', '6s', 'Ts', '3d', '8d', 'Ah', '4h', '8h']
    """
    # Find which player we are
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Extract the card display box for our player
    player_section_match = re.search(
        rf'Player{player_id}: Deadwood=\d+\n\+-+\+\n(.*?)\n\+-+\+',
        observation,
        re.DOTALL
    )
    
    hand = []
    if player_section_match:
        card_rows = player_section_match.group(1).strip().split('\n')
        for row in card_rows:
            # Find all cards in format: rank(A-K) + suit(s/h/d/c)
            cards_in_row = re.findall(r'([A2-9TJQK][shdc])', row)
            hand.extend(cards_in_row)
    
    return hand


def parse_discard_pile(observation: str) -> list[str]:
    """
    Extract all cards in the discard pile.
    
    Args:
        observation: Full game state string
        
    Returns:
        List of discarded cards in order (oldest to newest)
    """
    discard_match = re.search(r'Discard pile: (.*?)\n', observation)
    
    if not discard_match:
        return []
    
    pile_str = discard_match.group(1).strip()
    if not pile_str:
        return []
    
    # First try space-separated
    if ' ' in pile_str:
        return pile_str.split()
    
    # Otherwise, split into 2-char chunks
    cards = [pile_str[i:i+2] for i in range(0, len(pile_str), 2)]
    return cards


def parse_game_state(observation: str) -> GameState:
    """
    Parse observation into GameState.
    
    Args:
        observation: Full game state string
        
    Returns:
        GameState object
    """
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response - not a game state")
    
    parse_warnings = []

    # Player ID
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Hand
    hand = parse_hand_from_observation(observation)
    if not hand:
        parse_warnings.append("hand=[] (empty — shaping disabled)")

    # Deadwood
    deadwood_match = re.search(r'Deadwood=(\d+)', observation)
    deadwood = int(deadwood_match.group(1)) if deadwood_match else 0
    if not deadwood_match:
        parse_warnings.append("deadwood=0 (fallback — shaping will be 0)")

    # Phase
    phase_match = re.search(r'Phase: (\w+)', observation)
    phase = phase_match.group(1) if phase_match else 'Draw'
    if not phase_match:
        parse_warnings.append("phase='Draw' (fallback)")

    # Knock card
    knock_match = re.search(r'Knock card: (\d+)', observation)
    knock_card = int(knock_match.group(1)) if knock_match else 10
    
    # Upcard
    upcard_match = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard = upcard_match.group(1) if upcard_match else 'XX'
    
    # Discard pile
    discard_pile = parse_discard_pile(observation)
    
    # Stock size
    stock_match = re.search(r'Stock size: (\d+)', observation)
    stock_size = int(stock_match.group(1)) if stock_match else 0

    if parse_warnings:
        print(f"[PARSE_WARN] parse_game_state fallbacks: {', '.join(parse_warnings)}")
    
    return GameState(
        hand=hand,
        deadwood=deadwood,
        phase=phase,
        knock_card=knock_card,
        upcard=upcard,
        stock_size=stock_size,
        discard_pile=discard_pile,
        player_id=player_id,
    )
    

class RewardCalculator:
    """
    Deadwood-based reward calculator — normalized to [-1, 1] for validator alignment.

    Validator uses "higher is better" for ENVIRONMENTTASK scores, so all rewards
    must be in a consistent range where positive = good.

    Components:
    - Terminal: +1.0 win, +0.25 gin bonus, +0.1 knock bonus, -1.0 loss
    - Deadwood improvement: 0-0.5 weighted fraction of improvement
    - Invalid action penalty: -0.1 per step, total clipped to INVALID_TOTAL_CLIP

    Tournament adjustment:
    - KNOCK_BONUS rewards timely knock (deadwood > 0 wins) to prevent
      the agent from waiting too long for Gin when a knock would already win.
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
        """Per-step reward: only invalid action penalty."""
        if is_invalid:
            return self.invalid_penalty
        return 0.0

    @staticmethod
    def compute_discard_safety(states: list[GameState]) -> float:
        """
        Measure how often the agent's discards get picked up by the opponent.

        Walk consecutive state pairs: if the discard pile grew (agent discarded),
        record the new card. If the pile later shrinks or loses that card
        (opponent drew it face-up), count it as an unsafe discard.

        Returns a penalty in [-0.1, 0.0].  0.0 = perfectly safe discards.
        """
        if len(states) < 2:
            return 0.0

        agent_discards: list[str] = []
        unsafe_count = 0

        prev_pile = states[0].discard_pile
        for i in range(1, len(states)):
            curr_pile = states[i].discard_pile

            # Agent discarded: pile grew by one card
            if len(curr_pile) == len(prev_pile) + 1:
                new_card = curr_pile[-1]
                agent_discards.append(new_card)

            # Opponent drew face-up: pile shrank (top card taken)
            elif len(curr_pile) < len(prev_pile) and agent_discards:
                # The card that disappeared is the one the opponent took
                taken = set(prev_pile) - set(curr_pile)
                for card in taken:
                    if card in agent_discards:
                        unsafe_count += 1

            prev_pile = curr_pile

        if not agent_discards:
            return 0.0

        ratio = unsafe_count / len(agent_discards)
        return -0.1 * ratio

    def calculate_episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        initial_state: GameState | None,
        final_state: GameState | None,
        all_states: list[GameState] | None = None,
    ) -> float:
        """
        Combine deadwood improvement + terminal bonus + invalid penalties.
        Result is clipped to [-1, 1] so the validator's reverse-sort (higher=better) works correctly.
        """
        # 1. Deadwood improvement component — use DP optimal deadwood for accuracy
        if initial_state and final_state and initial_state.hand and final_state.hand:
            # Compute optimal (minimum possible) deadwood from actual hands via DP
            dw_initial = compute_optimal_deadwood(initial_state.hand)
            dw_final   = compute_optimal_deadwood(final_state.hand)
            if dw_initial > 0:
                raw_improvement = (dw_initial - dw_final) / dw_initial
                deadwood_component = raw_improvement * DEADWOOD_WEIGHT
            else:
                deadwood_component = 0.0  # already at 0 deadwood, no room to improve
        elif initial_state and final_state and initial_state.deadwood > 0:
            # Fallback to server-reported deadwood if hand parse failed
            raw_improvement = (initial_state.deadwood - final_state.deadwood) / initial_state.deadwood
            deadwood_component = raw_improvement * DEADWOOD_WEIGHT
        else:
            deadwood_component = 0.0

        # 2. Terminal bonus (win/loss/truncation)
        terminal = 0.0
        if done:
            if env_reward > 0.5:
                terminal = TERMINAL_WIN_REWARD
                if final_state and final_state.deadwood == 0:
                    terminal += GIN_BONUS   # Gin bonus: perfect hand (0 deadwood)
                else:
                    terminal += KNOCK_BONUS  # Knock bonus: won via timely knock (tournament)
            else:
                terminal = TERMINAL_LOSS_REWARD
        elif final_state:
            # Truncated: mild penalty proportional to remaining deadwood
            terminal = -final_state.deadwood / 100.0

        # 3. Invalid action penalty (accumulated, clipped)
        invalid_total = sum(r for r in step_rewards if r < 0)
        invalid_total = max(invalid_total, INVALID_TOTAL_CLIP)

        raw = deadwood_component + terminal + invalid_total
        # Clip to [-1, 1] for validator alignment (higher = better)
        return max(min(raw, TERMINAL_REWARD_CLIP), -TERMINAL_REWARD_CLIP)
    
    
REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

def remove_reasoning_tags(text: str) -> str:

    cleaned = text

    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]

        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]

    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def extract_action_id(completion_text: str) -> str:
    """
    Extract a clean numeric action ID from model completion text.
    """
    cleaned = remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5].strip()
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    match = re.search(r"-?\d+", cleaned)
    return match.group(0) if match else cleaned.strip()

    
class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.
    """
    def __init__(
        self,
        initial_max_turn=1,
        final_max_turn=13,
        rollouts_per_stage=1280,
        initial_hint_prob=0.8,
        final_hint_prob=0.0,
        hint_decay_optimizer_steps=100,
        warmup_rollouts=128,
        mcts_warmup_optimizer_steps=None,
        initial_mcts_sims=5,
        final_mcts_sims=25,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.hint_decay_optimizer_steps = hint_decay_optimizer_steps
        self.warmup_rollouts = warmup_rollouts
        self.mcts_warmup_optimizer_steps = (
            0 if mcts_warmup_optimizer_steps is None else mcts_warmup_optimizer_steps
        )
        self.initial_mcts_sims = initial_mcts_sims
        self.final_mcts_sims = final_mcts_sims

        self.total_rollouts = 0
        
    def get_max_turn(self):
        """Calculate current max_turn based on curriculum."""
        if self.total_rollouts < self.warmup_rollouts:
            # During warmup, use initial max_turn
            return self.initial_max_turn
        
        # Calculate stage (which batch of rollouts_per_stage we're in)
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        
        # Linearly increase max_turn
        current_max_turn = min(
            self.initial_max_turn + stage,
            self.final_max_turn
        )
        return current_max_turn
    
    def get_hint_prob(self, optimizer_step: Optional[int] = None):
        """Calculate current hint probability from optimizer-step progress."""
        current_step = 0 if optimizer_step is None else optimizer_step
        if self.hint_decay_optimizer_steps <= 0:
            return self.final_hint_prob
        progress = min(max(current_step, 0) / self.hint_decay_optimizer_steps, 1.0)
        current_prob = self.initial_hint_prob - progress * (
            self.initial_hint_prob - self.final_hint_prob
        )
        return max(current_prob, self.final_hint_prob)

    def get_mcts_sims(self, optimizer_step: Optional[int] = None):
        """Calculate current MCTS simulations based on curriculum progress."""
        current_step = 0 if optimizer_step is None else optimizer_step
        if self.mcts_warmup_optimizer_steps <= 0:
            return self.final_mcts_sims
        progress = min(max(current_step, 0) / self.mcts_warmup_optimizer_steps, 1.0)
        return int(
            self.initial_mcts_sims
            + progress * (self.final_mcts_sims - self.initial_mcts_sims)
        )

    def step(self, num_rollouts=1):
        """Increment rollout counter."""
        self.total_rollouts += num_rollouts

    def get_status(self, optimizer_step: Optional[int] = None):
        """Get current curriculum status for logging."""
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(optimizer_step),
            "mcts_sims": self.get_mcts_sims(optimizer_step),
        }
        

def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for game environments.
    """
    
    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_last_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                # Initialize with a test reset to ensure server is ready
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_last_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_last_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_last_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_last_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_last_prompt_and_completion_parallelized_curriculum.selected_game = selected_game

        rollout_warmup_rollouts = (
            trainer.args.rollout_warmup_rollouts
            if getattr(trainer.args, "rollout_warmup_rollouts", None) is not None
            else trainer.args.rollouts_per_stage
        )
        mcts_warmup_optimizer_steps = getattr(
            trainer.args, "mcts_warmup_optimizer_steps", None
        )
        hint_decay_optimizer_steps = 100

        # Initialize curriculum scheduler
        rollout_last_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.5,
            final_hint_prob=0.0,
            hint_decay_optimizer_steps=hint_decay_optimizer_steps,
            warmup_rollouts=rollout_warmup_rollouts,
            mcts_warmup_optimizer_steps=mcts_warmup_optimizer_steps,
            initial_mcts_sims=25,
            final_mcts_sims=25,
        )

        print(
            f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn={30}, "
            f"rollouts_per_stage={trainer.args.rollouts_per_stage}, "
            f"rollout_warmup_rollouts={rollout_warmup_rollouts}, "
            f"hint_decay_optimizer_steps={hint_decay_optimizer_steps}, "
            f"mcts_warmup_optimizer_steps={mcts_warmup_optimizer_steps}, "
            f"mcts_sims=25->25 (constant)"
        )

    # Retrieve static variables
    rank = rollout_last_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_last_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_last_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_last_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_last_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_last_prompt_and_completion_parallelized_curriculum.curriculum
    
    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    
    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob(current_optimizer_step)
    current_mcts_sims = curriculum.get_mcts_sims(current_optimizer_step)
    print(
        f"[CURRICULUM] Rollout {total_rollouts}, step {current_optimizer_step}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str):
        # Generate a random game_id for this episode
        game_id = int(prompt)

        # Select server based on index and rank
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        invalid_count = 0
        done = False
        train_reward = 0.0
        final_reward = 0.0
        turn_number = 0
        game_state_history: list[GameState] = []
        rewards = []
        calculator = RewardCalculator()
        dead_card_tracker = DeadCardTracker()
        bayesian_model = BayesianOpponentModel()  # Bayesian opponent meld inference
        bayes_hand = BayesianOpponentHandModel()  # Posterior over opponent hand / knock risk
        prev_discard_pile: list[str] = []  # track discard pile changes for Bayesian updates

        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), "opponent": "mcts", "mcts_max_simulations": current_mcts_sims, "mcts_num_rollouts": 1}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)
            initial_game_state = parse_game_state(formatted_observation)
            game_state_history.append(initial_game_state)

            # Seed dead-card tracker from initial discard pile
            dead_card_tracker.update_from_discard_pile(initial_game_state.discard_pile)
            prev_discard_pile = list(initial_game_state.discard_pile)
            # Fase 3: Initialize Bayesian opponent hand model with actual hand_size
            actual_hand_size = len(initial_game_state.hand)  # 7, 8, or 9 per game config
            bayes_hand._opp_hand_size = actual_hand_size if actual_hand_size > 0 else 7
            bayes_hand.initialize(
                our_hand=initial_game_state.hand,
                discard_pile=initial_game_state.discard_pile,
            )

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\nSETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\nMELDS (Valid Combinations):\n1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\nExamples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\nCARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\nGAME PHASES:\n1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\nEACH TURN:\n1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\nKNOCKING:\n- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\nSCORING: Winner scores difference in deadwood point values.\nCard Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\nIMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""
        
        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = (
                "\n\n# Strategy Tips\n"
                "- Early game: Draw from deck to see more cards\n"
                "- Build runs and sets to reduce deadwood\n"
                "- Track opponent's discards to guess their hand\n"
                "- Knock when you have ≤10 deadwood points and think you're ahead\n"
                "- Go for Gin (0 deadwood) when close for bonus points\n"
                "- In Layoff phase: use 'Dead cards' hint to find extension opportunities\n"
                "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
            )
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):                
            # Generate Rollout Completion
            # Only allow one thread to generate rollout completions at a time
            with rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            
            # Add completion to messages
            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = extract_action_id(completion_text)

            # --- UCB Draw Decision Shaping ---
            # Applied before /step so we can use the pre-action game state (phase + upcard)
            ucb_draw_reward = 0.0
            if game_state_history:
                ucb_draw_reward = draw_ucb_shaping(game_state_history[-1], action_to_send)

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                raw_observation = step_block.get("observation", "")
                formatted_observation = extract_and_format_observation(raw_observation)
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions in observation
            is_invalid = False
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1
                is_invalid = True

            immediate_reward = 0.0
            if done:
                final_reward = step_reward
            else:
                # Update dead-card tracker + Bayesian model; inject combined context
                dead_card_tracker.update_from_observation(formatted_observation)
                try:
                    current_hand = game_state_history[-1].hand if game_state_history else []
                except Exception:
                    current_hand = []
                dead_summary = dead_card_tracker.summary(current_hand)
                bayes_summary = bayesian_model.summary(current_hand)
                bayes_hand_summary = bayes_hand.summary(current_hand)  # Fase 3
                context_parts = [p for p in [dead_summary, bayes_summary, bayes_hand_summary] if p]
                obs_augmented = (
                    formatted_observation + "\n\n" + "\n".join(context_parts)
                    if context_parts else formatted_observation
                )
                messages.append({"role": "user", "content": obs_augmented})

                # Parse Game State and calculate step reward
                if not is_invalid:
                    try:
                        game_state = parse_game_state(formatted_observation)
                    except Exception as e:
                        print(f"Failed to parse game state: {e}")
                        immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0, is_invalid=True)
                    else:
                        game_state_history.append(game_state)
                        dead_card_tracker.update_from_discard_pile(game_state.discard_pile)
                        # Bayesian opponent updates based on discard_pile_delta
                        bayesian_model.update_from_discard_pile_delta(prev_discard_pile, game_state.discard_pile)
                        # Fase 3: Update BayesianOpponentHandModel
                        if len(game_state.discard_pile) < len(prev_discard_pile):
                            # Pile shrank → opponent drew upcard
                            drawn_card = prev_discard_pile[-1] if prev_discard_pile else None
                            if drawn_card:
                                bayes_hand.update_opp_drew_upcard(drawn_card)
                        elif len(game_state.discard_pile) > len(prev_discard_pile):
                            # Pile grew → opponent discarded
                            discarded_card = game_state.discard_pile[-1] if game_state.discard_pile else None
                            if discarded_card:
                                bayes_hand.update_opp_discarded(discarded_card)
                        else:
                            # No pile change → opponent likely drew from stock
                            bayes_hand.update_opp_drew_stock()
                        prev_discard_pile = list(game_state.discard_pile)
                        immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0)

                else:  # is_invalid is True
                    immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0, is_invalid=True)

            # Accumulate ucb_draw_reward into immediate reward (only when valid action)
            if not is_invalid:
                immediate_reward += ucb_draw_reward

            rewards.append(immediate_reward)
            turn_number += 1


        # Calculate episode reward (deadwood improvement + terminal + invalid penalties)
        initial_state = game_state_history[0] if game_state_history else None
        final_state = game_state_history[-1] if game_state_history else None
        episode_reward = calculator.calculate_episode_reward(rewards, final_reward, done, initial_state, final_state, all_states=game_state_history)
        train_reward = episode_reward

        initial_dw = game_state_history[0].deadwood if game_state_history else 0
        final_dw = game_state_history[-1].deadwood if game_state_history else 0

        # Single-line episode summary
        print(f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{episode_reward:6.2f} EnvR:{final_reward:5.1f} "
            f"DW:{initial_dw:2d}→{final_dw:2d} Inv:{invalid_count}")

        return index, {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "reward": train_reward,
            "final_score": final_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    executor = rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            # Fallback for failed episodes
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0,
                "final_score": 0.0,
            }
            
    # Update curriculum after batch
    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    
    # Log batch statistics
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    wins = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}")


    # ---- Aggregate ----
    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for game environments.
    Uses full prompt and completion IDs with action masking.
    """
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 16384 - 256      # Max prompt tokens before ending episode early
    
    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_full_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                # Initialize with a test reset to ensure server is ready
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
                res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
                res.raise_for_status()
                env_pool.append({"base_url": base_url})
                print(f"[INIT] Server {idx} ready")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        rollout_full_prompt_and_completion_parallelized_curriculum.rank = rank
        rollout_full_prompt_and_completion_parallelized_curriculum.env_pool = env_pool
        rollout_full_prompt_and_completion_parallelized_curriculum.num_servers = len(env_pool)
        rollout_full_prompt_and_completion_parallelized_curriculum.initialized = True
        rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore = Semaphore(1)
        rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range = games_to_task_id_range
        rollout_full_prompt_and_completion_parallelized_curriculum.selected_game = selected_game

        rollout_warmup_rollouts = (
            trainer.args.rollout_warmup_rollouts
            if getattr(trainer.args, "rollout_warmup_rollouts", None) is not None
            else trainer.args.rollouts_per_stage
        )
        mcts_warmup_optimizer_steps = getattr(
            trainer.args, "mcts_warmup_optimizer_steps", None
        )
        hint_decay_optimizer_steps = 100

        # Initialize curriculum scheduler
        rollout_full_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.5,
            final_hint_prob=0.0,
            hint_decay_optimizer_steps=hint_decay_optimizer_steps,
            warmup_rollouts=rollout_warmup_rollouts,
            mcts_warmup_optimizer_steps=mcts_warmup_optimizer_steps,
            initial_mcts_sims=25,
            final_mcts_sims=25,
        )

        print(
            f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn={30}, "
            f"rollouts_per_stage={trainer.args.rollouts_per_stage}, "
            f"rollout_warmup_rollouts={rollout_warmup_rollouts}, "
            f"hint_decay_optimizer_steps={hint_decay_optimizer_steps}, "
            f"mcts_warmup_optimizer_steps={mcts_warmup_optimizer_steps}, "
            f"mcts_sims=25->25 (constant)"
        )

    # Retrieve static variables
    rank = rollout_full_prompt_and_completion_parallelized_curriculum.rank
    env_pool = rollout_full_prompt_and_completion_parallelized_curriculum.env_pool
    num_servers = rollout_full_prompt_and_completion_parallelized_curriculum.num_servers
    games_to_task_id_range = rollout_full_prompt_and_completion_parallelized_curriculum.games_to_task_id_range
    selected_game = rollout_full_prompt_and_completion_parallelized_curriculum.selected_game
    curriculum = rollout_full_prompt_and_completion_parallelized_curriculum.curriculum
    
    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    
    # Get current curriculum parameters
    total_rollouts = curriculum.total_rollouts
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob(current_optimizer_step)
    current_mcts_sims = curriculum.get_mcts_sims(current_optimizer_step)
    print(
        f"[CURRICULUM] Rollout {total_rollouts}, step {current_optimizer_step}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str):
        # Generate a random game_id for this episode
        game_id = int(prompt)

        # Select server based on index and rank
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        invalid_count = 0
        done = False
        train_reward = 0.0
        final_reward = 0.0
        turn_number = 0
        game_state_history: list[GameState] = []
        rewards = []
        calculator = RewardCalculator()
        dead_card_tracker = DeadCardTracker()
        bayesian_model = BayesianOpponentModel()  # Bayesian opponent meld inference
        prev_discard_pile: list[str] = []  # track discard pile changes for Bayesian updates

        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), "opponent": "mcts", "mcts_max_simulations": current_mcts_sims, "mcts_num_rollouts": 1}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)
            initial_game_state = parse_game_state(formatted_observation)
            game_state_history.append(initial_game_state)

            # Seed dead-card and Bayesian trackers from initial state
            dead_card_tracker.update_from_discard_pile(initial_game_state.discard_pile)
            prev_discard_pile = list(initial_game_state.discard_pile)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\nSETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\nMELDS (Valid Combinations):\n1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\nExamples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\nCARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\nGAME PHASES:\n1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\nEACH TURN:\n1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\nKNOCKING:\n- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\nSCORING: Winner scores difference in deadwood point values.\nCard Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\nIMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""
        
        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = (
                "\n\n# Strategy Tips\n"
                "- Early game: Draw from deck to see more cards\n"
                "- Build runs and sets to reduce deadwood\n"
                "- Track opponent's discards to guess their hand\n"
                "- Knock when you have ≤10 deadwood points and think you're ahead\n"
                "- Go for Gin (0 deadwood) when close for bonus points\n"
                "- In Layoff phase: use 'Dead cards' hint to find extension opportunities\n"
                "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
            )
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):                
            # Generate Rollout Completion
            # Only allow one thread to generate rollout completions at a time
            with rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            action_to_send = extract_action_id(completion_text)

            # Always capture prompt_ids first (before any early exit) so episode always has valid prompt
            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids):]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))
                    prev_full_ids = prompt_ids.copy()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids
            messages.append({"role": "assistant", "content": completion_text})

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                raw_observation = step_block.get("observation", "")
                formatted_observation = extract_and_format_observation(raw_observation)
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions in observation
            is_invalid = False
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1
                is_invalid = True

            immediate_reward = 0.0
            if done:
                final_reward = step_reward
                messages.append({"role": "user", "content": formatted_observation})
            else:
                # Update dead-card tracker and inject summary into next user message
                dead_card_tracker.update_from_observation(formatted_observation)
                try:
                    current_hand = game_state_history[-1].hand if game_state_history else []
                except Exception:
                    current_hand = []
                dead_summary = dead_card_tracker.summary(current_hand)
                obs_with_dead_cards = (
                    formatted_observation + "\n\n" + dead_summary
                    if dead_summary else formatted_observation
                )
                messages.append({"role": "user", "content": obs_with_dead_cards})

                # Parse Game State and calculate step reward
                if not is_invalid:
                    try:
                        game_state = parse_game_state(formatted_observation)
                    except Exception as e:
                        print(f"Failed to parse game state: {e}")
                        immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0, is_invalid=True)
                    else:
                        game_state_history.append(game_state)
                        # Keep tracker current from parsed state (more accurate than raw obs)
                        dead_card_tracker.update_from_discard_pile(game_state.discard_pile)
                        immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0)
                else:  # is_invalid is True
                    immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0, is_invalid=True)

            rewards.append(immediate_reward)
            turn_number += 1

        # Calculate episode reward (deadwood improvement + terminal + invalid penalties)
        initial_state = game_state_history[0] if game_state_history else None
        final_state = game_state_history[-1] if game_state_history else None
        episode_reward = calculator.calculate_episode_reward(rewards, final_reward, done, initial_state, final_state, all_states=game_state_history)
        train_reward = episode_reward

        initial_dw = game_state_history[0].deadwood if game_state_history else 0
        final_dw = game_state_history[-1].deadwood if game_state_history else 0

        # Single-line episode summary
        print(f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{episode_reward:6.2f} EnvR:{final_reward:5.1f} "
            f"DW:{initial_dw:2d}→{final_dw:2d} Inv:{invalid_count}")

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        return index, {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask": episode_action_mask,
            "logprobs": episode_logprobs,
            "reward": train_reward,
            "final_score": final_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    executor = rollout_full_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            # Fallback for failed episodes
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "action_mask": [0],
                "logprobs": [1.0],
                "reward": 0.0,
                "final_score": 0.0,
            }
            
    # Update curriculum after batch
    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    
    # Log batch statistics
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    wins = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}")


    # ---- Aggregate ----
    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "action_mask": [r["action_mask"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }
    
    
def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)