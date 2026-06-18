"""Expert trajectory generator for Gin Rummy SFT training.

Adopted from tournament 20260518 winner 5EgpWgYv (yueliang) phase-aware expert
(`/Users/zulist/Documents/env/tournament_research_2026_05_18/winner_5EgpWgYv/scripts/envs/gin_rummy_trajectories.py`).

Modifications for PvP eval compatibility (Senin 2026-05-25):
1. System prompt sourced from PvP canonical (`pvp_format.SYSTEM_PROMPT_GIN_RUMMY`)
   instead of `gin_rummy_env._SYSTEM_PROMPT` — matches validator's prompt exactly.
2. User messages reformatted via `build_pvp_user_prompt_gin_rummy(env_obs, player_id)`
   so SFT trains the exact `Current State:\n...\nYour choice (ID only):` distribution.
3. Player perspective alternates per `game_id % 2` (PvP eval covers both via position swap).
4. Expert policy `get_expert_action(messages)` still reads the RAW env observation
   from message history because parsing (`parse_phase`, `parse_hand`, etc.) expects
   the raw env format, NOT the PvP-reformatted version.

Winner's expert logic:
- `compute_optimal_deadwood` (DP bitmask): globally-optimal meld selection
- `find_all_melds`: enumerates ALL valid melds (3+ sets, 3+ runs)
- `discard_score`: multi-factor (card value − meld penalty − pair bonus − adj bonus)
- `partial_set_can_complete` / `partial_run_can_complete`: dead-card tracking
- Phase routing: `choose_draw` / `choose_discard` / `choose_meld_or_layoff_action`

Expected score boost vs naive heuristic: +8-15% on R2/R4a gin (per
[[reference_winner_trajectory_logic]]).
"""

import re
from collections import Counter, defaultdict
from typing import Optional

import requests

from envs.gin_rummy_env import extract_and_format_observation
from envs.pvp_format import (
    SYSTEM_PROMPT_GIN_RUMMY,
    build_pvp_user_prompt_gin_rummy,
)

_TIMEOUT = 2400

# System prompt now sourced from PvP canonical (matches validator eval format)
_SYSTEM_PROMPT = SYSTEM_PROMPT_GIN_RUMMY

##############################################################################################
# Card utilities (verbatim from winner repo)
##############################################################################################

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
RANK_IDX   = {r: i for i, r in enumerate(RANK_ORDER)}


def card_value(card: str) -> int:
    return CARD_VALUES.get(card[0].upper(), 0) if len(card) >= 2 else 0


def get_rank(card: str) -> str:
    return card[0].upper()


def get_suit(card: str) -> str:
    return card[1].lower()


def find_all_melds(hand: list[str]) -> list[frozenset[str]]:
    melds: list[frozenset[str]] = []

    rank_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        rank_groups[get_rank(c)].append(c)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            melds.append(frozenset(cards[:3]))
        if len(cards) >= 4:
            melds.append(frozenset(cards[:4]))

    suit_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        suit_groups[get_suit(c)].append(c)
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_IDX[get_rank(c)])
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_IDX[get_rank(sorted_cards[j])] == RANK_IDX[get_rank(run[-1])] + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def _build_hand_state(hand: list[str]) -> tuple[int, list[int], list[int]]:
    n = len(hand)
    card_to_idx = {c: i for i, c in enumerate(hand)}
    meld_masks: list[int] = []
    for meld in find_all_melds(hand):
        mask, valid = 0, True
        for c in meld:
            if c not in card_to_idx:
                valid = False; break
            mask |= (1 << card_to_idx[c])
        if valid:
            meld_masks.append(mask)
    vals = [card_value(c) for c in hand]
    return n, vals, meld_masks


def compute_optimal_deadwood(hand: list[str]) -> int:
    if not hand:
        return 0
    n, vals, meld_masks = _build_hand_state(hand)
    memo: dict[int, int] = {}

    def _dp(used: int) -> int:
        if used in memo:
            return memo[used]
        best = sum(vals[i] for i in range(n) if not (used >> i & 1))
        for mm in meld_masks:
            if not (mm & used):
                best = min(best, _dp(used | mm))
        memo[used] = best
        return best

    return _dp(0)


def get_optimal_meld_cards(hand: list[str]) -> set[str]:
    if not hand:
        return set()
    n, vals, meld_masks = _build_hand_state(hand)
    memo: dict[int, tuple[int, int]] = {}

    def _dp(used: int) -> tuple[int, int]:
        if used in memo:
            return memo[used]
        best_dw    = sum(vals[i] for i in range(n) if not (used >> i & 1))
        best_final = used
        for mm in meld_masks:
            if not (mm & used):
                child_dw, child_final = _dp(used | mm)
                if child_dw < best_dw:
                    best_dw, best_final = child_dw, child_final
        memo[used] = (best_dw, best_final)
        return best_dw, best_final

    _, optimal_mask = _dp(0)
    return {hand[i] for i in range(n) if (optimal_mask >> i & 1)}


def meld_potential(upcard: str, hand: list[str]) -> int:
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    return max(0, compute_optimal_deadwood(hand) - compute_optimal_deadwood(hand + [upcard]))


def is_adjacent_in_suit(card: str, hand: list[str]) -> bool:
    idx  = RANK_IDX[get_rank(card)]
    suit = get_suit(card)
    for c in hand:
        if c == card or get_suit(c) != suit:
            continue
        if abs(RANK_IDX[get_rank(c)] - idx) <= 1:
            return True
    return False


def partial_run_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    idx  = RANK_IDX[get_rank(card)]
    suit = get_suit(card)
    indices_in_hand = sorted(RANK_IDX[get_rank(c)] for c in hand if get_suit(c) == suit)
    if idx not in indices_in_hand:
        return True
    seg = [idx]
    for i in sorted(indices_in_hand):
        if i == seg[-1] + 1:
            seg.append(i)
        elif i > seg[-1] + 1 and seg[0] <= idx <= seg[-1]:
            break
        elif i < idx:
            if not seg or i == seg[0] - 1:
                seg.insert(0, i)
    if len(seg) >= 3:
        return True
    lo, hi = seg[0], seg[-1]
    if lo > 0 and (RANK_ORDER[lo - 1] + suit) not in dead:
        return True
    if hi < 12 and (RANK_ORDER[hi + 1] + suit) not in dead:
        return True
    return False


def partial_set_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    rank      = get_rank(card)
    hand_suits = {get_suit(c) for c in hand if get_rank(c) == rank}
    if len(hand_suits) >= 3:
        return True
    for s in 'shdc':
        candidate = rank + s
        if s not in hand_suits and candidate not in dead:
            return True
    return False


##############################################################################################
# Observation parsers (verbatim from winner repo — expects RAW env format)
##############################################################################################

_RE_PHASE        = re.compile(r'Phase:\s*(\w+)')
_RE_PLAYER       = re.compile(r'You are Player (\d+)')
_RE_UPCARD       = re.compile(r'Upcard:\s*(\w+)')
_RE_DEADWOOD     = re.compile(r'Deadwood=(\d+)')
_RE_KNOCK_CARD   = re.compile(r'Knock card:\s*(\d+)')
_RE_LEGAL        = re.compile(r'^\s*(\d+)\s*->\s*Player:\s*\d+\s*Action:\s*(.+)$', re.MULTILINE)
_RE_DISCARD_PILE = re.compile(r'Discard pile[:\s]+([^\n]+)', re.IGNORECASE)
_RE_CARD         = re.compile(r'([A2-9TJQK][shdc])')
_RE_CARD_EXACT   = re.compile(r'^([A2-9TJQK][shdc])$')
_RE_MELD_GROUP   = re.compile(r'^([A2-9TJQK][shdc]){2,}$')


def parse_phase(obs: str) -> str:
    m = _RE_PHASE.search(obs)
    return m.group(1) if m else ''


def parse_hand(obs: str) -> list[str]:
    player_match = _RE_PLAYER.search(obs)
    pid     = player_match.group(1) if player_match else '0'
    section = re.search(
        rf'Player{pid}: Deadwood=\d+.*?\n\+-+\+\n(.*?)\n\+-+\+',
        obs, re.DOTALL
    )
    if not section:
        return []
    cards = []
    for row in section.group(1).strip().split('\n'):
        cards.extend(_RE_CARD.findall(row))
    return cards


def parse_upcard(obs: str) -> str:
    m = _RE_UPCARD.search(obs)
    return m.group(1) if m else 'XX'


def parse_deadwood(obs: str) -> int:
    m = _RE_DEADWOOD.search(obs)
    return int(m.group(1)) if m else 99


def parse_knock_card(obs: str) -> int:
    m = _RE_KNOCK_CARD.search(obs)
    return int(m.group(1)) if m else 10


def parse_legal_actions(obs: str) -> list[tuple[str, str]]:
    return _RE_LEGAL.findall(obs)


def parse_discard_pile(obs: str, upcard: Optional[str] = None) -> set[str]:
    cards: set[str] = set()
    m = _RE_DISCARD_PILE.search(obs)
    if m:
        cards.update(_RE_CARD.findall(m.group(1)))
    if upcard is None:
        upcard = parse_upcard(obs)
    if upcard and upcard != 'XX' and len(upcard) == 2:
        cards.add(upcard)
    return cards


##############################################################################################
# Strategy (verbatim from winner repo)
##############################################################################################

def _hand_stats(hand: list[str]) -> tuple[dict[str, int], set[str]]:
    rank_counts: dict[str, int] = Counter(get_rank(c) for c in hand)
    adj_cards = {c for c in hand if is_adjacent_in_suit(c, hand)}
    return rank_counts, adj_cards


def discard_score(card: str, hand: list[str], meld_cards: set[str],
                  rank_counts: dict[str, int], adj_cards: set[str],
                  dead_cards: Optional[set[str]] = None) -> int:
    score = card_value(card)
    if card in meld_cards:
        score -= 15
    has_pair = rank_counts[get_rank(card)] >= 2
    has_adj  = card in adj_cards
    if dead_cards is not None:
        if has_pair and partial_set_can_complete(card, hand, dead_cards):
            score -= 8
        if has_adj and partial_run_can_complete(card, hand, dead_cards):
            score -= 5
    else:
        if has_pair:
            score -= 8
        if has_adj:
            score -= 5
    return score


def choose_discard(hand: list[str], legal: list[tuple[str, str]], deadwood: int,
                   knock_card: int, dead_cards: Optional[set[str]] = None) -> str:
    if deadwood <= knock_card:
        knock_id = next((aid for aid, lbl in legal if lbl.strip().lower() == 'knock'), None)
        if knock_id:
            return knock_id

    meld_cards = get_optimal_meld_cards(hand)
    rank_counts, adj_cards = _hand_stats(hand)

    best_id, best_score = None, None
    for aid, label in legal:
        card_match = _RE_CARD_EXACT.match(label.strip())
        if not card_match:
            continue
        card = card_match.group(1)
        s = discard_score(card, hand, meld_cards, rank_counts, adj_cards, dead_cards)
        if best_score is None or s > best_score:
            best_score = s
            best_id = aid

    return best_id or legal[0][0]


def choose_draw(hand: list[str], upcard: str, legal: list[tuple[str, str]]) -> str:
    upcard_id = next((aid for aid, lbl in legal if 'Draw upcard' in lbl), None)
    stock_id  = next((aid for aid, lbl in legal if 'Draw stock'  in lbl), None)
    pass_id   = next((aid for aid, lbl in legal if lbl.strip() == 'Pass'), None)

    if upcard and upcard != 'XX' and upcard_id:
        if meld_potential(upcard, hand) > 0:
            return upcard_id

    if pass_id and not stock_id:
        return pass_id

    return stock_id or upcard_id or legal[0][0]


def choose_meld_or_layoff_action(legal: list[tuple[str, str]], hand: list[str],
                                  dead_cards: Optional[set[str]] = None) -> str:
    pass_id = next((aid for aid, lbl in legal if lbl.strip().lower() == 'pass'), None)

    for aid, label in legal:
        if _RE_MELD_GROUP.match(label.strip()):
            return aid

    meld_cards = get_optimal_meld_cards(hand)
    rank_counts, adj_cards = _hand_stats(hand)
    best_id, best_score = None, None
    for aid, label in legal:
        if aid == pass_id:
            continue
        card_match = _RE_CARD_EXACT.match(label.strip())
        if not card_match:
            continue
        card = card_match.group(1)
        s = discard_score(card, hand, meld_cards, rank_counts, adj_cards, dead_cards)
        if best_score is None or s > best_score:
            best_score = s
            best_id = aid

    return best_id or pass_id or legal[0][0]


##############################################################################################
# Expert action selector + episode runner
##############################################################################################

def get_expert_action(messages: list[dict]) -> str:
    """Return action ID from RAW env observation in last user message.

    NOTE: this reads `messages[-1]` user content directly. The expert parser
    expects RAW env observation format (not PvP-reformatted). Use the
    side-channel `_last_raw_obs` inside `generate_expert_episode` to keep raw
    obs available even though the conversation messages are PvP-reformatted.
    """
    obs        = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    phase      = parse_phase(obs)
    hand       = parse_hand(obs)
    upcard     = parse_upcard(obs)
    deadwood   = parse_deadwood(obs)
    knock_card = parse_knock_card(obs)
    legal      = parse_legal_actions(obs)
    dead_cards = parse_discard_pile(obs, upcard)

    if not legal:
        return "54"  # Pass — safe default

    if phase in ("Draw", "FirstUpcard"):
        return choose_draw(hand, upcard, legal)

    if phase == "Discard":
        return choose_discard(hand, legal, deadwood, knock_card, dead_cards)

    if phase in ("Knock", "Layoff", "Wall"):
        return choose_meld_or_layoff_action(legal, hand, dead_cards)

    return legal[0][0]


def _get_expert_action_from_raw(raw_obs: str) -> str:
    """Wrapper: build a single-message context from raw env obs for the parser."""
    return get_expert_action([{"role": "user", "content": raw_obs}])


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 200,
) -> "list[dict] | None":
    """Run one Gin Rummy game against the env server using the expert policy.

    User messages in the returned `messages` list are **PvP-formatted** so SFT
    trains the exact distribution PvP eval will feed at tournament Senin
    2026-05-25. The expert internally reads the RAW env observation (kept as
    `_last_raw_obs` side-channel) because the parser regex expects env format.

    PvP eval covers both player perspectives via position swap (each seed
    played twice with players swapped). We alternate `player_id = game_id % 2`
    so the SFT dataset includes both perspectives.

    max_turn=200 matches winner config (gin rummy episodes can run long).
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,  # mirrors winner config
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block       = res.json()["result"]
        episode_id  = block.get("episode_id", "")
        raw_observation = extract_and_format_observation(block.get("observation", ""))
    except Exception as exc:
        print(f"[gin_rummy_trajectories] Reset failed (game {game_id}): {exc}")
        return None

    # PvP eval alternates player perspectives. Use game_id parity.
    player_id = game_id % 2

    user_prompt = build_pvp_user_prompt_gin_rummy(raw_observation, player_id=player_id)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    # Side-channel: keep latest raw obs for the expert parser
    _last_raw_obs = raw_observation

    for _ in range(max_turn):
        # Expert reads RAW env observation (not PvP-reformatted)
        action = _get_expert_action_from_raw(_last_raw_obs)
        messages.append({"role": "assistant", "content": action})

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            raw_observation = extract_and_format_observation(step_block.get("observation", ""))
            done            = step_block.get("done", False)
        except Exception as exc:
            print(f"[gin_rummy_trajectories] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        _last_raw_obs = raw_observation
        user_prompt = build_pvp_user_prompt_gin_rummy(raw_observation, player_id=player_id)
        messages.append({"role": "user", "content": user_prompt})
    else:
        print(f"[gin_rummy_trajectories] max_turn={max_turn} reached (game {game_id})")

    return messages
