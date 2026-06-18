"""PvP-format prompt builders for SFT trajectory generation.

Mirrors `validator/evaluation/pvp/agents.py` from G.O.D `feature/pvp-eval-container`
branch (canonical eval format for tournament starting 2026-05-25).

Each agent class converts env server raw observation OR pyspiel state into the
exact PvP user-prompt format the validator will feed to our model during eval:

    Current State:
    {state_desc}

    You are Player {player_id}.
    Legal Actions:
    {action_id} -> {action_str}
    ...

    Your choice (ID only):

Match-target source: gradients-ai/G.O.D feature/pvp-eval-container
File: validator/evaluation/pvp/agents.py (cached at scripts/envs/pvp_assets/agents.py)
Prompts: core/config/pvp_game_prompts.yml (cached at scripts/envs/pvp_assets/pvp_game_prompts.yml)

SFT trajectory generators import from here to produce eval-format-matching data.
"""

import re
from pathlib import Path

import yaml


_ASSETS = Path(__file__).resolve().parent / "pvp_assets"
_PROMPTS_PATH = _ASSETS / "pvp_game_prompts.yml"


def _load_prompts() -> dict[str, str]:
    """Load PvP system prompt template + per-game rules."""
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# System prompt builder (shared)
# ---------------------------------------------------------------------------

def build_system_prompt(game_name: str) -> str:
    """Build PvP system prompt for the given game ("liars_dice", "leduc_poker", "gin_rummy")."""
    prompts = _load_prompts()
    rules_key = f"{game_name}_rules"
    if rules_key not in prompts:
        raise ValueError(f"Unknown game: {game_name} (no {rules_key} in pvp_game_prompts.yml)")
    return prompts["system_prompt_template"].format(
        game_name=game_name, rules=prompts[rules_key]
    )


def build_user_prompt(state_desc: str, player_id: int, legal_actions_block: str) -> str:
    """Build PvP user prompt with state + Player ID + Legal Actions + suffix.

    Args:
        state_desc: Game-specific state description (use the per-game formatters below)
        player_id: 0 or 1 (PvP eval covers both perspectives via position swap)
        legal_actions_block: Multi-line string with each action as "{id} -> {str}"
    """
    return (
        f"Current State:\n{state_desc}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{legal_actions_block}\n\n"
        "Your choice (ID only):"
    )


# ---------------------------------------------------------------------------
# Per-game state formatters
# ---------------------------------------------------------------------------

def format_liars_dice_state(
    dice: list[int],
    num_players: int = 2,
    current_player: int = 0,
    current_bid_quantity: "int | None" = None,
    current_bid_face: "int | None" = None,
) -> str:
    """Format Liar's Dice state per validator's LiarsDiceAgent.format_state.

    Matches `validator/evaluation/pvp/agents.py:LiarsDiceAgent.format_state`.
    """
    num_dice = len(dice)
    total_dice = num_dice * num_players

    lines = [
        f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {num_dice}",
        f"Total dice in game: {total_dice}",
        f"Players: {num_players}",
        f"Current player: Player {current_player}",
    ]

    if current_bid_quantity is not None and current_bid_face is not None:
        lines.append(
            f'\nCurrent bid: "{current_bid_quantity}-{current_bid_face}" '
            f"(at least {current_bid_quantity} dice showing {current_bid_face} across all players)"
        )
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        lines.append("No bid yet - you must make the first bid")

    return "\n".join(lines)


def format_leduc_poker_state(
    private_card: "int | None",
    round_num: int,
    pot: int,
    player_chips: int,
    opponent_chips: int,
    public_card: "int | None" = None,
    round1_seq: str = "",
    round2_seq: str = "",
) -> str:
    """Format Leduc Poker state per validator's LeducPokerAgent.format_state.

    card_id encoding: 0=J♠ 1=J♥ 2=Q♠ 3=Q♥ 4=K♠ 5=K♥ (rank=id//2, suit=id%2)
    Hand: PAIR when private_card.rank == public_card.rank
    """
    lines: list[str] = []

    if private_card is not None and private_card != -10000:
        lines.append(f"Your card: {_leduc_card_name(private_card)}")
    else:
        lines.append("Your card: (not dealt yet)")

    if public_card is not None and public_card != -10000:
        lines.append(f"Public card: {_leduc_card_name(public_card)}")
        if private_card is not None and private_card != -10000:
            if private_card // 2 == public_card // 2:
                lines.append("Hand: PAIR")

    lines.append(f"Round: {round_num}/2")
    lines.append(f"Pot: {pot} chips")
    lines.append(f"Your chips: {player_chips}")
    lines.append(f"Opponent chips: {opponent_chips}")

    if round1_seq:
        lines.append(f"Round 1 actions: {_parse_betting(round1_seq)}")
    if round2_seq:
        lines.append(f"Round 2 actions: {_parse_betting(round2_seq)}")

    return "\n".join(lines)


def _leduc_card_name(card_id: int) -> str:
    """0=J♠ 1=J♥ 2=Q♠ 3=Q♥ 4=K♠ 5=K♥ (per pvp/agents.py:_card_name)."""
    ranks = ["J", "Q", "K", "A"]  # A used only in 3+ player variants
    suits = ["♠", "♥"]  # ♠ ♥
    rank_idx = card_id // 2
    suit_idx = card_id % 2
    if rank_idx < len(ranks):
        return f"{ranks[rank_idx]}{suits[suit_idx]}"
    return f"Card_{card_id}"


def _parse_betting(seq: str) -> str:
    """Translate Leduc betting sequence like '1 1' → 'Call, Call'."""
    if not seq or not seq.strip():
        return "(none)"
    actions_map = {0: "Fold", 1: "Call", 2: "Raise"}
    numbers = [int(x) for x in seq.split() if x.isdigit()]
    if not numbers:
        return "(none)"
    return ", ".join(actions_map.get(a, f"Action{a}") for a in numbers)


# ---------------------------------------------------------------------------
# Parse env server's raw observation into PvP-formatted state
# ---------------------------------------------------------------------------

def reformat_liars_dice_observation(env_obs: str, player_id: int = 0) -> tuple[str, str]:
    """Convert env server's Liar's Dice observation into PvP-compatible
    (state_desc, legal_actions_block) tuple ready to feed into `build_user_prompt`.

    Env observation contains lines like:
        Your dice: [3, 5, 6]
        Total dice in game: 10
        Current bid: "2-5"
        Legal Actions:
        3 -> 1-1
        ...
    """
    # Parse from env observation
    dice_match = re.search(r"Your dice:\s*\[([^\]]+)\]", env_obs)
    dice = [int(x.strip()) for x in dice_match.group(1).split(",")] if dice_match else []

    total_dice_match = re.search(r"Total dice in game:\s*(\d+)", env_obs)
    total_dice = int(total_dice_match.group(1)) if total_dice_match else 10
    num_dice = len(dice) if dice else 5
    num_players = max(2, total_dice // max(num_dice, 1))

    bid_match = re.search(r'Current bid:\s*"(\d+)-(\d+)"', env_obs)
    bid_q = int(bid_match.group(1)) if bid_match else None
    bid_f = int(bid_match.group(2)) if bid_match else None

    state_desc = format_liars_dice_state(
        dice=dice,
        num_players=num_players,
        current_player=player_id,
        current_bid_quantity=bid_q,
        current_bid_face=bid_f,
    )

    actions_block = _extract_legal_actions_block(env_obs)
    return state_desc, actions_block


def reformat_leduc_poker_observation(env_obs: str, player_id: int = 0) -> tuple[str, str]:
    """Convert env server's Leduc observation into PvP-compatible
    (state_desc, legal_actions_block).

    Env observation format (verified affinetes):
        [Observer: 0]
        [Private: 2]
        [Round 1]
        [Pot: 2]
        [Money: 99 99]
        [Public: -10000]
        [Round1: 1]
        [Round2: ]
        Legal Actions:
        ...
    """
    private = _extract_pattern(env_obs, r"\[Private:\s*(-?\d+)\]")
    public = _extract_pattern(env_obs, r"\[Public:\s*(-?\d+)\]")
    round_num = _extract_pattern(env_obs, r"\[Round\s*(\d+)\]") or "1"
    pot = _extract_pattern(env_obs, r"\[Pot:\s*(\d+)\]") or "0"
    money = _extract_pattern(env_obs, r"\[Money:\s*([\d ]+)\]") or ""
    r1_seq = _extract_pattern(env_obs, r"\[Round1:\s*([^\]]*)\]") or ""
    r2_seq = _extract_pattern(env_obs, r"\[Round2:\s*([^\]]*)\]") or ""

    chips = money.split()
    player_chips = int(chips[player_id]) if len(chips) > player_id else 100
    opp_chips = int(chips[1 - player_id]) if len(chips) > (1 - player_id) else 100

    state_desc = format_leduc_poker_state(
        private_card=int(private) if private else None,
        round_num=int(round_num),
        pot=int(pot),
        player_chips=player_chips,
        opponent_chips=opp_chips,
        public_card=int(public) if public else None,
        round1_seq=r1_seq,
        round2_seq=r2_seq,
    )

    actions_block = _extract_legal_actions_block(env_obs)
    return state_desc, actions_block


def reformat_gin_rummy_observation(env_obs: str, player_id: int = 0) -> tuple[str, str]:
    """Gin Rummy uses raw observation_string per validator's GinRummyAgent.format_state.

    The env server's observation_string IS what pyspiel produces, so we use it
    as-is for state_desc. Only extract legal actions block separately.
    """
    actions_block = _extract_legal_actions_block(env_obs)
    # Strip the legal actions block from env_obs to get pure state description
    state_desc = re.sub(
        r"\n*Legal Actions:\n(?:[ \t]*\d+[ \t]*->[ \t]*\S.*(?:\n|$))+", "", env_obs
    ).strip()
    return state_desc, actions_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pattern(text: str, pattern: str) -> str:
    """Return first regex group match or empty string."""
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def _extract_legal_actions_block(env_obs: str) -> str:
    """Extract the `<id> -> <label>` lines from env observation.

    Returns multi-line string ready to plug into `build_user_prompt`.
    """
    actions = re.findall(r"^[ \t]*(\d+)[ \t]*->[ \t]*(.+)$", env_obs, re.MULTILINE)
    return "\n".join(f"{aid} -> {label.strip()}" for aid, label in actions)


# ---------------------------------------------------------------------------
# All-in-one entrypoint per game
# ---------------------------------------------------------------------------

def build_pvp_user_prompt_liars_dice(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_liars_dice_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_pvp_user_prompt_leduc_poker(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_leduc_poker_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_pvp_user_prompt_gin_rummy(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_gin_rummy_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


# ---------------------------------------------------------------------------
# Per-game system prompts (cached)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LIARS_DICE = build_system_prompt("liars_dice") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_LEDUC_POKER = build_system_prompt("leduc_poker") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_GIN_RUMMY = build_system_prompt("gin_rummy") if _PROMPTS_PATH.exists() else None
