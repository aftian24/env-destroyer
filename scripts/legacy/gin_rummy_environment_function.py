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


def would_complete_run(hand: list[str], card: str) -> bool:
    """Check if adding this card would complete a run (3+)"""
    current_runs = find_potential_runs(hand)
    current_complete = sum(1 for run in current_runs if len(run) >= 3)
    
    new_runs = find_potential_runs(hand, card)
    new_complete = sum(1 for run in new_runs if len(run) >= 3)
    
    return new_complete > current_complete


def would_improve_run(hand: list[str], card: str) -> bool:
    """Check if adding this card would improve a potential run (2 -> 3 cards)"""
    # Check if card fits into any existing 2-card sequence
    rank = get_rank(card)
    suit = get_suit(card)
    rank_idx = RANK_ORDER.index(rank)
    
    for existing_card in hand:
        if get_suit(existing_card) != suit:
            continue
        
        existing_rank_idx = RANK_ORDER.index(get_rank(existing_card))
        
        # Check if adjacent
        if abs(rank_idx - existing_rank_idx) == 1:
            # Found an adjacent card, now check if we have another to make 3
            test_hand = hand + [card]
            runs = find_potential_runs(test_hand)
            for run in runs:
                if card in run and len(run) >= 3:
                    return True
    
    return False


def would_complete_set(hand: list[str], card: str) -> bool:
    """Check if adding this card would complete a set (3+ same rank)"""
    rank = get_rank(card)
    same_rank_count = sum(1 for c in hand if get_rank(c) == rank)
    return same_rank_count >= 2  # Adding this would make 3+


def would_improve_set(hand: list[str], card: str) -> bool:
    """Check if adding this card would start a pair"""
    rank = get_rank(card)
    same_rank_count = sum(1 for c in hand if get_rank(c) == rank)
    return same_rank_count == 1  # We have 1, this makes a pair


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


def extract_and_format_observation(obs_text):
    """
    Extract and format observation to match evaluation format.
    Reconstructs legal actions from player hand.
    
    Args:
        obs_text: Raw observation text from result_block
        
    Returns:
        Formatted observation string matching evaluation format
    """
    
    # Case 1: Error message with legal actions already present
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        # Already formatted correctly, return as-is
        return obs_text
    
    # Case 2: Normal observation - extract state and reconstruct legal actions

    # Extract everything after "Current State:"
    state_match = re.search(
        r'Current State:\n(.*)',
        obs_text,
        re.DOTALL
    )
    
    if not state_match:
        # Fallback: return original if no "Current State:" found
        return obs_text
    
    state_text = state_match.group(0)
    
    # Extract player ID
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Put player ID between Current State and Legal Actions
    current_state_text, legal_action_text = state_text.split('Legal Actions:')
    formatted_state_text = current_state_text + f"You are Player {player_id}.\nLegal Actions:" + legal_action_text

    return formatted_state_text


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
    
    # Player ID
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Hand
    hand = parse_hand_from_observation(observation)
    
    # Deadwood
    deadwood_match = re.search(r'Deadwood=(\d+)', observation)
    deadwood = int(deadwood_match.group(1)) if deadwood_match else 0
    
    # Phase
    phase_match = re.search(r'Phase: (\w+)', observation)
    phase = phase_match.group(1) if phase_match else 'Draw'
    
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
    
    # Turn number (from stock size - starts at ~33-40, decreases each turn)
    initial_stock = 33  # Approximate starting stock for 10-card hands
    turn_number = max(0, initial_stock - stock_size)
    
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
    """Calculate rewards using state history"""
    
    def __init__(self, gamma: float = 0.95):
        """
        Args:
            gamma: Discount factor (0.99 = later rewards matter more)
        """
        self.gamma = gamma
        
        # Reward weights
        self.deadwood_weight = 0.5
        self.high_card_penalty = -0.2
        
        # Meld rewards
        self.pair_bonus = 2.0
        self.set_bonus = 10.0
        self.potential_run_bonus = 3.0
        self.run_bonus = 12.0
        
        # Penalties
        self.break_pair_penalty = -2.0
        self.break_set_penalty = -10.0
        self.break_run_penalty = -12.0
        
        # Strategic rewards
        self.knock_ready_bonus = 20.0
        self.discard_useful_penalty = -2.0
        self.missed_opportunity_penalty = -3.0
        self.picked_up_useless_upcard_penalty = -3.0
        
        
    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float
    ) -> float:
        """
        Calculate immediate reward for this step (no discounting here).
        
        Args:
            states: List of GameState objects (last item = current state)
            action: Action that was taken
            env_reward: Terminal reward from environment
            
        Returns:
            Immediate step reward (not discounted)
        """
        if len(states) < 2:
            return 0.0
        
        prev_state = states[-2]
        current_state = states[-1]
        
        reward = 0.0
        
        # R1: Deadwood reduction
        deadwood_change = prev_state.deadwood - current_state.deadwood
        reward += self.deadwood_weight * deadwood_change
        
        # R2: High card penalty
        reward += self.high_card_penalty * current_state.num_high_cards()
        
        # R3: Pair formation/breaking
        prev_pairs = prev_state.count_pairs()
        curr_pairs = current_state.count_pairs()
        pair_change = curr_pairs - prev_pairs
        
        if pair_change > 0:
            reward += self.pair_bonus * pair_change
        elif pair_change < 0:
            reward += self.break_pair_penalty * abs(pair_change)
        
        # R4: Set completion/breaking
        prev_sets = prev_state.count_sets()
        curr_sets = current_state.count_sets()
        set_change = curr_sets - prev_sets
        
        if set_change > 0:
            reward += self.set_bonus * set_change
        elif set_change < 0:
            reward += self.break_set_penalty * abs(set_change)
        
        # R5: Run completion/breaking
        prev_runs = prev_state.count_runs()
        curr_runs = current_state.count_runs()
        run_change = curr_runs - prev_runs
        
        if run_change > 0:
            reward += self.run_bonus * run_change
        elif run_change < 0:
            reward += self.break_run_penalty * abs(run_change)
        
        # R6: Potential run formation
        prev_potential_runs = prev_state.count_potential_runs()
        curr_potential_runs = current_state.count_potential_runs()
        potential_run_change = curr_potential_runs - prev_potential_runs
        
        if potential_run_change > 0:
            reward += self.potential_run_bonus * potential_run_change
        
        # R7: Knock readiness
        if current_state.can_knock() and not prev_state.can_knock():
            reward += self.knock_ready_bonus
        
        # R8: Discard awareness - did we discard something useful?
        if prev_state.phase == 'Discard' and len(current_state.discard_pile) > len(prev_state.discard_pile):
            newly_discarded = [c for c in current_state.discard_pile if c not in prev_state.discard_pile]
            if newly_discarded:
                discarded_card = newly_discarded[0]
                discarded_rank = get_rank(discarded_card)
                same_rank_in_hand = sum(1 for c in prev_state.hand if get_rank(c) == discarded_rank)
                
                if same_rank_in_hand >= 2:
                    reward += self.discard_useful_penalty
                
                if would_improve_run(prev_state.hand, discarded_card):
                    reward += self.discard_useful_penalty
        
        # R9: Missed opportunity - didn't pick up a useful upcard
        if prev_state.phase == 'Draw' and prev_state.upcard != 'XX':
            upcard = prev_state.upcard
            
            if action == '53':  # Drew from stock instead of upcard
                # Penalize missed opportunity to complete set or run
                if would_complete_set(prev_state.hand, upcard):
                    reward += self.missed_opportunity_penalty
                elif would_complete_run(prev_state.hand, upcard):
                    reward += self.missed_opportunity_penalty
            else: # Drew from upcard
                # Penalize if picked up a not useful upcard, only useful if it completes a set or run
                if not (
                    would_complete_set(prev_state.hand, upcard) or 
                    would_complete_run(prev_state.hand, upcard)
                ):
                    reward += self.picked_up_useless_upcard_penalty
        
        # R10: Terminal reward (only added on final step)
        if env_reward != 0.0:
            reward += max(min(env_reward * 100.0, 50.0), -50.0)
        
        return reward
    
    def calculate_discounted_return(self, rewards: list[float]) -> float:
        """
        Apply discounting to list of step rewards (called ONCE at episode end).
        Later rewards weighted more heavily.
        
        Args:
            rewards: List of immediate step rewards [r_0, r_1, r_2, ..., r_T]
            
        Returns:
            Discounted return where later rewards matter more
            
        Example:
            rewards = [1.0, 2.0, 5.0, 10.0]
            With gamma=0.99:
            - r_0 (early) gets weight gamma^3 = 0.99^3 ≈ 0.970
            - r_1 gets weight gamma^2 = 0.99^2 ≈ 0.980
            - r_2 gets weight gamma^1 = 0.99^1 = 0.990
            - r_3 (late) gets weight 1.0 (full weight)
        """
        if not rewards:
            return 0.0
        
        T = len(rewards)
        discounted_sum = 0.0
        
        for i, reward in enumerate(rewards):
            # Later rewards (higher i) get less discounting (closer to 1.0)
            discount_factor = self.gamma ** (T - 1 - i)
            discounted_sum += discount_factor * reward
        
        return discounted_sum
    
    
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
    
    
class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.
    """
    def __init__(
        self,
        initial_max_turn=1,
        final_max_turn=13,
        rollouts_per_stage=1280,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        warmup_rollouts=128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        
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
    
    def get_hint_prob(self):
        """Calculate current hint probability based on curriculum."""
        if self.total_rollouts < self.warmup_rollouts:
            # During warmup, always hint
            return self.initial_hint_prob
        
        # Linearly decay from initial to final over training
        # Decay over the course of reaching final_max_turn
        total_stages = self.final_max_turn - self.initial_max_turn
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)
    
    def step(self, num_rollouts=1):
        """Increment rollout counter."""
        self.total_rollouts += num_rollouts
        
    def get_status(self):
        """Get current curriculum status for logging."""
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(),
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
        
        # Initialize curriculum scheduler
        rollout_last_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.5,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=30, rollouts_per_stage={trainer.args.rollouts_per_stage}, warmup_rollouts={trainer.args.rollouts_per_stage}")

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
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

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
        
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}

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

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\nSETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\nMELDS (Valid Combinations):\n1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\nExamples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\nCARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\nGAME PHASES:\n1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\nEACH TURN:\n1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\nKNOCKING:\n- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\nSCORING: Winner scores difference in deadwood point values.\nCard Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\nIMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""
        
        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = "\n\n# Strategy Tips\n- Early game: Draw from deck to see more cards\n- Build runs and sets to reduce deadwood\n- Track opponent's discards to guess their hand\n- Knock when you have ≤10 deadwood points and think you're ahead\n- Go for Gin (0 deadwood) when close for bonus points"
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
            action_to_send = remove_reasoning_tags(completion_text)
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

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

            if done:
                final_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})
                
            # Parse Game State
            if not is_invalid and not done:
                # Normal, not invalid and not done
                try:
                    game_state = parse_game_state(formatted_observation)
                except Exception as e:
                    # Failed to parse game state, probaly invalid action in previous turn, skip parsing for this turn
                    print(f"Failed to parse game state: {e}")
                    immediate_reward = -10.0
                else:
                    game_state_history.append(game_state)
                    # Calculate Reward
                    immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0)
            elif is_invalid:
                # Invalid action
                immediate_reward = -10.0
            else:
                # Done
                # Re-calculate game reward
                game_reward = step_reward - 0.5
                immediate_reward = max(min(game_reward * 100.0, 50.0), -50.0)

            rewards.append(immediate_reward)
            turn_number += 1
            
        # Calculate discounted return
        discounted_return = calculator.calculate_discounted_return(rewards)
        train_reward = discounted_return
        
        initial_dw = game_state_history[0].deadwood if game_state_history else 0
        final_dw = game_state_history[-1].deadwood if game_state_history else 0
        
        # Single-line episode summary
        print(f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{discounted_return:6.2f} EnvR:{final_reward:5.1f} "
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
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}")


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
    MAX_PROMPT_LEN = 5000      # Max prompt tokens before ending episode early
    
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
        
        # Initialize curriculum scheduler
        rollout_full_prompt_and_completion_parallelized_curriculum.curriculum = CurriculumScheduler(
            initial_max_turn=trainer.args.initial_max_turn,
            final_max_turn=30,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.5,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=30, rollouts_per_stage={trainer.args.rollouts_per_stage}, warmup_rollouts={trainer.args.rollouts_per_stage}")

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
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

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
        
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}

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

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\nSETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\nMELDS (Valid Combinations):\n1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\nExamples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\nCARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\nGAME PHASES:\n1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\nEACH TURN:\n1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\nKNOCKING:\n- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\nSCORING: Winner scores difference in deadwood point values.\nCard Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\nIMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""
        
        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = "\n\n**Think short and act quickly!**\n\n# Strategy Tips\n- Early game: Draw from deck to see more cards\n- Build runs and sets to reduce deadwood\n- Track opponent's discards to guess their hand\n- Knock when you have ≤10 deadwood points and think you're ahead\n- Go for Gin (0 deadwood) when close for bonus points"
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

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE mismatch - tokenizer produced different IDs for same prefix text
                    # Graceful fallback: skip delta masking for this turn, just add completion
                    print(
                        f"Warning: BPE mismatch at turn {turn_number} (expected prefix {len(prev_full_ids)}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
                    # Reset prev_full_ids to current prompt to try to recover alignment
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids):]
                    if delta_prompt_ids:
                        episode_completion_ids.extend(delta_prompt_ids)
                        episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                        episode_action_mask.extend([0] * len(delta_prompt_ids))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids
            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

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

            if done:
                final_reward = step_reward
                messages.append({"role": "user", "content": formatted_observation})
            else:
                messages.append({"role": "user", "content": formatted_observation})
                
            # Parse Game State
            if not is_invalid and not done:
                # Normal, not invalid and not done
                try:
                    game_state = parse_game_state(formatted_observation)
                except Exception as e:
                    # Failed to parse game state, probaly invalid action in previous turn, skip parsing for this turn
                    print(f"Failed to parse game state: {e}")
                    immediate_reward = -10.0
                else:
                    game_state_history.append(game_state)
                    # Calculate Reward
                    immediate_reward = calculator.calculate_step_reward(game_state_history, action_to_send, 0.0)
            elif is_invalid:
                # Invalid action
                immediate_reward = -10.0
            else:
                # Done
                # Re-calculate game reward
                game_reward = step_reward - 0.5
                immediate_reward = max(min(game_reward * 100.0, 50.0), -50.0)
                
            rewards.append(immediate_reward)
            turn_number += 1
            
        # Calculate discounted return
        discounted_return = calculator.calculate_discounted_return(rewards)
        train_reward = discounted_return
        
        initial_dw = game_state_history[0].deadwood if game_state_history else 0
        final_dw = game_state_history[-1].deadwood if game_state_history else 0
        
        # Single-line episode summary
        print(f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{discounted_return:6.2f} EnvR:{final_reward:5.1f} "
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
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}")


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