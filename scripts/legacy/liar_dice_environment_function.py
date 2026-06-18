from trl.experimental.openenv import generate_rollout_completions
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import os
import random
import requests
import re
from dataclasses import dataclass
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Data structures and probability functions from liar_dice_strat.py
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    quantity: int
    face: int

    def __str__(self):
        return f"{self.quantity}-{self.face}"


@dataclass
class Action:
    action_id: int
    label: str
    bid: Bid | None
    prob: float = 0.0

    @property
    def is_liar(self) -> bool:
        return self.label.strip().lower() == "liar"

    @property
    def aggressiveness(self) -> float:
        return max(self.action_id / 59, 5/59) ** 0.5

    @property
    def score(self) -> float:
        # return self.prob if self.is_liar else self.prob * self.aggressiveness
        return self.prob


# ---------------------------------------------------------------------------
# Risky move reward parameters
# ---------------------------------------------------------------------------

# A bid with prob below this is a bluff (we're claiming something unlikely from our view)
BLUFF_PROB_THRESHOLD = 0.35

# A liar call with prob in [MIN, MAX] is risky: not confident enough to be "safe",
# not so wild it's just noise.
RISKY_LIAR_PROB_MIN = 0.35
RISKY_LIAR_PROB_MAX = 0.60

# Bonus added to the terminal reward per risky action that occurred in a winning episode
BLUFF_WIN_BONUS     = 0.5
RISKY_LIAR_WIN_BONUS = 0.5

# Cap how many risky actions contribute to the bonus (avoids incentivising spam)
RISKY_BONUS_MAX_COUNT = 2


@dataclass
class GameState:
    our_dice: list[int]
    total_dice: int
    current_bid: Bid | None
    actions: list[Action]

    @property
    def liar_action(self) -> Action | None:
        return next((a for a in self.actions if a.is_liar), None)

    @property
    def bid_actions(self) -> list[Action]:
        return [a for a in self.actions if not a.is_liar]


def bid_probability(bid: Bid | None, state: GameState) -> float:
    """P(bid is true) given the observable game state."""
    if bid is None:
        return 0.0

    our_dice   = state.our_dice   or []
    total_dice = state.total_dice or 0

    if bid.face == 6:
        our_count = sum(1 for d in our_dice if d == 6)
        p_hit = 1 / 6
    else:
        our_count = sum(1 for d in our_dice if d == bid.face or d == 6)
        p_hit = 2 / 6

    still_needed = bid.quantity - our_count
    if still_needed <= 0:
        return 1.0

    n_hidden = total_dice - len(our_dice)
    if n_hidden <= 0:
        return 0.0

    return 1.0 - binom.cdf(still_needed - 1, n=n_hidden, p=p_hit)


def _parse_bid_label(label: str) -> Bid | None:
    m = re.fullmatch(r"(\d+)-(\d+)", label.strip())
    return Bid(int(m.group(1)), int(m.group(2))) if m else None


def parse_game_state(messages: list[dict] | str) -> GameState:
    """Parse the last user message in a conversation into a GameState."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), None
    )
    if last_user_msg is None:
        raise ValueError("No user message found")

    # Our dice
    dice_match = re.search(r"Your dice:\s*\[([^\]]+)\]", last_user_msg)
    if not dice_match:
        raise ValueError("Could not parse 'Your dice'")
    our_dice = [int(x.strip()) for x in dice_match.group(1).split(",")]

    # Total dice
    total_match = re.search(r"Total dice in game:\s*(\d+)", last_user_msg)
    if not total_match:
        raise ValueError("Could not parse 'Total dice in game'")
    total_dice = int(total_match.group(1))

    # Current bid
    bid_match = re.search(r'Current bid:\s*"(\d+)-(\d+)"', last_user_msg)
    current_bid = Bid(int(bid_match.group(1)), int(bid_match.group(2))) if bid_match else None

    # Legal actions
    raw_actions = re.findall(r"^\s*(\d+)\s*->\s*(.+)$", last_user_msg, re.MULTILINE)
    if not raw_actions:
        raise ValueError("Could not parse legal actions")

    # Temporary state for probability calculations before the actions list is complete
    tmp_state = GameState(our_dice=our_dice, total_dice=total_dice, current_bid=current_bid, actions=[])

    # First pass: build actions with prob; aggressiveness/score filled after we know max_id
    actions = []
    for aid, label in raw_actions:
        bid  = _parse_bid_label(label)
        is_liar = label.strip().lower() == "liar"
        prob = (
            1.0 - bid_probability(current_bid, tmp_state) if is_liar and current_bid
            else (bid_probability(bid, tmp_state) if bid else 0.0)
        )
        actions.append(Action(action_id=int(aid), label=label.strip(), bid=bid, prob=prob))

    actions.sort(key=lambda a: a.action_id)

    return GameState(our_dice=our_dice, total_dice=total_dice, current_bid=current_bid, actions=actions)


class RewardCalculator:
    """
    Shaped reward calculator for Liar's Dice training.

    Step reward = action.score + terminal win/loss signal on the final step.
    """

    def __init__(
        self,
        gamma: float = 0.9,
    ):
        self.terminal_weight = 10.0
        self.gamma           = gamma

    def calculate_step_reward(
        self,
        action: Action | None,
        env_reward: float,
    ) -> float:

        reward = 0.0

        if action is not None:
            reward += action.score

        if env_reward != 0.0:
            terminal_reward = env_reward * self.terminal_weight
            reward  += terminal_reward

        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        """G = Σ γ^(T-1-i) * r_i  (later rewards get weight closer to 1.0)"""
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))


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
      

def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for Liar's Dice game environment.
    Uses full prompt and completion IDs with action masking.
    Threading infrastructure based on gin_rummy implementation.
    """
    
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384
    MAX_PROMPT_LEN = 5000
    
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

    selected_game = "liars_dice"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_full_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}
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
        """
        Run a single episode in a separate thread.
        Calculates reward based on action probability from game state.
        """
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
        bluff_count = 0        # bids made with low prob (bluffing)
        risky_liar_count = 0   # liar calls made without high confidence

        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = raw_observation
            initial_game_state = parse_game_state(formatted_observation)
            game_state_history.append(initial_game_state)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build System Prompt ---
        system_prompt = """"You are playing liars_dice.\n\n# Game Rules\nLIAR'S DICE RULES:\n\nSetup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\nGoal: Make bids about total dice across ALL players, or call \"Liar\" on opponent's bid.\n\nActions:\n- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.\n- Call Liar: Challenge the previous bid.\n\nBidding rules: Each bid must be higher than the previous bid. \"Higher\" means:\n  - Same face value but higher quantity (e.g., \"2 fours\" beats \"1 four\")\n  - Same quantity but higher face value (e.g., \"2 fives\" beats \"2 fours\")\n\nWild dice: 6s are WILD and count as ANY face value.\n- When counting dice for a bid, include 6s in the count\n- Example: Bid \"3 fours\" means at least 3 dice showing EITHER 4 OR 6\n\nWinning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.\n\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""""
        
        # Add strategy hints based on curriculum
        if use_hints:
            hint_prompt = """
# Strategy Tips
- Count your dice that match the bid (including 6s as wild)
- Call "Liar" when you're confident the bid is impossible
- Make conservative bids early, aggressive when opponent seems weak
"""
            system_prompt += hint_prompt
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):
            # Generate Rollout Completion (thread-safe)
            with rollout_full_prompt_and_completion_parallelized_curriculum.generation_semaphore:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length
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
                    # BPE mismatch - graceful fallback
                    print(
                        f"Warning: BPE mismatch at turn {turn_number} (expected prefix {len(prev_full_ids)}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
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
                action_to_send = action_to_send[:-4]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step Environment (POST /step) ---
            is_invalid = False
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                raw_observation = step_block.get("observation", "")
                formatted_observation = raw_observation
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions
            is_invalid = False
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1
                is_invalid = True

            if done:
                final_reward = step_reward
                messages.append({"role": "user", "content": formatted_observation})
            else:
                messages.append({"role": "user", "content": formatted_observation})
            
            # Calculate action probability from game state (before the action was taken)
            last_action_prob = 0.0
            if not is_invalid:
                # Parse game state from the observation BEFORE this action
                try:
                    # Get the previous messages up to (but not including) the last assistant message
                    previous_game_state = game_state_history[-1]
                    if not done:
                        game_state = parse_game_state(formatted_observation)
                    else:
                        # Game already done, only care about result
                        game_state = None
                    action_id = int(action_to_send.strip())
                except Exception as e:
                    # Failed to parse game state or action id, probably invalid action in previous turn, skip parsing for this turn
                    print(f"Failed to parse game state or action id: {e}")
                    immediate_reward = -1.0
                else:
                    taken_action = next((a for a in previous_game_state.actions if a.action_id == action_id), None)
                    last_action_prob = taken_action.prob if taken_action else 0.0

                    # Classify risky moves for bonus tracking
                    if taken_action is not None:
                        if taken_action.is_liar:
                            # Risky liar call: medium confidence (not sure, but calls anyway)
                            if RISKY_LIAR_PROB_MIN <= taken_action.prob <= RISKY_LIAR_PROB_MAX:
                                risky_liar_count += 1
                        else:
                            # Bluff: bid with low probability from our own dice view
                            if taken_action.prob < BLUFF_PROB_THRESHOLD:
                                bluff_count += 1

                    if not done:
                        game_state_history.append(game_state)
                        immediate_reward = calculator.calculate_step_reward(taken_action, 0.0)
                    else:
                        won = step_reward > 0.5
                        game_reward = step_reward - 0.5
                        immediate_reward = game_reward * 2.0
                        if won:
                            # Reward bluffs that paid off
                            bluff_bonus = BLUFF_WIN_BONUS * min(bluff_count, RISKY_BONUS_MAX_COUNT)
                            # Reward gutsy liar calls that paid off
                            risky_liar_bonus = RISKY_LIAR_WIN_BONUS * min(risky_liar_count, RISKY_BONUS_MAX_COUNT)
                            immediate_reward += bluff_bonus + risky_liar_bonus
            else:
                immediate_reward = -1.0

            rewards.append(immediate_reward)
            turn_number += 1

        # Calculate final training reward
        discounted_return = calculator.calculate_discounted_return(rewards)
        train_reward = discounted_return

        print(
            "[ID:{:<6} Hints:{} Done:{} T:{:>2d} | Reward:{:>8.2f} | LastProb:{:>7.3f} | "
            "EnvR:{:>6.1f} | Bluffs:{:<2} RiskyLiar:{:<2} Inv:{:<2}]".format(
                str(game_id)[:6], 
                int(use_hints), 
                int(done), 
                turn_number, 
                train_reward, 
                last_action_prob, 
                final_reward, 
                bluff_count, 
                risky_liar_count, 
                invalid_count)
        )

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




def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for Liar's Dice game environment.
    Returns only the last prompt and completion IDs (no action masking).
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

    selected_game = "liars_dice"

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(rollout_last_prompt_and_completion_parallelized_curriculum, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Initializing env on server {idx}: {base_url}")
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}
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
        last_action_prob = 0.0
        bluff_count = 0        # bids made with low prob (bluffing)
        risky_liar_count = 0   # liar calls made without high confidence

        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            raw_observation = result_block.get("observation", "")
            formatted_observation = raw_observation
            initial_game_state = parse_game_state(formatted_observation)
            game_state_history.append(initial_game_state)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build System Prompt ---
        system_prompt = """"You are playing liars_dice.\n\n# Game Rules\nLIAR'S DICE RULES:\n\nSetup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\nGoal: Make bids about total dice across ALL players, or call \"Liar\" on opponent's bid.\n\nActions:\n- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.\n- Call Liar: Challenge the previous bid.\n\nBidding rules: Each bid must be higher than the previous bid. \"Higher\" means:\n  - Same face value but higher quantity (e.g., \"2 fours\" beats \"1 four\")\n  - Same quantity but higher face value (e.g., \"2 fives\" beats \"2 fours\")\n\nWild dice: 6s are WILD and count as ANY face value.\n- When counting dice for a bid, include 6s in the count\n- Example: Bid \"3 fours\" means at least 3 dice showing EITHER 4 OR 6\n\nWinning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.\n\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""""

        # Add strategy hints based on curriculum
        if use_hints:
            hint_prompt = """
# Strategy Tips
- Count your dice that match the bid (including 6s as wild)
- Call "Liar" when you're confident the bid is impossible
- Make conservative bids early, aggressive when opponent seems weak
"""
            system_prompt += hint_prompt

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
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-4]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step Environment (POST /step) ---
            is_invalid = False
            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                raw_observation = step_block.get("observation", "")
                formatted_observation = raw_observation
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)

            except Exception as e:
                print(f"Step failed: {e}")
                step_reward = -0.01
                done = False
                invalid_count += 1

            # Check for invalid actions
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1
                is_invalid = True

            if done:
                final_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            # Calculate action probability from game state (before the action was taken)
            last_action_prob = 0.0
            if not is_invalid:
                try:
                    previous_game_state = game_state_history[-1]
                    if not done:
                        game_state = parse_game_state(formatted_observation)
                    else:
                        game_state = None
                    action_id = int(action_to_send.strip())
                except Exception as e:
                    print(f"Failed to parse game state or action id: {e}")
                    immediate_reward = -1.0
                else:
                    taken_action = next((a for a in previous_game_state.actions if a.action_id == action_id), None)
                    last_action_prob = taken_action.prob if taken_action else 0.0

                    # Classify risky moves for bonus tracking
                    if taken_action is not None:
                        if taken_action.is_liar:
                            # Risky liar call: medium confidence (not sure, but calls anyway)
                            if RISKY_LIAR_PROB_MIN <= taken_action.prob <= RISKY_LIAR_PROB_MAX:
                                risky_liar_count += 1
                        else:
                            # Bluff: bid with low probability from our own dice view
                            if taken_action.prob < BLUFF_PROB_THRESHOLD:
                                bluff_count += 1

                    if not done:
                        game_state_history.append(game_state)
                        immediate_reward = calculator.calculate_step_reward(taken_action, 0.0)
                    else:
                        won = step_reward > 0.5
                        game_reward = step_reward - 0.5
                        immediate_reward = game_reward * 2.0
                        if won:
                            # Reward bluffs that paid off
                            bluff_bonus = BLUFF_WIN_BONUS * min(bluff_count, RISKY_BONUS_MAX_COUNT)
                            # Reward gutsy liar calls that paid off
                            risky_liar_bonus = RISKY_LIAR_WIN_BONUS * min(risky_liar_count, RISKY_BONUS_MAX_COUNT)
                            immediate_reward += bluff_bonus + risky_liar_bonus
            else:
                immediate_reward = -1.0

            rewards.append(immediate_reward)
            turn_number += 1

        # Calculate discounted return
        discounted_return = calculator.calculate_discounted_return(rewards)
        train_reward = discounted_return

        print(
            "[ID:{:<6} Hints:{} Done:{} T:{:>2d} | Reward:{:>8.2f} | LastProb:{:>7.3f} | "
            "EnvR:{:>6.1f} | Bluffs:{:<2} RiskyLiar:{:<2} Inv:{:<2}]".format(
                str(game_id)[:6], 
                int(use_hints), 
                int(done), 
                turn_number, 
                train_reward, 
                last_action_prob, 
                final_reward, 
                bluff_count, 
                risky_liar_count, 
                invalid_count)
        )

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


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)