import os
import re
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from trl.experimental.openenv import generate_rollout_completions


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
    
    # Remove "Waiting for Player -2 to move..." if present
    state_text = re.sub(
        r'\n\nWaiting for Player -2 to move\.\.\.$',
        '',
        state_text
    )
    
    # Detect player ID (look for "You are Player X")
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    
    # Extract player hand to reconstruct legal actions
    hand_pattern = rf'P{player_id} hand: ([\d\s]+)'
    hand_match = re.search(hand_pattern, state_text)
    
    if not hand_match:
        # Can't find hand, return state without legal actions
        return state_text
    
    # Parse hand cards
    hand_str = hand_match.group(1).strip()
    cards = [int(card) for card in hand_str.split()]
    
    # Reconstruct legal actions
    # The action ID corresponds to the bid value - 1 (0-indexed)
    # But we need to map to the actual available cards
    legal_actions = []
    for i, card in enumerate(cards):
        # Action ID is the card value minus 1
        action_id = card - 1
        legal_actions.append(f"{action_id} -> [P{player_id}]Bid: {card}")
    
    # Format the complete observation
    formatted = state_text + "\n\nYou are Player " + str(player_id) + ".\nLegal Actions:\n"
    formatted += "\n".join(legal_actions)
    formatted += "\n\nYour choice (ID only):"
    
    return formatted


def extract_prize_card(obs_text):
    """
    Extract the current prize card value from observation.
    
    Args:
        obs_text: Observation text
        
    Returns:
        Prize card value (int) or None if not found
    """
    # Look for "Point card: X"
    match = re.search(r'Current point card:\s*(\d+)', obs_text)
    if match:
        return int(match.group(1))
    return None


def extract_bid_from_action(action_text, obs_text):
    """
    Extract the bid value from the action.
    
    Args:
        action_text: The action string (should be action ID)
        obs_text: The observation to help parse legal actions
        
    Returns:
        Bid card value (int) or None if cannot parse
    """
    try:
        action_id = int(action_text.strip())
        # The bid value is action_id + 1
        return action_id + 1
    except Exception:
        return None
    

def get_hand_cards(observation_text: str, player_id: int = 0) -> list[int]:
    """Count how many cards remain in the player's hand."""
    pattern = rf"P{player_id} hand:\s*([\d ]+)"
    match = re.search(pattern, observation_text)
    if not match:
        return []
    string_cards = match.group(1).strip().split()
    return [int(card) for card in string_cards]


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
        initial_hint_prob=0.75,
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


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

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

    selected_game = "goofspiel"
    
    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
        # Get local rank
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        
        # Determine endpoint
        if not server_list:
            # Fallback (though likely fatal for the task)
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        # Store endpoint on the function to avoid re-parsing
        rollout_first_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing environment on rank {rank} at {base_url}...")
            payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
            create_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            create_res.raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. Rank: {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_endpoint = rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # --- 3. Batch Loop ---
    # We use a random game_id for the batch, or you could sample per item if preferred
    game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        done = False
        solved = False
        train_reward = 0
        turn_number = 0
        
        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]
            
            # Get episode id for rest of interactions
            episode_id = result_block.get("episode_id", "")

            # Construct Initial Observation
            current_observation = result_block.get("observation", "")
            format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
            current_observation += format_instructions


        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        
        messages.append({"role": "user", "content": current_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                episode_completion_ids = completion_ids
                episode_logprobs = logprobs

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
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                done = step_block.get("done", False)
                
                # Format next observation
                formatted_observation = step_state
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                step_reward = -0.01
                done = False

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1
        
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)

        

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards
    }


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """
    Parallelized rollout function for game environments.
    """
    # --- Constants ---
    STRATEGY_REWARD = 1.0
    INVALID_PENALTY = -0.1
    
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

    selected_game = "goofspiel"

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
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
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
            final_max_turn=13,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=13, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.75, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

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
        done = False
        turn_number = 0
        target_training_turn = current_max_turn - 1
        
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

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

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing goofspiel.\n\n# Game Rules\nGOOFSPIEL RULES:\nSetup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.\nGoal: Win the most points by bidding on prize cards.\n\nEach turn:\n1. Reveal top prize card (worth its face value in points)\n2. Players simultaneously play one bid card from their hand\n3. Highest bidder wins the prize card (adds its value to score)\n4. If bids tie, prize card is discarded (no one gets points)\n\nWinning: Player with most points after all rounds wins.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""

        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = "\n\nDon't think for long. The best strategies is to bid the card with same value as the point card \n\nExample: \nIf the point card is 1, bid using card 1, likely action ID 0\nIf the point card is 13, bid using card 13, likely action ID 12\nIf the point card is 10, bid using card 10, likely action ID 9\nAlways bid following this strategy to maximize your winning chance."
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}]

        # Strategy forcing for turns before target training turn
        while not done and (turn_number < target_training_turn):
            messages.append({"role": "user", "content": formatted_observation})

            hand_cards = get_hand_cards(formatted_observation)
            if len(hand_cards) <= 1:
                target_training_turn = turn_number
                break
            
            prize_card = extract_prize_card(formatted_observation)
            action_id = prize_card - 1

            messages.append({"role": "assistant", "content": str(action_id)})

            # --- Step Environment (POST /step) ---
            try:
                formatted_observation = ""
                step_payload = {"action": str(action_id), "episode_id": episode_id}
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

            turn_number += 1

        if done:
            print(
                f"[GT] Game {game_id} ended during strategy forcing phase at turn {turn_number}. "
                f"Returning fallback."
            )
            return index, None

        messages.append({"role": "user", "content": formatted_observation})
        prize_card = extract_prize_card(formatted_observation)

        with rollout_last_prompt_and_completion_parallelized_curriculum.generation_semaphore:
            rollout_out = generate_rollout_completions(
                trainer, prompts=[messages], as_chat=True
            )[0]

        prompt_ids = rollout_out.get("prompt_ids", [])
        completion_ids = rollout_out.get("completion_ids", [])
        logprobs = rollout_out.get("logprobs", [])
        completion_text = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        
        messages.append({"role": "assistant", "content": completion_text})

        # Parse action from model output
        action_to_send = remove_reasoning_tags(completion_text)
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]
        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        # Check strategy adherence for training turn
        bid_card = extract_bid_from_action(action_to_send, formatted_observation)
        strategy_followed = (
            bid_card is not None and prize_card is not None and bid_card == prize_card
        )

        # Step environment with model's action
        invalid_action = False
        
        invalid_action = False
        try:
            action_id_parsed = int(action_to_send.strip())
            hand_cards = get_hand_cards(formatted_observation)
            if action_id_parsed not in hand_cards:
                print(f"Invalid action: {action_id_parsed} not in hand cards: {hand_cards}")
                invalid_action = True
        except Exception:
            invalid_action = True
            print(f"Invalid action: {action_to_send}")
            
        if invalid_action:
            print(f"Messages: {messages}")
            reward = INVALID_PENALTY
        elif strategy_followed:
            # Calculate scale reward for response length, longer responses get lower reward
            response_length = len(completion_ids)
            prompt_length = len(prompt_ids)
            len_reward_scale = max(0.2, min(5, prompt_length / response_length))
            reward = STRATEGY_REWARD * len_reward_scale
        else:
            reward = 0.0
            
        print("--------------------------------")
        print(
            f"[GT] game={game_id} train_turn={target_training_turn} "
            f"strategy={strategy_followed} "
            f"reward={reward:.3f} hints={use_hints}"
        )
        print("--------------------------------")
        
        return index, {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "reward": reward,
            "strategy_followed": strategy_followed,
        }

    # Execute episodes in parallel
    results = [None] * len(prompts)
    executor = rollout_last_prompt_and_completion_parallelized_curriculum.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            # Fallback for failed / short-circuited episodes
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0,
                "strategy_followed": False,
            }

    # Update curriculum
    curriculum.step(len(prompts))

    # Log batch stats
    valid = [r for r in results if r is not None]
    if valid:
        avg_strat = sum(1 for r in valid if r["strategy_followed"]) / len(valid)
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        print(
            f"[GT-BATCH] Strategy: {avg_strat:.1%}, Avg Reward: {avg_reward:.3f}"
        )

    return {
        "prompt_ids": [r["prompt_ids"] for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs": [r["logprobs"] for r in results],
        "env_rewards": [r["reward"] for r in results],
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
    MAX_PROMPT_LEN = 4225      # Max prompt tokens before ending episode early
    
    # --- Reward Shaping Parameters ---
    STRATEGY_REWARD_WEIGHT = 0.5  # Weight for strategy adherence vs final score
    STEP_STRATEGY_REWARD = 0.1    # Immediate reward for following strategy at each step

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

    selected_game = "goofspiel"

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
                payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts"}
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
            final_max_turn=13,
            rollouts_per_stage=trainer.args.rollouts_per_stage,
            initial_hint_prob=0.75,
            final_hint_prob=0.0,
            warmup_rollouts=trainer.args.rollouts_per_stage,
        )
        print(f"[CURRICULUM] Initialized with initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=13, rollouts_per_stage={trainer.args.rollouts_per_stage}, initial_hint_prob=0.75, final_hint_prob=0.0, warmup_rollouts={trainer.args.rollouts_per_stage}")

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
        turn_number = 0
        
        # Track strategy adherence
        strategy_followed_count = 0
        total_strategy_opportunities = 0
        step_rewards = [] 
        all_steps_correct = True
        # Determine if this episode gets hints
        use_hints = random.random() < current_hint_prob

        # --- Reset Environment (POST /reset) ---
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts"}

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

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            return index, None

        # --- Build Conversation History ---
        # Fisrt make system prompt
        system_prompt = "You are playing goofspiel.\n\n# Game Rules\nGOOFSPIEL RULES:\nSetup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.\nGoal: Win the most points by bidding on prize cards.\n\nEach turn:\n1. Reveal top prize card (worth its face value in points)\n2. Players simultaneously play one bid card from their hand\n3. Highest bidder wins the prize card (adds its value to score)\n4. If bids tie, prize card is discarded (no one gets points)\n\nWinning: Player with most points after all rounds wins.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action \"0 -> roll\": respond \"0\"\n- For action \"89 -> a3\": respond \"89\""

        # Add suggestion for playing strategy based on curriculum
        if use_hints:
            suggestion_prompt = "\n\nThe best strategies is to bid the card with same value as the point card \n\nExample: \nIf the point card is 1, bid using card 1, likely action ID 0\nIf the point card is 13, bid using card 13, likely action ID 12\nIf the point card is 10, bid using card 10, likely action ID 9\nAlways bid following this strategy to maximize your winning chance."
            system_prompt += suggestion_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_observation}]

        # --- Interaction Loop ---
        while not done and (turn_number < current_max_turn):
            # Extract prize card before taking action
            prize_card = extract_prize_card(formatted_observation)
            
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
                
            # --- Check Strategy Adherence ---
            bid_card = extract_bid_from_action(action_to_send, formatted_observation)
            if bid_card is not None:
                total_strategy_opportunities += 1
                if bid_card == prize_card and all_steps_correct:
                    strategy_followed_count += 1
                    # Give immediate reward for following strategy
                    step_reward = STEP_STRATEGY_REWARD
                    step_rewards.append(step_reward)
                else:
                    all_steps_correct = False
                    step_rewards.append(0.0)
            else:
                # Invalid action - counts as a strategy opportunity that was NOT followed
                total_strategy_opportunities += 1
                all_steps_correct = False
                step_rewards.append(0.0)

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
            if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
                invalid_count += 1

            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]
            
        # --- Calculate Final Reward with Strategy Shaping ---
        # Strategy adherence ratio
        strategy_ratio = strategy_followed_count / total_strategy_opportunities if total_strategy_opportunities > 0 else 0.0
        
        # Combine immediate step rewards
        immediate_rewards = sum(step_rewards)
        
        # For short episodes (curriculum learning), prioritize strategy adherence
        # For full episodes, blend strategy with final score
        if not done:
            # Partial episode - focus on strategy
            shaped_reward = immediate_rewards + strategy_ratio
        else:
            # Full episode - blend strategy adherence with final score
            shaped_reward = (
                STRATEGY_REWARD_WEIGHT * strategy_ratio +
                (1 - STRATEGY_REWARD_WEIGHT) * train_reward +
                immediate_rewards
            )

        # Apply invalid action penalty
        shaped_reward = shaped_reward - 0.05 * float(invalid_count)

        # Log in one line
        print("============")
        print(f"id: {game_id}, max_turn: {current_max_turn}, hints: {use_hints}", f"Strategy: {strategy_followed_count}/{total_strategy_opportunities} ({strategy_ratio:.2%})")
        print("============")

        return index, {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask": episode_action_mask,
            "logprobs": episode_logprobs,
            "reward": shaped_reward,
            "strategy_ratio": strategy_ratio,
            "final_score": train_reward,
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
                "strategy_ratio": 0.0,
                "final_score": 0.0,
            }
            
    # Update curriculum after batch
    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    
    # Log batch statistics
    avg_strategy = sum(r["strategy_ratio"] for r in list_results) / len(list_results) if list_results else 0
    avg_final = sum(r["final_score"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Avg Strategy Adherence: {avg_strategy:.2%}, Avg Final Score: {avg_final:.3f}")

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