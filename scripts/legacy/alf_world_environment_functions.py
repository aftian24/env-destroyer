import os
import random
import requests
from threading import Semaphore, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from trl.experimental.openenv import generate_rollout_completions


def alfworld_rollout_first_prompt_and_completion_parallelized(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(alfworld_rollout_first_prompt_and_completion_parallelized, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url, env_id}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Creating env on server {idx}: {base_url}")
                res = requests.post(f"{base_url}/create", timeout=300)
                res.raise_for_status()
                env_id = res.json()["id"]
                env_pool.append(
                    {
                        "base_url": base_url,
                        "env_id": env_id,
                        "lock": Lock(),
                    }
                )
                print(f"[INIT] Server {idx} ready (env_id={env_id})")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        alfworld_rollout_first_prompt_and_completion_parallelized.rank = rank
        alfworld_rollout_first_prompt_and_completion_parallelized.env_pool = env_pool
        alfworld_rollout_first_prompt_and_completion_parallelized.num_servers = len(env_pool)
        alfworld_rollout_first_prompt_and_completion_parallelized.initialized = True
        alfworld_rollout_first_prompt_and_completion_parallelized.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        alfworld_rollout_first_prompt_and_completion_parallelized.generation_semaphore = Semaphore(1)

    # Retrieve static variables
    rank = alfworld_rollout_first_prompt_and_completion_parallelized.rank
    env_pool = alfworld_rollout_first_prompt_and_completion_parallelized.env_pool
    num_servers = alfworld_rollout_first_prompt_and_completion_parallelized.num_servers

    tokenizer = trainer.processing_class
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    def run_single_prompt(index, prompt: str):
        try:
            game_id = int(prompt)
        except ValueError:
            raise ValueError(f"Prompt must be numeric string, got: {prompt}")

        server_idx = (game_id + rank) % num_servers
        server = env_pool[server_idx]
        
        with server["lock"]:
            env_id = server["env_id"]
            env_endpoint = server["base_url"]

            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            invalid_count = 0
            done = False
            solved = False
            turn_number = 0

            # --- Reset Environment (POST /reset) ---
            # Reuse existing env_id, just change the game
            payload = {"id": env_id, "game": game_id, "world_type": "Text"}
            
            try:
                reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
                reset_res.raise_for_status()
                reset_data = reset_res.json()
                
                # Construct Initial Observation
                current_observation = reset_data["observation"]
                current_available_actions = reset_data["available_actions"]
                formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
            except Exception as e:
                print(f"Failed to reset environment (Game {game_id}): {e}")
                print(reset_data)
                return index, None

            # --- Build Conversation History ---
            messages = []
            for message in conversation_start:
                if message["from"] == "human":
                    messages.append({"role": "user", "content": message["value"]})
                elif message["from"] == "gpt":
                    messages.append({"role": "assistant", "content": message["value"]})
            
            messages.append({"role": "user", "content": formatted_observation})

            # --- Interaction Loop ---
            while not done and (turn_number < max_turns):
                # Generate Rollout Completion
                # Only allow one thread to generate rollout completions at a time
                with alfworld_rollout_first_prompt_and_completion_parallelized.generation_semaphore:
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
                step_reward = 0.0
                step_done = False
                step_state = ""

                try:
                    step_payload = {"id": env_id, "action": action_to_send}
                    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                    step_res.raise_for_status()
                    step_data = step_res.json()
                    
                    # Extract response data
                    step_state = step_data["observation"]
                    step_reward = step_data["reward"]
                    step_done = step_data["done"]
                    current_available_actions = step_data["available_actions"]
                    
                    # Format next observation
                    formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
                    
                except Exception as e:
                    print(f"Step failed: {e}")
                    formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                    step_reward = 0.0
                    step_done = False

                # Update Loop State
                if step_done and step_reward > 0:
                    solved = True

                if "Nothing happens" in step_state:
                    invalid_count += 1
                
                done = step_done

                if not done:
                    messages.append({"role": "user", "content": formatted_observation})

                turn_number += 1

            train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)
            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "logprobs": episode_logprobs,
                "reward": train_reward
            }

    results = [None] * len(prompts)
    list_results = []

    executor = alfworld_rollout_first_prompt_and_completion_parallelized.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "logprobs": [1.0],
                "reward": 0.0
            }
    
    list_results.extend([results[i] for i in range(len(results)) if results[i] is not None])

    # ---- Aggregate ----
    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def alfworld_rollout_full_prompt_and_completion_parallelized(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early
    
    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(alfworld_rollout_first_prompt_and_completion_parallelized, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        # random.seed(rank)

        if not server_urls:
            raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

        env_pool = []  # list of dicts: {base_url, env_id}

        for idx, base_url in enumerate(server_urls):
            try:
                print(f"[INIT] Creating env on server {idx}: {base_url}")
                res = requests.post(f"{base_url}/create", timeout=300)
                res.raise_for_status()
                env_id = res.json()["id"]
                env_pool.append(
                    {
                        "base_url": base_url,
                        "env_id": env_id,
                        "lock": Lock(),
                    }
                )
                print(f"[INIT] Server {idx} ready (env_id={env_id})")
            except Exception as e:
                raise RuntimeError(f"Failed to init server {base_url}: {e}")

        alfworld_rollout_first_prompt_and_completion_parallelized.rank = rank
        alfworld_rollout_first_prompt_and_completion_parallelized.env_pool = env_pool
        alfworld_rollout_first_prompt_and_completion_parallelized.num_servers = len(env_pool)
        alfworld_rollout_first_prompt_and_completion_parallelized.initialized = True
        alfworld_rollout_first_prompt_and_completion_parallelized.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
        alfworld_rollout_first_prompt_and_completion_parallelized.generation_semaphore = Semaphore(1)

    # Retrieve static variables
    rank = alfworld_rollout_first_prompt_and_completion_parallelized.rank
    env_pool = alfworld_rollout_first_prompt_and_completion_parallelized.env_pool
    num_servers = alfworld_rollout_first_prompt_and_completion_parallelized.num_servers

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    def run_single_prompt(index, prompt: str):
        try:
            game_id = int(prompt)
            # game_id = random.randint(0, DATA_LEN - 1)
        except ValueError:
            raise ValueError(f"Prompt must be numeric string, got: {prompt}")

        server_idx = (game_id + rank) % num_servers
        server = env_pool[server_idx]
        
        with server["lock"]:
            env_id = server["env_id"]
            env_endpoint = server["base_url"]

            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
            invalid_count = 0
            done = False
            solved = False
            turn_number = 0

            # --- Reset Environment (POST /reset) ---
            # Reuse existing env_id, just change the game
            payload = {"id": env_id, "game": game_id, "world_type": "Text"}
            
            try:
                reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
                reset_res.raise_for_status()
                reset_data = reset_res.json()
                
                # Construct Initial Observation
                current_observation = reset_data["observation"]
                current_available_actions = reset_data["available_actions"]
                formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
            except Exception as e:
                print(f"Failed to reset environment (Game {game_id}): {e}")
                print(reset_data)
                return index, None

            # --- Build Conversation History ---
            messages = []
            for message in conversation_start:
                if message["from"] == "human":
                    messages.append({"role": "user", "content": message["value"]})
                elif message["from"] == "gpt":
                    messages.append({"role": "assistant", "content": message["value"]})
            
            messages.append({"role": "user", "content": formatted_observation})

            # --- Interaction Loop ---
            while not done and (turn_number < max_turns):
                # Generate Rollout Completion
                # Only allow one thread to generate rollout completions at a time
                with alfworld_rollout_first_prompt_and_completion_parallelized.generation_semaphore:
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
                        delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
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
                step_reward = 0.0
                step_done = False
                step_state = ""

                try:
                    step_payload = {"id": env_id, "action": action_to_send}
                    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                    step_res.raise_for_status()
                    step_data = step_res.json()
                    
                    # Extract response data
                    step_state = step_data["observation"]
                    step_reward = step_data["reward"]
                    step_done = step_data["done"]
                    current_available_actions = step_data["available_actions"]
                    
                    # Format next observation
                    formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
                    
                except Exception as e:
                    print(f"Step failed: {e}")
                    formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                    step_reward = 0.0
                    step_done = False

                # Update Loop State
                if step_done and step_reward > 0:
                    solved = True

                if "Nothing happens" in step_state:
                    invalid_count += 1
                
                done = step_done

                if not done:
                    messages.append({"role": "user", "content": formatted_observation})

                turn_number += 1

            # Truncate episode if completion sequence exceeds max length
            if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
                print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
                episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
                episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
                episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

            train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)
            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "action_mask": episode_action_mask,
                "logprobs": episode_logprobs,
                "reward": train_reward
            }

    results = [None] * len(prompts)
    list_results = []

    executor = alfworld_rollout_first_prompt_and_completion_parallelized.thread_pool

    futures = [
        executor.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]

    for f in as_completed(futures):
        idx, res = f.result()
        if res is not None:
            results[idx] = res
        else:
            results[idx] = {
                "prompt_ids": [1],
                "completion_ids": [1],
                "action_mask": [0],
                "logprobs": [1.0],
                "reward": 0.0
            }

    list_results.extend([results[i] for i in range(len(results)) if results[i] is not None])
    # ---- Aggregate ----
    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "action_mask": [r["action_mask"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def alfworld_rollout_full_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(alfworld_rollout_full_prompt_and_completion, "initialized", False):
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
        alfworld_rollout_full_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            # Store env_id on the function
            alfworld_rollout_full_prompt_and_completion.env_id = create_res.json()["id"]
            alfworld_rollout_full_prompt_and_completion.initialized = True
            print(f"Environment initialized. ID: {alfworld_rollout_full_prompt_and_completion.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_id = alfworld_rollout_full_prompt_and_completion.env_id
    env_endpoint = alfworld_rollout_full_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []
    all_episode_action_masks: list[list[int]] = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    # --- 3. Batch Loop ---
    # We use a random game_id for the batch, or you could sample per item if preferred
    game_id = random.randint(0, DATA_LEN - 1)

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []
        prev_full_ids: list[int] | None = None
        invalid_count = 0
        done = False
        solved = False
        turn_number = 0
        
        # --- Reset Environment (POST /reset) ---
        # Reuse existing env_id, just change the game
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            
            # Construct Initial Observation
            current_observation = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        for message in conversation_start:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})
        
        messages.append({"role": "user", "content": formatted_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
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
                    delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
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
            step_reward = 0.0
            step_done = False
            step_state = ""

            try:
                step_payload = {"id": env_id, "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                # Extract response data
                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_available_actions = step_data["available_actions"]
                
                # Format next observation
                formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                step_reward = 0.0
                step_done = False

            # Update Loop State
            if step_done and step_reward > 0:
                solved = True

            if "Nothing happens" in step_state:
                invalid_count += 1
            
            done = step_done

            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)
        all_episode_action_masks.append(episode_action_mask)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards,
        "action_mask": all_episode_action_masks
    }


def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) if r is not None else 0.0 for r in rewards] if rewards is not None else [0.0] * len(completions)
