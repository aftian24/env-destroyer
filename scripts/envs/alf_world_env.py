import functools
import os
import random
from concurrent.futures import as_completed
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import init_env_pool, rollout_reward_func  # re-exported for callers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN     = 24576
_TIMEOUT            = 2400

_CONVERSATION_START = [
    {
        "role": "user",
        "content": (
            'Interact with a household to solve a task. Imagine you are an intelligent agent in a household '
            'environment and your target is to perform actions to complete the task goal. At the beginning of your '
            'interactions, you will be given the detailed description of the current environment and your goal to '
            'accomplish. For each of your turn, you will be given a list of actions which you can choose one to '
            'perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose '
            '"THOUGHT", you should first think about the current condition and plan for your future actions, and '
            'then output your action in this turn. Your output must strictly follow this format:'
            '"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly '
            'output the action in this turn. Your output must strictly follow this format:"Action:\nyour next '
            'action". After your each turn, the environment will give you immediate feedback based on which you '
            'plan your next few steps. if the envrionment output "Nothing happened", that means the previous '
            'action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from '
            'the given available actions. Any actions except provided available actions will be regarded as '
            'illegal. \n2. Think when necessary, try to act directly more in the process.'
        ),
    },
    {
        "role": "assistant",
        "content": "OK. I'll follow your instructions and try my best to solve the task.",
    },
]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _ensure_initialized() -> None:
    """Set up AlfWorld server pool once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    # AlfWorld uses /create (not /reset) and needs a Lock per server
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(
        reset_payload={},
        reset_endpoint="create",
        lock_per_server=True,
    )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
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
    max_turns: int,
) -> tuple[int, "dict | None"]:
    """
    Run one AlfWorld episode.

    ``use_full_prompt=True``  — accumulates all turns with action masking.
    ``use_full_prompt=False`` — keeps only the first turn's token IDs
                               (equivalent to "first_prompt" mode).
    """
    try:
        game_id = int(prompt)
    except ValueError:
        raise ValueError(f"Prompt must be a numeric string, got: {prompt}")

    server_idx = (game_id + rank) % num_servers
    server     = env_pool[server_idx]

    with server["lock"]:
        env_id       = server["env_id"]
        env_endpoint = server["base_url"]

        episode_prompt_ids:    list[int]   = []
        episode_completion_ids: list[int]  = []
        episode_logprobs:      list[float] = []
        episode_action_mask:   list[int]   = []
        prev_full_ids: "list[int] | None"  = None

        invalid_count = 0
        done          = False
        solved        = False
        turn_number   = 0

        # --- Reset environment ---
        try:
            reset_res = requests.post(
                f"{env_endpoint}/reset",
                json={"id": env_id, "game": game_id, "world_type": "Text"},
                timeout=_TIMEOUT,
            )
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            current_observation       = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
        except Exception as exc:
            print(f"Failed to reset environment (Game {game_id}): {exc}")
            return index, None

        messages = list(_CONVERSATION_START)  # shallow copy of system messages
        messages.append({"role": "user", "content": formatted_observation})

        # --- Interaction loop ---
        while not done and turn_number < max_turns:
            with generation_semaphore:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids     = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs       = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # --- Token accumulation ---
            if use_full_prompt:
                if len(prompt_ids) > _MAX_PROMPT_LEN:
                    print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
                    done = True
                    break

                if turn_number == 0:
                    episode_prompt_ids = prompt_ids
                    prev_full_ids = prompt_ids.copy()
                else:
                    if prev_full_ids is None:
                        prev_full_ids = prompt_ids.copy()
                    elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                        print(f"Warning: token shift at turn {turn_number}. Skipping delta mask.")
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
            else:
                # first-prompt mode: keep turn-0 tokens only
                if turn_number == 0:
                    episode_prompt_ids    = prompt_ids
                    episode_completion_ids = completion_ids
                    episode_logprobs       = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step environment ---
            step_reward = 0.0
            step_done   = False
            step_state  = ""
            try:
                step_res = requests.post(
                    f"{env_endpoint}/step",
                    json={"id": env_id, "action": action_to_send},
                    timeout=_TIMEOUT,
                )
                step_res.raise_for_status()
                step_data = step_res.json()
                step_state  = step_data["observation"]
                step_reward = step_data["reward"]
                step_done   = step_data["done"]
                current_available_actions = step_data["available_actions"]
                formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
            except Exception as exc:
                print(f"Step failed: {exc}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation
                step_reward = 0.0
                step_done   = False

            if step_done and step_reward > 0:
                solved = True
            if "Nothing happens" in step_state:
                invalid_count += 1

            done = step_done
            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # --- Truncate if needed (full-prompt only) ---
        if use_full_prompt and len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]

        train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)

        result: dict = {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
        }
        if use_full_prompt:
            result["action_mask"] = episode_action_mask
        return index, result


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool, max_turns: int) -> dict[str, list]:
    _ensure_initialized()

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        max_turns=max_turns,
    )

    _fallback: dict = {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0}
    if use_full_prompt:
        _fallback["action_mask"] = [0]

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    list_results = [r for r in results if r is not None]

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


def alfworld_rollout_first_prompt_and_completion_parallelized(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — keeps only the first turn's token IDs (no action mask)."""
    return _dispatch(prompts, trainer, use_full_prompt=False, max_turns=max_turns)


def alfworld_rollout_full_prompt_and_completion_parallelized(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True, max_turns=max_turns)


def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) if r is not None else 0.0 for r in rewards] if rewards is not None else [0.0] * len(completions)
