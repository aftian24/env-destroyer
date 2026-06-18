import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore

import requests


# ---------------------------------------------------------------------------
# Game ID ranges (single source of truth — also imported by train_grpo_env.py)
# ---------------------------------------------------------------------------

GAMES_TO_TASK_ID_RANGE: dict[str, tuple[int, int]] = {
    "goofspiel":   (0,         99_999_999),
    "liars_dice":  (100000000, 199_999_999),
    "leduc_poker": (200000000, 299_999_999),
    "gin_rummy":   (300000000, 399_999_999),
    "othello":     (400000000, 499_999_999),
    "backgammon":  (500000000, 599_999_999),
    "hex":         (600000000, 699_999_999),
    "clobber":     (700000000, 799_999_999),
}


# ---------------------------------------------------------------------------
# Curriculum scheduler (base class — subclass to extend per-game logic)
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.

    Subclass and override ``get_max_turn`` or ``get_hint_prob`` to implement
    custom scheduling logic for a specific game environment.
    """

    def __init__(
        self,
        initial_max_turn: int,
        final_max_turn: int,
        rollouts_per_stage: int,
        initial_hint_prob: float,
        final_hint_prob: float,
        warmup_rollouts: int,
    ) -> None:
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        """Return current max turns. Override for non-linear schedules."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted = self.total_rollouts - self.warmup_rollouts
        stage = adjusted // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        """Return current hint probability. Override for custom decay."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = self.final_max_turn - self.initial_max_turn
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted / total_decay_rollouts, 1.0) if total_decay_rollouts > 0 else 1.0
        current = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        """Advance internal rollout counter. Override to add custom logic."""
        self.total_rollouts += num_rollouts

    def get_status(self) -> dict:
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(),
        }


# ---------------------------------------------------------------------------
# Environment pool initialisation
# ---------------------------------------------------------------------------

def init_env_pool(
    reset_payload: dict,
    reset_endpoint: str = "reset",
    lock_per_server: bool = False,
) -> tuple[int, list[dict], int, ThreadPoolExecutor, Semaphore]:
    """
    Initialise the environment server pool once per process.

    Reads ``LOCAL_RANK`` and ``ENVIRONMENT_SERVER_URLS`` from the environment.
    Sends a warm-up request to each server to verify it is reachable.

    Args:
        reset_payload:    JSON body for the warm-up POST request.
        reset_endpoint:   API endpoint to POST (``"reset"`` or ``"create"``).
        lock_per_server:  If True, attach a ``threading.Lock`` to each pool
                          entry (required by AlfWorld's per-server serialisation).

    Returns:
        ``(rank, env_pool, num_servers, thread_pool, generation_semaphore)``
        Each ``env_pool`` entry is a dict with at least ``"base_url"``.
        When ``lock_per_server=True`` it also has ``"env_id"`` and ``"lock"``.
    """
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    env_pool: list[dict] = []
    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Connecting to server {idx}: {base_url}")
            res = requests.post(f"{base_url}/{reset_endpoint}", json=reset_payload, timeout=300)
            res.raise_for_status()
            entry: dict = {"base_url": base_url}
            if lock_per_server:
                entry["env_id"] = res.json()["id"]
                entry["lock"] = Lock()
            env_pool.append(entry)
            print(f"[INIT] Server {idx} ready")
        except Exception as exc:
            raise RuntimeError(f"Failed to init server {base_url}: {exc}") from exc

    thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
    generation_semaphore = Semaphore(1)
    return rank, env_pool, len(env_pool), thread_pool, generation_semaphore


# ---------------------------------------------------------------------------
# Generic reward passthrough (identical across all game environments)
# ---------------------------------------------------------------------------

def rollout_reward_func(completions, **kwargs):
    """Generic reward passthrough used by all game environments."""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
