"""
Generate game trajectories against env servers and save as an HF DatasetDict
(train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Run from /workspace/scripts/:
  python -m envs.generate_trajectories --environment_name liars_dice \
      --output_path /path/to/dataset --num_games 100000

  python -m envs.generate_trajectories --environment_name gin_rummy \
      --output_path /path/to/dataset --num_games 8000 --max_turn 200

  python -m envs.generate_trajectories --environment_name leduc_poker \
      --output_path /path/to/dataset --num_games 200000 --max_turn 10 \
      --sample-by-score --score-power 3

Score-based sampling (winner 5EgpWgYv leduc_poker strategy):
  Some generators (e.g. leduc_poker) return (messages, score) tuples. When
  --sample-by-score is set, each game is kept with probability
  clamp(score, 0, 1) ** score_power. --wins-only is a stricter filter that
  discards any game where score <= 0. For generators that return only
  messages (no score), all games are kept regardless of these flags.
"""

import argparse
import random
from collections import Counter
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed

from datasets import Dataset, DatasetDict

from envs.shared_env import GAMES_TO_TASK_ID_RANGE, init_env_pool
from envs.sft_env_configs import get_sft_trajectory_generator


# ── Process-pool worker ───────────────────────────────────────────────────────
# Each worker process loads the expert generator once via _worker_init, then
# handles multiple games sequentially. Using processes (not threads) gives each
# worker its own GIL so CPU-bound expert computation runs truly in parallel
# without contention. --num_workers controls how many concurrent env server
# connections are open, letting you tune without overloading either side.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)


def _worker_play(
    game_id: int, endpoint: str, max_turn: int
) -> "list[dict] | tuple[list[dict], float] | None":
    """Return type varies per generator:
    - liars_dice / gin_rummy: returns ``list[dict]`` (messages only)
    - leduc_poker (random+score): returns ``tuple[list[dict], float]`` (messages, score)
    Score-based sampling (see main()) is only applied when the tuple form is returned.
    """
    return _GENERATE_FN(game_id, endpoint, max_turn)

# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_RATIO    = 0.01
MIN_ASSISTANT_TURNS = 1


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """
    Split a conversation into overlapping sub-conversations.
    Each window: [system] + window_turns × (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    if not messages:
        return None
    messages = [{"role": m["role"], "content": str(m["content"])} for m in messages]
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_name", required=True)
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=50000)
    p.add_argument("--max_turn",    type=int, default=30)
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = num_servers. "
                        "Each process holds one concurrent env server connection; "
                        "raise to increase throughput, lower to reduce env server load.")
    p.add_argument("--seed", type=int, default=42)
    # Score-based sampling (winner's leduc_poker strategy). Only effective when
    # the generator returns a (messages, score) tuple — pure list[dict] returns
    # are kept regardless.
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games where score <= 0. Tuple-returning generators only.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Keep each game with probability clamp(score, 0, 1) ** score-power. "
                        "Tuple-returning generators only.")
    p.add_argument("--score-power", type=float, default=1.0,
                   help="Exponent applied to the clamped score when sampling (default: 1.0). "
                        "Higher values bias more strongly toward high-scoring games.")
    # Time-bounded mode — caps total wall-time for trajectory gen regardless of
    # num_games progress. Critical for PvP tournament where total budget is 1.5h
    # and data gen must leave headroom for tokenize + SFT.
    p.add_argument("--max-time-seconds", type=int, default=0,
                   help="Stop generating after this many seconds, even if num_games "
                        "not reached. 0 (default) = no time cap (legacy behavior).")
    args = p.parse_args()
    if args.window_step == 0:
        args.window_step = args.window_turns // 2 or 1

    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[args.environment_name]

    reset_payload = {
        "task_id": task_id_min,
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    _, env_pool, num_servers, _, _ = init_env_pool(reset_payload)

    num_workers = args.num_workers or max(1, num_servers)

    random.seed(args.seed)
    game_ids = random.sample(range(task_id_min + 1, task_id_max), args.num_games)
    tasks = [
        (gid, env_pool[i % num_servers]["base_url"], args.max_turn)
        for i, gid in enumerate(game_ids)
    ]

    conversations: list[list[dict]] = []
    skipped = 0
    score_filtered = 0  # games dropped by --wins-only or --sample-by-score
    all_scores: list[float] = []
    use_score_filter = args.wins_only or args.sample_by_score
    # Progress logging: trajectory gen for 100k+ games can take 10-60min (depends on
    # num_workers + env server count + MCTS sims). Silent execution makes it
    # impossible to distinguish "running normally" from "hung". Log every 1% or
    # every 100 completions (whichever larger). Also logs the worker count + env
    # server count so anyone tailing the log can spot 1-worker bottlenecks.
    import sys, time
    print(
        f"[trajectory_gen] Starting: env={args.environment_name} "
        f"games={args.num_games} workers={num_workers} servers={num_servers} "
        f"max_turn={args.max_turn} score_filter={use_score_filter} "
        f"max_time_seconds={args.max_time_seconds or 'unlimited'}",
        flush=True,
    )
    if use_score_filter:
        print(
            f"[trajectory_gen] Score filter: wins_only={args.wins_only} "
            f"sample_by_score={args.sample_by_score} score_power={args.score_power}",
            flush=True,
        )
    _t_start = time.time()
    _log_every = max(100, args.num_games // 100)
    _time_cap_hit = False  # set True once max_time_seconds exceeded
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(args.environment_name,),
    ) as pool:
        futures = {pool.submit(_worker_play, gid, ep, mt): gid for gid, ep, mt in tasks}
        completed = 0
        for future in as_completed(futures):
            # Time-bounded early stop: when max_time_seconds set and exceeded,
            # cancel remaining futures and break. We still process this future
            # (already completed) so no work is wasted.
            _elapsed_now = time.time() - _t_start
            if args.max_time_seconds and _elapsed_now >= args.max_time_seconds and not _time_cap_hit:
                print(
                    f"[trajectory_gen] TIME LIMIT reached ({_elapsed_now:.0f}s "
                    f">= {args.max_time_seconds}s). Cancelling remaining futures.",
                    flush=True,
                )
                _time_cap_hit = True
                # Best-effort cancel of pending futures
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
            # Future.result() raises CancelledError for futures we just cancelled,
            # or other exceptions if a worker crashed. Don't crash the whole loop —
            # treat as "skip this game". (Bug fix 2026-05-25: previously
            # CancelledError propagated up and terminated entire trajectory_gen.)
            try:
                result = future.result()
            except CancelledError:
                # Expected when time-cap triggered .cancel() — skip silently
                completed += 1
                continue
            except Exception as _exc:
                # Worker crashed — log + skip but continue with remaining futures
                print(f"[trajectory_gen] worker future error: {type(_exc).__name__}: {_exc}",
                      flush=True)
                skipped += 1
                completed += 1
                continue

            # Unpack score when the generator returns (messages, score)
            if isinstance(result, tuple):
                raw_messages, score = result
                all_scores.append(score)
            else:
                raw_messages, score = result, None

            # Apply score-based filters when a score is available
            drop_by_score = False
            if score is not None and use_score_filter:
                if args.wins_only and score <= 0:
                    drop_by_score = True
                elif args.sample_by_score:
                    prob = max(0.0, min(1.0, score)) ** args.score_power
                    if random.random() >= prob:
                        drop_by_score = True
            if drop_by_score:
                score_filtered += 1
                completed += 1
            else:
                cleaned = _clean(raw_messages)
                if cleaned is None:
                    skipped += 1
                else:
                    conversations.append(cleaned)
                completed += 1
            if completed % _log_every == 0 or completed == args.num_games:
                _elapsed = time.time() - _t_start
                _rate = completed / max(_elapsed, 1e-3)
                _eta = (args.num_games - completed) / max(_rate, 1e-3)
                print(
                    f"[trajectory_gen] {completed}/{args.num_games} "
                    f"({100*completed/args.num_games:.1f}%) "
                    f"valid={len(conversations)} skipped={skipped} "
                    f"score_filtered={score_filtered} rate={_rate:.1f}/s "
                    f"elapsed={_elapsed:.0f}s eta={_eta:.0f}s",
                    flush=True,
                )
            # After time cap hit, break out of the iteration loop as soon as
            # remaining cancellations propagate. Only break here so already-
            # completed futures aren't discarded.
            if _time_cap_hit and (time.time() - _t_start) >= args.max_time_seconds + 5:
                # 5s grace for cancellation propagation
                print(f"[trajectory_gen] Stopping loop after {completed} games "
                      f"({len(conversations)} valid).", flush=True)
                break

    if all_scores:
        wins = sum(1 for s in all_scores if s > 0)
        print(
            f"[trajectory_gen] Score stats: min={min(all_scores):.3f} "
            f"max={max(all_scores):.3f} wins(>0)={wins}/{len(all_scores)} "
            f"({100*wins/len(all_scores):.1f}%)  score_filtered={score_filtered}",
            flush=True,
        )

    if not conversations:
        raise RuntimeError("No valid conversations generated. Check ENVIRONMENT_SERVER_URLS.")

    windowed: list[list[dict]] = []
    for conv in conversations:
        windows = _sliding_windows(conv, args.window_turns, args.window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    splits = dataset.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
    dd = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    dd.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
