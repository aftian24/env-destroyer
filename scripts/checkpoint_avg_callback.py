"""
Adaptive training callback: checkpoint averaging + overfitting rollback.

Combines three mechanisms:
1. Sliding-window checkpoint averaging (last 3 eval-point snapshots)
2. Overfitting detection (eval loss >5% above best for 2 consecutive evals)
3. Rollback + escalating response (restore best weights, cut LR, bump NEFTune)

On overfitting detection:
- Restores model to best checkpoint weights
- Cuts LR by 50%
- Bumps NEFTune noise alpha (5 -> 10 -> 15) if NEFTune is active
- Continues training with more regularization

The model gets a second chance from the best point with gentler settings,
instead of wasting remaining time on a diverging trajectory.
"""

import datetime
import gc
import os
import shutil
import time
from collections import deque
from typing import Optional

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import is_main_process

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

# Overfitting thresholds (relative to best eval loss)
_OVERFIT_THRESHOLD = 0.05   # 5% worse than best = overfitting signal
_OVERFIT_PATIENCE = 3       # 3 consecutive evals above threshold = confirmed
_MAX_ROLLBACKS = 2          # cap rollbacks to avoid burning the time budget

# NEFTune escalation steps
_NEFTUNE_LEVELS = [5, 10, 15]


class AdaptiveTrainingCallback(TrainerCallback):
    """Checkpoint averaging + overfitting detection with rollback.

    Usage:
        cb = AdaptiveTrainingCallback(window=3)
        trainer = Trainer(..., callbacks=[cb, ...])
        cb.trainer = trainer
        trainer.train()
    """

    def __init__(
        self,
        window: int = 3,
        device: str = "cpu",
        use_reward_accuracy: bool = False,
        averaging_mode: str = "ram",
        output_dir: str | None = None,
        disk_members: int = 4,
        soup_max: int = 8,
    ):
        self.window = window
        self.device = device
        self.use_reward_accuracy = use_reward_accuracy

        # How checkpoint averaging is done:
        #   "ram"  — in-RAM window averaging during training (non-sharded, fits).
        #   "disk" — average consolidated checkpoints from disk at train end
        #            (sharded / too big for RAM); in-RAM snapshots only capture
        #            shards under FSDP/DeepSpeed, so window averaging is invalid.
        #   "off"  — best-checkpoint selection only, no averaging.
        self.averaging_mode = averaging_mode
        self.output_dir = output_dir
        self.disk_members = disk_members

        # Checkpoint averaging state.
        # `snapshots` is retained only for the (currently inert) rollback path;
        # the live selection mechanism is the RAM-gated greedy-soup `pool` below.
        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=window)
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_loss: float = float("inf")
        self.best_source: str = "none"

        # ── Greedy-soup pool ──
        # Up to `soup_max` lowest-dev-loss weight snapshots, combined once at
        # train end (greedy soup). Admission is gated on MEASURED free host RAM
        # (see _can_admit_snapshot): a full-FT snapshot can be many GB, so the
        # pool grows only while headroom remains and otherwise evicts its worst
        # member. end_time (set by the caller) bounds the soup's eval time.
        self.soup_max = soup_max
        self.pool: list[dict] = []
        self._snap_bytes: Optional[int] = None
        # Set by the caller (train_instruct): the submission dir to write the
        # final soup into, and the task end_time that bounds the soup's evals.
        self._submission_dir: str | None = None
        self.end_time: str = ""

        # Overfitting detection state
        self.overfit_counter = 0
        self.rollback_count = 0
        self.neftune_level_idx = 0

        # Re-entry guard
        self._evaluating = False
        self.trainer = None

    def _get_metric(self, metrics):
        """Extract the tracking metric. Lower = better (accuracy is negated)."""
        if self.use_reward_accuracy:
            acc = metrics.get("eval_rewards/accuracies")
            if acc is not None:
                return -acc
        return metrics.get("eval_loss")

    @staticmethod
    def _unwrap(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    @torch.no_grad()
    def _snapshot(self, model) -> dict[str, torch.Tensor]:
        unwrapped = self._unwrap(model)
        return {
            n: p.data.cpu().clone()
            for n, p in unwrapped.named_parameters()
            if p.requires_grad
        }

    def _restore(self, model, state: dict[str, torch.Tensor]):
        unwrapped = self._unwrap(model)
        for n, p in unwrapped.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(p.device))

    @torch.no_grad()
    def _compute_avg(self) -> dict[str, torch.Tensor]:
        avg = {}
        n = len(self.snapshots)
        for name in self.snapshots[0]:
            avg[name] = sum(s[name] for s in self.snapshots) / n
        return avg

    # ── DDP helpers (keep collectives in lockstep across ranks) ──
    def _bcast_params(self, model):
        if torch.distributed.is_initialized():
            for p in self._unwrap(model).parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)

    def _bcast_flag(self, model, value: bool) -> bool:
        if not torch.distributed.is_initialized():
            return value
        t = torch.tensor([1.0 if value else 0.0], device=next(model.parameters()).device)
        torch.distributed.broadcast(t, src=0)
        return t.item() > 0.5

    def _bcast_scalar(self, model, value) -> float:
        if not torch.distributed.is_initialized():
            return value if value is not None else float("inf")
        t = torch.tensor([value if value is not None else float("inf")],
                         device=next(model.parameters()).device)
        torch.distributed.broadcast(t, src=0)
        return t.item()

    def _restore_best(self, model):
        if is_main_process(LOCAL_RANK) and self.best_state is not None:
            self._restore(model, self.best_state)
        self._bcast_params(model)

    # ── Greedy-soup pool (RAM-measured) ──
    def _trainable_snapshot_bytes(self, model) -> int:
        """Bytes one CPU snapshot of the trainable params costs (cached)."""
        if self._snap_bytes is None:
            self._snap_bytes = sum(
                p.numel() * p.element_size()
                for p in self._unwrap(model).parameters() if p.requires_grad
            )
        return self._snap_bytes

    @staticmethod
    def _available_ram_bytes() -> Optional[int]:
        """Free host RAM in bytes (psutil if present, else /proc/meminfo). None
        if neither is readable — callers then stay conservative."""
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return None

    def _can_admit_snapshot(self, model) -> bool:
        """True only if a new CPU snapshot fits with headroom. A snapshot costs
        ~trainable bytes (bf16); the end-of-train soup also allocates a float32
        accumulator (~2x a snapshot), so require a margin of max(2GB, 3x snap).
        When free RAM can't be read, fall back to a tiny pool (seed only)."""
        snap = self._trainable_snapshot_bytes(model)
        avail = self._available_ram_bytes()
        if avail is None:
            return len(self.pool) < 2
        margin = max(2 * 1024**3, 3 * snap)
        return (avail - margin) >= snap

    def _consider_for_pool(self, model, loss, step: int) -> None:
        """Admit (loss, step, snapshot) into the top-k pool if RAM allows; else
        evict the worst member when this one is strictly better. Rank-0 only.
        Keeps the pool sorted best-first and bounded by soup_max AND free RAM."""
        if loss is None or loss != loss:   # None / NaN
            return
        if len(self.pool) < self.soup_max and self._can_admit_snapshot(model):
            self.pool.append({"loss": loss, "step": step, "state": self._snapshot(model)})
        elif self.pool:
            worst = max(self.pool, key=lambda e: e["loss"])
            if loss < worst["loss"]:
                # Free the evicted snapshot BEFORE allocating the new one so peak
                # RAM doesn't transiently hold both.
                self.pool.remove(worst)
                worst["state"] = None
                del worst
                gc.collect()
                self.pool.append({"loss": loss, "step": step, "state": self._snapshot(model)})
        # else: full, RAM-bound, and not better — skip.

        self.pool.sort(key=lambda e: e["loss"])
        snap_gb = self._trainable_snapshot_bytes(model) / 1e9
        avail = self._available_ram_bytes()
        avail_str = f"{avail / 1e9:.0f}GB" if avail is not None else "?"
        print(
            f"[sn56][soup] pool={len(self.pool)}/{self.soup_max} "
            f"(snap~{snap_gb:.2f}GB/ea, RAM livre {avail_str})",
            flush=True,
        )

    def _soup_remaining_secs(self) -> Optional[float]:
        if not self.end_time:
            return None
        try:
            end = datetime.datetime.strptime(
                self.end_time, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        except Exception:
            return None

    @torch.no_grad()
    def _load_soup_into_model(self, model, soup_sum: dict, n: int) -> None:
        """Write soup_sum / n into the model's trainable params (rank-0 only)."""
        inv = 1.0 / n
        for name, p in self._unwrap(model).named_parameters():
            if name in soup_sum:
                p.data.copy_((soup_sum[name] * inv).to(p.dtype))

    @torch.no_grad()
    def _greedy_soup(self, model) -> bool:
        """Combine pooled snapshots into a greedy model soup (Wortsman 2022).

        Seeded with the best single (lowest dev loss), it walks the rest in
        increasing-loss order and ACCEPTS a candidate only if it lowers dev loss
        — so the result is never worse than the best single. DDP-safe: rank 0
        owns the pool, all ranks run every evaluate(). Time-bounded: the soup may
        spend at most half of the remaining wall-clock (the dev-pass uses the
        rest). Leaves the soup weights in the model and adopts them as best."""
        is_main = is_main_process(LOCAL_RANK)
        k = int(self._bcast_scalar(model, float(len(self.pool) if is_main else 0)))
        if k <= 1:
            return False  # nothing to combine (0/1 snapshot = best single already)

        cands = sorted(self.pool, key=lambda e: e["loss"]) if is_main else None
        # float32 accumulator: averaging many bf16 snapshots in bf16 loses bits.
        soup_sum = {n: t.float().clone() for n, t in cands[0]["state"].items()} if is_main else None
        if is_main:
            self._restore(model, cands[0]["state"])
        self._bcast_params(model)

        t0 = time.perf_counter()
        seed_loss = self._eval_synced(model)
        if seed_loss != seed_loss or seed_loss == float("inf"):
            return False
        per_eval = max(1.0, time.perf_counter() - t0)
        remaining = self._soup_remaining_secs()
        budget = (remaining * 0.5) if remaining is not None else float("inf")

        best, n = seed_loss, 1
        for i in range(1, k):
            # Stop if the next eval would blow the soup's time budget.
            if (time.perf_counter() - t0) + per_eval > budget:
                if is_main:
                    print(f"[sn56][soup] tempo esgotado após {i} cand(s)", flush=True)
                break
            if is_main:   # tentative average → model params
                inv = 1.0 / (n + 1)
                st = cands[i]["state"]
                for name, p in self._unwrap(model).named_parameters():
                    if name in soup_sum and name in st:
                        p.data.copy_(((soup_sum[name] + st[name].float()) * inv).to(p.dtype))
            self._bcast_params(model)
            cand_loss = self._eval_synced(model)
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if accept:
                if is_main:
                    st = cands[i]["state"]
                    for name in soup_sum:
                        if name in st:
                            soup_sum[name] += st[name].float()
                n += 1
                best = cand_loss
            elif is_main:
                # reject: restore the current soup average on rank 0. No
                # broadcast needed — the next iteration overwrites every rank's
                # weights with a fresh tentative (and broadcasts) before its
                # eval; the post-loop reload+broadcast covers a trailing reject.
                self._load_soup_into_model(model, soup_sum, n)
        # The model now holds the final soup average (last op left it there on
        # accept; on reject we restored it). Re-broadcast to be certain.
        if is_main:
            self._load_soup_into_model(model, soup_sum, n)
        self._bcast_params(model)

        if is_main:
            self.best_state = self._snapshot(model)
            self.best_loss = best
            self.best_source = f"soup(n={n}/{k})"
            print(f"[sn56][soup] n={n}/{k}, loss {seed_loss:.4f} -> {best:.4f}", flush=True)
        self.best_loss = self._bcast_scalar(model, best if is_main else best)
        return True

    def _eval_synced(self, model) -> float:
        """trainer.evaluate() (a DDP collective — all ranks call it), returning a
        rank-synced scalar metric. inf on failure."""
        self._evaluating = True
        try:
            loss = self._get_metric(self.trainer.evaluate())
        except Exception as e:
            print(f"[sn56][soup] eval falhou ({e})", flush=True)
            loss = None
        finally:
            self._evaluating = False
        return self._bcast_scalar(model, loss)

    # ── Disk-based averaging (sharded / too-big models) ──
    @torch.no_grad()
    def _disk_average_and_submit(self, model):
        """Train-end fallback: uniform-average the last K consolidated
        checkpoints from disk (streamed tensor by tensor, O(1) RAM), then
        load-and-eval the result. Submit only if it beats best on the dev set.
        Guarded: if the averaged weights can't be loaded (e.g. sharded load
        path), leave the submission (already the best checkpoint) untouched.
        """
        import glob
        from safetensors.torch import safe_open

        is_main = is_main_process(LOCAL_RANK)
        avg_dir = None
        if is_main and self.output_dir:
            ckpts = sorted(
                glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
            )
            ckpts = [c for c in ckpts if glob.glob(os.path.join(c, "*.safetensors"))]
            ckpts = ckpts[-self.disk_members:]
            if len(ckpts) >= 2:
                avg_dir = self._stream_average_checkpoints(ckpts)

        # On any miss leave the submission untouched — under sharding the
        # in-memory best_state is only a shard, so restoring it would corrupt
        # the model; the submission dir already holds the consolidated best.
        if not self._bcast_flag(model, avg_dir is not None):
            return

        loaded = True
        if is_main:
            try:
                full = {}
                for sf in glob.glob(os.path.join(avg_dir, "*.safetensors")):
                    with safe_open(sf, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            full[k] = f.get_tensor(k)
                self._unwrap(model).load_state_dict(full, strict=False)
            except Exception as e:
                print(f"[sn56][disco] carga da média falhou ({e}), mantendo melhor", flush=True)
                loaded = False
        if not self._bcast_flag(model, loaded):
            return

        self._bcast_params(model)
        self._evaluating = True
        try:
            avg_loss = self._get_metric(self.trainer.evaluate())
        except Exception as e:
            print(f"[sn56][disco] avaliação da média falhou ({e})", flush=True)
            avg_loss = None
        finally:
            self._evaluating = False
        avg_loss = self._bcast_scalar(model, avg_loss)

        if is_main:
            print(f"[sn56][disco] média={avg_loss:.4f} vs melhor={self.best_loss:.4f}", flush=True)
        if avg_loss < self.best_loss - 0.002 * abs(self.best_loss):
            if is_main and avg_dir:
                self._submit_dir(avg_dir, avg_loss)
                print("[sn56][disco] submissão atualizada com a média", flush=True)

    @torch.no_grad()
    def _stream_average_checkpoints(self, ckpts: list[str]) -> str | None:
        """Uniform-average matching safetensors across checkpoint dirs, one
        tensor at a time (O(1) RAM). Returns a temp dir with the averaged
        weights + copied config/tokenizer, or None on failure.
        """
        import glob
        from safetensors.torch import safe_open, save_file
        try:
            opens = []
            for c in ckpts:
                files = glob.glob(os.path.join(c, "*.safetensors"))
                opens.append({os.path.basename(f): f for f in files})
            keys = None
            for o in opens:
                ks = set()
                for b, p in o.items():
                    with safe_open(p, framework="pt", device="cpu") as h:
                        ks |= {(b, k) for k in h.keys()}
                keys = ks if keys is None else (keys & ks)
            n = len(ckpts)
            merged: dict[str, torch.Tensor] = {}
            for (b, k) in keys:
                acc = None
                src_dtype = None
                for o in opens:
                    with safe_open(o[b], framework="pt", device="cpu") as h:
                        t = h.get_tensor(k)
                        src_dtype = t.dtype
                        t = t.float()
                    acc = t if acc is None else acc + t
                merged[k] = (acc / n).to(src_dtype)
            out = (self._submission_dir or ckpts[-1]).rstrip("/") + ".disk_avg"
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out, exist_ok=True)
            save_file(merged, os.path.join(out, "model.safetensors"))
            # Copy non-weight files (config, tokenizer) from the latest checkpoint.
            for fn in os.listdir(ckpts[-1]):
                src = os.path.join(ckpts[-1], fn)
                if not fn.endswith(".safetensors") and os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out, fn))
            return out
        except Exception as e:
            print(f"[sn56][disco] média em disco falhou ({e})", flush=True)
            return None

    def _submit_dir(self, src_dir: str, metric: float):
        """Atomically replace the submission dir with src_dir's contents (rank 0)."""
        sub = getattr(self, "_submission_dir", None)
        if not sub:
            return
        with open(os.path.join(src_dir, "loss.txt"), "w") as f:
            f.write(f"disk_avg,{metric}")
        if os.path.exists(sub):
            shutil.rmtree(sub)
        os.rename(src_dir, sub)

    @torch.no_grad()
    def _drop_memorized_samples(self, model):
        """After rollback, drop samples the model has memorized (low loss).

        Quick no-grad forward pass → keep top 50% by loss (the learning
        frontier). Bottom 50% have near-zero loss = zero gradient signal.
        """
        dataset = getattr(self.trainer.train_dataset, "eval_dataset", None)
        if dataset is None or len(dataset) < 100:
            return

        # Try using data_filter module if available, fall back to trainer.evaluate
        try:
            from data_filter import compute_sample_losses
            device = str(next(model.parameters()).device)
            losses = compute_sample_losses(model, dataset, batch_size=128, device=device)
        except ImportError:
            # data_filter not on this branch — skip sample filtering
            return

        import numpy as np
        nonzero = [l for l in losses if l > 0]
        if len(nonzero) < 100:
            return

        median_loss = float(np.median(nonzero))

        # Don't remove samples (would break DataLoader mid-epoch).
        # Instead mask labels to -100 — sample stays but contributes zero loss/gradient.
        n_masked = 0
        for sample, loss in zip(dataset, losses):
            if 0 < loss < median_loss:
                labels = sample.get("labels", [])
                if isinstance(labels, torch.Tensor):
                    sample["labels"] = torch.full_like(labels, -100)
                elif isinstance(labels, list):
                    sample["labels"] = [-100] * len(labels)
                n_masked += 1

        if n_masked > 0:
            print(
                f"Escondi {n_masked}/{len(dataset)} memorized samples "
                f"(loss < {median_loss:.4f}), zero gradient contribution",
                flush=True,
            )

    def _rollback_and_respond(self, model, state):
        """Rollback to best checkpoint and escalate regularization.

        Rank 0 holds best_state (CPU snapshots are rank-0 only to save memory).
        Restore on rank 0, then broadcast to all ranks for DDP sync.
        """
        self.rollback_count += 1

        # Restore best weights from disk (submission dir) — more reliable
        # than in-memory restore which can fail silently with wrapped models.
        # Fall back to in-memory restore if disk load fails.
        _restored_from_disk = False
        if is_main_process(LOCAL_RANK):
            submission_dir = getattr(self, "_submission_dir", None)
            if submission_dir and os.path.exists(submission_dir):
                try:
                    from safetensors.torch import load_file
                    import glob
                    safetensor_files = glob.glob(os.path.join(submission_dir, "*.safetensors"))
                    if safetensor_files:
                        unwrapped = self._unwrap(model)
                        full_state = {}
                        for sf in safetensor_files:
                            full_state.update(load_file(sf, device=str(next(unwrapped.parameters()).device)))
                        unwrapped.load_state_dict(full_state, strict=True)
                        _restored_from_disk = True
                        print(f"pesos recuperados do disco ({submission_dir})", flush=True)
                except Exception as e:
                    print(f"disco falhou ({e}), tentando memória", flush=True)
                    _restored_from_disk = False

            if not _restored_from_disk and self.best_state is not None:
                self._restore(model, self.best_state)

        if torch.distributed.is_initialized():
            _unwrapped = self._unwrap(model)
            for p in _unwrapped.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            for b in _unwrapped.buffers():
                torch.distributed.broadcast(b.data, src=0)

        # Cut LR by 25% — must update both optimizer AND scheduler base_lrs
        # so the cosine scheduler doesn't overwrite the cut on the next step.
        if self.trainer is not None and self.trainer.optimizer is not None:
            for pg in self.trainer.optimizer.param_groups:
                old_lr = pg["lr"]
                pg["lr"] = old_lr * 0.75
            # Update scheduler base so cosine decay starts from the cut LR
            if hasattr(self.trainer, "lr_scheduler") and self.trainer.lr_scheduler is not None:
                sched = self.trainer.lr_scheduler
                if hasattr(sched, "base_lrs"):
                    sched.base_lrs = [lr * 0.75 for lr in sched.base_lrs]
            new_lr = self.trainer.optimizer.param_groups[0]["lr"]
            # Reset optimizer momentum — stale m/v from overfit trajectory
            # would push restored weights back toward overfitting.
            for pg in self.trainer.optimizer.param_groups:
                for p in pg["params"]:
                    opt_state = self.trainer.optimizer.state.get(p)
                    if opt_state:
                        if "exp_avg" in opt_state:
                            opt_state["exp_avg"].zero_()
                        if "exp_avg_sq" in opt_state:
                            opt_state["exp_avg_sq"].zero_()
            print(
                f"Não não não, voltando pro melhor momento #{self.rollback_count}: "
                f"restored best weights (loss={self.best_loss:.4f}), "
                f"LR cut to {new_lr:.2e}, optimizer state reset",
                flush=True,
            )

        # Escalate NEFTune if active
        if self.trainer is not None:
            model_to_check = self._unwrap(self.trainer.model)
            emb = model_to_check.get_input_embeddings() if hasattr(model_to_check, "get_input_embeddings") else None
            if emb is not None and hasattr(emb, "neftune_noise_alpha") and emb.neftune_noise_alpha is not None:
                self.neftune_level_idx = min(
                    self.neftune_level_idx + 1, len(_NEFTUNE_LEVELS) - 1
                )
                new_alpha = _NEFTUNE_LEVELS[self.neftune_level_idx]
                emb.neftune_noise_alpha = new_alpha
                print(
                    f"Sacudindo mais um pouco, caos={new_alpha}",
                    flush=True,
                )

        # Drop memorized samples — focus on the learning frontier
        try:
            self._drop_memorized_samples(model)
        except Exception as e:
            print(f"Filtragem falhou: {e}", flush=True)

        # Reset overfitting counter. Seed snapshot window with best state
        # so averaging can resume from the known-good weights (1/3 filled).
        self.overfit_counter = 0
        self.snapshots.clear()
        if is_main_process(LOCAL_RANK) and self.best_state is not None:
            self.snapshots.append({k: v.clone() for k, v in self.best_state.items()})

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mb = n_params * 4 / 1e6 * self.window
        print(
            f"Rede de segurança pronta: window={self.window}, "
            f"overfit_threshold={_OVERFIT_THRESHOLD:.0%}, patience={_OVERFIT_PATIENCE}, "
            f"max_rollbacks={_MAX_ROLLBACKS}, "
            f"{n_params/1e6:.1f}M params (~{mb:.0f}MB for snapshots)",
            flush=True,
        )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self._evaluating or model is None or self.trainer is None:
            return
        # Snapshots and avg eval are rank-0 only (CPU memory + extra eval).
        # Overfitting detection uses metrics which are synced across ranks.
        # Rollback (_restore) runs on all ranks to keep DDP in sync.
        is_main = is_main_process(LOCAL_RANK)

        base_loss = self._get_metric(metrics) if metrics else None
        if base_loss is None:
            return

        # --- Best-checkpoint tracking + RAM-gated greedy-soup pool ---
        # We no longer average a sliding window every eval (an extra eval each
        # time). Instead we keep the lowest-dev-loss snapshots in a RAM-measured
        # pool and combine them ONCE, at train end, via greedy soup — global
        # (best checkpoints regardless of when they occurred) and monotone (never
        # worse than the best single). Snapshots and the pool are rank-0 only.
        is_new_best = base_loss < self.best_loss
        if is_new_best:
            self.best_loss = base_loss
            if is_main:
                self.best_state = self._snapshot(model)
                self.best_source = f"base@step{state.global_step}"
            self.overfit_counter = 0

        # Pool admission (RAM-gated). Only in RAM mode: under sharding ("disk")
        # the in-RAM snapshots are just shards, so soup is invalid there.
        if is_main and self.averaging_mode == "ram":
            self._consider_for_pool(model, base_loss, state.global_step)

        # Sync best_loss across ranks so overfitting detection is consistent.
        if torch.distributed.is_initialized():
            bl_tensor = torch.tensor([self.best_loss], device=next(model.parameters()).device)
            torch.distributed.broadcast(bl_tensor, src=0)
            self.best_loss = bl_tensor.item()

        # Log (rank-0 only)
        delta_pct = (base_loss - self.best_loss) / self.best_loss * 100 if self.best_loss > 0 else 0
        if is_main:
            print(
                f"passo={state.global_step}: "
                f"atual={base_loss:.4f} (pool {len(self.pool)}/{self.soup_max}), "
                f"melhor={self.best_loss:.4f} de {self.best_source}, "
                f"delta={delta_pct:+.1f}%",
                flush=True,
            )

        # --- Overfitting detection (all ranks — rollback must be in sync) ---
        if not is_new_best and base_loss > self.best_loss * (1 + _OVERFIT_THRESHOLD):
            self.overfit_counter += 1
            print(
                f"Hmm ficando confortável demais... {delta_pct:+.1f}% fora "
                f"(contador={self.overfit_counter}/{_OVERFIT_PATIENCE})",
                flush=True,
            )

            if self.overfit_counter >= _OVERFIT_PATIENCE:
                # Overfitting confirmed: stop early instead of the old cut-LR /
                # retry-on-train rollback (which never helped). The best weights
                # are restored at train end; the final dev-data pass (in
                # train_instruct) then nudges that best checkpoint and saves.
                # All ranks reach this identically (best_loss/base_loss are synced
                # above), so should_training_stop is set in lockstep — no DDP skew.
                print(
                    "Overfit confirmado (3x), parando cedo — melhor sera "
                    "restaurado e o dev-pass roda no fim",
                    flush=True,
                )
                control.should_training_stop = True
        elif not is_new_best:
            # Within plateau zone — reset counter
            self.overfit_counter = 0

    def on_save(self, args, state, control, model=None, **kwargs):
        if not is_main_process(LOCAL_RANK):
            return
        if self.best_state is None or "avg" not in self.best_source:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.exists(checkpoint_dir):
            return

        # Stash CURRENT weights (not last snapshot — model has trained further)
        current_state = self._snapshot(model)
        self._restore(model, self.best_state)
        try:
            unwrapped = self._unwrap(model)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(checkpoint_dir)
        finally:
            # Restore current training weights, not a stale snapshot
            self._restore(model, current_state)

        print(
            f"Salvei o checkpoint com os pesos misturados "
            f"(source={self.best_source})",
            flush=True,
        )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Disk-averaging fallback for sharded / too-big models: average the
        # consolidated checkpoints on disk and submit if better. Independent of
        # best_state (which is only a shard under sharding). Non-fatal — leaves
        # the best already in the submission dir on any error.
        if self.averaging_mode == "disk" and self.trainer is not None:
            try:
                self._disk_average_and_submit(model)
            except Exception as e:
                print(f"[sn56][disco] falhou ({e}), mantendo melhor", flush=True)
            print(
                f"Forma final: {self.best_source} "
                f"(nota={self.best_loss:.4f}, voltas={self.rollback_count})",
                flush=True,
            )
            return

        # RAM / off path: greedy-soup the pooled snapshots (monotone — never
        # worse than the best single), then restore the winning weights and
        # persist them to the submission so the soup takes effect even when the
        # downstream dev-pass is skipped. The submission already holds the best
        # single (written during training), so any soup failure is harmless.
        if self.averaging_mode == "ram" and self.trainer is not None:
            try:
                self._greedy_soup(model)
            except Exception as e:
                print(f"[sn56][soup] falhou ({e}), usando melhor único", flush=True)

        if self.best_state is None and not torch.distributed.is_initialized():
            print("Nenhuma foto tirada", flush=True)
            return
        self._restore_best(model)

        # Persist the (souped) best weights to the submission. Rank-0 only,
        # weights-only atomic swap (reuses the dev-pass saver, proven for both
        # LoRA and full-FT). If the dev-pass runs next it starts from these
        # in-memory soup weights and re-saves; if it's skipped, the soup is
        # already the submission.
        if is_main_process(LOCAL_RANK) and getattr(self, "_submission_dir", None):
            try:
                from dev_pass import _save_weights_only
                _save_weights_only(
                    self._unwrap(model), self._submission_dir,
                    lambda m: print(m, flush=True),
                )
            except Exception as e:
                print(f"[sn56][soup] persistência falhou ({e})", flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(
            f"Forma final: {self.best_source} "
            f"(nota={self.best_loss:.4f}, voltas={self.rollback_count})",
            flush=True,
        )
