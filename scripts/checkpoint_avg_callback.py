"""
Weight-averaging callback for HuggingFace Trainer.

Three cooperating mechanisms:
1. RAM-gated greedy-soup pool: collects low-loss weight snapshots during
   training and combines them once at the end (Wortsman 2022).  Admission is
   limited by measured free host RAM so it never OOMs.
2. Disk-streaming average: fallback for sharded/large models where in-RAM
   snapshots are per-shard and invalid for averaging; streams checkpoint
   safetensors tensor-by-tensor (O(1) RAM).
3. Overfitting early-stop: monitors eval loss and stops training when loss
   rises above best by a threshold for N consecutive evals, then restores
   best weights before the optional dev-pass.
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

_RANK = int(os.getenv("LOCAL_RANK", "0"))

# ── Overfitting detection ──
_LOSS_RISE_RATIO = 0.05        # eval loss 5% above best → bad eval
_CONSECUTIVE_BAD_EVALS = 3     # how many bad evals before early stop
_ROLLBACK_CAP = 2              # max early-stops (guards time budget)

# NEFTune escalation on confirmed overfit
_NOISE_ALPHA_LADDER = [5, 10, 15]


def _peel_wrapper(model):
    """Unwrap DDP / FSDP / pipeline wrappers to the base module."""
    while hasattr(model, "module"):
        model = model.module
    return model


class AdaptiveTrainingCallback(TrainerCallback):
    """Checkpoint averaging and overfitting early-stop callback.

    Intended usage:
        cb = AdaptiveTrainingCallback(window=3, averaging_mode="ram")
        trainer = Trainer(..., callbacks=[cb, ...])
        cb.trainer = trainer   # must be set before train()
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
        # Public averaging config
        self.window = window
        self.device = device
        self.use_reward_accuracy = use_reward_accuracy
        self.averaging_mode = averaging_mode  # "ram" | "disk" | "off"
        self.output_dir = output_dir
        self.disk_members = disk_members
        self.soup_max = soup_max

        # Best-checkpoint state (rank-0 only)
        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=window)
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_loss: float = float("inf")
        self.best_source: str = "none"

        # Greedy-soup pool
        self.pool: list[dict] = []
        self._snap_bytes: Optional[int] = None

        # Set by caller before train()
        self._submission_dir: str | None = None
        self.end_time: str = ""

        # Overfitting counters
        self.overfit_counter = 0
        self.rollback_count = 0
        self.neftune_level_idx = 0

        # Re-entry guard (soup triggers extra evaluate() calls)
        self._evaluating = False
        self.trainer = None

    # ── Metric extraction ──

    def _extract_metric(self, metrics: dict) -> Optional[float]:
        """Return the scalar to minimise. Accuracy is negated so lower = better."""
        if self.use_reward_accuracy:
            acc = metrics.get("eval_rewards/accuracies")
            if acc is not None:
                return -float(acc)
        val = metrics.get("eval_loss")
        return float(val) if val is not None else None

    # ── Weight snapshot / restore helpers ──

    @torch.no_grad()
    def _capture_weights(self, model) -> dict[str, torch.Tensor]:
        base = _peel_wrapper(model)
        return {
            name: param.data.cpu().clone()
            for name, param in base.named_parameters()
            if param.requires_grad
        }

    def _apply_weights(self, model, weight_dict: dict[str, torch.Tensor]) -> None:
        base = _peel_wrapper(model)
        for name, param in base.named_parameters():
            if name in weight_dict:
                param.data.copy_(weight_dict[name].to(param.device))

    def _restore_best_to_model(self, model) -> None:
        if is_main_process(_RANK) and self.best_state is not None:
            self._apply_weights(model, self.best_state)
        self._sync_params(model)

    # ── DDP collective helpers ──

    def _sync_params(self, model) -> None:
        """Broadcast rank-0 parameters to all other ranks."""
        if torch.distributed.is_initialized():
            for p in _peel_wrapper(model).parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)

    def _sync_flag(self, model, value: bool) -> bool:
        if not torch.distributed.is_initialized():
            return value
        buf = torch.tensor(
            [1.0 if value else 0.0],
            device=next(model.parameters()).device,
        )
        torch.distributed.broadcast(buf, src=0)
        return buf.item() > 0.5

    def _sync_scalar(self, model, value) -> float:
        safe = value if (value is not None and value == value) else float("inf")
        if not torch.distributed.is_initialized():
            return safe
        buf = torch.tensor([safe], device=next(model.parameters()).device)
        torch.distributed.broadcast(buf, src=0)
        return buf.item()

    # ── RAM-gated greedy-soup pool ──

    def _bytes_per_snapshot(self, model) -> int:
        if self._snap_bytes is None:
            self._snap_bytes = sum(
                p.numel() * p.element_size()
                for p in _peel_wrapper(model).parameters()
                if p.requires_grad
            )
        return self._snap_bytes

    @staticmethod
    def _free_host_ram() -> Optional[int]:
        """Available host RAM in bytes, or None if unreadable."""
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return None

    def _has_room_for_snapshot(self, model) -> bool:
        snap = self._bytes_per_snapshot(model)
        free = self._free_host_ram()
        if free is None:
            return len(self.pool) < 2  # conservative fallback
        headroom = max(2 * 1024 ** 3, 3 * snap)
        return (free - headroom) >= snap

    def _try_admit_to_pool(self, model, loss: float, step: int) -> None:
        """Add snapshot to pool if loss is competitive and RAM allows."""
        if loss != loss:  # NaN guard
            return
        admitted = False
        if len(self.pool) < self.soup_max and self._has_room_for_snapshot(model):
            self.pool.append({"loss": loss, "step": step, "state": self._capture_weights(model)})
            admitted = True
        elif self.pool:
            worst = max(self.pool, key=lambda e: e["loss"])
            if loss < worst["loss"]:
                # Free evicted snapshot before allocating new one
                self.pool.remove(worst)
                worst["state"] = None
                del worst
                gc.collect()
                self.pool.append({"loss": loss, "step": step, "state": self._capture_weights(model)})
                admitted = True

        if admitted:
            self.pool.sort(key=lambda e: e["loss"])

        snap_gb = self._bytes_per_snapshot(model) / 1e9
        free = self._free_host_ram()
        free_str = f"{free / 1e9:.0f}GB" if free is not None else "unknown"
        print(
            f"[env][pool] size={len(self.pool)}/{self.soup_max} "
            f"snap~{snap_gb:.2f}GB free={free_str}",
            flush=True,
        )

    def _remaining_seconds(self) -> Optional[float]:
        if not self.end_time:
            return None
        try:
            deadline = datetime.datetime.strptime(
                self.end_time, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            return (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        except Exception:
            return None

    # ── Greedy-soup combination (RAM path) ──

    @torch.no_grad()
    def _write_average_to_model(self, model, acc: dict, count: int) -> None:
        scale = 1.0 / count
        for name, param in _peel_wrapper(model).named_parameters():
            if name in acc:
                param.data.copy_((acc[name] * scale).to(param.dtype))

    def _run_evaluate(self, model) -> float:
        """All-rank evaluate() call; returns a rank-synced scalar."""
        self._evaluating = True
        metric = None
        try:
            metric = self._extract_metric(self.trainer.evaluate())
        except Exception as exc:
            print(f"[env][soup] evaluate failed: {exc}", flush=True)
        finally:
            self._evaluating = False
        return self._sync_scalar(model, metric)

    @torch.no_grad()
    def _greedy_soup(self, model) -> bool:
        """Combine pool snapshots via greedy soup (Wortsman 2022).

        Starts from the best single snapshot and greedily admits each
        successive candidate only when it improves eval loss.  Result is
        monotonically better than or equal to the best single.
        """
        is_main = is_main_process(_RANK)
        pool_size = int(self._sync_scalar(model, float(len(self.pool) if is_main else 0)))
        if pool_size <= 1:
            return False

        ranked = sorted(self.pool, key=lambda e: e["loss"]) if is_main else None

        # float32 accumulator prevents bf16 precision loss during summation
        acc = {n: t.float().clone() for n, t in ranked[0]["state"].items()} if is_main else None
        if is_main:
            self._apply_weights(model, ranked[0]["state"])
        self._sync_params(model)

        t0 = time.perf_counter()
        seed_loss = self._run_evaluate(model)
        if seed_loss != seed_loss or seed_loss == float("inf"):
            return False

        time_per_eval = max(1.0, time.perf_counter() - t0)
        budget = (self._remaining_seconds() or float("inf")) * 0.5

        current_best, accepted = seed_loss, 1
        for idx in range(1, pool_size):
            elapsed = time.perf_counter() - t0
            if elapsed + time_per_eval > budget:
                if is_main:
                    print(f"[env][soup] time budget reached after {idx} candidate(s)", flush=True)
                break

            if is_main:
                inv = 1.0 / (accepted + 1)
                candidate_state = ranked[idx]["state"]
                for name, param in _peel_wrapper(model).named_parameters():
                    if name in acc and name in candidate_state:
                        param.data.copy_(
                            ((acc[name] + candidate_state[name].float()) * inv).to(param.dtype)
                        )
            self._sync_params(model)

            candidate_loss = self._run_evaluate(model)
            if candidate_loss == candidate_loss and candidate_loss < current_best - 1e-6:
                if is_main:
                    cand_st = ranked[idx]["state"]
                    for name in acc:
                        if name in cand_st:
                            acc[name] += cand_st[name].float()
                accepted += 1
                current_best = candidate_loss
            elif is_main:
                self._write_average_to_model(model, acc, accepted)

        if is_main:
            self._write_average_to_model(model, acc, accepted)
        self._sync_params(model)

        if is_main:
            self.best_state = self._capture_weights(model)
            self.best_loss = current_best
            self.best_source = f"soup(n={accepted}/{pool_size})"
            print(
                f"[env][soup] accepted={accepted}/{pool_size} "
                f"loss {seed_loss:.4f} -> {current_best:.4f}",
                flush=True,
            )
        self.best_loss = self._sync_scalar(model, current_best if is_main else current_best)
        return True

    # ── Disk-streaming average (sharded / large-model path) ──

    @torch.no_grad()
    def _stream_average_checkpoints(self, ckpt_dirs: list[str]) -> Optional[str]:
        """Average matching safetensors across checkpoint dirs, O(1) RAM.
        Returns path to a new dir with averaged weights, or None on failure.
        """
        import glob
        from safetensors.torch import safe_open, save_file
        try:
            file_maps = []
            for ckpt in ckpt_dirs:
                files = glob.glob(os.path.join(ckpt, "*.safetensors"))
                file_maps.append({os.path.basename(f): f for f in files})

            # Intersect keys across all checkpoints
            shared_keys = None
            for fmap in file_maps:
                ks = set()
                for basename, path in fmap.items():
                    with safe_open(path, framework="pt", device="cpu") as h:
                        ks |= {(basename, k) for k in h.keys()}
                shared_keys = ks if shared_keys is None else (shared_keys & ks)

            n_ckpts = len(ckpt_dirs)
            merged: dict[str, torch.Tensor] = {}
            for (basename, key) in shared_keys:
                accumulator = None
                orig_dtype = None
                for fmap in file_maps:
                    with safe_open(fmap[basename], framework="pt", device="cpu") as h:
                        tensor = h.get_tensor(key)
                        orig_dtype = tensor.dtype
                        tensor = tensor.float()
                    accumulator = tensor if accumulator is None else accumulator + tensor
                merged[key] = (accumulator / n_ckpts).to(orig_dtype)

            out_path = (self._submission_dir or ckpt_dirs[-1]).rstrip("/") + ".disk_avg"
            if os.path.exists(out_path):
                shutil.rmtree(out_path)
            os.makedirs(out_path, exist_ok=True)
            save_file(merged, os.path.join(out_path, "model.safetensors"))

            # Copy non-weight files from the latest checkpoint
            for fname in os.listdir(ckpt_dirs[-1]):
                src = os.path.join(ckpt_dirs[-1], fname)
                if not fname.endswith(".safetensors") and os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out_path, fname))
            return out_path
        except Exception as exc:
            print(f"[env][disk] streaming average failed: {exc}", flush=True)
            return None

    def _atomic_replace_submission(self, src_dir: str, metric: float) -> None:
        """Replace submission dir with src_dir contents (rank-0 only)."""
        dest = getattr(self, "_submission_dir", None)
        if not dest:
            return
        with open(os.path.join(src_dir, "loss.txt"), "w") as fh:
            fh.write(f"disk_avg,{metric}")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(src_dir, dest)

    @torch.no_grad()
    def _disk_average_and_submit(self, model) -> None:
        """Average last K consolidated checkpoints from disk; submit if better."""
        import glob
        from safetensors.torch import safe_open

        is_main = is_main_process(_RANK)
        avg_dir = None

        if is_main and self.output_dir:
            all_ckpts = sorted(
                glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
            )
            valid_ckpts = [c for c in all_ckpts if glob.glob(os.path.join(c, "*.safetensors"))]
            candidates = valid_ckpts[-self.disk_members:]
            if len(candidates) >= 2:
                avg_dir = self._stream_average_checkpoints(candidates)

        if not self._sync_flag(model, avg_dir is not None):
            return

        load_ok = True
        if is_main:
            try:
                combined = {}
                for sf in glob.glob(os.path.join(avg_dir, "*.safetensors")):
                    with safe_open(sf, framework="pt", device="cpu") as fh:
                        for key in fh.keys():
                            combined[key] = fh.get_tensor(key)
                _peel_wrapper(model).load_state_dict(combined, strict=False)
            except Exception as exc:
                print(f"[env][disk] load failed: {exc}, keeping best", flush=True)
                load_ok = False

        if not self._sync_flag(model, load_ok):
            return

        self._sync_params(model)
        self._evaluating = True
        avg_metric = None
        try:
            avg_metric = self._extract_metric(self.trainer.evaluate())
        except Exception as exc:
            print(f"[env][disk] eval failed: {exc}", flush=True)
        finally:
            self._evaluating = False
        avg_metric = self._sync_scalar(model, avg_metric)

        if is_main:
            print(f"[env][disk] avg={avg_metric:.4f} vs best={self.best_loss:.4f}", flush=True)
        if avg_metric < self.best_loss - 0.002 * abs(self.best_loss):
            if is_main and avg_dir:
                self._atomic_replace_submission(avg_dir, avg_metric)
                print("[env][disk] submission updated with disk average", flush=True)

    # ── Post-rollback sample filtering ──

    def _mask_memorized_samples(self, model) -> None:
        """Mask labels of memorized (very-low-loss) samples to -100.
        These contribute zero gradient and slow learning after rollback.
        """
        dataset = getattr(self.trainer.train_dataset, "eval_dataset", None)
        if dataset is None or len(dataset) < 100:
            return
        try:
            from data_filter import compute_sample_losses
            device = str(next(model.parameters()).device)
            losses = compute_sample_losses(model, dataset, batch_size=128, device=device)
        except ImportError:
            return

        import numpy as np
        nonzero = [l for l in losses if l > 0]
        if len(nonzero) < 100:
            return

        threshold = float(np.median(nonzero))
        masked = 0
        for sample, loss in zip(dataset, losses):
            if 0 < loss < threshold:
                labels = sample.get("labels", [])
                if isinstance(labels, torch.Tensor):
                    sample["labels"] = torch.full_like(labels, -100)
                elif isinstance(labels, list):
                    sample["labels"] = [-100] * len(labels)
                masked += 1

        if masked:
            print(
                f"[env] masked {masked}/{len(dataset)} memorized samples "
                f"(loss < {threshold:.4f})",
                flush=True,
            )

    # ── HuggingFace TrainerCallback hooks ──

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        est_mb = n * 4 / 1e6 * self.window
        print(
            f"[env][cb] ready — window={self.window} "
            f"rise_ratio={_LOSS_RISE_RATIO:.0%} patience={_CONSECUTIVE_BAD_EVALS} "
            f"max_stops={_ROLLBACK_CAP} "
            f"{n/1e6:.1f}M params (~{est_mb:.0f}MB snapshots)",
            flush=True,
        )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self._evaluating or model is None or self.trainer is None:
            return
        is_main = is_main_process(_RANK)

        cur_loss = self._extract_metric(metrics) if metrics else None
        if cur_loss is None:
            return

        # ── Best-checkpoint tracking ──
        improved = cur_loss < self.best_loss
        if improved:
            self.best_loss = cur_loss
            if is_main:
                self.best_state = self._capture_weights(model)
                self.best_source = f"base@step{state.global_step}"
            self.overfit_counter = 0

        if is_main and self.averaging_mode == "ram":
            self._try_admit_to_pool(model, cur_loss, state.global_step)

        # Sync best_loss across ranks
        if torch.distributed.is_initialized():
            bl = torch.tensor([self.best_loss], device=next(model.parameters()).device)
            torch.distributed.broadcast(bl, src=0)
            self.best_loss = bl.item()

        pct = (cur_loss - self.best_loss) / self.best_loss * 100 if self.best_loss > 0 else 0
        if is_main:
            print(
                f"[env][eval] step={state.global_step} "
                f"loss={cur_loss:.4f} (pool={len(self.pool)}/{self.soup_max}) "
                f"best={self.best_loss:.4f} from={self.best_source} "
                f"delta={pct:+.1f}%",
                flush=True,
            )

        # ── Overfitting detection ──
        if not improved and cur_loss > self.best_loss * (1 + _LOSS_RISE_RATIO):
            self.overfit_counter += 1
            print(
                f"[env][overfit] loss {pct:+.1f}% above best "
                f"({self.overfit_counter}/{_CONSECUTIVE_BAD_EVALS})",
                flush=True,
            )
            if self.overfit_counter >= _CONSECUTIVE_BAD_EVALS:
                print(
                    "[env][overfit] confirmed — stopping early, "
                    "best weights will be restored at train end",
                    flush=True,
                )
                control.should_training_stop = True
        elif not improved:
            self.overfit_counter = 0

    def on_save(self, args, state, control, model=None, **kwargs):
        if not is_main_process(_RANK):
            return
        if self.best_state is None or "avg" not in self.best_source:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(ckpt_dir):
            return

        # Stash current weights, write best into the checkpoint, then restore
        current = self._capture_weights(model)
        self._apply_weights(model, self.best_state)
        try:
            base = _peel_wrapper(model)
            if hasattr(base, "save_pretrained"):
                base.save_pretrained(ckpt_dir)
        finally:
            self._apply_weights(model, current)

        print(f"[env][save] wrote averaged weights to {ckpt_dir}", flush=True)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.averaging_mode == "disk" and self.trainer is not None:
            try:
                self._disk_average_and_submit(model)
            except Exception as exc:
                print(f"[env][disk] failed: {exc}, keeping best", flush=True)
            print(
                f"[env][end] source={self.best_source} "
                f"loss={self.best_loss:.4f} rollbacks={self.rollback_count}",
                flush=True,
            )
            return

        if self.averaging_mode == "ram" and self.trainer is not None:
            try:
                self._greedy_soup(model)
            except Exception as exc:
                print(f"[env][soup] failed: {exc}, using best single", flush=True)

        if self.best_state is None and not torch.distributed.is_initialized():
            print("[env][end] no snapshots captured", flush=True)
            return

        self._restore_best_to_model(model)

        if is_main_process(_RANK) and getattr(self, "_submission_dir", None):
            try:
                from dev_pass import _save_weights_only
                _save_weights_only(
                    _peel_wrapper(model),
                    self._submission_dir,
                    lambda msg: print(msg, flush=True),
                )
            except Exception as exc:
                print(f"[env][end] persist failed: {exc}", flush=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(
            f"[env][end] source={self.best_source} "
            f"loss={self.best_loss:.4f} rollbacks={self.rollback_count}",
            flush=True,
        )
