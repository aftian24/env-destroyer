"""
SFT trainer for EnvTask imitation learning.

Baseline: train_instruct.py (production machinery: callbacks, batch-size adjustment,
  success.txt, LoRA helpers).
Differences from train_instruct.py:
  1. Dataset: loaded via load_from_disk (HF DatasetDict) instead of MyDataset.
  2. Trainer: SFTTrainer with tokenize_and_mask (assistant-only loss) instead of Trainer.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
import transformers
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import is_main_process
from trl import SFTConfig, SFTTrainer

from customized_trainer import (
    CustomEvalSaveCallback,
    WhenToEvalHandler,
    resize_if_needed,
    set_generation_config,
)
from state_manager import get_state, set_state
from tournament_env_utils import log_tournament_environment
from utility import log_info
from checkpoint_avg_callback import AdaptiveTrainingCallback

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFTEnvTrainingArgs(SFTConfig):
    request_path: Optional[str] = field(default=None)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


# ---------------------------------------------------------------------------
# Model helpers (mirrored from train_instruct.py)
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, torch.nn.Linear)):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def load_lora_model(training_args: SFTEnvTrainingArgs, model_path: str,
                    lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )

    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split() if m.strip()]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=PeftTaskType.CAUSAL_LM,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    return model


def load_model(training_args: SFTEnvTrainingArgs, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        log_info("Using LIGER kernel")
        model_class = AutoLigerKernelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
    )
    return model


# ---------------------------------------------------------------------------
# Tokenisation + masking (from game-trajectories-sft/train.py)
# ---------------------------------------------------------------------------

def tokenize_and_mask(dataset: DatasetDict, tokenizer, max_length: int = 4096) -> DatasetDict:
    """Apply chat template and mask non-assistant tokens so loss is assistant-only.

    Safety hardening (2026-05-24 patch):
    - try/except per example: malformed messages don't crash the whole job
    - bounds clamp on p/r indices: tokenizer rounding can produce off-by-one
    - skip examples where no assistant token survives truncation (sum(mask)==0)
    - count + report skipped examples for telemetry
    """
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def _process(example):
        try:
            msgs = example["messages"]
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            mask = [0] * len(ids)
            for i, msg in enumerate(msgs):
                if msg["role"] != "assistant":
                    continue
                p = len(tokenizer.apply_chat_template(msgs[:i],   tokenize=True, add_generation_prompt=True))
                r = len(tokenizer.apply_chat_template(msgs[:i+1], tokenize=True, add_generation_prompt=False))
                # Bounds clamp — tokenizer rounding occasionally returns lengths
                # that exceed len(ids) by 1, causing IndexError on slice.
                p = max(0, min(p, len(ids)))
                r = max(0, min(r, len(ids)))
                for j in range(p, r):
                    mask[j] = 1
            # Truncate to max_length AFTER masking (preserves earlier assistant tokens)
            if len(ids) > max_length:
                ids, mask = ids[:max_length], mask[:max_length]
            # Skip if no assistant token survived (e.g., assistant tokens all
            # past max_length truncation point — no loss signal possible)
            if sum(mask) == 0:
                return {"input_ids": [eos_id], "assistant_masks": [0], "_skip": True}
            return {"input_ids": ids, "assistant_masks": mask, "_skip": False}
        except Exception as exc:
            # Don't fail entire dataset on one malformed example; report + skip.
            print(f"[tokenize] skip example due to {type(exc).__name__}: {exc}",
                  flush=True)
            return {"input_ids": [eos_id], "assistant_masks": [0], "_skip": True}

    mapped = dataset.map(_process, num_proc=4, desc="tokenize+mask")

    # Filter out skipped examples + report counts
    if "_skip" in mapped[list(mapped.keys())[0]].column_names:
        before = {s: len(mapped[s]) for s in mapped.keys()}
        mapped = mapped.filter(lambda x: not x["_skip"], num_proc=4)
        mapped = mapped.remove_columns(["_skip"])
        after = {s: len(mapped[s]) for s in mapped.keys()}
        for split in before:
            dropped = before[split] - after[split]
            if dropped > 0:
                print(f"[tokenize] {split}: dropped {dropped}/{before[split]} "
                      f"({100*dropped/max(before[split], 1):.2f}%) malformed/empty examples",
                      flush=True)
    return mapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argument_parser = transformers.HfArgumentParser((SFTEnvTrainingArgs, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    # Tournament: parse + log validator-injected context.
    # Logged once on main process to avoid stdout flood under DDP.
    # dataset_type may be dict OR JSON string depending on caller — handle both.
    _ds_type = train_info.get("dataset_type") or train_request.get("dataset_type") or {}
    if isinstance(_ds_type, str):
        try:
            _ds_type = json.loads(_ds_type)
        except (json.JSONDecodeError, TypeError):
            _ds_type = {}
    env_name_for_log = (
        (_ds_type or {}).get("environment_name", "")
        or train_request.get("environment_name", "")
    )
    if is_main_process(LOCAL_RANK):
        log_tournament_environment(env_name_for_log)

    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load pre-generated trajectories and apply assistant masking
    raw: DatasetDict = load_from_disk(train_request["dataset_path"])
    dataset = tokenize_and_mask(raw, tokenizer, max_length=training_args.max_length or 4096)

    train_ds = dataset["train"]
    dev_ds = dataset["validation"]

    # Batch size adjustment (mirrors train_instruct.py)
    original_steps = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )

    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = max(int(max_batch_size_theory), 1)

    if (training_args.per_device_train_batch_size > max_batch_size_theory
            and train_request.get("adjust_batch_size", True)):
        training_args.per_device_train_batch_size = max_batch_size_theory

    # Load model
    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    try:
        model.config.use_cache = False
    except Exception:
        pass

    set_generation_config(train_request["model_name"], model)

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    max_steps = train_request.get("max_steps", -1)
    training_args.save_only_model = True

    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)

    success_file = os.path.join(training_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)

    checking_step = train_request.get("checking_step", 70)
    if checking_step >= total_steps_per_epoch:
        checking_step = max(total_steps_per_epoch - 2, 1)

    # ── Greedy-soup checkpoint averaging (Wortsman 2022) ─────────────────────
    # Averages the N best checkpoints by validation loss instead of returning
    # just the final or best-single checkpoint, improving OOD generalization.
    _shard_ds = getattr(training_args, "deepspeed", None) is not None
    _sharded = _shard_ds or len(training_args.fsdp) > 0
    _trainable_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    _avg_mode = "disk" if (_sharded or 6 * _trainable_bytes > 100e9) else "ram"
    _cm = train_request.get("checking_mode")
    is_final_run = _cm in ("none", None)
    ckpt_avg = (
        AdaptiveTrainingCallback(window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir)
        if is_final_run else None
    )
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]
        ckpt_avg.end_time = train_request["end_time"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        callbacks=[
            *([] if ckpt_avg is None else [ckpt_avg]),
            CustomEvalSaveCallback(
                WhenToEvalHandler(
                    train_request["end_time"],
                    train_request["save_before_remaining_time"],
                    periodic_save_steps=periodic_save_steps,
                    steps_per_epoch=total_steps_per_epoch,
                    max_steps=max_steps,
                ),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
            )
        ],
    )

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer
    trainer.train()

    if is_main_process(LOCAL_RANK):
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Done", "finish")


if __name__ == "__main__":
    main()
