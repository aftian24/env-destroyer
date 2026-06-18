#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from state_manager import get_state, set_state

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import pathlib

import lr_utils
import train_cst
import training_paths as train_paths
from core.models.utility_models import TaskType
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
from instruct_config import get_training_json as get_instruct_training_json
from grpo_env_config import get_training_json as get_env_training_json
from sft_env_config import (
    get_training_json as get_sft_env_training_json,
    get_training_json_multi_env as get_sft_env_training_json_multi_env,
)
from envs import supports_sft
from tournament_env_utils import (
    get_miner_datasets,
    log_tournament_environment,
    parse_baseline_stats,
)
from transformers import AutoConfig


def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    # print(f"Running command: {cmd}", flush=True)
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
    wandb_mode: str = "offline",
    wandb_project: str = None,
    wandb_entity: str = None,
):
    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        # print(f"New train command: {train_cmd}", flush=True)
                    else:
                        print(f"batch size is 1, cannot reduce further", flush=True)
                        if task_type == TaskType.GRPOTASK.value:
                            # disable vllm
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            # print(f"disable VLLM {train_cmd}", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": wandb_mode,
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }
        
        # Add API key - wandb expects WANDB_API_KEY
        # Check both WANDB_API_KEY and WANDB_TOKEN (for backwards compatibility)
        wandb_token = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
        if wandb_token:
            training_env_vars["WANDB_API_KEY"] = wandb_token
        
        # Add project and entity for online mode
        if wandb_mode == "online":
            if wandb_project:
                training_env_vars["WANDB_PROJECT"] = wandb_project
            if wandb_entity:
                training_env_vars["WANDB_ENTITY"] = wandb_entity

        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        # check if training is successfully here so we can break the loop; if output_dir contains file: "successs.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if os.path.exists(os.path.join(output_dir, "success.txt")):
            return True
        time.sleep(5)
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    lowest_loss = min([run["current_loss"] for run in train_runs])
    for run in train_runs:
        if run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.18,
        TaskType.DPOTASK.value: 0.18,
        TaskType.GRPOTASK.value: 0.2,
        TaskType.CHATTASK.value: 0.18,
    }
    return log_scale_map[task_type]


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask", "EnvTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.0
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["offline", "online", "disabled"],
        help="Wandb mode: offline (default), online, or disabled",
        default="offline",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Wandb project name (required for online mode)",
        default=None,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Wandb entity/team name (optional for online mode)",
        default=None,
    )

    args = parser.parse_args()
    
    # Validate wandb configuration for online mode
    if args.wandb_mode == "online" and not args.wandb_project:
        args.wandb_project = "Gradients-On-Demand"
        print(
            f"Info: --wandb-mode is set to 'online' but --wandb-project not provided. "
            f"Using default project: '{args.wandb_project}'",
            flush=True,
        )
    
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    # BASELINE_STATS: validator passes per-env baseline (starting loss/reward stats)
    # as a JSON-encoded env var. Schema: `EnvBaselineStats` from
    # `core/models/model_prep_models.py` —
    #   {"task_type": "env",
    #    "weights": {<layer_group>: {"frobenius": ..., "rms": ..., "max_abs": ...}},
    #    "env_stats": {<env_name>: {"num_episodes", "mean_score", "std_score",
    #                                "min_score", "max_score", "median_score"}}}
    #
    # We currently just LOG this — useful telemetry for:
    # (a) Confirming validator's MODEL_PREP_ENABLED_BY_TASK_TYPE includes env
    # (b) Detecting which envs the validator pre-evaluated (= envs in this task)
    # (c) Future: adaptive reward shaping or early-stopping based on baseline
    baseline_stats_raw = os.environ.get("BASELINE_STATS", "")
    if baseline_stats_raw:
        try:
            baseline_stats = json.loads(baseline_stats_raw)
            env_stats = baseline_stats.get("env_stats", {})
            print(f"[text_trainer] BASELINE_STATS received: task_type="
                  f"{baseline_stats.get('task_type', '?')}, "
                  f"envs={list(env_stats.keys())}", flush=True)
            for env, stats in env_stats.items():
                print(f"  {env}: mean={stats.get('mean_score', '?')} "
                      f"std={stats.get('std_score', '?')} "
                      f"min/max=[{stats.get('min_score', '?')}, "
                      f"{stats.get('max_score', '?')}] "
                      f"episodes={stats.get('num_episodes', '?')}", flush=True)
        except Exception as exc:
            print(f"[text_trainer] BASELINE_STATS parse failed (non-fatal): {exc}",
                  flush=True)
            baseline_stats = None
    else:
        baseline_stats = None
        print(f"[text_trainer] No BASELINE_STATS env var present "
              f"(validator's MODEL_PREP_ENABLED may be off, or organic task).",
              flush=True)

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "baseline_stats": baseline_stats,  # passed through for downstream consumers
        "checking_mode": "first_time",
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.ENVIRONMENTTASK.value:
        # PvP tournament 2026-05-25+: validator sends `environment_names: list[str]`
        # for multi-env tasks (R1=2 envs, R2=4 envs, R3=6 envs). ALL envs are
        # PvP-evaluated head-to-head → SFT data MUST cover all listed envs.
        # Discord 2026-05-23: "you can't optimise for a single env training since
        # you'll be pipped on all the others."
        #
        # Routing:
        #   - environment_names list with len >= 2 → multi-env path (scaled per-env
        #     trajectory gen + merge + single SFT)
        #   - environment_names list with len == 1 → single-env (treat first as legacy)
        #   - environment_name str (legacy) → single-env
        env_names = dataset_type_dict.get("environment_names")
        is_multi_env = (
            env_names
            and isinstance(env_names, list)
            and len(env_names) >= 2
        )

        if is_multi_env:
            # Drop envs this trainer has no SFT generator for (e.g. intercode) instead of
            # failing the whole task. We still train on the supported subset so the model
            # competes where it can rather than producing no submission at all (which
            # otherwise scores NULL on every env, including the ones it could have won).
            supported_envs = [e for e in env_names if supports_sft(e)]
            dropped = [e for e in env_names if not supports_sft(e)]
            if dropped:
                print(f"[text_trainer] Dropping envs with no SFT generator: {dropped}. "
                      f"Training on supported subset: {supported_envs}.", flush=True)
            if not supported_envs:
                raise ValueError(
                    f"Multi-env task contains no SFT-supported envs: {env_names}. "
                    f"Add a trajectory generator + register in envs/sft_env_configs.py."
                )
            # Reflect the filtered set so the single-env builder (which reads
            # dataset_type.environment_names) picks up a supported env, not a dropped one.
            dataset_type_dict["environment_names"] = supported_envs
            if len(supported_envs) >= 2:
                print(f"[text_trainer] Multi-env task detected: {supported_envs} "
                      f"(n={len(supported_envs)}). Running scaled per-env trajectory gen + "
                      f"merge + single SFT.", flush=True)
                log_tournament_environment(",".join(supported_envs))
                train_info = get_sft_env_training_json_multi_env(train_info, supported_envs)
                tokenize_cmd = train_info["generate_cmd"]
            else:
                env_name = supported_envs[0]
                print(f"[text_trainer] Single supported env after dropping unsupported: "
                      f"{env_name}.", flush=True)
                log_tournament_environment(env_name)
                train_info = get_sft_env_training_json(train_info)
                tokenize_cmd = train_info["generate_cmd"]
        else:
            # Single-env path (legacy + R0/organic)
            if env_names and len(env_names) == 1:
                env_name = env_names[0]
                print(f"[text_trainer] Single-env from environment_names list: "
                      f"{env_name}", flush=True)
            else:
                env_name = dataset_type_dict.get("environment_name", "")
            log_tournament_environment(env_name)
            if supports_sft(env_name):
                train_info = get_sft_env_training_json(train_info)
                tokenize_cmd = train_info["generate_cmd"]
            else:
                train_info = get_env_training_json(train_info)
                tokenize_cmd = ""
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    if tokenize_cmd:
        run_cmd_with_log(
            tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
        )

    original_train_cmd = train_cmd
    train_success = False
    state = get_state()
    state = {}
    set_state(state) # reset first
    state["mode"] = "initial"
    # at first the state is always running the train_cmd

    set_state(state)
    # TODO Run something magic here
    count = 0
    while True:
        state = get_state()
        train_cmd = original_train_cmd  # will replace based on the state later
        c_train_info = copy.deepcopy(train_info)
        final_output_dir = None
        if args.task_type == TaskType.GRPOTASK.value or args.task_type == TaskType.ENVIRONMENTTASK.value:
            state["mode"] = "finish" # do not run this for GRPO task
            c_train_info["train_request"]["checking_mode"] = "none"
        else:
            if state["mode"] == "initial":
                c_train_info["train_request"]["checking_mode"] = "first_time"
                
            elif state["mode"] == "continue":
                c_train_info["train_request"]["checking_mode"] = "second_time"
                n_runs = state["next_runs"]
                if "lrs" not in state: # first time of continue
                    current_lr = float(state["train"]["lr"])
                    state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
                    assert len(state["lrs"]) == n_runs, f"Number of learning rates {state['lrs']} should be equal to number of runs {n_runs}"
                    state["runs"] = []
                
                set_state(state)
                state["runs"].append(state["train"].copy())
                delete_poor_checkpoints(state["runs"])
                if len(state["runs"]) < n_runs:
                    index = len(state["runs"])
                    current_lr = state["lrs"][index]
                    train_cmd = replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                else: # the final run
                    # first find from runs the best loss
                    c_train_info["train_request"]["checking_mode"] = "none"
                    index = np.argmin([run["current_loss"] for run in state["runs"]])
                    print(f"BL;{index};{state['runs'][index]['current_loss']}; {state['lrs'][index]}", flush=True)
                    train_cmd = state["runs"][index]["train_cmd"]  #replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                    final_output_dir = state["runs"][index]["output_dir"]
                    state["mode"] = "finish"
            else: # the state = finish; no need to run more
                assert state["mode"] == "finish"
                break
        
        set_state(state)
        if train_cmd:
            run_output_dir = output_dir + f"_{count}" if not final_output_dir else final_output_dir
            train_cmd = replace_args_in_cmd(train_cmd, "output_dir", run_output_dir)
            
            current_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_{count}.json")
            with open(current_request_path, "w") as f:
                json.dump(c_train_info, f, indent=4, ensure_ascii=False)
            
            train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)
            
            state["train"] = {
                "train_cmd": train_cmd,
                "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
                "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
                "output_dir": run_output_dir
            }
            state["train"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            set_state(state)
            
            log_path = state["train"]["log_path"]
            # print(f"Run training with train_info: {c_train_info}", flush=True)
            success = run_training(
                train_cmd,
                log_path,
                args.task_id,
                args.retries,
                args.task_type,
                args.expected_repo_name,
                args.wandb_mode,
                args.wandb_project,
                args.wandb_entity,
            )
            time.sleep(5)
            if not success:
                print(f"Training failed for task {args.task_id} at count={count}", flush=True)
                break 
            else:
                print(f"Training successfully done for task {args.task_id} at count={count}", flush=True)
                break
        
        count += 1

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
