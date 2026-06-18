from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    disable_action_mask,
    get_use_vllm,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_grpo_lr
from envs.env_configs import get_env_config

allow_find_lk_lr = False

# Per-size orchestrator decisions: lr, launch mode, GPU count, adapter/quant flags.
# Training dynamics live in DEFAULT_HYPERPARAMS in train_grpo_env.py.
SIZE_CONFIG: dict[str, dict] = {
    "0_1_b":  {"lr": 3e-5, "distributed": "ddp", "gpu_count": 1, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "1_2_b":  {"lr": 1e-5, "distributed": "ddp", "gpu_count": 1, "use_lora": False, "use_vllm": True,  "use_4bit": False},
    "2_4_b":  {"lr": 1e-5, "distributed": "ddp", "gpu_count": 2, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "4_5_b":  {"lr": 8e-6, "distributed": "ddp", "gpu_count": 2, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "5_6_b":  {"lr": 8e-6, "distributed": "ddp", "gpu_count": 2, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "6_9_b":  {"lr": 8e-6, "distributed": "ddp", "gpu_count": 4, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "9_12_b": {"lr": 6e-6, "distributed": "ddp", "gpu_count": 4, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "12_15_b":{"lr": 5e-6, "distributed": "ddp", "gpu_count": 4, "use_lora": True,  "use_vllm": True,  "use_4bit": False},
    "15_20_b":{"lr": 5e-6, "distributed": "ddp", "gpu_count": 4, "use_lora": True,  "use_vllm": False, "use_4bit": False},
    "20_40_b":{"lr": 4e-6, "distributed": "ddp", "gpu_count": 8, "use_lora": True,  "use_vllm": False, "use_4bit": True},
    "40_80_b":{"lr": 3e-6, "distributed": "ddp", "gpu_count": 8, "use_lora": True,  "use_vllm": False, "use_4bit": True},
}


def get_size_label(param_nums: int) -> str:
    if param_nums < 1_000_000_000:
        return "0_1_b"
    elif param_nums < 2_000_000_000:
        return "1_2_b"
    elif param_nums < 4_000_000_000:
        return "2_4_b"
    elif param_nums < 5_000_000_000:
        return "4_5_b"
    elif param_nums < 6_000_000_000:
        return "5_6_b"
    elif param_nums < 9_000_000_000:
        return "6_9_b"
    elif param_nums < 12_000_000_000:
        return "9_12_b"
    elif param_nums < 15_000_000_000:
        return "12_15_b"
    elif param_nums < 20_000_000_000:
        return "15_20_b"
    elif param_nums < 40_000_000_000:
        return "20_40_b"
    elif param_nums < 80_000_000_000:
        return "40_80_b"
    else:
        print(f"Model size {param_nums} is not supported, falling back to 40_80_b")
        return "40_80_b"


def get_grpo_config(param_nums: int) -> dict:
    """Backward-compat wrapper — returns SIZE_CONFIG for the resolved size."""
    return SIZE_CONFIG[get_size_label(param_nums)]


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "disable_fa",
        "disable_action_mask",
        "environment_name",
        "size_label",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    start_cmd = "python"
    run_type = config["distributed"]
    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    if run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_grpo_env.py \
    --request_path {request_path} \
    --environment_name {environment_name} \
    --bf16 True \
    --report_to wandb \
    --output_dir /workspace/data/trained_model \
    --num_train_epochs {epoch_num} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 1 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --vllm_mode colocate \
    --disable_fa {disable_fa} \
    --disable_action_mask {disable_action_mask} \
    --loss_type dr_grpo \
    --num_iterations 2 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --do_eval False \
    --size_label {size_label}"""
    )
    # DAPO Clip-Higher (epsilon_low=0.2, epsilon_high=0.28): asymmetric PPO clip
    # allows larger positive policy updates → more exploration of high-advantage actions.
    # Per goy-2 branch of 56susnet/jembut. Default symmetric clip (0.2, 0.2) tested as
    # standard; this asymmetric variant has empirical evidence in DAPO paper (ByteDance
    # arXiv 2503.14476) for improved RLHF convergence.

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear"
        )

    if config.get("use_vllm", True):
        template += " --use_vllm True"
    else:
        template += " --use_vllm False"

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("tensor_parallel", False):
        template = template + f" --vllm_tensor_parallel_size {gpu_nums}"

    if config.get("use_4bit", False):
        template = (
            template
            + " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
        )

    print(f"template: {template}", flush=True)
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    size_label = get_size_label(param_nums)
    if model_name in ["mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.2"]:
        size_label = "6_9_b"
    gpu_cfg = SIZE_CONFIG[size_label]
    print(f"size_label: {size_label}, gpu_cfg: {gpu_cfg}")

    run_config = {
        # Bumped 2 -> 4 per Loki trace of winning leduc miner 5GKSa6y1 (tournament
        # tourn_8d983ccab32305f1_20260511, task 0d4ddaa8). Winner trained 3975
        # steps with epoch_num=4; competitors at epoch=2 lost (under-training).
        "epoch_num": 4,
        "learning_rate": gpu_cfg["lr"],
        "min_lr_rate": 0.25,
        "use_liger": False,
        "optimizer": "paged_adamw_8bit",
        "use_lora": gpu_cfg.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "disable_action_mask": disable_action_mask(model_name),
        "gpu_nums": gpu_cfg["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": gpu_cfg.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "use_vllm": get_use_vllm(model_architecture, model_name),
        "tensor_parallel": gpu_cfg.get("tensor_parallel", False),
        "use_4bit": gpu_cfg.get("use_4bit", False),
        "environment_name": train_info.get("dataset_type", {}).get("environment_name"),
        "size_label": size_label,
    }

    if model_name == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5":
        run_config["use_lora"] = True

    if not gpu_cfg.get("use_vllm", True):
        run_config["use_vllm"] = False

    run_config["learning_rate"] *= train_info["reg_ratio"]

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 10
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 75

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    return {"train_request": train_request, "run_cmd": run_cmd}
