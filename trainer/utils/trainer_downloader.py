import argparse
import asyncio
import os
import shutil
import tempfile
import json
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

import trainer.utils.training_paths as train_paths
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from trainer import constants as cst


hf_api = HfApi()


async def download_text_dataset(task_id, dataset_url, file_format, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    if file_format == FileFormat.S3.value:
        input_data_path = train_paths.get_text_dataset_path(task_id)

        if not os.path.exists(input_data_path):
            local_path = await download_s3_file(dataset_url)
            shutil.copy(local_path, input_data_path)

    elif file_format == FileFormat.HF.value:
        repo_name = dataset_url.replace("/", "--")
        input_data_path = os.path.join(dataset_dir, repo_name)

        if not os.path.exists(input_data_path):
            snapshot_download(repo_id=dataset_url, repo_type="dataset", local_dir=input_data_path, local_dir_use_symlinks=False)

    return input_data_path, file_format


def write_environment_task_proxy_dataset(
    out_path: str,
    dataset_size: int = 1000,
    prompt_text: str = "Interact with this environment.",
    prompt_field: str = "prompt",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [{prompt_field: prompt_text} for _ in range(dataset_size)]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {out_path} with field '{prompt_field}'")


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        local_dir = os.path.expanduser(local_dir)
        local_filename = f"{repo_id.replace('/', '_')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(final_path):
            print(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            print(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        raise e


async def download_axolotl_base_model(repo_id: str, save_dir: str) -> str:
    model_dir = os.path.join(save_dir, repo_id.replace("/", "--"))
    if os.path.exists(model_dir):
        print(f"Model {repo_id} already exists at {model_dir}. Skipping download.")
        return model_dir
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=model_dir, local_dir_use_symlinks=False)
    return model_dir


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-type",
        required=True,
        choices=[TaskType.INSTRUCTTEXTTASK.value, TaskType.DPOTASK.value, TaskType.GRPOTASK.value, TaskType.CHATTASK.value, TaskType.ENVIRONMENTTASK.value],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file-format")
    
    # Optional args that might technically be passed but ignored for text tasks
    # (Leaving out model-type completely as it causes issues if enum values are missing)
    
    args, unknown = parser.parse_known_args()

    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    adapters_dir = cst.HUGGINGFACE_CACHE_PATH
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)

    print(f"Downloading datasets to: {dataset_dir}", flush=True)
    print(f"Downloading models to: {model_dir}", flush=True)

    if args.task_type == TaskType.ENVIRONMENTTASK.value:
        model_path = await download_axolotl_base_model(args.model, model_dir)
        input_data_path = train_paths.get_text_dataset_path(args.task_id)
        write_environment_task_proxy_dataset(
            out_path=input_data_path,
            dataset_size=1000,
            prompt_text="Interact with this environment.",
            prompt_field="prompt",
        )
    else:
        dataset_path, _ = await download_text_dataset(args.task_id, args.dataset, args.file_format, dataset_dir)
        model_path = await download_axolotl_base_model(args.model, model_dir)

    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
