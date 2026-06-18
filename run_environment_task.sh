#!/bin/bash

# Default GPU configuration (use all GPUs if not specified)
GPUS="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpus)
      GPUS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -g, --gpus GPU_IDS    Specify which GPUs to use (default: 'all')"
      echo "                        Examples: 'all', '0', '0,1', '\"device=0,1\"'"
      echo "  -h, --help            Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                    # Use all available GPUs"
      echo "  $0 -g 0               # Use only GPU 0"
      echo "  $0 -g 0,1             # Use GPUs 0 and 1"
      echo "  $0 --gpus '\"device=0,2\"'  # Use GPUs 0 and 2 (quoted for Docker)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

TASK_ID="1"

# List of models to test (add or remove models as needed)
MODELS=(
  # "unsloth/Llama-3.2-3B-Instruct"
  # "Qwen/Qwen3-4B-Instruct-2507"
  # "mistralai/Mistral-7B-Instruct-v0.3"
  # "mistralai/Mistral-7B-Instruct-v0.2"
  # "Qwen/Qwen2.5-3B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2-7B-Instruct"
  # "codellama/CodeLlama-7b-Instruct-hf"
  # "NousResearch/Hermes-3-Llama-3.2-3B"
)

DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
DATASET_TYPE='{
  "environment_name": "liars_dice"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=3
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# For uploading the outputs
HUGGINGFACE_TOKEN=""
WANDB_TOKEN=""
HUGGINGFACE_USERNAME=""
EXPECTED_REPO_NAME=""
DOCKER_BUILDKIT=0.15

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
LOGS_DIR="$(pwd)/training_logs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 777 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 777 "$OUTPUTS_DIR"
mkdir -p "$LOGS_DIR"
chmod 777 "$LOGS_DIR"

# Create Docker network if it doesn't exist (needed for environment servers and trainer)
docker network create agent_eval_net 2>/dev/null || true

# Build the downloader image
DOWNLOADER_IMAGE="trainer-downloader:latest"
TRAINER_IMAGE="standalone-text-trainer:latest"

docker build -t "$DOWNLOADER_IMAGE" -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build -t "$TRAINER_IMAGE" -f dockerfiles/standalone-text-trainer.dockerfile .

# Start environment servers and get URLs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/.environment_server_urls.txt"
echo "Starting environment servers..."
sudo chmod +x "$SCRIPT_DIR/run_environment_env.sh"
"$SCRIPT_DIR/run_environment_env.sh"
# Read URLs from file
if [ -f "$URLS_FILE" ]; then
  ENVIRONMENT_SERVER_URLS=$(cat "$URLS_FILE")
  rm -f "$URLS_FILE"  # Clean up the temporary file
  echo "Environment server URLs: $ENVIRONMENT_SERVER_URLS"
else
  echo "Error: Failed to get environment server URLs" >&2
  exit 1
fi

# Loop through each model
for MODEL in "${MODELS[@]}"; do
  echo "=========================================="
  echo "Starting training for model: $MODEL"
  echo "=========================================="
  
  # Create a safe model name for directories (replace / with _)
  MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
  LOCAL_EXPECTED_REPO_NAME="${EXPECTED_REPO_NAME}_${MODEL_SAFE}"   # Unique repo name per model
  LOCAL_FOLDER="$OUTPUTS_DIR/$TASK_ID/${LOCAL_EXPECTED_REPO_NAME}"
  
  #Download model and dataset
  echo "Downloading model and dataset..."
  docker run --rm \
    --volume "$CHECKPOINTS_DIR:/cache:rw" \
    --name downloader-image \
    -e HF_TOKEN="$HUGGINGFACE_TOKEN" \
    "$DOWNLOADER_IMAGE" \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --file-format "$FILE_FORMAT" \
    --task-type "EnvTask"

  ###############################
  # Start the training container #
  ###############################
  TRAINING_CONTAINER_NAME="grpo-text-trainer-${MODEL_SAFE}"
  TRAINING_TIMEOUT_SECONDS=$((HOURS_TO_COMPLETE * 3600))

  # Run the training container in detached mode
  echo "Using GPUs: $GPUS"
  # Tournament-update env var pass-through:
  # - MINER_DATASETS_DIR / MINER_DATASETS: validator-mounted whitelisted SFT datasets
  #   (feature/miner-dataset-whitelist; max 2 datasets, read-only mount).
  # - BASELINE_STATS: precomputed baseline scores per env
  #   (feature/model-prep-container; allows skipping early exploration).
  # - AUGMENTED_MODEL: hint that the model repo was anonymized
  #   (feature/model-prep-container; scrubs _name_or_path).
  MINER_DATASETS_DIR="${MINER_DATASETS_DIR:-}"
  MINER_DATASETS="${MINER_DATASETS:-}"
  BASELINE_STATS="${BASELINE_STATS:-}"
  AUGMENTED_MODEL="${AUGMENTED_MODEL:-}"
  HF_TOKEN_PASS="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
  # Local-test convenience: if MINER_DATASETS_DIR points to a host path, mount it
  # read-only at the same path inside the container so the validator's contract
  # (datasets readable at MINER_DATASETS_DIR) is preserved during local testing.
  MINER_DATASETS_MOUNT_ARG=""
  if [ -n "$MINER_DATASETS_DIR" ] && [ -d "$MINER_DATASETS_DIR" ]; then
    MINER_DATASETS_MOUNT_ARG="--volume $MINER_DATASETS_DIR:$MINER_DATASETS_DIR:ro"
    echo "Mounting miner datasets: $MINER_DATASETS_DIR (read-only)"
  fi
  docker run -d --gpus "$GPUS" \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    --cpus=8 \
    --network agent_eval_net \
    --volume "$CHECKPOINTS_DIR:/cache:rw" \
    --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
    $MINER_DATASETS_MOUNT_ARG \
    --name $TRAINING_CONTAINER_NAME \
    --ipc=host \
    -e ENVIRONMENT_SERVER_URLS="$ENVIRONMENT_SERVER_URLS" \
    -e WANDB_TOKEN="$WANDB_TOKEN" \
    -e HF_TOKEN="$HF_TOKEN_PASS" \
    -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
    -e MINER_DATASETS_DIR="$MINER_DATASETS_DIR" \
    -e MINER_DATASETS="$MINER_DATASETS" \
    -e BASELINE_STATS="$BASELINE_STATS" \
    -e AUGMENTED_MODEL="$AUGMENTED_MODEL" \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$TRAINER_IMAGE" \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --task-type "EnvTask" \
    --file-format "$FILE_FORMAT" \
    --hours-to-complete "$HOURS_TO_COMPLETE" \
    --expected-repo-name "$LOCAL_EXPECTED_REPO_NAME" \
    --wandb-mode "online" \
    --max-steps 900

  TRAIN_CONTAINER_STATUS=0
  
  # Create log file with timestamp
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="$LOGS_DIR/training_${MODEL_SAFE}_${TIMESTAMP}.log"
  echo "Logging training output to: $LOG_FILE"
  echo "Waiting up to $HOURS_TO_COMPLETE hour(s) ($TRAINING_TIMEOUT_SECONDS seconds) for training container to finish..."

  # Wait for the container to finish or timeout, save logs to file and display
  timeout $TRAINING_TIMEOUT_SECONDS docker logs -f $TRAINING_CONTAINER_NAME 2>&1 | tee "$LOG_FILE" || true

  # Check if the container is still running (timeout hit)
  if [ "$(docker inspect -f '{{.State.Running}}' $TRAINING_CONTAINER_NAME 2>/dev/null)" == "true" ]; then
    echo "Time limit exceeded ($HOURS_TO_COMPLETE hour(s)). Stopping training container $TRAINING_CONTAINER_NAME..."
    docker stop $TRAINING_CONTAINER_NAME
    echo "Training container $TRAINING_CONTAINER_NAME stopped."
  else
    # Optionally fetch the exit code
    TRAIN_CONTAINER_STATUS=$(docker inspect -f '{{.State.ExitCode}}' $TRAINING_CONTAINER_NAME 2>/dev/null)
    echo "Training container $TRAINING_CONTAINER_NAME finished with exit code $TRAIN_CONTAINER_STATUS."
  fi

  # Remove the training container (cleanup)
  docker rm $TRAINING_CONTAINER_NAME 2>/dev/null || true

  # Upload the trained model to HuggingFace only when output exists and uploader script is available.
  if [ -d "$LOCAL_FOLDER" ]; then
    sudo chmod -R 777 "$LOCAL_FOLDER"

    UPLOAD_SCRIPT="./trainer/utils/hf_upload.py"
    if [ -f "$UPLOAD_SCRIPT" ]; then
      if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
      elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
      else
        echo "No python interpreter found for upload. Skipping upload."
        PYTHON_BIN=""
      fi

      if [ -n "$PYTHON_BIN" ]; then
        HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
        HUGGINGFACE_USERNAME=$HUGGINGFACE_USERNAME \
        WANDB_TOKEN=$WANDB_TOKEN \
        TASK_ID=$TASK_ID \
        EXPECTED_REPO_NAME=$LOCAL_EXPECTED_REPO_NAME \
        LOCAL_FOLDER=$LOCAL_FOLDER \
        MODEL=$MODEL \
        "$PYTHON_BIN" "$UPLOAD_SCRIPT"
      fi
    else
      echo "Upload script not found at $UPLOAD_SCRIPT. Skipping upload."
    fi
  else
    echo "Output folder not found: $LOCAL_FOLDER. Skipping upload."
  fi
  
  echo "=========================================="
  echo "Completed training for model: $MODEL"
  echo "Log saved to: $LOG_FILE"
  echo "=========================================="
  echo ""
done

echo "=========================================="
echo "All training completed!"
echo "Training logs are saved in: $LOGS_DIR"
echo "=========================================="

# Cleanup environment servers (done once after all models)
echo "Cleaning up environment servers..."
NUM_SERVERS=4
for i in $(seq 0 $((NUM_SERVERS-1))); do
  docker rm -f "agentgym-server-$i" 2>/dev/null || true
done
echo "Environment servers cleaned up."