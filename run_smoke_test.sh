#!/bin/bash
#
# 1-GPU smoke test orchestrator — mirrors run_environment_task.sh tetapi:
#   - 1 GPU (default: GPU 0)
#   - Smaller model (Qwen2.5-0.5B-Instruct, fits 16-24GB VRAM dengan LoRA)
#   - Per-game testing (gin_rummy / liars_dice / leduc_poker)
#   - max-steps low (quick verification, ~5-15 min per game)
#   - Verbose logging for [TOURNAMENT_ENV] context block
#
# USAGE:
#   bash run_smoke_test.sh <ENV_NAME> [--gpus GPU_ID] [--steps N] [--model MODEL_ID]
#
# Examples:
#   bash run_smoke_test.sh leduc_poker
#   bash run_smoke_test.sh gin_rummy --gpus 0 --steps 100
#   bash run_smoke_test.sh liars_dice --model Qwen/Qwen2.5-1.5B-Instruct
#
# Pre-requisites:
#   - HUGGINGFACE_TOKEN env var set (for model download + optional upload)
#   - WANDB_TOKEN env var set (optional, for telemetry)
#   - Docker + GPU drivers installed

set -e

# ---------- Defaults ----------
ENV_NAME="${1:-leduc_poker}"
shift || true
GPUS="0"
MAX_STEPS=100
MODEL="Qwen/Qwen2.5-0.5B-Instruct"   # 0.5B → bucket 0_1_b → 1 GPU + LoRA
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)   GPUS="$2"; shift 2 ;;
    --steps)  MAX_STEPS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      grep '^#' "$0" | head -30
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------- Validation ----------
case "$ENV_NAME" in
  gin_rummy|liars_dice|leduc_poker) ;;
  *) echo "ERROR: ENV_NAME must be one of: gin_rummy, liars_dice, leduc_poker" >&2; exit 1 ;;
esac

if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "WARNING: HUGGINGFACE_TOKEN not set — model download may fail for gated models."
fi

# ---------- Test identifiers ----------
TASK_ID="smoketest-${ENV_NAME}-$(date +%s)"
HF_USER="${HUGGINGFACE_USERNAME:-Zaydensth}"
EXPECTED_REPO_NAME="smoketest-${ENV_NAME}-$(date +%Y%m%d-%H%M)"

DATASET_TYPE="{\"environment_name\": \"${ENV_NAME}\"}"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1   # 1h smoke test (vs 3h tournament)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINTS_DIR="$(pwd)/smoke_checkpoints/${TASK_ID}"
OUTPUTS_DIR="$(pwd)/smoke_outputs/${TASK_ID}"
LOGS_DIR="$(pwd)/smoke_logs/${TASK_ID}"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"

# ---------- Banner ----------
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  env-jagger 1-GPU smoke test                               ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  ENV_NAME        : $ENV_NAME"
echo "║  MODEL           : $MODEL"
echo "║  GPUS            : $GPUS"
echo "║  MAX_STEPS       : $MAX_STEPS"
echo "║  TASK_ID         : $TASK_ID"
echo "║  HF_USER         : $HF_USER"
echo "║  EXPECTED_REPO   : $EXPECTED_REPO_NAME"
echo "║  HOURS           : $HOURS_TO_COMPLETE"
echo "╚════════════════════════════════════════════════════════════╝"

if [ "$DRY_RUN" = "true" ]; then
  echo "[DRY-RUN] Would now: docker build + start env servers + run trainer"
  exit 0
fi

# ---------- Docker setup ----------
docker network create agent_eval_net 2>/dev/null || true

DOWNLOADER_IMAGE="trainer-downloader:latest"
TRAINER_IMAGE="standalone-text-trainer:latest"

echo ""
echo "[STEP 1] Building Docker images (this may take 10-15 min on first run)..."
docker build -t "$DOWNLOADER_IMAGE" -f dockerfiles/trainer-downloader.dockerfile . | tail -10
docker build -t "$TRAINER_IMAGE" -f dockerfiles/standalone-text-trainer.dockerfile . | tail -10

# ---------- Start env servers ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/.environment_server_urls.txt"
echo ""
echo "[STEP 2] Starting environment servers..."
sudo chmod +x "$SCRIPT_DIR/run_environment_env.sh"
"$SCRIPT_DIR/run_environment_env.sh"
if [ -f "$URLS_FILE" ]; then
  ENVIRONMENT_SERVER_URLS=$(cat "$URLS_FILE")
  rm -f "$URLS_FILE"
  echo "  Environment URLs: $ENVIRONMENT_SERVER_URLS"
else
  echo "  ERROR: Failed to get env server URLs" >&2
  exit 1
fi

# ---------- Download model ----------
MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
LOCAL_EXPECTED_REPO_NAME="${EXPECTED_REPO_NAME}_${MODEL_SAFE}"

echo ""
echo "[STEP 3] Downloading model + dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image-${TASK_ID} \
  -e HF_TOKEN="$HUGGINGFACE_TOKEN" \
  "$DOWNLOADER_IMAGE" \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

# ---------- Run training ----------
TRAINING_CONTAINER_NAME="grpo-smoketest-${MODEL_SAFE}-${TASK_ID:0:8}"
TRAINING_TIMEOUT_SECONDS=$((HOURS_TO_COMPLETE * 3600))

echo ""
echo "[STEP 4] Starting training container ($TRAINING_CONTAINER_NAME)..."
echo "         Using GPU(s): $GPUS"
echo "         Max steps: $MAX_STEPS (smoke test mode)"

# Build GPU spec — single-ID uses `device=N`, multi-ID needs `"device=0,1,2,3"` literal.
# Without literal quotes, docker mis-parses multi-ID as Count+DeviceIDs conflict
# ("cannot set both Count and DeviceIDs on device request").
if [[ "$GPUS" == *","* ]]; then
  DOCKER_GPU_ARG=("--gpus" "\"device=$GPUS\"")
else
  DOCKER_GPU_ARG=("--gpus" "device=$GPUS")
fi

docker run -d "${DOCKER_GPU_ARG[@]}" \
  --cpus=8 \
  --network agent_eval_net \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name $TRAINING_CONTAINER_NAME \
  --ipc=host \
  -e ENVIRONMENT_SERVER_URLS="$ENVIRONMENT_SERVER_URLS" \
  -e WANDB_TOKEN="${WANDB_TOKEN:-}" \
  -e HF_TOKEN="$HUGGINGFACE_TOKEN" \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -e MINER_DATASETS_DIR="${MINER_DATASETS_DIR:-}" \
  -e MINER_DATASETS="${MINER_DATASETS:-}" \
  -e BASELINE_STATS_PATH="${BASELINE_STATS_PATH:-}" \
  -e BASELINE_STATS="${BASELINE_STATS:-}" \
  -e AUGMENTED_MODEL="${AUGMENTED_MODEL:-}" \
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
  --wandb-mode "${WANDB_MODE:-offline}" \
  --max-steps "$MAX_STEPS"

# ---------- Monitor ----------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/${ENV_NAME}_${MODEL_SAFE}_${TIMESTAMP}.log"
echo ""
echo "[STEP 5] Monitoring training (log: $LOG_FILE)..."
echo "         Timeout: ${HOURS_TO_COMPLETE}h (${TRAINING_TIMEOUT_SECONDS}s)"
echo "         Watch for: [TOURNAMENT_ENV] block, torchrun start, reward trajectory"
echo ""

# Capture log + watch live (also tee to console for first 200 lines)
timeout $TRAINING_TIMEOUT_SECONDS docker logs -f $TRAINING_CONTAINER_NAME 2>&1 | tee "$LOG_FILE" | grep -E '\[TOURNAMENT_ENV\]|torchrun|loss|reward|FAIL|ERROR|success.txt|Exception' &
PID=$!
wait $PID || true

# ---------- Container exit status ----------
if [ "$(docker inspect -f '{{.State.Running}}' $TRAINING_CONTAINER_NAME 2>/dev/null)" == "true" ]; then
  echo ""
  echo "[STEP 6] Timeout reached. Stopping container..."
  docker stop $TRAINING_CONTAINER_NAME
fi
EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' $TRAINING_CONTAINER_NAME 2>/dev/null || echo "unknown")
echo "         Container exit code: $EXIT_CODE"

# ---------- Cleanup ----------
docker logs $TRAINING_CONTAINER_NAME > "${LOG_FILE}.full" 2>&1 || true
docker rm $TRAINING_CONTAINER_NAME 2>/dev/null || true

# ---------- Stop env servers ----------
echo ""
echo "[STEP 7] Cleaning up env servers..."
docker ps --filter "label=affine_environment" -q | xargs -r docker stop 2>/dev/null || true
docker ps -a --filter "label=affine_environment" -q | xargs -r docker rm 2>/dev/null || true

# ---------- Result summary ----------
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Smoke Test Result Summary                                 ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Env             : $ENV_NAME"
echo "║  Model           : $MODEL"
echo "║  Task ID         : $TASK_ID"
echo "║  Exit code       : $EXIT_CODE"
echo "║  Log full        : ${LOG_FILE}.full"
echo "║  Checkpoint dir  : $CHECKPOINTS_DIR"
echo "║  Outputs dir     : $OUTPUTS_DIR"
echo "╚════════════════════════════════════════════════════════════╝"

# ---------- Quick health-check verifications ----------
echo ""
echo "[VERIFY] Quick health checks on the log:"
if grep -q "\[TOURNAMENT_ENV\]" "${LOG_FILE}.full"; then
  echo "  ✅ [TOURNAMENT_ENV] context block logged"
else
  echo "  ❌ [TOURNAMENT_ENV] block MISSING — check tournament_env_utils.py wiring"
fi
if grep -q "torchrun" "${LOG_FILE}.full"; then
  echo "  ✅ torchrun started"
else
  echo "  ❌ torchrun never started — check entrypoint.sh or grpo_env_config.py"
fi
if grep -qE "reward.*[-0-9]+\.[0-9]+" "${LOG_FILE}.full"; then
  echo "  ✅ Reward trajectory logged"
else
  echo "  ⚠️  No reward trajectory found — model may not have completed any rollout"
fi
if grep -q "success.txt" "${LOG_FILE}.full"; then
  echo "  ✅ success.txt written (training completed)"
else
  echo "  ⚠️  success.txt NOT written — training may have been cut off"
fi
if grep -qE "Traceback|RuntimeError|OutOfMemoryError" "${LOG_FILE}.full"; then
  echo "  ❌ Errors detected — inspect ${LOG_FILE}.full"
  echo "     First 5 error lines:"
  grep -nE "Traceback|RuntimeError|OutOfMemoryError" "${LOG_FILE}.full" | head -5 | sed 's/^/       /'
else
  echo "  ✅ No fatal errors detected"
fi

echo ""
echo "Done. Next:"
echo "  - To test next game: bash run_smoke_test.sh <next_env> --gpus $GPUS"
echo "  - Inspect full log: less ${LOG_FILE}.full"
echo "  - Check HF upload: ls $OUTPUTS_DIR"
