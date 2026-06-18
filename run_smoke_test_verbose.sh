#!/bin/bash
#
# VERBOSE 1-GPU smoke/full-train orchestrator — prints EVERY step output live.
# Unlike run_smoke_test.sh, this does NOT grep-filter console output, so you
# see all GRPO metrics, batch wins, episode logs, curriculum events live.
#
# Output stream:
#   - Console: EVERYTHING (no filter) — useful for screen attach / SSH tail
#   - File:    EVERYTHING (same as console) at $LOGS_DIR/<env>_<model>_<ts>.log
#
# USAGE:
#   bash run_smoke_test_verbose.sh <ENV_NAME> [--gpus GPU_IDS] [--steps N] [--model MODEL_ID]
#
# Examples:
#   bash run_smoke_test_verbose.sh leduc_poker --gpus 0 --steps 300
#   bash run_smoke_test_verbose.sh gin_rummy --steps 100
#   bash run_smoke_test_verbose.sh liars_dice --model Qwen/Qwen2.5-1.5B-Instruct
#
# Pre-requisites:
#   - HUGGINGFACE_TOKEN env var set (for model download)
#   - WANDB_TOKEN env var set (optional)
#   - Docker + GPU drivers installed
#
# Tip: Run in tmux/screen for detachable session:
#   screen -dmS train bash run_smoke_test_verbose.sh leduc_poker --steps 300
#   screen -r train     # attach
#   Ctrl+A then D       # detach

set -e

# ---------- Defaults ----------
ENV_NAME="${1:-leduc_poker}"
shift || true
GPUS="0"
MAX_STEPS=100
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)   GPUS="$2"; shift 2 ;;
    --steps)  MAX_STEPS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | head -30; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$ENV_NAME" in
  gin_rummy|liars_dice|leduc_poker) ;;
  *) echo "ERROR: ENV_NAME must be gin_rummy / liars_dice / leduc_poker" >&2; exit 1 ;;
esac

# ---------- Test identifiers ----------
TASK_ID="verbose-${ENV_NAME}-$(date +%s)"
HF_USER="${HUGGINGFACE_USERNAME:-Zaydensth}"
EXPECTED_REPO_NAME="verbose-${ENV_NAME}-$(date +%Y%m%d-%H%M)"
DATASET_TYPE="{\"environment_name\": \"${ENV_NAME}\"}"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-3}"   # default 3h, override via env var
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINTS_DIR="$(pwd)/verbose_checkpoints/${TASK_ID}"
OUTPUTS_DIR="$(pwd)/verbose_outputs/${TASK_ID}"
LOGS_DIR="$(pwd)/verbose_logs/${TASK_ID}"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"

# ---------- Banner ----------
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  VERBOSE TRAIN — prints every step output                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  ENV_NAME        : $ENV_NAME"
echo "║  MODEL           : $MODEL"
echo "║  GPUS            : $GPUS"
echo "║  MAX_STEPS       : $MAX_STEPS"
echo "║  HOURS           : $HOURS_TO_COMPLETE"
echo "║  TASK_ID         : $TASK_ID"
echo "║  HF_USER         : $HF_USER"
echo "║  EXPECTED_REPO   : $EXPECTED_REPO_NAME"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "All output (including every GRPO step + episode log) streams live below."
echo "Log file: $LOGS_DIR/${ENV_NAME}_$(echo "$MODEL" | sed 's/\//_/g')_$(date +%Y%m%d_%H%M%S).log"
echo ""

# ---------- Docker setup ----------
docker network create agent_eval_net 2>/dev/null || true

DOWNLOADER_IMAGE="trainer-downloader:latest"
TRAINER_IMAGE="standalone-text-trainer:latest"

echo "[STEP 1/5] Building Docker images (cached if exists)..."
docker build -t "$DOWNLOADER_IMAGE" -f dockerfiles/trainer-downloader.dockerfile . 2>&1
echo ""
docker build -t "$TRAINER_IMAGE" -f dockerfiles/standalone-text-trainer.dockerfile . 2>&1
echo ""

# ---------- Start env servers ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/.environment_server_urls.txt"
echo "[STEP 2/5] Starting environment servers..."
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
echo ""

# ---------- Download model + dataset ----------
MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
LOCAL_EXPECTED_REPO_NAME="${EXPECTED_REPO_NAME}_${MODEL_SAFE}"

echo "[STEP 3/5] Downloading model + dataset..."
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
echo ""

# ---------- Run training ----------
TRAINING_CONTAINER_NAME="grpo-verbose-${MODEL_SAFE}-${TASK_ID:0:10}"
TRAINING_TIMEOUT_SECONDS=$((HOURS_TO_COMPLETE * 3600))

echo "[STEP 4/5] Starting training container: $TRAINING_CONTAINER_NAME"
echo "         Using GPU(s): $GPUS, Max steps: $MAX_STEPS"
echo ""

docker run -d --gpus "device=$GPUS" \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --cpus=16 \
  --network agent_eval_net \
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
  -e PYTHONUNBUFFERED=1 \
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

# ---------- Live stream ALL output ----------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/${ENV_NAME}_${MODEL_SAFE}_${TIMESTAMP}.log"
echo "[STEP 5/5] Streaming live output (NO grep filter — see every step)..."
echo "           Log file: $LOG_FILE"
echo "           Timeout: ${HOURS_TO_COMPLETE}h (${TRAINING_TIMEOUT_SECONDS}s)"
echo ""
echo "════════════════════════════════════════════════════════════"
echo " LIVE TRAINER OUTPUT BELOW (everything streams unfiltered)"
echo "════════════════════════════════════════════════════════════"

# CRITICAL: -t flag forces line-buffering so output streams in real-time
# stdbuf -oL ensures even pipe buffering doesn't delay output
stdbuf -oL -eL timeout $TRAINING_TIMEOUT_SECONDS docker logs -f -t $TRAINING_CONTAINER_NAME 2>&1 | stdbuf -oL tee "$LOG_FILE" || true

# ---------- Cleanup ----------
echo ""
echo "════════════════════════════════════════════════════════════"
echo " Training output ended. Cleaning up..."
echo "════════════════════════════════════════════════════════════"

if [ "$(docker inspect -f '{{.State.Running}}' $TRAINING_CONTAINER_NAME 2>/dev/null)" == "true" ]; then
  echo "[STEP 6] Timeout reached. Stopping container..."
  docker stop $TRAINING_CONTAINER_NAME
fi
EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' $TRAINING_CONTAINER_NAME 2>/dev/null || echo "unknown")
echo "         Container exit code: $EXIT_CODE"

# Save backup full log
docker logs $TRAINING_CONTAINER_NAME > "${LOG_FILE}.full" 2>&1 || true
docker rm $TRAINING_CONTAINER_NAME 2>/dev/null || true

# Stop env servers
echo ""
echo "[STEP 7] Cleaning up env servers..."
docker stop $(docker ps --filter "name=agentgym-server" -q) 2>/dev/null || true
docker rm $(docker ps -a --filter "name=agentgym-server" -q) 2>/dev/null || true

# ---------- Result summary ----------
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  VERBOSE TRAIN — Summary                                   ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Env             : $ENV_NAME"
echo "║  Model           : $MODEL"
echo "║  Task ID         : $TASK_ID"
echo "║  Exit code       : $EXIT_CODE"
echo "║  Log (live)      : $LOG_FILE"
echo "║  Log (backup)    : ${LOG_FILE}.full"
echo "║  Checkpoint dir  : $CHECKPOINTS_DIR"
echo "║  Outputs dir     : $OUTPUTS_DIR"
echo "╚════════════════════════════════════════════════════════════╝"

# ---------- Stats summary ----------
if [ -f "${LOG_FILE}.full" ]; then
  echo ""
  echo "[STATS] Training statistics:"
  STEP_COUNT=$(grep -c "rewards/rollout_reward_func" "${LOG_FILE}.full" 2>/dev/null || echo 0)
  BATCH_COUNT=$(grep -c "\[BATCH\]" "${LOG_FILE}.full" 2>/dev/null || echo 0)
  ERROR_COUNT=$(grep -cE "Traceback|RuntimeError|OutOfMemoryError" "${LOG_FILE}.full" 2>/dev/null || echo 0)
  echo "  GRPO steps completed: $STEP_COUNT / $MAX_STEPS"
  echo "  Batches finished: $BATCH_COUNT"
  echo "  Fatal errors: $ERROR_COUNT"

  if [ "$BATCH_COUNT" -gt 0 ]; then
    # Calc avg win rate over last 10 batches
    python3 -c "
import re, sys
with open('${LOG_FILE}.full') as f:
    lines = f.readlines()
data = []
for line in lines:
    m = re.search(r'Wins: (\d+)/(\d+).*AvgReturn:\s*([-0-9.]+)', line)
    if m:
        data.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
if data:
    last10 = data[-10:]
    w = sum(d[0] for d in last10)
    g = sum(d[1] for d in last10)
    r = sum(d[2] for d in last10) / len(last10)
    overall = 100 * sum(d[0] for d in data) / sum(d[1] for d in data)
    print(f'  Win rate (overall): {overall:.1f}%')
    print(f'  Win rate (last 10 batches): {100*w/g:.1f}%')
    print(f'  Avg return (last 10 batches): {r:.2f}')
" 2>/dev/null
  fi
fi

echo ""
echo "Done."
