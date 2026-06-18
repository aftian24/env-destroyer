#!/bin/bash
#
# AUGMENTATION TEST — verify our code handles validator-augmented models.
# Combines best of:
#   - Friend's approach (Jordansky/augmented-* pre-augmented test models +
#     fake BASELINE_STATS_PATH injection)
#   - Our setup (diagonalge/mcts-api official validator image, verbose stream,
#     win rate aggregation)
#
# WHY THIS TEST EXISTS:
#   Env tournament currently doesn't augment models (MODEL_PREP_ENABLED_ENV=False),
#   but our code includes defensive handling for augmented scenarios. This test
#   verifies that defensive code actually works against real augmented models
#   produced by 56susnet/G.O.D feature/model-prep-container pipeline.
#
# USAGE:
#   bash run_augmentation_test.sh <GAME> [--model AUGMENTED_REPO] [--steps N]
#
# Examples (test with pre-augmented Qwen3-4B):
#   bash run_augmentation_test.sh leduc_poker
#   bash run_augmentation_test.sh gin_rummy --steps 100
#
# Defaults to Jordansky/augmented-f560e4e6ee71e78d (Qwen3-4B + weight_scaling 1.07)

set -e

GAME="${1:-leduc_poker}"
shift || true
GPUS="0"
MAX_STEPS=50
# Default: Qwen3-4B variant (~7.7GB, fits 1-GPU H100 80GB easily)
AUGMENTED_MODEL="Jordansky/augmented-f560e4e6ee71e78d"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)   GPUS="$2"; shift 2 ;;
    --steps)  MAX_STEPS="$2"; shift 2 ;;
    --model)  AUGMENTED_MODEL="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | head -25; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

case "$GAME" in
  gin_rummy|liars_dice|leduc_poker) ;;
  *) echo "ERROR: GAME must be gin_rummy / liars_dice / leduc_poker"; exit 1 ;;
esac

# ── Test identifiers ──────────────────────────────────────────────────────
TASK_ID="aug-${GAME}-$(date +%s)"
HF_USER="${HUGGINGFACE_USERNAME:-Zaydensth}"
EXPECTED_REPO_NAME="aug-test-${GAME}-$(date +%Y%m%d-%H%M)"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
DATASET_TYPE="{\"environment_name\":\"$GAME\"}"
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1

CHECKPOINTS_DIR="$(pwd)/aug_checkpoints/${TASK_ID}"
OUTPUTS_DIR="$(pwd)/aug_outputs/${TASK_ID}"
LOGS_DIR="$(pwd)/aug_logs/${TASK_ID}"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$LOGS_DIR"

# ── Inject FAKE BASELINE_STATS (friend's pattern — triggers our parser) ──
# This file is mounted into trainer container at /cache/baseline_stats_test.json
# Validator normally writes this; we fake it for testing.
BASELINE_STATS_FILE="$CHECKPOINTS_DIR/baseline_stats_test.json"
cat > "$BASELINE_STATS_FILE" << EOF
{
  "task_type": "env",
  "weights": {
    "by_group": {
      "attn.q_proj": {"weight_rms": 0.012, "weight_norm": 3.1, "max_abs": 0.5},
      "mlp.gate_proj": {"weight_rms": 0.015, "weight_norm": 4.2, "max_abs": 0.6}
    }
  },
  "env_stats": {
    "gin_rummy": {"num_episodes": 50, "mean_score": 0.4, "std_score": 0.10, "min_score": 0.0, "max_score": 0.9, "median_score": 0.4},
    "liars_dice": {"num_episodes": 50, "mean_score": 0.5, "std_score": 0.15, "min_score": 0.0, "max_score": 0.95, "median_score": 0.5},
    "leduc_poker": {"num_episodes": 50, "mean_score": 0.45, "std_score": 0.12, "min_score": 0.0, "max_score": 0.88, "median_score": 0.44}
  }
}
EOF

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  AUGMENTATION COMPATIBILITY TEST                           ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Game:             $GAME"
echo "║  Augmented model:  $AUGMENTED_MODEL"
echo "║  GPUs:             $GPUS"
echo "║  Max steps:        $MAX_STEPS"
echo "║  BASELINE_STATS:   $BASELINE_STATS_FILE (FAKE for testing)"
echo "║  MCTS image:       diagonalge/mcts-api:latest (validator official)"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ── Docker setup ──────────────────────────────────────────────────────────
docker network create agent_eval_net 2>/dev/null || true
docker build -t trainer-downloader:latest -f dockerfiles/trainer-downloader.dockerfile . | tail -3
docker build -t standalone-text-trainer:latest -f dockerfiles/standalone-text-trainer.dockerfile . | tail -3
docker build -t hf-uploader:latest -f dockerfiles/hf-uploader.dockerfile . 2>&1 | tail -3 || echo "  (hf-uploader build skipped — will skip upload step)"

# ── Start env servers (uses our diagonalge-based run_environment_env.sh) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/.environment_server_urls.txt"
echo "[STEP 2] Starting environment servers (diagonalge/mcts-api official)..."
"$SCRIPT_DIR/run_environment_env.sh"
ENVIRONMENT_SERVER_URLS=$(cat "$URLS_FILE")
rm -f "$URLS_FILE"
echo "  URLs: $ENVIRONMENT_SERVER_URLS"
echo ""

# ── Download augmented model ──────────────────────────────────────────────
MODEL_SAFE=$(echo "$AUGMENTED_MODEL" | sed 's/\//_/g')

echo "[STEP 3] Downloading augmented model: $AUGMENTED_MODEL ..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name "downloader-${TASK_ID:0:20}" \
  -e HF_TOKEN="${HUGGINGFACE_TOKEN:-dummy}" \
  trainer-downloader:latest \
  --task-id "$TASK_ID" \
  --model "$AUGMENTED_MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"
echo ""

# ── Run training with augmented model + fake BASELINE_STATS_PATH ──────────
TRAINING_CONTAINER_NAME="grpo-aug-${MODEL_SAFE:0:20}-${TASK_ID:0:8}"
TIMEOUT_SEC=$((HOURS_TO_COMPLETE * 3600))

echo "[STEP 4] Starting training container ($TRAINING_CONTAINER_NAME)..."
echo "         Mounting BASELINE_STATS_PATH=/cache/baseline_stats_test.json"

docker run -d --gpus "device=$GPUS" \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --shm-size=32g \
  --network agent_eval_net \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name "$TRAINING_CONTAINER_NAME" \
  --ipc=host \
  -e ENVIRONMENT_SERVER_URLS="$ENVIRONMENT_SERVER_URLS" \
  -e WANDB_TOKEN="${WANDB_TOKEN:-}" \
  -e WANDB_INIT_TIMEOUT=300 \
  -e HF_TOKEN="${HUGGINGFACE_TOKEN:-dummy}" \
  -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-dummy}" \
  -e MINER_DATASETS_DIR="${MINER_DATASETS_DIR:-}" \
  -e MINER_DATASETS="${MINER_DATASETS:-}" \
  -e BASELINE_STATS_PATH="/cache/baseline_stats_test.json" \
  -e AUGMENTED_MODEL="1" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  standalone-text-trainer:latest \
  --task-id "$TASK_ID" \
  --model "$AUGMENTED_MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --wandb-mode "${WANDB_MODE:-offline}" \
  --max-steps "$MAX_STEPS"

# ── Live stream all output ────────────────────────────────────────────────
LOG_FILE="$LOGS_DIR/${GAME}_${MODEL_SAFE}_$(date +%H%M%S).log"
echo "[STEP 5] Streaming training output (live)..."
echo "         Log: $LOG_FILE"
echo "════════════════════════════════════════════════════════════"

stdbuf -oL -eL timeout $TIMEOUT_SEC docker logs -f -t $TRAINING_CONTAINER_NAME 2>&1 | stdbuf -oL tee "$LOG_FILE" || true

# ── Cleanup ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
if [ "$(docker inspect -f '{{.State.Running}}' $TRAINING_CONTAINER_NAME 2>/dev/null)" == "true" ]; then
  docker stop $TRAINING_CONTAINER_NAME
fi
EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' $TRAINING_CONTAINER_NAME 2>/dev/null || echo "?")
docker logs $TRAINING_CONTAINER_NAME > "${LOG_FILE}.full" 2>&1 || true
docker rm $TRAINING_CONTAINER_NAME 2>/dev/null || true
docker stop $(docker ps --filter "name=agentgym-server" -q) 2>/dev/null || true
docker rm $(docker ps -a --filter "name=agentgym-server" -q) 2>/dev/null || true

# ── Compatibility verification ────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  AUGMENTATION TEST RESULT                                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Exit code:       $EXIT_CODE"
echo "║  Augmented model: $AUGMENTED_MODEL"
echo "╚════════════════════════════════════════════════════════════╝"

if [ -f "${LOG_FILE}.full" ]; then
  echo ""
  echo "[VERIFY] Code-path compatibility checks:"
  if grep -q "TOURNAMENT_ENV" "${LOG_FILE}.full"; then
    echo "  ✅ TOURNAMENT_ENV block logged"
    if grep -q "BASELINE_STATS_PATH.*=.*baseline_stats_test.json" "${LOG_FILE}.full"; then
      echo "  ✅ BASELINE_STATS_PATH correctly detected from env var"
    fi
    if grep -qE "baseline.*(mean=|leduc_poker|liars_dice|gin_rummy)" "${LOG_FILE}.full"; then
      echo "  ✅ BASELINE_STATS_PATH parsed (baseline scores logged)"
    fi
  fi
  if grep -q "rewards/rollout_reward_func" "${LOG_FILE}.full"; then
    GRPO_STEPS=$(grep -c "rewards/rollout_reward_func" "${LOG_FILE}.full")
    echo "  ✅ GRPO training completed ($GRPO_STEPS step logs)"
  fi
  if grep -q "Loading safetensors checkpoint shards" "${LOG_FILE}.full"; then
    echo "  ✅ Model weights loaded successfully (augmented model OK)"
  fi
  if grep -q "size_label" "${LOG_FILE}.full"; then
    SIZE=$(grep -oE "size_label: [0-9_b]+" "${LOG_FILE}.full" | head -1)
    echo "  ✅ Size label routing worked: $SIZE"
  fi
  if grep -qE "Traceback|RuntimeError|OutOfMemoryError" "${LOG_FILE}.full"; then
    echo "  ❌ Errors detected — inspect ${LOG_FILE}.full"
    grep -nE "Traceback|RuntimeError|OutOfMemoryError" "${LOG_FILE}.full" | head -3
  else
    echo "  ✅ No fatal errors"
  fi
fi

echo ""
echo "Done. To verify on HF: open https://huggingface.co/$AUGMENTED_MODEL"
echo "Compare with vanilla: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"
