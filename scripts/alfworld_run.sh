#!/usr/bin/env bash
set -e

source /opt/AgentGym/.venv/bin/activate

NUM_SERVERS=$1
if [ -z "$NUM_SERVERS" ]; then
    NUM_SERVERS=$(python3 - <<EOF
import os
print(min(os.cpu_count() or 1, 8))
EOF
)
elif [ "$NUM_SERVERS" -gt 8 ]; then
    NUM_SERVERS=8
fi

START_PORT=${START_PORT:-8000}
LOG_DIR=${LOG_DIR:-/workspace/logs}

mkdir -p "$LOG_DIR"

echo "Starting $NUM_SERVERS ALFWorld servers..."

ALFWORLD_PIDS=()
SERVER_ADDR_LIST=""

for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((START_PORT + i))

    alfworld --host 0.0.0.0 --port "$PORT" \
        > "$LOG_DIR/alfworld_$PORT.log" 2>&1 &

    PID=$!
    ALFWORLD_PIDS+=("$PID")

    ADDR="http://localhost:$PORT"
    SERVER_ADDR_LIST="${SERVER_ADDR_LIST:+$SERVER_ADDR_LIST,}$ADDR"

    echo "  → port $PORT (pid=$PID)"
done

# Export for parent script
export ALFWORLD_PIDS
export ALFWORLD_SERVER_ADDRS="$SERVER_ADDR_LIST"

echo "ALFWorld servers started:"
echo "$ALFWORLD_SERVER_ADDRS"

echo "$ALFWORLD_SERVER_ADDRS" > /tmp/alfworld_servers.txt

deactivate
