#!/bin/bash

# Create network if it doesn't exist
docker network create agent_eval_net 2>/dev/null || true

# Number of servers to start
NUM_SERVERS=4

# Start servers
for i in $(seq 0 $((NUM_SERVERS-1))); do
  host_port=$((8001 + i))
  container_name="agentgym-server-$i"
  
  # Remove existing container if it exists
  docker rm -f "$container_name" 2>/dev/null || true
  
  # Start the server - redirect container ID output to stderr
  # Image: diagonalge/mcts-api:latest (matches validator's official
  # MCTS_API_DOCKER_IMAGE per gradients-ai/G.O.D core/constants.py — ensures
  # training opponent behavior matches what validator uses during eval).
  docker run -d \
    --name "$container_name" \
    --network agent_eval_net \
    -p $host_port:8000 \
    diagonalge/mcts-api:latest >&2
  
  echo "Started $container_name on host port $host_port" >&2
done

# Output URLs (using container names for Docker network access)
# Format: comma-separated URLs
urls=()
for i in $(seq 0 $((NUM_SERVERS-1))); do
  urls+=("http://agentgym-server-$i:8000")
done

# Write URLs to a file to avoid stdout contamination
# Use a temporary file in the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/.environment_server_urls.txt"
echo "${urls[*]}" | tr ' ' ',' > "$URLS_FILE"
echo "URLs written to $URLS_FILE" >&2