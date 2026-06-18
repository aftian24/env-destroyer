#!/usr/bin/env bash
set -e

echo "=== Setting up AgentGym ALFWorld ==="

# Clone AgentGym
git clone https://github.com/WooooDyy/AgentGym /opt/AgentGym

cd /opt/AgentGym

# Create venv with Python 3.9 (ALFWorld requirement)
uv venv .venv --python 3.9 --seed
source .venv/bin/activate

# Setup ALFWorld environment
cd agentenv-alfworld
bash ./setup.sh

echo "=== AgentGym ALFWorld setup completed ==="
