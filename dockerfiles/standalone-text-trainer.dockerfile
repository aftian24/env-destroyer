FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

# System dependencies
RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    redis-server \
    && rm -rf /var/lib/apt/lists/*
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts

# # Setup AlfWorld server env
# COPY scripts/alfworld_setup.sh /workspace/scripts/alfworld_setup.sh
# COPY scripts/alfworld_run.sh /workspace/scripts/alfworld_run.sh
# RUN chmod +x /workspace/scripts/alfworld_setup.sh
# RUN /workspace/scripts/alfworld_setup.sh

# Install main dependencies
COPY scripts/grpo_requirements.txt /workspace/scripts/grpo_requirements.txt
RUN python -m venv /workspace/.grpo_env
RUN bash -c "source /workspace/.grpo_env/bin/activate && \
    pip install uv && \
    uv pip install --no-build-isolation -r /workspace/scripts/grpo_requirements.txt && \
    uv pip install --no-build-isolation flash-attn==2.8.3 && \
    git clone --depth 1 https://github.com/WooooDyy/AgentGym && \
    uv pip install --no-build-isolation AgentGym/agentenv && \
    deactivate"

# Copy current folder to /workspace/auto_ml
COPY scripts /workspace/scripts
# # Make entrypoint script executable
# RUN chmod +x /workspace/scripts/alfworld_run.sh

RUN chmod +x /workspace/scripts/run_text_trainer.sh
# RUN chmod +x /workspace/scripts/entrypoint.sh

ENTRYPOINT ["./run_text_trainer.sh"]