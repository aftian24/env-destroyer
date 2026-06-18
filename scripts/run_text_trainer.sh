#!/bin/bash
set -e

redis-server --daemonize yes
sleep 10

# Workaround for bitsandbytes 0.45.x: when NVML init races (4-rank SFT on H100
# PCIe), bnb's get_native_library() falls back to libbitsandbytes_cpu.so which
# is NOT bundled in the wheel → OSError cascade kills training. We symlink the
# missing CPU file to a CUDA-built variant. The bnb loader inspects loaded DLL
# for 'get_context' attribute to detect CUDA-built lib (cextension.py:73), so
# the symlink works correctly even when CPU fallback path is taken.
BNB_DIR="/workspace/.grpo_env/lib/python3.12/site-packages/bitsandbytes"
if [ -d "$BNB_DIR" ] && [ ! -f "$BNB_DIR/libbitsandbytes_cpu.so" ]; then
    # Pick the highest available CUDA version (cuda128 > cuda126 > ...)
    LATEST_CUDA_LIB=$(ls -1 "$BNB_DIR"/libbitsandbytes_cuda*.so 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CUDA_LIB" ]; then
        ln -sf "$LATEST_CUDA_LIB" "$BNB_DIR/libbitsandbytes_cpu.so"
        echo "[bnb-workaround] symlinked libbitsandbytes_cpu.so -> $(basename $LATEST_CUDA_LIB)"
    fi
fi

echo "*****Running text trainer"
source /workspace/.grpo_env/bin/activate
python3 /workspace/scripts/text_trainer.py "$@"
deactivate