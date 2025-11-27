#!/bin/bash
#SBATCH --job-name=RAG4RE_Full_Ollama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:40g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=1-00:00
#SBATCH --output=rag4re_ollama_output.log

set -euo pipefail

echo "=== Node & GPU info ==="
hostname
nvidia-smi || true
echo "======================="

# ---- Ollama paths ----
export OLLAMA_HOME="$HOME/ollama"
export PATH="$OLLAMA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="$HOME/.ollama/models"
export OLLAMA_DEBUG=1
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=3600m
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OLLAMA_BIN="$OLLAMA_HOME/bin/ollama"

echo "Using ollama at: $OLLAMA_BIN"
echo "Using model dir: $OLLAMA_MODELS"
[ -x "$OLLAMA_BIN" ] || { echo "ERROR: $OLLAMA_BIN not executable"; exit 1; }
mkdir -p "$OLLAMA_MODELS"

echo "========================================"
echo "Starting Ollama server at $(date)"
echo "========================================"
"$OLLAMA_BIN" serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

cleanup() { echo "Stopping Ollama (pid $OLLAMA_PID)"; kill $OLLAMA_PID 2>/dev/null || true; }
trap cleanup EXIT

# Wait until server responds
echo "Waiting for Ollama to start..."
for i in $(seq 1 60); do
  if "$OLLAMA_BIN" list >/dev/null 2>&1; then
    echo "Ollama is up."
    break
  fi
  sleep 2
  kill -0 $OLLAMA_PID 2>/dev/null || { echo "Ollama died early"; tail -n 120 ollama_server.log || true; exit 1; }
done

echo "Available models:"
"$OLLAMA_BIN" list || true

MODEL_TAG="qwen3:14b"

if ! "$OLLAMA_BIN" show "$MODEL_TAG" >/dev/null 2>&1; then
  echo "ERROR: Model $MODEL_TAG not available in $OLLAMA_MODELS"
  "$OLLAMA_BIN" list || true
  exit 1
fi

# --- Python/venv ----
cd "$SLURM_SUBMIT_DIR"
if [ -d "thesis_env" ]; then
    source thesis_env/bin/activate
else
    echo "WARNING: venv not found at $SLURM_SUBMIT_DIR/thesis_env; using system python"
fi

echo "========================================"
echo "Starting Relation Extraction/RAG at $(date)"
echo "========================================"
set +e
srun -c "${SLURM_CPUS_PER_TASK}" python3 bws_re_reasoning_qwen.py \
    --train_file /home/lnuj3/thesis/processed_train.json \
    --dev_file /home/lnuj3/thesis/processed_test.json \
    --num_shots 55 \
    --model "$MODEL_TAG"
RE_EXIT_CODE=$?
set -e

echo "========================================"
echo "Completed at $(date)"
echo "Exit code: $RE_EXIT_CODE"
echo "========================================"

if [ "$RE_EXIT_CODE" -ne 0 ]; then
    echo "Last 120 lines of ollama_server.log:"
    tail -n 120 ollama_server.log || true
fi

exit $RE_EXIT_CODE
