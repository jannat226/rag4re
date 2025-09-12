#!/bin/bash
#SBATCH --job-name=RAG4RE
#SBATCH --gres=gpu:40g:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=1-00:00

source thesis_env/bin/activate

# Use first command-line argument as MODE, default to "sentences" if not provided
MODE=${1:-sentences}

if [ "$MODE" == "sentences" ]; then
    OUTFILE="output-sentences.log"
    SCRIPT="baselineWithSentences-DS.py"
elif [ "$MODE" == "zero_shot" ]; then
    OUTFILE="output-zeroshot.log"
    SCRIPT="baselineForZeroShot.py"
elif [ "$MODE" == "llama" ]; then
    OUTFILE="output-llama.log"
    SCRIPT="baselineWithLlama.py"
else
    OUTFILE="output-default.log"
    SCRIPT="baselineWithSentences-DS.py"
fi

python3 $SCRIPT > $OUTFILE 2>&1

#sbatch run.sh sentences