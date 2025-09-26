#!/bin/bash
#SBATCH --job-name=RAG4RE-FullTestData
#SBATCH --gres=gpu:40g:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=1-00:00
#SBATCH --output=output-bws_re_gemini.log

source thesis_env/bin/activate

python3 bws_re_gemini.py \
    --train_file /home/lnuj3/thesis/processed_train.json \
    --dev_file /home/lnuj3/thesis/processed_test.json \
    --num_shots 10
