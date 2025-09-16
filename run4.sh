#!/bin/bash
#SBATCH --job-name=RAG4RE
#SBATCH --gres=gpu:40g:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=1-00:00
#SBATCH --output=output-baselineWithLlama-backup.log


# module load python/3.11
source thesis_env/bin/activate
# pip install -r requirements.txt
# pip freeze > requirements.txt
# python preprocessing.py --test_in dev5.json --test_out processed_test.json
# python preprocessing.py --train_in small_train.json --train_out processed_train.json --val_out processed_val.json
# python3 baselineWithLlama-test.py

# python3 bws-backup-ollama.py 

# python3 baseline_with_llama_for_ten_shot.py
# python3 baselineWithOllama_only_abstract.py
# python3 baselineWithLlama.py
# python3 baselineWithSentences_for_ten_shot-Ollama.py
# python3 baselineForZeroShot.py
# python3 baselineWithSentences.py
python3 baselineWithLlama-backup.py
# python3 baseline_fineTuned.py

# python3 baseline_with_llama_for_ten_shot.py
# python3 baselineWithSentences_for_ten_shot.py