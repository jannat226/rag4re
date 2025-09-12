#!/bin/bash
#SBATCH --job-name=RAG4RE
#SBATCH --gres=gpu:40g:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=1-00:00
#SBATCH --output=output43.log
ollama serve 