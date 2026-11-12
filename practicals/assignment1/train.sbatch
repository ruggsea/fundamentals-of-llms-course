#!/bin/bash
#SBATCH --job-name=llm-student
#SBATCH --account=idea_stud
#SBATCH --partition=gpu
#SBATCH --nodelist=IT010146
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from practicals/. Use the course account on the general GPU queue.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PATH="$HOME/.local/bin:$PATH"
DATA_DIR="${1:?Usage: sbatch [allocation options] assignment1/train.sbatch DATA_DIR RUN_DIR [trainer options]}"
RUN_DIR="${2:?Provide a new output directory}"
shift 2
nvidia-smi
.venv/bin/python assignment1/code/train.py --data "$DATA_DIR" --out "$RUN_DIR" \
    --config assignment1/configs/final.json --device cuda --precision float32 \
    --threads "$SLURM_CPUS_PER_TASK" "$@"
