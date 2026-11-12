#!/bin/bash
#SBATCH --job-name=llm-prepare
#SBATCH --account=idea_stud
#SBATCH --partition=hpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export RAYON_NUM_THREADS="$SLURM_CPUS_PER_TASK"
RAW_TEXT="${1:?Usage: sbatch assignment1/prepare.sbatch RAW_TEXT NEW_DATA_DIR VAL_FRACTION SPLIT_SEED}"
DATA_DIR="${2:?Provide a new data directory}"
VAL_FRACTION="${3:?Provide the validation fraction (between 0 and 1)}"
SPLIT_SEED="${4:?Provide the integer split seed}"
if [ -e "$DATA_DIR" ]; then
    printf 'Output already exists: %s\n' "$DATA_DIR" >&2
    exit 1
fi
.venv/bin/python assignment1/code/tools.py split-data --input "$RAW_TEXT" --out "$DATA_DIR/splits" \
    --val-fraction "$VAL_FRACTION" --seed "$SPLIT_SEED"
.venv/bin/python assignment1/code/tools.py train-tokenizer --input "$DATA_DIR/splits/train.txt" \
    --out "$DATA_DIR/tokenizer" --vocab-size 4096
.venv/bin/python assignment1/code/tools.py prepare \
    --input "$DATA_DIR/splits/laptop-train.txt" --val-input "$DATA_DIR/splits/laptop-val.txt" \
    --out "$DATA_DIR/laptop" --tokenizer "$DATA_DIR/tokenizer/tokenizer.json"
.venv/bin/python assignment1/code/tools.py prepare \
    --input "$DATA_DIR/splits/train.txt" --val-input "$DATA_DIR/splits/val.txt" \
    --out "$DATA_DIR/final" --tokenizer "$DATA_DIR/tokenizer/tokenizer.json"
