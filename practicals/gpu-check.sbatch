#!/bin/bash
#SBATCH --job-name=llm-setup
#SBATCH --partition=gpu
#SBATCH --account=idea_stud
#SBATCH --nodelist=IT010146
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="/cl_tmp/$USER/llm-course/cache/uv"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

hostname
nvidia-smi
.venv/bin/python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable in this allocation"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones(4, device="cuda", requires_grad=True)
(x * x).sum().backward()
torch.cuda.synchronize()
assert torch.equal(x.grad, torch.full_like(x, 2))
print("GPU computation and gradients passed")
PY
