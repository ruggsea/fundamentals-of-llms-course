# Practical setup: your computer and GSC1

Use your computer for editing, tests and small experiments. Use GSC1 for larger
training runs through Slurm, the cluster's job scheduler. You can develop locally
without a GPU.

Use Slurm account **`idea_stud`** for course jobs: partition **`hpc`** for CPU
preparation and **`gpu`** for final training. The steps below cover local setup
first, then the cluster.

## 1. Install uv and get a working copy

[uv](https://docs.astral.sh/uv/) manages Python versions, virtual environments and
packages. Follow its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for your operating system, then open a new terminal and check:

```sh
uv --version
```

Clone the course repository, then create the environment:

```sh
git clone https://github.com/ruggsea/fundamentals-of-llms-course.git
cd fundamentals-of-llms-course/practicals
uv python install 3.12
uv venv --python 3.12 .venv
```

A virtual environment keeps these dependencies separate from other Python projects.
Create it separately on each computer and on the cluster; do not copy `.venv`
between machines or commit it to Git. The commands below assume your working
directory is `practicals/`.

Activate the environment using the command for your shell:

| System / shell | Command |
|---|---|
| Linux or macOS, Bash/Zsh | `source .venv/bin/activate` |
| Windows, PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows, Command Prompt | `.venv\Scripts\activate.bat` |

If PowerShell blocks activation, you can use `.venv\Scripts\python.exe` in place
of `python` and `uv pip install --python .venv\Scripts\python.exe ...` to install
packages without changing execution policy.

Install the initial non-PyTorch packages:

```sh
uv pip install numpy pytest
```

## 2. Choose the PyTorch build for your machine

PyTorch performs tensor operations and computes gradients. Its installation and
available acceleration depend on your operating system, GPU and driver.
Follow **one** of the paths below. You do not need `torchvision` or `torchaudio`
for text-only work.

The [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
provides supported platform/build combinations. [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
explains how to select the matching packages. Use stable releases rather than
nightly builds for course work.

### Linux or Windows without an NVIDIA GPU

```sh
uv pip install torch --torch-backend=cpu
```

CPU execution is sufficient for checking components and running tiny models.
Training time depends on the experiment; a successful setup check is not a promise
that a full assignment run will be fast on a laptop.

### macOS with Apple Silicon

Use a native Apple Silicon terminal/Python installation rather than an Intel
Python running through Rosetta:

```sh
uv pip install torch
```

PyTorch uses Apple's Metal Performance Shaders backend, called `mps`, for GPU
operations. CUDA is for NVIDIA GPUs and does not apply here. Check MPS availability:

```sh
python -c "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

A program must explicitly place its model and tensors on `mps`; installing the
package alone does not enable acceleration. Some operations or precision modes
can differ from CUDA. Start with float32 and compare a small CPU run when debugging.
See the [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

For an Intel Mac, check whether the selected PyTorch release supplies a compatible
wheel. If it does not, use the course cluster rather than attempting a source
build as your first setup step.

### Linux or Windows with an NVIDIA GPU

First check the driver:

```sh
nvidia-smi
```

Then install a driver-compatible PyTorch build:

```sh
uv pip install torch --torch-backend=auto
```

Here `auto` selects a backend using the local NVIDIA driver. Run this on the
machine with the GPU. For a remote cluster, follow the explicit-build instructions
below rather than detecting hardware on its login node.

Check the result:

```sh
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The CUDA version shown by `nvidia-smi` describes driver compatibility; it is not
necessarily an installed CUDA toolkit. Prebuilt PyTorch wheels provide their CUDA
runtime dependencies. A separate toolkit/module is usually needed only when
building CUDA extensions. Consult the installation selector if your GPU or driver
is too old for the selected build.

For AMD or Intel GPU acceleration, consult the official selector and uv guide for
hardware/platform support. Use the CPU path if your hardware is not supported.

## 3. Check that computation and gradients work

With the environment activated, run:

```sh
python -c "import sys, torch, numpy; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); x = torch.tensor([2.0], requires_grad=True); (x*x).sum().backward(); print('Gradient:', x.grad.item())"
```

The executable should be inside your `.venv`, and the gradient should be `4.0`.
This checks CPU execution. To test your GPU as well, run **one** of:

```sh
# NVIDIA
python -c "import torch; x = torch.ones(4, device='cuda', requires_grad=True); (x*x).sum().backward(); torch.cuda.synchronize(); print(x.device, x.grad)"

# Apple Silicon
python -c "import torch; x = torch.ones(4, device='mps', requires_grad=True); (x*x).sum().backward(); torch.mps.synchronize(); print(x.device, x.grad)"
```

These should print the requested device and four gradients equal to `2`.
An error should be investigated before starting training.

Save your installed package versions:

```sh
uv pip freeze > environment.txt
```

Keep this file with your results. It helps you reproduce a run and report setup
problems. Create the environment separately on each machine.

## 4. Connect to GSC1

GSC1 is the University of Graz computing cluster. Course students have been added
to the Slurm account `idea_stud`. Your GSC1 username is the same as your **UNIGRAZonline username**.
Open a terminal on your computer and connect with:

```sh
ssh YOUR_UNIGRAZONLINE_USERNAME@l1.gsc1.uni-graz.at
```

Replace `YOUR_UNIGRAZONLINE_USERNAME` with your own username. Follow the
[GSC connection instructions](https://hpc-wiki.uni-graz.at/wiki:gsc1) for network,
VPN and SSH-key requirements. Contact the teaching team if you cannot log in.

| Setting | What to use |
|---|---|
| SSH username | Your UNIGRAZonline username |
| Slurm account | `idea_stud` — the course allocation |
| CPU partition | `hpc` — data preparation |
| GPU partition | `gpu` — model training |
| GPU request | `gpu:rtx2080ti:1` — one NVIDIA RTX 2080 Ti (11 GB) |
| GPU node | `IT010146` |

The account identifies the course allocation; the partition selects a queue.
The supplied scripts set both. If your account cannot use these queues, contact
the teaching team with the submission error.

Check your account membership after logging in:

```sh
sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS
```

You should see `idea_stud` in the account column. If it is missing, contact the
teaching team. Choosing a different account in your job script will not fix access.

## 5. Set up your files and environment on GSC1

Use the login node to edit files, install packages and submit jobs.
**Run training and GPU computations through Slurm, not directly on the login node!**

### Store your work

| Location | Use |
|---|---|
| `$HOME` | Small files such as code and settings. Space is limited. |
| `/cl_tmp/$USER` | Your working files, datasets, environment and results. Visible from login and compute nodes. Not backed up. |
| `/usertmp/$USER` | Temporary files on one compute node. Avoid this for the course unless you need it; copy results out before the job ends. |

Create a working folder and check free space:

```sh
mkdir -p "/cl_tmp/$USER/llm-course"
df -h /cl_tmp
```

Copy your working checkout into this folder, or clone your own GitHub repository
containing your implementation. Open its `practicals/`
directory before running the commands below. Keep a separate copy of important
results: scratch storage is not a backup.

### Install the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on GSC1 if
`uv --version` does not work. Create a new environment on the cluster; do not copy
your laptop's `.venv`.

From `practicals/`:

```sh
export UV_CACHE_DIR="/cl_tmp/$USER/llm-course/cache/uv"
mkdir -p "$UV_CACHE_DIR"

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install numpy pytest
uv pip install -r assignment1/code/requirements.txt
uv pip install torch --torch-backend=cu128
uv pip check
uv pip freeze > environment.txt
```

Use the explicit CUDA build shown here on GSC1. Automatic GPU detection on the
login node can select the wrong build because that node has no allocated GPU.
The GPU check in the next section verifies that computation works in your job.

Install packages before submitting a job. The supplied scripts use your existing
`.venv`; they do not install anything while holding a GPU.

### Hugging Face account and login

Create a [Hugging Face account](https://huggingface.co/join) in your laptop's
browser. You will use it to upload your trained models for submission.

In your **GSC terminal**, with the course environment activated, run:

```sh
hf auth login
hf auth whoami
```

Choose browser login. Open the displayed link on your laptop, enter the code and
approve the request. `whoami` should print your Hugging Face username. This logs
in the command-line tool on GSC; you do not need a browser on the cluster.
[Login documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login).

Login credentials stay in your home directory. Put downloaded model files on scratch:

```sh
export HF_HUB_CACHE="/cl_tmp/$USER/llm-course/cache/huggingface/hub"
mkdir -p "$HF_HUB_CACHE"
```

After training, follow the [submission instructions](assignment1/submission.md)
to upload directly from GSC.

## 6. Check the GPU with a short job

From `practicals/`, submit [gpu-check.sbatch](gpu-check.sbatch):

```sh
mkdir -p logs
sbatch gpu-check.sbatch
```

Create `logs/` **before** submitting: Slurm needs it when opening the output files.

The script requests one GPU, two CPU cores and 8 GB of system RAM for up to five
minutes. It checks a small GPU calculation and its gradients. You can run this
check before completing the assignment code.

Slurm prints a job ID. Replace `JOB_ID` below with that number:

```sh
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
cat logs/llm-setup-JOB_ID.out
cat logs/llm-setup-JOB_ID.err
```

| Result | What it means / what to do |
|---|---|
| `PENDING` or `PD` | The job is waiting. `Priority` or `Resources` usually means you need to wait for a turn. |
| `RUNNING` or `R` | The job has started. Check its log files. |
| `COMPLETED`, exit code `0:0` | The job finished successfully. Check the success message below. |
| `FAILED`, `TIMEOUT` or `OUT_OF_MEMORY` | Read the error log before changing the script or submitting again. |

For a successful GPU check, the output must contain:

```text
GPU computation and gradients passed
```

To cancel a job you no longer need:

```sh
scancel JOB_ID
```

## 7. Submit a training job

Before submitting:

- Complete the missing assignment functions and run the CPU tests.
- Prepare the dataset as described in the assignment.
- Complete the GPU check above.
- Check the assignment's data and compute instructions before a larger run.

From `practicals/`:

```sh
mkdir -p logs
sbatch assignment1/train.sbatch data/COURSE_DATA runs/baseline
```

Replace `COURSE_DATA` with your prepared dataset folder. Use a new run directory
for each new run. The script selects account `idea_stud`, partition `gpu` and
one RTX 2080 Ti on `IT010146`. Training uses float32, as on the CPU.

The default requests in [assignment1/train.sbatch](assignment1/train.sbatch) are:

| Script setting | Default | Meaning |
|---|---|---|
| `--gres=gpu:rtx2080ti:1` | One RTX 2080 Ti | GPU allocated to the job |
| `--nodelist=IT010146` | `IT010146` | Node containing these GPUs |
| `--cpus-per-task=4` | Four CPU cores | CPU resources for the process |
| `--mem=16G` | 16 GB | System RAM, not GPU memory |
| `--time=06:00:00` | Six hours | Maximum runtime before Slurm stops the job |

Use the assignment's laptop configuration for development and its final
configuration for the two main runs. CPU preparation uses
[assignment1/prepare.sbatch](assignment1/prepare.sbatch), which requests no GPU.
`#SBATCH` lines are read by Slurm; shell variables such as `$USER` are not expanded
inside those lines.

Use the same monitoring commands as for the GPU check. Training logs are named
`logs/llm-student-JOB_ID.out` and `logs/llm-student-JOB_ID.err`. Model checkpoints
and training metrics go into the run directory you specified.

An accepted submission means the job is queued, not that training has succeeded.
Check its final state and logs. If a job fails or reaches its time limit, fix the
cause before starting a new run in a new output directory.

## 8. Get help or use a coding agent

Work through setup one step at a time: install, check the environment, check the
GPU, then train. Give your agent this guide and the assignment. Ask it to explain
any proposed change before applying it.

For GSC jobs, keep account `idea_stud` and select `hpc` for CPU preparation or
`gpu` for training. Verify the working directory, `.venv`, data paths and output
paths before changing resources.

When asking for help, include:

- The command and full error message.
- Your operating system and Python/PyTorch versions.
- On GSC1: the job ID, job script and relevant log files.

Do not share passwords, access tokens or private SSH keys.

Official GSC documentation:

- [Connection and cluster overview](https://hpc-wiki.uni-graz.at/wiki:gsc1)
- [Accounts](https://hpc-wiki.uni-graz.at/wiki:account)
- [Slurm jobs](https://hpc-wiki.uni-graz.at/wiki:slurm)
- [GPU jobs](https://hpc-wiki.uni-graz.at/wiki:gpu)
- [Support](https://hpc-wiki.uni-graz.at/wiki:contact)
