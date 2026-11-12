# Assignment 1: build and train a small language model

**Submission deadline: Sunday 13 December 2026, 23:59 (Graz time).**

**Hand-in: one GitHub repository containing the report, code and supporting files,
with a link to one public Hugging Face repository containing both models.**
See [Submission](#submission) below.

**Individual interviews/defences: 16–18 December 2026**, in booked 20-minute slots.

1. Split the data, train one tokenizer and prepare the token files.
2. Implement the model and training functions.
3. Check training with the laptop configuration.
4. Train the final baseline on one GPU.
5. Change one model feature and train the modified version.
6. Compare the results, submit the models and explain the work.

## Setup

Follow [setup.md](../setup.md). Run commands from `practicals/` with its `.venv`
activated:

```sh
uv pip install -r assignment1/code/requirements.txt
```

Multiline commands use Bash syntax. In PowerShell or Command Prompt, join the
lines and remove the trailing `\` characters.

[Training data](https://huggingface.co/datasets/idealab-cs2/fundamentals-of-llms-training-data): about 1.5 GiB of English web text selected
from FineWeb. The dataset page explains the source, extraction and filtering.
Download this version into `data/raw/`:

```sh
hf download idealab-cs2/fundamentals-of-llms-training-data training-data.txt source.json \
  --repo-type dataset --revision 9624da983649aa6c90f4dbb260b2b9c87de5350c --local-dir data/raw
```

The public dataset download does not require login. The commands below use
`data/raw/training-data.txt`.

The download is a UTF-8 text file. Documents are separated by `<|endoftext|>`.
The preparation tools split whole documents into training and validation sets,
then encode them with your tokenizer and save token IDs in `train.bin` and `val.bin`.

During training, the loader draws sequences of token IDs. The model learns to
predict the next token at each position.

## Scope

| Implement | Supplied |
|---|---|
| Causal attention, transformer block and model forward pass | Layer constructors, RMSNorm and rotary positions |
| Next-token targets and one optimizer update | Training loop, AdamW, learning-rate schedule, logs and checkpoints |
| Three tests and one model modification | BPE implementation, data preparation, basic tests, evaluation and generation |

Implement the five functions marked `TODO` in
`model.py` and `train.py`; use PyTorch layers and autograd. Keep the supplied
interfaces working so the models can be evaluated and reused in Assignment 2.

Use the laptop configuration while developing. The required final experiment is
**two runs on one GPU each**: the baseline and one modification, each trained on
about **500 million tokens**.
Keep the baseline architecture and training settings fixed for the comparison.

Distributed training, gradient accumulation, tokenizer implementation and
large parameter searches are outside the assignment. The training and submission
scripts handle the surrounding infrastructure.

## Starter overview

The model predicts the next token from the tokens before it:

```text
Text → trained tokenizer → token IDs
Token IDs → embeddings → transformer blocks → final RMSNorm → next-token scores
```

Each block contains causal attention and a gated feed-forward layer, with
normalization and residual connections. The output projection shares the embedding
weights. Train the tokenizer once, then keep it fixed while training the models.

| File or folder | Purpose |
|---|---|
| `code/model.py` | Layers are defined; complete the attention, block and model forward passes. |
| `code/train.py` | Complete next-token targets and one optimizer update. |
| `code/components.py` | Supplied RMSNorm and rotary positions. |
| `code/helpers/data.py` | Split documents, train the tokenizer and encode the data. |
| `code/helpers/` | Supplied data loading, training loop, logging, checkpoints and evaluation. |
| `code/tools.py` | Runs the supplied utilities shown below. |
| `configs/` | Model and training settings for laptop and final runs. |
| `tests/` | Basic checks and the file for your three tests. |

Edit `model.py`, `train.py` and `tests/test_student.py`. You do not need to edit
the helpers. The training loop draws token windows, calls your target and update
functions, and saves validation results and checkpoints.

## Coding agents

Read the relevant lecture notes and supplied materials first. Work on one component at a time:

1. Ask the agent to explain implementation options.
2. Choose an approach and explain why it fits.
3. Work through a small example: tensor values, shapes or a diagram.
4. Implement it, test it and investigate unexpected results.
5. Continue once you can explain the code and its test.

## 1. Train the tokenizer and prepare laptop data

Run this section on your laptop. Use a small model and small subsets of the
training and validation data to check your code. We call these subsets **laptop data**.
For the two main GPU runs, use the full training split and a larger validation
set: **final data**.

Use the supplied byte-pair encoding (BPE) implementation to train **one tokenizer
with a 4,096-token vocabulary**. Use that same tokenizer for every laptop and final run.

Choose your training/validation split and explain your choice in the report.
The validation fraction is a number between 0 and 1: `0.2` means about 20% of
documents go to validation and 80% to training. The examples use **`0.01` (1%)**,
roughly 5,000 validation documents, and integer seed **`0`** as starting values.
The seed makes the split reproducible. Larger validation sets take longer to score.

Split whole documents:

```sh
python assignment1/code/tools.py split-data --input data/raw/training-data.txt \
  --out data/a1/splits --val-fraction 0.01 --seed 0
```

Check the document counts in `data/a1/splits/split.json` and verify that the
training and validation sets have no documents in common. Keep this split for
both model versions. The laptop files are smaller subsets of these same splits.

Train the tokenizer on the training split. The helper samples up to 32 MiB from
across that split:

```sh
python assignment1/code/tools.py train-tokenizer --input data/a1/splits/train.txt \
  --out data/a1/tokenizer --vocab-size 4096
```

Encode a short sentence and inspect its token IDs and byte pieces:

```sh
python assignment1/code/tools.py tokenize --tokenizer data/a1/tokenizer/tokenizer.json \
  --text "A small bird landed."
```

Encode the laptop data:

```sh
python assignment1/code/tools.py prepare \
  --input data/a1/splits/laptop-train.txt --val-input data/a1/splits/laptop-val.txt \
  --out data/a1/laptop --tokenizer data/a1/tokenizer/tokenizer.json
```

Preparation writes `tokenizer.json`, `train.bin`, `val.bin` and `meta.json`, which
records data counts and checksums. Use a new output directory when preparing data again.

## 2. Implement the model and training functions

Complete these functions in `assignment1/code/`:

| File | Function | Task |
|---|---|---|
| `model.py` | `Attention.forward` | Causal multi-head attention |
| `model.py` | `Block.forward` | Attention, feed-forward layer, normalization and residual connections |
| `model.py` | `GPT.forward` | Token scores and prediction loss |
| `train.py` | `make_targets` | Inputs and next-token targets |
| `train.py` | `train_step` | One optimizer update |

### Attention

Implement the attention calculation below. Both laptop and final GPU runs use
your implementation.

`B` is batch size, `T` sequence length, `C` model width, `H` head count and
`D = C / H` the width of one head.

1. Project `[B, T, C]` inputs into queries, keys and values using `qkv`.
2. Reshape each to `[B, H, T, D]`.
3. Apply the supplied `rotate` helper to queries and keys using the first `T`
   rows of `cos` and `sin`.
4. Calculate `scores = Q @ K.transpose(-2, -1) / sqrt(D)`.
5. Let position `i` attend only to positions `j <= i`; set other scores to negative infinity.
6. Apply softmax over the last dimension in float32, then cast to the values' dtype.
7. Multiply the attention weights by the values.
8. Merge heads into `[B, T, C]` and apply `self.out`.

### Transformer block

Connect the supplied modules:

```text
x = x + attention(attn_norm(x))
h = ffn_norm(x)
x = x + down(activation(gate(h)) * up(h))
```

Use RMSNorm for both norms and the configured activation: SiLU for the baseline.

### Model and loss

1. Embed token IDs `[B, T]` into `[B, T, C]`.
2. Apply all transformer blocks and the final RMSNorm.
3. Project to vocabulary scores using the **token embedding weights**.
   These scores, called *logits*, have shape `[B, T, vocabulary size]`.
4. If targets are given, calculate mean cross-entropy from float32 logits,
   ignoring targets equal to `-100`. Targets are already shifted.
5. Return `(logits, loss)`, or `(logits, None)` without targets.

Reject inputs that are not two-dimensional or have a sequence length outside
`1 <= T <= context`. Initialization and position tables are supplied.

### Targets and optimizer update

`make_targets` receives `[B, T+1]` token windows. Return `[B, T]` inputs and targets
so each position predicts the following token. Preserve order, integer dtype and device.

In `train_step`:

1. Set training mode and clear old gradients.
2. Run the forward pass inside the supplied `precision_context`.
3. Check that the loss is finite and calculate gradients with backpropagation.
4. Clip the total gradient norm to `1.0`; reject nonfinite gradients.
   Keep the value returned by `clip_grad_norm_`: the norm **before clipping**.
5. Update the weights with the optimizer.
6. Return `(loss, grad_norm)` as two Python floats.

### Checks

Test the model and training functions as you implement them:

```sh
python -m pytest assignment1/tests -q
```

The supplied tests check shapes, finite loss and whether a training step runs.
Then check that the model can learn one fixed batch: use one layer, width 16,
two heads, context 8 and two fixed sequences of nine token IDs. Use `make_targets`
to get inputs and targets, then repeat 100 updates on that same batch. The loss
should clearly decrease.

## 3. Check training locally

After implementing the five functions and passing the basic tests, run:

```sh
python assignment1/code/train.py --config assignment1/configs/laptop.json \
  --data data/a1/laptop --out runs/laptop-baseline --device cpu
```

The laptop configuration has two layers, width 128, four heads, context 128 and
500 updates: about one million parameters. Use it to
check that loss decreases, inspect outputs and try the modification before the
final runs. You can use
`--device mps` or `--device cuda` if available.

Start the larger model from random weights; the laptop checkpoint is for development.

## 4. Train the final baseline

`configs/final.json` uses eight layers, width 512, eight heads, context 512 and a
batch of 32 sequences: about 28 million parameters with the 4,096-token vocabulary.
Its 30,518 updates process **500,006,912 tokens**, including repeated samples.
This count is the training budget, not the number of distinct tokens in the dataset.

The default job uses one RTX 2080 Ti on `IT010146`, with float32 arithmetic.
Use it for both runs. If your modification needs more GPU memory, adapt the job
request and record the change in your report.

### Prepare final data

On your laptop, encode the full splits with the tokenizer you already trained:

```sh
python assignment1/code/tools.py prepare \
  --input data/a1/splits/train.txt --val-input data/a1/splits/val.txt \
  --out data/a1/final --tokenizer data/a1/tokenizer/tokenizer.json
```

`data/a1/laptop` is for local checks; `data/a1/final` is for both final runs.

### Copy to GSC

Complete the [GSC setup](../setup.md#5-set-up-your-files-and-environment-on-gsc1).
From your **local terminal**, in `practicals/`, copy the completed code and final
data. Replace `USERNAME` and `YOUR_REPOSITORY` with your GSC username and checkout
folder:

```sh
ssh USERNAME@l1.gsc1.uni-graz.at "mkdir -p /cl_tmp/USERNAME/llm-course/YOUR_REPOSITORY/practicals/data/a1"
scp -r assignment1 \
  USERNAME@l1.gsc1.uni-graz.at:/cl_tmp/USERNAME/llm-course/YOUR_REPOSITORY/practicals/
scp -r data/a1/final \
  USERNAME@l1.gsc1.uni-graz.at:/cl_tmp/USERNAME/llm-course/YOUR_REPOSITORY/practicals/data/a1/
```

### Start training

Connect to GSC and run these commands in the **GSC terminal**:

```sh
cd /cl_tmp/$USER/llm-course/YOUR_REPOSITORY/practicals
source .venv/bin/activate
mkdir -p logs
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
```

Check the job's logs and validation loss. A completed run has `step: 30518` and
`tokens: 500006912` in the last line of `metrics.jsonl`. Check this for both runs
before submitting. See [setup.md](../setup.md#7-submit-a-training-job) for monitoring.

| Run file | Contents |
|---|---|
| `metrics.jsonl` | Training and validation results |
| `best.pt` | Checkpoint with the lowest validation loss |
| `last.pt` | Most recent evaluated checkpoint |
| `run.json` | Model, training and environment settings |
| `source/` | Code used for that run |

The trainer logs loss and gradient norm every update. It evaluates validation
data before training, every **100 updates on laptop / 1,000 on final runs**, and
at the end. These intervals are set by `eval_every` in the configurations.

Each run starts from random weights and needs a new output directory.
If a job fails, fix the cause and start a new run. Keep the saved source for each version.

## 5. Implement and train one modification

An ablation compares the baseline with a version where **one feature** changes.
Choose one change, for example:

| Difficulty | Change |
|---|---|
| Easy | Replace SiLU with GELU in the gated feed-forward layer. |
| Medium | Give the input embedding and output projection separate weights. |
| Harder | Use grouped-query attention: several query heads share key/value heads. |

More ideas: [Raschka's architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

1. Write down the expected effect.
2. Implement and test the change with the laptop configuration.
3. Copy the updated `assignment1/` folder to GSC using the command above. Train
   the changed model from random weights with the final configuration and save
   it in `runs/a1-variant`.
4. Keep data, tokenizer, seed, training-token count, batch size, learning-rate
   schedule and evaluation fixed. Keep unchanged parts' initial weights the same
   where possible. Record any change in parameter count or computation.

For the activation modification, add `--activation gelu` to the training command
once your block implements both activations.

## 6. Compare the results

Generate loss plots from the saved logs:

```sh
python assignment1/code/tools.py plot-runs runs/a1-baseline runs/a1-variant \
  --out comparison.png
```

The plots show training loss, validation loss, validation bits per byte and the
gradient norm before clipping. Norms above `1.0` mean clipping was applied.

Report the **best validation bits per byte** for each run (lower is better):
total prediction loss in nats divided by `ln(2)` and the number of UTF-8 text bytes. This lets us compare
models using different tokenizers. The supplied evaluator counts every target
once and includes end-token loss; end tokens add no text bytes.

Use your validation set to compare your two models. The teaching team will
evaluate **both submitted models on a separate held-out dataset**, using bits
per byte. This dataset is not provided to students and is the same for everyone.

### Generate text (optional)

To inspect your models, generate text using each run's saved source:

```sh
python runs/a1-baseline/source/tools.py generate runs/a1-baseline/best.pt \
  --prompt "The purpose of a library is" --tokens 40 --seed 0 --device cpu
python runs/a1-variant/source/tools.py generate runs/a1-variant/best.pt \
  --prompt "The purpose of a library is" --tokens 40 --seed 0 --device cpu
```

You may try any prompt. Generated text is not a submission requirement.

### Optional: Weights & Biases

[Weights & Biases](https://docs.wandb.ai/) displays training curves in a web
dashboard. It is **optional**; the local logs, plotting script and report are sufficient.

To use it, install `wandb`, run `wandb login`, then add
`--wandb-project YOUR_PROJECT` to the training command. Local logs are saved either way.

## Deliverables

### Tests

Add **three tests** to `assignment1/tests/test_student.py`. Test model or training
behaviour, including your modification where useful. Tests can also cover the tokenizer.

Each test must check behaviour beyond the supplied tests. Choose inputs with an
expected result you can explain. For each test, describe a bug it catches and
something it would miss.

For one test, demonstrate that it catches a bug: introduce the bug, show the
failure, fix it and show the passing result. Submit the corrected code, the bug's
code change and both test outputs.

### Files and report

- [ ] Source code and commands to run it.
- [ ] Three tests, explanations and the bug demonstration.
- [ ] Both model checkpoints, tokenizer, settings and logs, uploaded to Hugging Face.
- [ ] The Hugging Face link, full revision ID and upload receipt in the GitHub repository.
- [ ] Report of **at most two pages**, containing:
  - Training and validation loss plots comparing the two runs.
  - A table of vocabulary size, parameter count, total trained tokens, best
    validation bits per byte, full-run runtime and device for each model.
  - Validation fraction, split seed and resulting document counts; two sentences
    explaining your choice of validation size.
  - Expected effect of the modification, results and conclusion.
  - A brief account of agent use and sources.

Test explanations may go in separate files outside the page limit.

### Submission

Hand in one **GitHub repository** containing your report, source code, commands,
tests and bug demonstration. Its README must explain how
to run the code and link to the uploaded models.

Upload both models to one **public Hugging Face model repository** using the
[submission instructions](submission.md). The supplied script stores the baseline
and modified model in separate `baseline/` and `variant/` directories. Keep the
large checkpoint files on Hugging Face rather than in GitHub.

Include the Hugging Face repository link, full revision ID and upload receipt
(`a1-receipt.json`) in your GitHub repository. The revision ID identifies the exact
uploaded version to grade.

Email the GitHub link to **lorenz.prattes@uni-graz.at** and
**ruggero.lazzaroni@uni-graz.at** by the deadline, with the subject
**final submission: assignment 1**. Ensure both instructors can access the repository.

## 7. Explanation

Be ready to:

1. Explain the train/validation split and how token IDs become model inputs and targets.
2. Explain one test and demonstrate how it catches a bug.
3. Run one optimizer update on a tiny CPU batch. Show inputs, next-token targets,
   logits, loss, and one parameter's gradient and value before and after the update.
4. Explain gradient clearing, clipping and how AdamW updates the weights.
5. Explain the ablation results.
