# Build a small GPT, start to finish

This folder is the whole course in eight Python files. You train a byte-level BPE tokenizer, encode a corpus,
train a small GPT on it, measure it, teach it a chat format and then a preference. Everything runs on a laptop
CPU in about four minutes.

The code is short on purpose. Every file fits on a few screens, and every function does one thing the slides
explain. Read a file next to its lecture.

## Setup, once

```bash
cd code
uv venv .venv && uv pip install --python .venv/bin/python torch numpy
source .venv/bin/activate
```

Torch and numpy are the only dependencies. The corpus, `data/tiny.txt`, is the first megabyte of TinyStories
(Eldan and Li 2023): 1,258 short stories written for small models. It is in the repo, so nothing needs the
network after setup. If you delete it, `python tokenizer.py` downloads it again.

## Run it, in this order

| step | command | what it does | time on an M4 laptop |
|---|---|---|---|
| 0 | `python tests.py` | eleven fast checks; run them first | 3 s |
| 1 | `python tokenizer.py` | trains 1,791 BPE merges, writes `data/tokenizer.json` | 4 s |
| 2 | `python data.py` | encodes the stories into `data/train.bin` and `data/val.bin` | 3 s |
| 3 | `python train.py` | pretrains the GPT for 2M tokens, writes `runs/example/` | 3 min |
| 4 | `python eval.py` | bits per byte, perplexity, a 40-item multiple-choice test | 3 s |
| 5 | `python sft.py` | adds chat tokens, fine-tunes on 30 question-answer pairs | 7 s |
| 6 | `python dpo.py` | DPO on 20 preference triples | 20 s |

The whole sequence took 247 s from an empty `runs/` folder. The numbers below are from that run. With the same
seed you should get the same numbers on a CPU; a GPU will differ in the last digits.

## The files

| file | lines | what it is | lectures |
|---|---|---|---|
| `tokenizer.py` | 120 | byte-level BPE: `train`, `encode`, `decode`, special tokens, save and load | 2, and 10 for the chat tokens |
| `data.py` | 72 | text to a uint16 token file, `<\|endoftext\|>` between stories, `get_batch` with the shift by one | 5, 6 |
| `model.py` | 160 | RMSNorm, attention with RoPE, SwiGLU, the block, `GPT` with `generate` and `logprob`, `count_params` | 3, 4, 8 |
| `train.py` | 137 | AdamW, warmup then cosine or WSD, clipping, token budget, held-out loss, checkpoints, `metrics.jsonl` | 5 |
| `eval.py` | 89 | held-out bits per byte and perplexity; multiple choice scored two ways, bootstrap CI | 9 |
| `sft.py` | 76 | chat template, new rows in the embedding, loss on the answer tokens only | 10 |
| `dpo.py` | 74 | DPO against a frozen reference, the implicit reward margin | 11 |
| `tests.py` | 125 | what each step must get right, as plain asserts | all |

Data files: `data/tiny.txt` (the corpus), `data/mc.jsonl` (40 multiple-choice items, three options each),
`data/sft.jsonl` (30 question-answer pairs, half about this course), `data/dpo.jsonl` (20 prompt, chosen,
rejected triples). The scripts write `data/tokenizer.json`, `data/meta.json` and the two `.bin` files.

## What each step prints

**Tests.** All eleven print `PASS`. `pytest tests.py` also works if you have pytest. The first test trains BPE
on lecture 2's toy corpus (`low` ×5, `lower` ×2, `widest` ×3, `newest` ×6) and checks the twelve merges against
the CS336 handout's list: `s t, e st, o w, l ow, w est, n e, ne west, w i, wi d, wid est, low e, lowe r`.

**Tokenizer.** The merges come from the stories, so the first ones are English fragments.

```
1791 merges. The first ten: [b'he', b' t', b' a', b' s', b' w', b'nd', b' the', b'ed', b' to', b' b']
41 bytes, 11 tokens: ['Once', ' upon', ' a', ' time', ',', ' there', ' was', ' a', ' little', ' dog', '.']
```

**Data.** The last 10% of stories are held out. One token covers 3.7 bytes of text.

```
train: 1133 documents, 239,791 tokens, 890,183 bytes, 3.71 bytes per token
val: 125 documents, 23,934 tokens, 89,916 bytes, 3.76 bytes per token
```

**Training.** The model has 1,115,264 parameters: 4 layers, 4 heads, width 128, windows of 128 tokens,
vocabulary 2,048. It sees 976 batches of 16 windows, 1,998,848 tokens in total. A line every 100 steps:

| step | tokens | learning rate | train loss | held-out loss | held-out bits per byte |
|---|---|---|---|---|---|
| 0 | 0 | | | 7.660 | 2.94 |
| 100 | 204,800 | 0.00298 | 4.379 | 4.247 | 1.63 |
| 300 | 614,400 | 0.00254 | 3.490 | 3.443 | 1.32 |
| 500 | 1,024,000 | 0.00171 | 2.803 | 3.177 | 1.22 |
| 800 | 1,638,400 | 0.00053 | 2.670 | 2.960 | 1.14 |
| 976 | 1,998,848 | 0.00030 | 2.482 | 2.962 | 1.14 |

At step 0 the held-out loss is 7.66 nats, and ln 2,048 = 7.62: the untrained model spreads its guesses evenly.
From step 500 on, training loss falls faster than held-out loss. The budget is 8.3 passes over 240k tokens,
so the model starts to memorise its training stories. The full log is in `runs/example/metrics.jsonl`
(untracked). Then it writes:

```
Once upon a time, there was a poppy named Jane. De loved to go on a piece of different things. One day, Jony
had a big bowl and it was going to find something new. She was very big and didn't know what to do.
```

**Evaluation.** Bits per byte here counts every held-out token once, so it differs a little from the random
windows in the training log. Two more seeds (`python train.py --seed 1 --out runs/seed1`, same for 2) give the
spread:

```
python eval.py runs/example/ckpt.pt runs/seed1/ckpt.pt runs/seed2/ckpt.pt

runs/example/ckpt.pt: held-out bits per byte 1.1413, perplexity 19.53 per token, multiple choice 0.600 by likelihood, 0.225 by generation
runs/seed1/ckpt.pt: held-out bits per byte 1.1463, perplexity 19.79 per token, multiple choice 0.625 by likelihood, 0.225 by generation
runs/seed2/ckpt.pt: held-out bits per byte 1.1421, perplexity 19.57 per token, multiple choice 0.650 by likelihood, 0.200 by generation

40 items, chance 0.333, n_seeds=3
log-likelihood    accuracy 0.625  95% CI [0.483, 0.758]  (n_seeds=3)
generate + regex  accuracy 0.217  95% CI [0.108, 0.342]  (n_seeds=3)
```

The same model, on the same 40 items, is above chance when you compare the probabilities of the three answers
and below chance when you let it write and look for an answer in its text. In 31 of the 40 items its 8 greedy tokens
contain none of the three words. That is lecture 9's point about scoring rules, in one run.

**SFT.** The template is `<|user|>question<|assistant|>answer<|endoftext|>`. The two new tokens get two new
rows in the embedding table. Only the 586 answer tokens count in the loss; the prompt tokens are masked.

```
step  25  answer loss 0.446
step 100  answer loss 0.003
'What color is the sky?'     -> 'The sky is blue.<|endoftext|>'
'What does BPE do?'          -> 'BPE finds the pair of tokens that comes together most often and merges it into one new token.<|endoftext|>'
'What color is the sun?'     -> 'The sun is very hot.<|endoftext|>'
'What is a dog?'             -> 'A dog has four legs.<|endoftext|>'
```

The first two questions are in the training pairs. The last two are not, and the model answers with the
closest training answer it has. With 30 examples and a million parameters, SFT teaches the format and a
lookup table, nothing more.

**DPO.** β = 0.1, 50 steps over all 20 triples, starting from the SFT model, which is also the frozen
reference.

```
step  1  loss 0.6931  reward margin +0.000  chosen ahead in 0% of pairs
step 10  loss 0.2003  reward margin +2.116  chosen ahead in 100% of pairs
step 50  loss 0.0176  reward margin +4.763  chosen ahead in 100% of pairs
log p(chosen) summed: -297.6 -> -225.5;  log p(rejected): -610.1 -> -1500.0
```

At step 1 the policy is the reference, so every margin is exactly 0 and the loss is ln 2 = 0.693. Most of the
margin comes from pushing the rejected answers down, not from pulling the chosen ones up.

## Things to try

- `python train.py --schedule wsd --out runs/wsd`: the learning rate stays flat until 80% of the budget,
  then decays. Compare the two `metrics.jsonl` files.
- `python train.py --tokens 4000000 --out runs/long`: double the budget. Does held-out loss still fall?
- `python train.py --resume`: continue a run from its last checkpoint.
- In `model.py`, `count_params(50257, 12, 12, 768, 1024, style="gpt2")` returns lecture 4's 124,439,808,
  GPT-2 small with its tied output head. The same config in our modern block, `style="modern"`, gives
  123,551,232: no biases, no position table, a SwiGLU of width 2,048.
- On a GPU, the same commands use CUDA and bf16 autocast. Raise `--d_model`, `--n_layer` and `--tokens`
  together.

## Where this closes the gaps in the student log

`slides/review/student.md` lists five places where the slides stop before the code runs. Each has a file.

1. The model in torch: `model.py`, with the heads glued back together and `W_O` mapping back to width d.
2. Text to batches: `data.py`, with `<|endoftext|>` between documents and `get_batch` doing the shift.
3. A training script that runs: `train.py`, with the schedule written out, held-out loss and checkpoints.
4. Generation and scoring for your own model: `GPT.generate` and `GPT.logprob` in `model.py`.
5. Post-training plumbing: the chat tokens in `tokenizer.py`, `resize_vocab` and the prompt mask in `sft.py`,
   the reference model and answer log-probs in `dpo.py`.
