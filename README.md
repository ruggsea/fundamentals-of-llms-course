# Fundamentals of LLMs

Winter semester 2026/27 · Uni Graz · [Course registration](https://online.uni-graz.at/kfu_online/wbLv.wbShowLVDetail?pStpSpNr=1005778&pSpracheNr=1) · [Course page](https://cs2.uni-graz.at/teaching/fundamentals-of-llms/)

Slides: https://ruggsea.github.io/fundamentals-of-llms-course/ · Reference code: [`code/`](code/)

## Overview

Build, train and evaluate a language model, from n-grams to transformers. The course covers tokenization, attention, training data, inference, post-training, reasoning and tool use. Two individual projects put this into practice.

You need solid Python: functions, classes, file I/O and debugging. Prior machine learning or NLP experience helps but is not required; allow extra time in the first two weeks if these are new to you.

An elective for the Data Science and Mathematics master's programmes.

| | |
|---|---|
| When | Thursdays, 15:15–17:00 |
| Where | SR 127.11 · IDea_Lab, Leechgasse 34 |

## Schedule

Slides and reading lists will be added as we work through them.

| # | date | topic | slides |
|---|---|---|---|
| 01 | 08 Oct | The guessing game. | [slides](https://ruggsea.github.io/fundamentals-of-llms-course/slides/?deck=lectures/01-guessing-game.md) |
| 02 | 15 Oct | Tokenizers. | after the session |
| 03 | 22 Oct | Attention, by hand. | after the session |
| 04 | 29 Oct | The transformer block. | after the session |
| 05 | 05 Nov | Training. Assignment 1 released. | after the session |
| 06 | 12 Nov | Data cleaning. | after the session |
| 07 | 19 Nov | Data mixing. | after the session |
| 08 | 26 Nov | Inference. | after the session |
| 09 | 03 Dec | Evaluation. Room: SR 15.05 (room change). | after the session |
| 10 | 10 Dec | Post-training. | after the session |
| | 16–18 Dec | Assignment 1 defences, bookable slots | |
| 11 | 17 Dec | DPO & GRPO. | after the session |
| 12 | 07 Jan | Reasoning. | after the session |
| 13 | 14 Jan | Agents & tools. | after the session |
| 14 | 21 Jan | Reading a technical report. | after the session |
| 15 | 28 Jan | Project feedback. Last session. | after the session |
| | 3–5 Feb | Assignment 2 defences, bookable slots | |

## Reading material

The reading material from the slides, per lecture.

**01 · The guessing game** (08 Oct)

- Shannon 1951, [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), sections 1–3.
- Shannon 1948, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), section 3 only.
- Ben Hoyt, [Markov chain in 20 lines of Python](https://benhoyt.com/writings/markov-chain/). Run it on a book you like.
- Cameron Wolfe, [Language Model Training and Inference: From Concept to Code](https://cameronrwolfe.substack.com/p/language-model-training-and-inference).
- Stanford CS336, [lecture 1](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py), first half.

Additional resources and reading material will be posted in the Discord.

## Contact

For general questions about the course, ask Lorenz. If you have specific requests, you can also contact Ruggero. Addresses are on the [course page](https://cs2.uni-graz.at/teaching/fundamentals-of-llms/).

The course communication channel will be sent out via email. For illness or other circumstances affecting a deadline or defence, contact us promptly and provide the relevant documentation.

## Assessment

Two individual assignments, each followed by a defence. Active attendance is required.

| Component | Weight | Date |
|---|---|---|
| Assignment 1 | 30% | Wednesday 9 December 2026, 23:59 |
| A1 defence | 10% | 16–18 December 2026 |
| Assignment 2 | 40% | Monday 15 February 2027, 23:59 — proposed |
| A2 defence | 20% | 3–5 February 2027 |

### Grading scheme

The final grade is based on the weighted total of the four components above. Passing scores are divided into four equal bands.

| Weighted total | Grade |
|---|---|
| 87.5–100% | 1 |
| 75–<87.5% | 2 |
| 62.5–<75% | 3 |
| 50–<62.5% | 4 |
| <50% | 5 |

Insufficient understanding of the submitted work demonstrated in a defence can lead to an overall negative grade.

The A2 report deadline is not yet confirmed. The A2 code and results deadline and the submission channel will be announced separately. All times are local to Graz.

## Using AI

Working with generative AI, especially coding agents, is encouraged in this course. You should think of AI as a tool that can help you achieve bigger, more interesting tasks. The tasks at hand do however require deep understanding of the material, even with the help of AI agents. Therefore you will have to find ways to balance vibecoding and personal engagement with content and code. Additionally, you will need to demonstrate your understanding to us in the defences.

Briefly document which tools you used and for what.

## Assignments

### A1 · Train a small GPT

Train and evaluate a small language model. Details will be added here shortly.

### A2 · Post-train and evaluate

Post-train a model and compare it with the unchanged base model. Details will be added here shortly.

## Defences

Each assignment has an individual defence. You should be able to explain your work and results.

Slots will be available for booking within the specified time windows. Booking and preparation details will follow.

## Compute

You will get access to the GSC cluster for the course. Access instructions and the compute allowance will be announced here.

## Reference code

[`code/`](code/): a small GPT from tokenizer to DPO, runs on a laptop CPU. Setup and usage in [`code/README.md`](code/README.md).
