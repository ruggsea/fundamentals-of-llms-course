# Fundamentals of LLMs

Winter semester 2026/27 · Uni Graz · [Course registration](https://online.uni-graz.at/kfu_online/wbLv.wbShowLVDetail?pStpSpNr=1005778&pSpracheNr=1) · [Course page](https://cs2.uni-graz.at/teaching/fundamentals-of-llms/)

Slides: https://ruggsea.github.io/fundamentals-of-llms-course/

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
| 02 | 15 Oct | Tokenizers. Ruggero's hour, then Lorenz on agents. | [slides](https://ruggsea.github.io/fundamentals-of-llms-course/slides/?deck=lectures/02-tokenizers.md) · [Agentic coding slides](https://ruggsea.github.io/fundamentals-of-llms-course/agentic-coding/slides/) · [Getting started with coding agents](https://github.com/ruggsea/fundamentals-of-llms-course/blob/main/agentic-coding/getting-started.md) |
| 03 | 29 Oct | Attention, by hand. | [slides](https://ruggsea.github.io/fundamentals-of-llms-course/slides/?deck=lectures/03-attention.md) |
| 04 | 05 Nov | The transformer block. | after the session |
| 05 | 12 Nov | Training. Assignment 1 released. | after the session |
| 06 | 19 Nov | Data cleaning. | after the session |
| 07 | 26 Nov | Data mixing. | after the session |
| 08 | 03 Dec | Inference. Room: SR 15.05 (room change). | after the session |
| 09 | 10 Dec | Evaluation. | after the session |
| | 16–18 Dec | Assignment 1 defences, bookable slots | |
| 10 | 17 Dec | Post-training. | after the session |
| 11 | 07 Jan | DPO & GRPO. | after the session |
| 12 | 14 Jan | Reasoning. | after the session |
| 13 | 21 Jan | Agents & tools. | after the session |
| 14 | 28 Jan | Reading a technical report. Last session. | after the session |
| | 3–5 Feb | Assignment 2 defences, bookable slots | |

## Reading material

The reading material from the slides, per lecture.

**01 · The guessing game** (08 Oct)

- Shannon 1951, [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), sections 1–3.
- Shannon 1948, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), section 3 only.
- Ben Hoyt, [Markov chain in 20 lines of Python](https://benhoyt.com/writings/markov-chain/). Run it on a book you like.
- Cameron Wolfe, [Language Model Training and Inference: From Concept to Code](https://cameronrwolfe.substack.com/p/language-model-training-and-inference).
- Stanford CS336, [lecture 1](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py), first half.

**02 · Tokenizers** (15 Oct)

- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). Builds a production-style tokenizer from scratch; the first hour covers the material of this lecture. A [text version](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html) by fast.ai covers the same content.
- Sennrich, Haddow, Birch 2016, [Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), sections 1–3.
- Sebastian Raschka, [BPE tokenizer from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb). The same algorithm, implemented step by step in a notebook.
- Radford et al. 2019, [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), section 2.2. The byte-level BPE that GPT-2 uses.
- Stanford CS336, [assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics), section 2. Assignment 1 is to train a small language model; this is the tokenizer part of it.
- Gage 1994, [A New Algorithm for Data Compression](https://www.cs.cmu.edu/~wcohen/10-605/papers/bpe.pdf). The data-compression paper BPE started as.
- janus, [SolidGoldMagikarp (plus prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLYq/solidgoldmagikarp-plus-prompt-generation-or-glossary), 2023. What happens when a vocabulary token is never activated during training.
- Beren Millidge, [Integer tokenization is insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/), 2023. How the GPT-2/3 tokenizers split integers, and why that hurts arithmetic.
- Yu et al. 2023, [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185). Modelling raw bytes directly, without a tokenizer.
- Andrej Karpathy, [minbpe](https://github.com/karpathy/minbpe), 2024. A minimal reference implementation of the algorithm, with exercises.
- Tao et al. 2024, [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies](https://arxiv.org/abs/2407.13623). How to choose the vocabulary size jointly with model and data size.
- [tiktokenizer](https://tiktokenizer.vercel.app/). Type text, see how different OpenAI tokenizers cut it.

**03 · Attention, by hand** (29 Oct)

- Grant Sanderson (3Blue1Brown), [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention). Queries, keys, masking and heads, with pictures; there is a video and a text version.
- Sebastian Raschka, [LLMs from scratch, chapter 3](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb). The attention of this lecture, implemented step by step in a notebook.
- Dan Jurafsky and James Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3rd edition draft, ch. 7, "Transformers and Pretraining". Free online. The same climb from a weighted sum, and historical notes on where the query, key and value names come from.
- Bahdanau, Cho, Bengio 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), sections 1 to 3. Where attention was first proposed, for translation.
- Vaswani et al. 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.2. The paper that introduced the transformer; this section defines scaled dot-product and multi-head attention.
- Sutskever, Vinyals, Le 2014, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215). The one-vector translation model attention was built to fix.
- Peter Bloem, [Transformers from scratch](https://peterbloem.nl/blog/transformers). Self-attention built up from a plain weighted sum, with PyTorch code.
- Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/). GPT-2, piece by piece: the token embeddings, and the attention we computed today, at GPT-2's real size.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/). GPT-2 running in the browser; type a prompt and look at its attention.
- Callum McDougall, [Induction heads, illustrated](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated), 2023. Two heads working together to copy text, step by step in diagrams.
- Adam Casson, [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops). Where the compute of a transformer goes, and when attention's share grows.

Additional resources and reading material will be posted in the Discord.

## Contact

For general questions about the course, ask Lorenz (lorenz.prattes@uni-graz.at). If you have specific requests, you can also contact Ruggero (ruggero.lazzaroni@uni-graz.at).

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

To set up a coding agent, follow [Getting started with coding agents](https://github.com/ruggsea/fundamentals-of-llms-course/blob/main/agentic-coding/getting-started.md). It covers Pi with GitHub Copilot Student or OpenCode Go. The [agentic coding slides](https://ruggsea.github.io/fundamentals-of-llms-course/agentic-coding/slides/) from lecture 02 explain how these agents work and where they fail.

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

Code and setup instructions will be added here.
