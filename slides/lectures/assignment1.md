<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Fundamentals of LLMs · Assignment 1</p>

# Build and train a small language model

</div>
<div class="pic" style="background-image:url(assets/babbage.jpg)"></div>

---

## Data and tokenizer

Split the course data into training and validation sets.

Train your own **4,096-token BPE tokenizer** on the training split using the supplied code.

Use it to encode small subsets for laptop checks and the full splits for final training.

---

## Implementation

Complete five functions in `model.py` and `train.py`:

| Function | Task |
|---|---|
| `Attention.forward` | Causal multi-head attention |
| `Block.forward` | Transformer block |
| `GPT.forward` | Token scores and loss |
| `make_targets` | Next-token targets |
| `train_step` | One optimizer update |

---

## Laptop checks

Pass the supplied tests and check that the model can learn one fixed batch.

Train the small model (**2 layers, ~1M parameters**) for 500 updates. Inspect the loss and generated text, then test your modification.

---

## Final training on GSC

After implementing locally, copy your code and full prepared data to GSC and set up its Python environment.

Train two models, each on **one GPU for 500 million tokens**:

| Run | Model |
|---|---|
| Baseline | 8 layers, ~28M parameters |
| Modification | Same configuration with one feature changed, e.g. SiLU → GELU |

Keep the data, tokenizer, seed and training schedule fixed.

---

## Tests and comparison

Add **three tests** and explain how it works.

Compare loss curves, best validation bits per byte and runtime. Explain whether the modification had its expected effect.


---

## Submission

Put the **report (at most two pages)**, code, commands and tests in a **GitHub repository**.

Upload both models to **one public Hugging Face repository**. Include its link, revision ID and receipt in GitHub.

Follow the [assignment checklist](../practicals/assignment1/assignment.md#deliverables) for the report and supporting files.

---

## Interview

Explain your data split, demonstrate a test and discuss the model comparison.

Run one optimizer update on a tiny CPU batch, showing inputs, targets, token scores, loss, a gradient and how a parameter changes.

---

## Dates and hand-in

**Deadline:** Sunday 13 December 2026, 23:59, Graz time.

**Interviews:** 16–18 December, individual 20-minute slots.

Email the GitHub URL to **both** lorenz.prattes@uni-graz.at and ruggero.lazzaroni@uni-graz.at.

Subject: **`final submission: assignment 1`**

---

## Instructions

- [Assignment sheet](../practicals/assignment1/assignment.md)
- [Setup and cluster access](../practicals/setup.md)
- [Submission instructions](../practicals/assignment1/submission.md)
