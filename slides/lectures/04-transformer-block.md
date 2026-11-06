<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Advanced Machine Learning · Lecture 4 · 5 Nov 2026</p>

# One block, by hand

<p class="tagline">Five words, two numbers, and every weight written out.</p>
<p class="kicker">Jana Lasser · Ruggero Marino Lazzaroni · Lorenz Prattes<br>Uni Graz · Winter 2026/27 · DAT.C3111UB</p>
</div>
<div class="pic" style="background-image:url(assets/babbage.jpg)"></div>

---

<p class="kicker">The sentence</p>

## The five words.

<div class="quote">the dog chased the cat</div>

<p class="shape">every figure from here on draws these five words</p>

---

<p class="kicker">The order</p>

## Six steps, in this order.

1. **Project.** Each word's arrow × `Wq`, `Wk`, `Wv` gives a query, a key and a value.
2. **Rotate.** Turn the query and the key by the word's position. Not the value.
3. **Score.** One dot product per pair of words → one number per pair.
4. **Share.** Softmax turns each row of scores into weights that add to 1.
5. **Mix.** Each word's output is the value vectors, shortened by those weights, added up.
6. **Project again.** One more matrix, `W_O`, puts that back into the word's own numbers.

Note: This is the map. Slides 5 to 22 are these six steps one at a time, then the residual, the norm and the MLP, which wrap around them.

---

<p class="kicker">1 · Tokens</p>

## Words.

<div class="tokline"><span class="tok">the</span><span class="tok">dog</span><span class="tok">chased</span><span class="tok">the</span><span class="tok">cat</span></div>

We cut on spaces. Real tokenizers do something smarter; that was lecture 2.

---

<p class="kicker">1 · Tokens</p>

## Ids.

```python run
WORDS = ["the", "dog", "cat", "chased", "sat", "old"]
at = {w: i for i, w in enumerate(WORDS)}

sentence = ["the", "dog", "chased", "the", "cat"]
ids = [at[w] for w in sentence]
print(ids, len(ids))
```

<pre class="output fragment">[0, 1, 3, 0, 2] 5</pre>

<p class="shape">ids · shape (5,) · five integers, one per word</p>

---

<p class="kicker">2 · Embeddings</p>

## Each word is two numbers.

| word | dim 1: carries content | dim 2: needs content |
|---|---|---|
| the | 0.00 | 1.00 |
| dog | 1.00 | 0.00 |
| cat | 0.80 | 0.20 |
| chased | 0.10 | 1.00 |
| sat | 0.00 | 0.90 |
| old | 0.60 | 0.40 |

<p class="shape">E · shape (6, 2) · one row per word, chosen by hand, nothing trained</p>

Note: Nouns carry content, articles and verbs need content. That is the whole design and it is what makes the attention weights readable later.

---

<p class="kicker">2 · Embeddings</p>

## Looking the ids up in that table.

```python run
import numpy as np
E = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2],
              [0.1, 1.0], [0.0, 0.9], [0.6, 0.4]])
X = E[ids]
print(X.shape)
print(X)
```

<pre class="output fragment">(5, 2)
[[0.  1. ]
 [1.  0. ]
 [0.1 1. ]
 [0.  1. ]
 [0.8 0.2]]</pre>

<p class="shape">X · shape (5, 2) · five arrows, two numbers each</p>

---

<p class="kicker">2 · Embeddings</p>

## The five arrows.

<img class="fig" src="figures/04_a_space.png" alt="the six word embeddings as arrows in a plane">

<p class="shape">red: the words in the sentence · grey: the words we are not using</p>

Nothing in `X` says which word came first. That is the next problem.

---

<p class="kicker">3 · Q, K, V</p>

## Three matrices, three roles.

<div class="mrow">
  <span class="mshape">Wq (2, 2)</span>
  <span class="m c2"><span>0.0</span><span>0.0</span><span>4.0</span><span>0.0</span></span>
  <span class="mshape">query: what this word is asking for</span>
</div>
<div class="mrow">
  <span class="mshape">Wk (2, 2)</span>
  <span class="m c2"><span>1.0</span><span>0.0</span><span>0.0</span><span>0.0</span></span>
  <span class="mshape">key: what this word offers as a match</span>
</div>
<div class="mrow">
  <span class="mshape">Wv (2, 2)</span>
  <span class="m c2"><span>1.0</span><span>0.0</span><span>0.0</span><span>1.0</span></span>
  <span class="mshape">value: what this word hands over</span>
</div>

```python run
Wq = np.array([[0.0, 0.0], [4.0, 0.0]])     # (2, 2): 2 numbers in, 2 out
Wk = np.array([[1.0, 0.0], [0.0, 0.0]])     # (2, 2): 2 in, 2 out
Wv = np.eye(2)                              # (2, 2): the value IS the word's arrow
Q, K, V = X @ Wq, X @ Wk, X @ Wv
print(K.round(2))
```

<pre class="output fragment">[[0.  0. ]
 [1.  0. ]
 [0.1 0. ]
 [0.  0. ]
 [0.8 0. ]]</pre>

<p class="shape">Q, K, V · shape (5, 2) each · same shape in, same shape out · not rotated yet</p>

---

<p class="kicker">4 · RoPE</p>

## Rotate the query and the key by the word's position.

```python run
def turn(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)

probe = np.array([[1.0, 0.0]])
print(turn(probe, 0.5).round(3), turn(probe, 1.5).round(3))
```

<pre class="output fragment">[[0.878 0.479]] [[0.071 0.997]]</pre>

<img class="fig" src="figures/04_a_rotate.png" alt="one arrow turned by 0 to 5 positions">

<p class="shape">one radian pair per position · the arrow keeps its length, only the angle changes</p>

Note: Half a radian per position here. The formula gives one radian at width two, which turns too fast for a five-word sentence. Real models rotate 64 pairs at 64 different speeds.

---

<p class="kicker">4 · RoPE</p>

## Only the distance between two words survives the rotation.

<img class="fig" src="figures/04_a_gap.png" alt="two pairs of arrows with the same gap and the same score">

<p class="shape">left: positions 1 and 4 · right: positions 2 and 5 · gap 3 both times, same dot product</p>

Two words at positions 105 and 102 get exactly the score they would get at 5 and 2. The absolute position
disappears; only the difference is left.

---

<p class="kicker">4 · RoPE</p>

## What it does not touch.

- **Not the embeddings.** `X` is left alone. RoPE happens inside every attention layer, not at the bottom of the model.
- **Not the values.** Only the query and the key are turned. The value is what actually gets passed on, so it stays untwisted.
- **Not the length of anything.** A rotation changes the angle and nothing else.
- **Not a single learned weight.** The angles are fixed numbers, not parameters.

---

<p class="kicker">4 · RoPE</p>

## Turning q and k, at their positions.

```python run
pos = np.arange(len(X)) * 0.5
Q, K = turn(Q, pos), turn(K, pos)
print(pos.round(2))
print(K.round(2))
```

<pre class="output fragment">[0.  0.5 1.  1.5 2. ]
[[ 0.    0.  ]
 [ 0.88  0.48]
 [ 0.05  0.08]
 [ 0.    0.  ]
 [-0.33  0.73]]</pre>

<p class="shape">Q and K · shape (5, 2) · V is untouched</p>

---

<p class="kicker">5 · Scores</p>

## One dot product per pair of words.

```python run
scores = Q @ K.T / np.sqrt(2)               # (5, 5): one number per PAIR of words
print(scores.shape)
print(scores.round(2))
```

<pre class="output fragment">(5, 5)
[[ 0.    2.48  0.15  0.   -0.94]
 [ 0.    0.    0.    0.    0.  ]
 [ 0.    2.48  0.28  0.    1.22]
 [ 0.    1.53  0.25  0.    1.99]
 [ 0.    0.04  0.03  0.    0.45]]</pre>

<p class="shape">scores · shape (5, 5) · row = the word asking, column = the word answering</p>

Row `chased` is a word that needs something, and its biggest number is 2.48, next to `dog`. Row `dog` is all
zeros, because a noun has no query.

Note: Dividing by the square root of the key width is not decoration. Dot products grow with the number of dimensions, and a large score makes softmax put almost everything on one word, which flattens the gradient. At width two it is the square root of two. Say it in one line and move on.

---

<p class="kicker">6 · Mask</p>

## A word may not look forward.

```python run
future = np.triu(np.ones((5, 5), bool), 1)
print(future.astype(int))
scores = np.where(future, -np.inf, scores)
```

<pre class="output fragment">[[0 1 1 1 1]
 [0 0 1 1 1]
 [0 0 0 1 1]
 [0 0 0 0 1]
 [0 0 0 0 0]]</pre>

<p class="shape">scores · shape (5, 5) · the upper triangle becomes minus infinity</p>

Minus infinity, not zero. Zero is a score that the next step could still give weight to.

---

<p class="kicker">7 · Softmax</p>

## Each row becomes weights that add to one.

```python run
e = np.exp(scores - scores.max(-1, keepdims=True))
W = e / e.sum(-1, keepdims=True)
print(W.round(2))
print(W.sum(-1).round(2))
```

<pre class="output fragment">[[1.   0.   0.   0.   0.  ]
 [0.5  0.5  0.   0.   0.  ]
 [0.07 0.84 0.09 0.   0.  ]
 [0.13 0.58 0.16 0.13 0.  ]
 [0.18 0.18 0.18 0.18 0.28]]
[1. 1. 1. 1. 1.]</pre>

<p class="shape">W · shape (5, 5) · every row adds to 1</p>

---

<p class="kicker">7 · Softmax</p>

## The verb's row: 84% of it goes to its subject.

<img class="fig" src="figures/04_a_weights.png" alt="the attention row of the verb">

<p class="shape">W[2] · shape (5,) · 0.84 on "dog", then almost nothing</p>

---

<p class="kicker">8 · Weighted sum</p>

## Each word pulls in the values, shortened by its weights.

<img class="fig" src="figures/04_a_weighted.png" alt="the weighted sum of value arrows">

<p class="shape">W (5, 5) @ V (5, 2) → (5, 2) · five arrows in, five arrows out</p>

---

<p class="kicker">9 · W_O</p>

## The keys and queries are gone. One matrix puts the result back.

```python run
glued = np.array([2.0, 3.0])                   # head 1 found 2.0, head 2 found 3.0
print(glued.sum())                             # adding: 5.0, the heads are gone
print(glued @ np.eye(2))                       # identity: [2. 3.], kept apart
mix = np.array([[1.0, 1.0], [1.0, -1.0]])      # a W_O that mixes
print(glued @ mix)                             # [5. -1.]: both heads in both
```

<pre class="output fragment">5.0
[2. 3.]
[ 5. -1.]</pre>

<p class="shape">W_O (d, d) · column j is the recipe for output number j</p>

With one head this model uses the identity, so `W_O` changes nothing here. It earns its place when there are
several heads, because it is the only matrix that can mix them.

---

<p class="kicker">9 · W_O</p>

## The same step in this model.

```python run
Wo = np.eye(2)                              # (2, 2): one head, so nothing to mix
y = (W @ V) @ Wo
print(y.shape)
print(y.round(2))
```

<pre class="output fragment">(5, 2)
[[0.   1.  ]
 [0.5  0.5 ]
 [0.85 0.16]
 [0.6  0.42]
 [0.43 0.59]]</pre>

<p class="shape">y · shape (5, 2) · the same shape as X, on purpose</p>

Look at row `chased`: 0.84 of `dog` has been copied into it.

---

<p class="kicker">10 · Residual</p>

## The word's own arrow is kept and added to.

<img class="fig" src="figures/04_a_residual.png" alt="adding the attention update to the word's own arrow">

<p class="shape">h = x + Δattention · shape (5, 2) · nothing is replaced</p>

```python run
h = X + y
print(X[2], y[2].round(2), h[2].round(2))
```

<pre class="output fragment">[0.1 1. ] [0.85 0.16] [0.95 1.16]</pre>

---

<p class="kicker">11 · Norm</p>

## Divide each arrow by its own length.

<img class="fig" src="figures/04_a_circle.png" alt="arrows before and after the norm">

<p class="shape">h (5, 2) → (5, 2) · five different lengths become one</p>

```python run
g1 = np.ones(2)
unit = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)
n = unit * g1
print(np.linalg.norm(h, axis=1).round(2))
print(np.linalg.norm(n, axis=1).round(2))
```

<pre class="output fragment">[2.   1.58 1.5  1.54 1.46]
[1.41 1.41 1.41 1.41 1.41]</pre>

---

<p class="kicker">12 · MLP</p>

## Widen, bend, narrow.

```python run
W1 = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
W2 = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
hidden = n @ W1
print(hidden.shape)
print(hidden[2].round(2))
print(np.maximum(hidden, 0)[2].round(2))
```

<pre class="output fragment">(5, 4)
[0.89 1.1  0.89 -1.1 ]
[0.89 1.1  0.89 0.  ]</pre>

<p class="shape">n (5, 2) @ W1 (2, 4) → (5, 4) · i.e. 2 in, 4 hidden out</p>

The fourth number was the second one with its sign flipped, and the bend deleted it.

---

<p class="kicker">12 · MLP</p>

## Without the bend the two matrices are one.

```python run
print(W1 @ W2)
print(np.allclose(n @ W1 @ W2, n @ (W1 @ W2)))
```

<pre class="output fragment">[[1.  0. ]
 [0.  0.5]]
True</pre>

<p class="shape">W1 (2, 4) @ W2 (4, 2) → (2, 2)</p>

Two hundred layers without an activation are one matrix. The bend is what buys depth.

---

<p class="kicker">12 · MLP</p>

## Add this to the stream too.

```python run
f = np.maximum(hidden, 0) @ W2
h = h + f
print(f[2].round(2), h[2].round(2))
```

<pre class="output fragment">[0.89 0.55] [1.84 1.71]</pre>

<p class="shape">x_out = h + Δmlp · shape (5, 2) · unchanged since the embedding</p>

Attention let words talk to each other. The MLP let each word think on its own.

---

<p class="kicker">The block</p>

## One stream, two boxes, two additions.

<img class="fig" src="figures/04_a_flow.png" alt="the block as a dataflow diagram">
<p class="shape">x (5, 2) → attention → + → norm → MLP → + → the next block</p>

Only the red box lets one word's numbers end up inside another word's numbers.

---

<p class="kicker">13 · Unembed</p>

## The table again, sideways.

```python run
head = E.T                                  # (2, 6): one column per vocabulary word
logits = (h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)) @ head
print(head.shape, logits.shape)
print(logits[-1].round(2))
```

<pre class="output fragment">(2, 6) (5, 6) [0.62 1.27 1.14 0.75 0.56 1.01]</pre>

<p class="shape">logits · shape (5, 6) · one score per vocabulary word, for every position</p>

---

<p class="kicker">14 · Next token</p>

## Softmax again, now over the vocabulary.

```python run
p = np.exp(logits[-1]) / np.exp(logits[-1]).sum()
print([(w, round(float(pi), 3)) for w, pi in zip(WORDS, p)])
```

<pre class="output fragment">[('the', 0.123), ('dog', 0.235), ('cat', 0.206), ('chased', 0.139), ('sat', 0.115), ('old', 0.181)]</pre>

<p class="shape">p · shape (6,) · adds to 1 · the largest is 0.24</p>

It cannot choose. We chose every weight in it, so it has nothing to say about the next word. That is what
training will fix, next week.

---

<p class="kicker">The whole thing</p>

## Fourteen lines.

```python
ids      = [at[w] for w in sentence]                    # (5,)
X        = E[ids]                                       # (5, 2)
Q, K, V  = X @ Wq, X @ Wk, X @ Wv                       # (5, 2) each
Q, K     = turn(Q, pos), turn(K, pos)                   # roPE on Q and K only
scores   = Q @ K.T / np.sqrt(2)                         # (5, 5)
scores   = np.where(future, -np.inf, scores)            # causal mask
W        = np.exp(scores - scores.max(-1, keepdims=True))
W        = W / W.sum(-1, keepdims=True)                 # (5, 5) weights
y        = (W @ V) @ Wo                                 # (5, 2)
h        = X + y                                        # residual
n        = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6) * g1
h        = h + np.maximum(n @ W1, 0) @ W2               # residual
logits   = (h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)) @ head
p        = np.exp(logits[-1]) / np.exp(logits[-1]).sum()
```

Twelve lines are the block. Two are the table lookups at either end. Every weight is in `figures/hand.py`.

---

<p class="kicker">Lab</p>

## Your own model is these lines with bigger numbers.

<img class="sticker tr sm" src="stickers/gear.png">

```python
import torch
from model import GPT
model = GPT(vocab_size=2048, n_layer=4, n_head=4, d_model=128, block_size=128)
x = torch.randint(0, 2048, (16, 128))
logits, _ = model(x)
print(logits.shape)
```

<pre class="output fragment">torch.Size([16, 128, 2048])</pre>

`d = 128` instead of 2, four blocks instead of one, sixteen sentences instead of one.

---

<p class="kicker">Reading</p>

## Three things to read, and one to run.

<img class="sticker br sm" src="stickers/owl.png">

- Grant Sanderson (3Blue1Brown), [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) and [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention).
- Su et al. 2021, [RoFormer](https://arxiv.org/abs/2104.09864), section 3.2. Rotary positions, the two-dimensional case first.
- Vaswani et al. 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), sections 3.1 to 3.3.
- Run it: `python slides/figures/hand.py` prints every array in this deck.
