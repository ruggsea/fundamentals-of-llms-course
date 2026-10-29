<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Advanced Machine Learning · Lecture 3 · 29 October 2026</p>

# Attention, by hand

<p class="tagline">Every word asks a question of the words before it. A dictionary where every key matches a little, computed with a pen, then found inside GPT-2.</p>
<p class="kicker">Jana Lasser · Ruggero Marino Lazzaroni · Lorenz Prattes<br>Uni Graz · Winter 2026/27 · DAT.C3111UB</p>
</div>
<div class="pic" style="background-image:url(assets/babbage.jpg)"></div>

Note: Babbage's Difference Engine: a machine that computes tables. Today's machine computes a table too, a different one for every sentence. The lecture in one sentence, to say at the start and at the end: each word asks a question, compares it with a label on every earlier word, and takes a weighted average of what those words carry. Pacing for 90 minutes: recap and problem 8, history 12, soft lookup up to the exercise 25, the exercise plus break 15, queries keys values 15, causal and heads with the lab 10, GPT-2 and what is missing 10, close 3. Running behind: drop the dial live slide (the point survives in its note) and the shuffle press-run (the table slide carries the message).

---

<p class="kicker">Last week</p>

## Last week text became integers, and each integer gets a row of numbers.

- A tokenizer turned "Hello, 🌍! 你好!" into 8 integers. Our own TinyStories tokenizer packs 3.7 bytes into a token.
- Each token id selects one row of a table: GPT-2 small has 50,257 rows of 768 numbers, learned in training.
- A row is the same in every sentence. That is today's problem.

Note: One-minute recap from deck 02: the o200k_base example and the lab slide's 3.71 bytes per token. 50,257 × 768 = 38.6M parameters, a third of GPT-2 small. The A1 leaderboard ranks by bits per byte on held-out text.

---

<p class="kicker">The problem</p>

## To guess what follows "it", the model has to know what "it" is.

<p style="font-size: 1.15em;">The cat sat on the mat because <b>it</b> was tired.</p>

<p style="font-size: 1.15em;">The cat sat on the mat because <b>it</b> was soft.</p>

<div class="fragment">

- "it" gets the same row in both sentences. Here it is the cat; there it is the mat.
- Lecture 1's counting model saw only the last few words. "cat" is six words back.
- So each word's row has to take in the words around it. Attention is how.

</div>

Note: Let the room answer what "it" is in each sentence before the fragment. The second sentence is decided by "soft", which comes after "it": a model that reads left to right has to keep both readings open (Part 4 shows two heads doing exactly that). Six words back: cat, sat, on, the, mat, because. A 5-gram model sees four words of context.

---

<!-- .slide: class="divider" data-background-image="assets/stradanus.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 1</p>

## Attention was invented for translation.
</div>

Note: Fine print, for whoever asks: the soft weighting of a decoder's inputs appeared first in Graves 2013, for handwriting. Bahdanau et al. brought it to translation and named it "attention" (Jurafsky & Martin, 3rd edition draft, ch. 14, historical notes).

---

<p class="kicker">History · 1</p>

## Before attention, the recurrent network was the language model.

<div class="cols">
<div>

![Karpathy 2015, The Unreasonable Effectiveness of Recurrent Neural Networks](assets/papers/karpathy2015_blog.png)

<p class="caption">Karpathy's char-rnn demo, 21 May 2015. Its LSTMs were trained on Shakespeare, Wikipedia, LaTeX and the Linux kernel.</p>

</div>
<div>

- A recurrent network (RNN) reads one symbol at a time and keeps a running vector. Its guess for the next symbol depends on everything read so far, squeezed into one vector.
- 2010: word by word, a mixture of RNN language models cut perplexity about in half against a strong n-gram model.
- 2011: letter by letter on 100 MB of English Wikipedia. 2013: the same trick writes text and handwriting.
- Then the same machine was pointed at translation, and that is where attention comes from.

</div>
</div>

<p class="source"><a target="_blank" href="https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf">Mikolov et al. 2010</a> · <a target="_blank" href="https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf">Sutskever, Martens, Hinton 2011</a> · <a target="_blank" href="https://arxiv.org/abs/1308.0850">Graves 2013</a> · <a target="_blank" href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">Karpathy, May 2015</a></p>

Note: The picture is a crop of the blog's first screen, 21 May 2015. Honest dating: the post is eight months after Bahdanau's paper, three slides down; the letter-guessing line and the translation line ran side by side, and we walk them in the order the ideas met, not the order of the posts. Karpathy's models are LSTMs (long short-term memory, a stronger RNN); we say recurrent network and leave the cell design out. Mikolov et al., Interspeech 2010: "around 50% reduction of perplexity by using mixture of several RNN LMs, compared to a state of the art backoff language model" (Wall Street Journal); "backoff" is the n-gram world's smoothing, lecture 1. Sutskever, Martens and Hinton 2011: about 100 MB of English Wikipedia, "predicting the next character in a stream of text". Graves 2013: "predicting one data point at a time", text and handwriting. Perplexity is lecture 1's surprise score; lower is better.

---

<p class="kicker">History · 2</p>

## In 2014 a neural network read a sentence into one vector, then wrote the translation.

<div class="cols">
<div>

![Title and abstract of Sutskever, Vinyals and Le 2014](assets/papers/sutskever2014_abstract.png)

</div>
<div>

- A recurrent network (RNN) reads one word at a time and keeps a running vector.
- At the end of the English sentence, that one vector is all the decoder (the network that writes the French) gets.
- It worked: 34.8 BLEU (a translation score from 0 to 100) on English to French, ahead of a phrase-based system's 33.3.

</div>
</div>

<p class="source"><a target="_blank" href="https://arxiv.org/abs/1409.3215">Sutskever, Vinyals, Le 2014 · Sequence to Sequence Learning with Neural Networks</a></p>

Note: The abstract: "uses a multilayered Long Short-Term Memory (LSTM) to map the input sentence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector." BLEU is a translation score from 0 to 100, how many word sequences overlap a human translation; we meet scores like it in lecture 9. 34.8 vs 33.3: their abstract, WMT'14. The reader half of the pair is the encoder; the writer half is the decoder. Same year: Cho et al. 2014, the RNN encoder-decoder with GRUs. Lecture 2's history slide had RNNs writing Shakespeare one letter at a time; this is the same machine, pointed at translation. Screenshot from the arXiv PDF, page 1.

---

<p class="kicker">History · 3</p>

## One vector was the bottleneck, so Bahdanau let the decoder look back at every word.

<div class="cols">
<div>

![Title and abstract of Bahdanau, Cho and Bengio 2014](assets/papers/bahdanau2014_abstract.png)

</div>
<div>

- Their abstract: "we conjecture that the use of a fixed-length vector is a bottleneck".
- The fix: for each word it writes, the decoder "(soft-)searches" the source sentence.
- It scores every English word, turns the scores into weights that sum to one, and takes the weighted average. That is today's lecture, in 2014.

</div>
</div>

<p class="source"><a target="_blank" href="https://arxiv.org/abs/1409.0473">Bahdanau, Cho, Bengio 2014 · Neural Machine Translation by Jointly Learning to Align and Translate</a> (ICLR 2015)</p>

Note: Section 1: "a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences". Section 3.1, eq. (5) and (6): the context vector c_i is a weighted sum of the encoder's vectors h_j, with weights α_ij = softmax of scores e_ij. Their score is a small neural network, a(s_{i−1}, h_j); the Transformer replaces it with a dot product, which is what we compute today. The title has neither word; the body calls it an "attention mechanism", and the name stuck. The soft weighting of a decoder's inputs is older: Graves 2013 used it for handwriting, and Bahdanau et al. brought it to translation and named it (Jurafsky & Martin, 3rd edition draft, ch. 14, historical notes).

---

<p class="kicker">History · 4</p>

## The first attention maps: each French word looks at the English words it translates.

<div class="cols wide-right">
<div>

Rows: the French the model wrote. Columns: the English it read. White is weight near 1.

<div class="fragment">

"zone économique européenne" looks back at "European Economic Area" in reverse order. Nobody told it French puts the adjective after.

</div>

</div>
<div>

<img src="assets/papers/bahdanau2014_alignment_a.png" alt="Bahdanau et al. 2014, Figure 3a: alignment weights between an English sentence and its French translation" style="height: 640px;">

</div>
</div>

<p class="source">Bahdanau, Cho, Bengio 2014, Figure 3(a)</p>

Note: Their caption: "Each pixel shows the weight α_ij of the annotation of the j-th source word for the i-th target word (see Eq. (6)), in grayscale (0: black, 1: white)." The model learned the alignment from translation pairs alone. Keep this picture in mind: by the end of today the room computes one of these rows by hand.

---

<p class="kicker">History · 5</p>

## In 2017 the Transformer kept the attention and threw away the RNN.

<div class="cols">
<div>

![Title and abstract of Vaswani et al. 2017](assets/papers/vaswani2017_abstract.png)

</div>
<div>

- "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely".
- In Bahdanau, the decoder read another sentence. Here every word reads its own: self-attention.
- No word waits for the one before it, so a whole sentence is computed at once. That made training fast on graphics cards (GPUs).
- GPT (2018), GPT-2 (2019) and every model in this course are stacks of its blocks.

</div>
</div>

<p class="source"><a target="_blank" href="https://arxiv.org/abs/1706.03762">Vaswani et al. 2017 · Attention Is All You Need</a></p>

Note: Abstract: 28.4 BLEU English to German, "improving over the existing best results, including ensembles, by over 2 BLEU", and 41.8 on English to French "after training for 3.5 days on eight GPUs". The paper is an encoder-decoder for translation; GPT keeps only the decoder half, which is what we build. Convolutions: another pattern machine of the same years, out of scope today. A block is the repeating unit of the stack, next week. The paper's first page grants permission to reproduce its tables and figures for scholarly works with attribution. Its Figure 1, the block diagram, waits for lecture 4, once every box has a name.

---

<!-- .slide: class="divider" data-background-image="assets/diderot.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 2</p>

## A dictionary where every key matches a little.

<p class="quote">"At its heart, attention is really just a weighted sum of context vectors, with a lot of complications added to how the weights are computed and what gets summed."</p>
<p class="quote-src">Jurafsky & Martin, Speech and Language Processing</p>
</div>

---

<p class="kicker">Soft lookup · 1</p>

## A Python dict matches one key exactly, or fails.

```python run
animals = {"cat": "Katze", "dog": "Hund"}
print(animals["cat"])
print(animals["kitten"])
```

<div class="fragment">

A *soft* dictionary would answer "kitten" anyway: compare it with every key, see that it is most like "cat", and return mostly the value of "cat".

</div>

Note: Press run: "Katze", then a KeyError. The key that is asked for is the query. Attention keeps the three parts of a dict, query, key and value, and replaces "equal or not" by "how similar".

---

<p class="kicker">Soft lookup · 2</p>

## Four words get four readable numbers each.

| word | animal | object | action | pronoun |
|---|---|---|---|---|
| cat | 1 | 0 | 0 | 0 |
| sat | 0 | 0 | 1 | 0 |
| mat | 0 | 1 | 0 | 0 |
| it | 0 | 0 | 0 | 1 |

<p class="caption">Four words from "the cat sat on the mat because it". We picked the numbers so they can be read. A real model's 768 numbers are learned and mean nothing you can name.</p>

The query: "it" is looking for an animal, $q = [2, 0, 0, 0]$. The keys and values are the four rows.

Note: Each row is the token's row of the embedding table from lecture 2, written small. We drop "the", "on", "because" so everything fits on a slide. Where the query comes from (a learned table) is Part 3; for now we write it down.

---

<p class="kicker">Soft lookup · 3</p>

## Step 1: a score is the dot product of the query with each key.

$$ q \cdot k = q_1 k_1 + q_2 k_2 + q_3 k_3 + q_4 k_4 $$

<p class="inwords">Multiply position by position and add. q · cat = 2·1 + 0·0 + 0·0 + 0·0 = 2. For sat, mat and it every product has a 0 in it: the score is 0.</p>

<div class="fragment">

| key | cat | sat | mat | it |
|---|---|---|---|---|
| score | 2 | 0 | 0 | 0 |

The dot product is large when two vectors point the same way.

</div>

Note: Do one on the board. This is the "how similar" of the soft dict. With these readable vectors the scores are whole numbers; in a real model they are anything.

---

<p class="kicker">Soft lookup · 4</p>

## Step 2: softmax turns scores into weights that sum to one.

<div style="font-size: 0.95em; display: flex; justify-content: center;">

| key | cat | sat | mat | it | sum |
|---|---|---|---|---|---|
| score | 2 | 0 | 0 | 0 | |
| <span class="fragment" data-fragment-index="1">$e^{\text{score}}$</span> | <span class="fragment" data-fragment-index="1">7.389</span> | <span class="fragment" data-fragment-index="1">1</span> | <span class="fragment" data-fragment-index="1">1</span> | <span class="fragment" data-fragment-index="1">1</span> | <span class="fragment" data-fragment-index="1">10.389</span> |
| <span class="fragment" data-fragment-index="2">weight</span> | <span class="fragment" data-fragment-index="2">0.711</span> | <span class="fragment" data-fragment-index="2">0.096</span> | <span class="fragment" data-fragment-index="2">0.096</span> | <span class="fragment" data-fragment-index="2">0.096</span> | <span class="fragment" data-fragment-index="2">1.000</span> |

</div>

<div class="fragment" data-fragment-index="1"><p class="inwords">Raise e (2.718) to each score. Everything becomes positive and the order stays: e² = 7.389, e⁰ = 1.</p></div>

<div class="fragment" data-fragment-index="2">

$$ w_i = \frac{e^{s_i}}{\sum_j e^{s_j}} $$

<p class="inwords">Divide each by the sum of all four; j runs over the four keys. 7.389 / 10.389 = 0.711 for "cat".</p>

</div>

Note: Why e and not just the scores: scores can be negative, weights cannot. Softmax is the same "scores to probabilities" step that turns a model's output into a next-token guess (lecture 4). figures/03_attention_by_hand.py prints every number.

---

<p class="kicker">Soft lookup · 5</p>

## Step 3: the output is the weighted average of the values.

$$ z = 0.711\, v_{\text{cat}} + 0.096\, v_{\text{sat}} + 0.096\, v_{\text{mat}} + 0.096\, v_{\text{it}} $$

<p class="inwords">Take 0.711 of the "cat" row, 0.096 of each other row, and add them up number by number.</p>

<div class="fragment">

$$ z = [0.711,\ 0.096,\ 0.096,\ 0.096] $$

<p class="inwords">Read it with the column names: animal 0.71, object 0.10, action 0.10, pronoun 0.10. The new row for "it" is mostly cat. The numbers repeat the weights only because our four values are one-hot rows.</p>

</div>

Note: This z replaces the row of "it" for the next layer. That is the whole point: after attention, "it" carries what it refers to. The weights tell you where the values came from, not why: say this now, it comes back with GPT-2 in Part 5.

---

<p class="kicker">Soft lookup · 6</p>

## The three steps are three lines of numpy. Press run.

```python run
import numpy as np
X = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
q = np.array([2, 0, 0, 0])                     # "it" looks for an animal
scores = X @ q                                 # step 1: dot products
weights = np.exp(scores) / np.exp(scores).sum()   # step 2: softmax
z = weights @ X                                # step 3: weighted average
print("weights:", weights.round(3))
print("z:      ", z.round(3))
```

Note: `X @ q` is the dot product of q with every row of X at once. `weights @ X` is the weighted sum of the rows. Change q to [0, 2, 0, 0] and "it" turns into the mat.

---

<p class="kicker">Live</p>

## Turn the dial up and the soft dictionary becomes the Python dict.

<div id="dial-demo">

<p><input type="range" id="dial-s" min="0" max="10" step="0.5" value="1" aria-label="multiply the scores"><span id="dial-val"></span></p>

<div id="dial-bars"></div>

<p class="caption">"it" asks for an animal; its scores are [1, 0, 0, 0] times the dial. Softmax turns small gaps into big ones.</p>

</div>

Note: Drag slowly. At 0 every word gets 0.25: a blurry average. At 1: 0.475 on cat. At 2: 0.711, our slide. At 6: 0.993. At 10 the weight is 1.000 on cat and 0 elsewhere: an exact lookup, the dict from five slides ago. The hard dict is one end of a dial, and a model can learn to sit anywhere on it. We come back to the dial in Part 3: dividing by √d keeps it from being turned too far.

---

<!-- .slide: class="statement" -->

<div>

# Now you: do two lookups, by hand.

1. "it" asks with $q = [1, 0, 0, 0]$. Compute the four weights and the output z.
2. Write a query that makes "it" look at "mat". What is z now?

<p>e⁰ = 1 · e¹ = 2.718 · e² = 7.389 · e³ = 20.09</p>

| word | animal | object | action | pronoun |
|---|---|---|---|---|
| cat | 1 | 0 | 0 | 0 |
| sat | 0 | 0 | 1 | 0 |
| mat | 0 | 1 | 0 | 0 |
| it | 0 | 0 | 0 | 1 |

</div>

Note: 10 minutes, pairs, pen and paper. Then check with the numpy slide. Answers (figures/03_attention_by_hand.py): 1. scores [1, 0, 0, 0], e: 2.718, 1, 1, 1, sum 5.718, weights 0.475, 0.175, 0.175, 0.175; z = [0.475, 0.175, 0.175, 0.175], half cat. A weaker question gives a blurrier answer: the dial again. 2. q = [0, 2, 0, 0] (or any query with its largest number in the object column): weights 0.711 on mat, z = [0.096, 0.711, 0.096, 0.096]. Point: the query decides what a word looks for. Then the break.

<img class="sticker br" src="stickers/manicule-right.png">

---

<!-- .slide: class="divider" data-background-image="assets/press.jpg" data-background-size="cover" data-background-position="center" -->

<div class="band">
<p class="kicker">Part 3</p>

## The model learns what to ask.
</div>

---

<p class="kicker">Queries, keys, values · 1</p>

## Each word plays three roles.

- In a library, the question you bring, the label on the spine and the text inside the book are three different things.
- A word is all three at once. "it" looks for an animal, is labelled a pronoun, and hands over whatever it carries.
- So each role gets its own learned table: $W_Q$ for queries, $W_K$ for keys, $W_V$ for values.

Note: Until now we wrote the query by hand and used the rows themselves as keys and values. With W_Q different from W_K, the question a word asks is no longer "who looks like me". The three come from the same row x; only the table differs. The names are older than the Transformer: "the terminology of key, query, and value, came from memory networks", "by using an embedding of a query to match keys representing content in an associative memory" (Jurafsky & Martin, 3rd edition draft, ch. 7, historical notes; Sukhbaatar et al. 2015, Weston et al. 2015, Graves et al. 2014). Our soft dictionary is that metaphor, grown a spine.

---

<p class="kicker">Queries, keys, values · 2</p>

## Three tables turn each row into a query, a key and a value.

$$ Q = X W_Q \qquad K = X W_K \qquad V = X W_V $$

<p class="inwords">Each W is a table of numbers; multiplying a row by it is called a projection. Our hand-made W<sub>Q</sub> has a 4 where the pronoun row meets the animal column, and another where the action row meets it: "it" and "sat" both go looking for an animal. W<sub>K</sub> and W<sub>V</sub> leave the rows as they are.</p>

<div class="fragment">

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V $$

<p class="inwords">The three steps for every word at once. QKᵀ is the table of every query's dot product with every key; the ᵀ (transpose) swaps K's rows and columns. d<sub>k</sub> is how many numbers a key has, here 4, so the scores are divided by √4 = 2. The 4 in W<sub>Q</sub> divided by 2 gives our score of 2.</p>

</div>

Note: Vaswani et al. 2017, eq. (1). Kᵀ, "K transposed", is K with rows and columns swapped. Real models learn these tables in training (lecture 5); nobody writes a 4 by hand. Why √d_k: three slides on.

---

<p class="kicker">Queries, keys, values · 3</p>

## Every word asks at once: one 4 × 4 table of weights.

<div class="cols wide-right">
<div>

```python run
import numpy as np
X = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
W_q = np.zeros((4, 4)); W_q[3, 0] = 4; W_q[2, 0] = 4   # it, sat -> animal
Q, K, V = X @ W_q, X, X
S = Q @ K.T / np.sqrt(4)
W = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
print(W.round(3))
```

Each row is one word asking, and sums to 1.

</div>
<div>

![Attention weights for cat, sat, mat, it](figures/03_attention_weights.png)

</div>
</div>

Note: figures/03_attention_by_hand.py. Rows: the word asking. Columns: the word looked at. "sat" and "it" put 0.711 on "cat". "cat" and "mat" ask nothing (their query is all zeros), so they spread 0.25 everywhere. sum(axis=1, keepdims=True) divides each row by its own sum. Ask: what is wrong with row "cat"? It looks at words that come after it. Part 4.

---

<p class="kicker">Queries, keys, values · 4</p>

## With random tables the weights come out nearly flat. Training sharpens them.

```python run
import numpy as np
X = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
rng = np.random.default_rng(0)
W_q = 0.5 * rng.normal(size=(4, 4))     # small random numbers
W_k = 0.5 * rng.normal(size=(4, 4))
S = (X @ W_q) @ (X @ W_k).T / np.sqrt(4)
W = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
print(W.round(2))
```

Note: Every row stays near 0.25: a model before training is the dial at almost zero. Change the seed and run again. Training (lecture 5) moves the numbers in W_Q and W_K until the weights help predict the next token. Raschka, LLMs from scratch ch. 3, section 3.4 has the same thing as a torch class.

---

<p class="kicker">Queries, keys, values · 5</p>

## Dividing by √d keeps the dial from being turned too far.

![Largest attention weight with and without scaling](figures/03_scaling_sqrt_dk.png)

<p class="caption">Random queries and keys, 64 keys, 2,000 queries per point. Without the division, a key size of 1,024 puts 0.95 of the weight on one key.</p>

Note: Script: figures/03_scaling_sqrt_dk.py. x is d_k, the numbers per key; y is the biggest weight in a row, averaged. A dot product adds d_k products, so with more numbers per key the scores spread wider: their standard deviation grows as √d_k (2.0 at d_k = 4, 32 at d_k = 1,024). Wider scores are the dial turned up. Dividing by √d_k puts it back. Vaswani et al., section 3.2.1: "We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients." The gradient part is theirs and is lecture 5's business.

---

<!-- .slide: class="divider" data-background-image="assets/semaphore.jpg" data-background-size="cover" data-background-position="center" -->

<div class="band">
<p class="kicker">Part 4</p>

## No peeking, and many questions at once.
</div>

---

<p class="kicker">Causal · 1</p>

## One pass plays every round of Shannon's game at once.

![Four input tokens, four attention outputs, four next-token guesses](figures/03_next_token.png)

Each position guesses the next token. If position t could see token t + 1, it would copy the answer. Reading only what came before is what causal means here.

Note: Lecture 1: the guesser sees only the text so far. A sentence of T tokens gives T guesses in one pass, which is why training is cheap (lecture 5). How z becomes a guess over the vocabulary is lecture 4. Script: figures/03_next_token.py. Grey arrows: what each position may read.

---

<p class="kicker">Causal · 2</p>

## The causal mask blanks everything above the diagonal.

<div class="cols wide-right">
<div>

```python
future = np.triu(np.ones((4, 4)), k=1) == 1
S[future] = -np.inf
```

A hidden score becomes −∞ before the softmax, and $e^{-\infty} = 0$. The row still sums to 1.

<div class="fragment">

"sat" can now see only "cat" and itself: 0.881 on cat, 0.119 on itself. "cat" sees only itself.

</div>

</div>
<div>

![Causal attention weights for cat, sat, mat, it](figures/03_causal_weights.png)

</div>
</div>

Note: np.triu(..., k=1) marks the cells above the diagonal: the future. Masking after the softmax would leave rows that no longer sum to 1; masking before gives the renormalisation for free. sat: e²/(e² + e⁰) = 7.389/8.389 = 0.881. "it" is last, so it sees everything and keeps 0.711 on cat. figures/03_attention_by_hand.py, causal weights.

---

<p class="kicker">Causal · 3</p>

## Without the mask, the past changes when you edit the future. Press run.

```python run
import numpy as np
def attend(X, W_q, causal):
    S = (X @ W_q) @ X.T / np.sqrt(4)
    if causal: S[np.triu(np.ones((4, 4)), k=1) == 1] = -np.inf
    W = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
    return W @ X
W_q = np.zeros((4, 4)); W_q[3, 0] = 4; W_q[2, 0] = 4
cat_sat_mat_it = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
cat_sat_mat_cat = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
for causal in (False, True):
    print("mask" if causal else "no mask", attend(cat_sat_mat_it, W_q, causal)[0].round(2),
          attend(cat_sat_mat_cat, W_q, causal)[0].round(2))
```

Note: We change only the last word, "it" to "cat", and print the output for the FIRST word. Without the mask, "cat"'s output changes: it has read a word that has not been written yet, and in training it would learn to copy the answer. With the mask it is [1, 0, 0, 0] both times. code/tests.py checks exactly this on your model (test_attention_is_causal).

---

<p class="kicker">Multi-head</p>

## Two heads ask two questions: one finds the animal, the other the object.

![Two causal attention tables: head 1 sends it to cat, head 2 sends it to mat](figures/03_heads.png)

<p class="caption">Head 2's W<sub>Q</sub> has its 4 where the pronoun row meets the object column. "it" gets one answer from each, and the layers after decide between them.</p>

Note: Back to the two sentences at the start: "tired" or "soft" comes after "it", so a left-to-right model has to keep both readings. One head cannot: it gives one weighted average. In a real model the heads are not separate models. The query, key and value tables are cut into H column blocks, each head uses its own block, and the outputs are glued back together and mixed by one more table, W_O (lecture 4). The cut is a reshape, so 8 heads of 64 numbers cost about the same as one head of 512. Vaswani et al., section 3.2.2: h = 8 heads, d_k = 64. "The layers after" means later blocks of the stack, next week's word. Script: figures/03_attention_by_hand.py.

---

<p class="kicker">Lab</p>

## Your model now has causal multi-head attention: CausalSelfAttention in model.py.

<img class="sticker tr sm" src="stickers/gear.png">

```python
q, k, v = self.qkv(x).split(d, dim=-1)
q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
scores = scores.masked_fill(future, float("-inf"))
weights = F.softmax(scores, dim=-1)
y = weights @ v
y = y.transpose(1, 2).reshape(B, T, d)
return self.out(y)
```

<pre class="output fragment">PASS test_forward_shapes_and_initial_loss
PASS test_attention_is_causal</pre>

<p class="caption">Every line is one of today's slides, written in torch (lecture 4): projections, heads, scaled scores, mask, softmax, blend, W_O.</p>

Note: Command: `cd code && python tests.py`. The causality check changes only the last token and asserts that every earlier position's output stays the same: the "press run" slide, as a test. The file is code/model.py, CausalSelfAttention.forward; the slide skips two lines (the B, T, d unpack and apply_rope), and apply_rope is lecture 4. self.qkv holds W_Q, W_K and W_V side by side, self.out is W_O. B is the batch size, T the number of tokens in the window.

---

<!-- .slide: class="divider" data-background-image="assets/luyken.jpg" data-background-size="cover" data-background-position="center" -->

<div class="band">
<p class="kicker">Part 5</p>

## GPT-2 has been doing this all along.
</div>

---

<p class="kicker">Live</p>

## GPT-2 runs the same formula on our sentence. Click a word, pick a head.

<div id="attn-demo">

<div class="attn-controls">
<select id="attn-head" aria-label="layer and head"></select>
<label><input type="checkbox" id="attn-bits"> show surprisal in bits</label>
</div>

<div id="attn-out"></div>

<p class="caption" id="attn-info"></p>

<p class="caption">GPT-2 small, 12 layers × 12 heads, the same formula as today with learned tables of 768 numbers. The clicked word is the query; red is the weight each earlier token gets.</p>

</div>

Note: Data: slides/assets/gpt2_attention.json, from figures/03_gpt2_attention.py (Hugging Face gpt2). Opens on layer 4 head 3 with the query "it": 84% of its weight on "cat", the pattern of our hand-made head 1. Then pick layer 4 head 11: every word puts 100% on the word just before it. Many heads put most weight on the first token, "The": a known "attention sink", it means nothing. Click around: most heads are not readable at all. Surprisal is lecture 1's bits for each token given the ones before it. One sentence: a demonstration, not a measurement (next slide).

---

<p class="kicker">GPT-2</p>

## We found that head by searching all 144 heads. A weight says where a head looked, not why.

- In layer 4, head 3 sends 84% of "it" to "cat". It is the strongest such head in 3 of the 5 sentences of our check, figures/03_gpt2_headsearch.py.
- On "The girl dropped the cup because she was tired." it gives "girl" only 21%. On the farmer sentence, 5%.
- So we do not claim it resolves pronouns. Nobody hand-assigned a role to it; training made it, and many heads do several things at once.

Note: The check is in the repo: figures/03_gpt2_headsearch.py, five sentences of the same shape (cat/mat, dog/ball, girl/cup, farmer/horse, bird/nest), gpt2, one sentence per row, n=1. Head 4.3's weight from the pronoun to the subject: 0.84, 0.59, 0.21, 0.05, 0.76; the strongest of all 144 heads on that pair in three of the five (the others go to layer 0 head 6 with 0.56 on the girl sentence, and layer 11 head 8 with 0.34 on the farmer sentence). Picking the best head after looking is the same trap as reporting the best of many runs: lecture 9. Attention maps as explanations are contested in the literature; we leave it there.

---

<p class="kicker">What is missing</p>

## Attention is one part of a block. Next week we add the other three.

| part | what it does |
|---|---|
| MLP (multi-layer perceptron) | works on each word alone, after attention has mixed them |
| residual | adds each part's output to its input, so the row is edited, not replaced |
| norm | rescales each row so stacking 12 or 96 blocks stays stable |
| positions | tell attention where each word sits; today it has no idea |

Note: One slide, no detail: each row is a section of lecture 4 (MLP and SwiGLU, the residual stream, RMSNorm, RoPE). The last row is the cliffhanger on the next slide.

---

<p class="kicker">What is missing</p>

## Shuffle the words and attention gives the same answers, shuffled.

```python run
import numpy as np
def attend(X):                           # no mask, Q = K = V = X
    S = X @ X.T / np.sqrt(X.shape[1])
    return (np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)) @ X
cat, sat, mat, it = np.eye(4)
print(attend(np.stack([cat, sat, mat, it]))[0].round(2))   # cat sat mat it
print(attend(np.stack([mat, it, sat, cat]))[3].round(2))   # mat it sat cat
```

The row for "cat" is the same in both orders. Attention reads a bag of words. Lecture 4 gives it positions.

Note: Order-blindness of attention without positions. The causal mask leaks a little order (a token can see how many came before it), but it does not say which word sat where. Lecture 4 opens with the same demo on "dog bites man" and then RoPE.

---

<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Next week · 5 November</p>

# The transformer block

<p class="tagline">Attention plus an MLP, norms, residuals and positions: a short list of choices, each one tested by ablation.</p>
<p class="kicker">Thursdays 15:15 · SR 127.11 · Leechgasse 34</p>
</div>
<div class="pic" style="background-image:url(assets/babbage.jpg)"></div>

---

<p class="kicker">Reading</p>

## Read these before next Thursday.

<img class="sticker br sm" src="stickers/owl.png">

- Grant Sanderson (3Blue1Brown), [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention). Queries, keys, masking and heads, with pictures; there is a video and a text version.
- Sebastian Raschka, [LLMs from scratch, chapter 3](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb). The attention of this lecture, implemented step by step in a notebook.
- Dan Jurafsky and James Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3rd edition draft, ch. 7, "Transformers and Pretraining". Free online. The same climb from a weighted sum, and historical notes on where the query, key and value names come from.
- Bahdanau, Cho, Bengio 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), sections 1 to 3. Where attention was first proposed, for translation.
- Vaswani et al. 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.2. The paper that introduced the transformer; this section defines scaled dot-product and multi-head attention.

Note: Figures in this deck: slides/figures/03_*.py; the live demo's data comes from 03_gpt2_attention.py; paper screenshots in assets/papers/ are crops of the arXiv PDFs.

---

<p class="kicker">Reading</p>

## If you want more, read these too.

<img class="sticker br sm" src="stickers/owl.png">

- Sutskever, Vinyals, Le 2014, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215). The one-vector translation model attention was built to fix.
- Peter Bloem, [Transformers from scratch](https://peterbloem.nl/blog/transformers). Self-attention built up from a plain weighted sum, with PyTorch code.
- Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/). How word vectors are learned, and why their directions carry meaning.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/). GPT-2 running in the browser; type a prompt and look at its attention.
- Callum McDougall, [Induction heads, illustrated](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated), 2023. Two heads working together to copy text, step by step in diagrams.
- Adam Casson, [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops). Where the compute of a transformer goes, and when attention's share grows.

Note: Synced to the course README under the lecture 3 section.
