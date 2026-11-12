<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Advanced Machine Learning · Lecture 5 · 12 Nov 2026</p>

# Training it

<p class="tagline">One loss, one loop, a few knobs.</p>
<p class="kicker">Jana Lasser · Ruggero Marino Lazzaroni · Lorenz Prattes<br>Uni Graz · Winter 2026/27 · DAT.C3111UB</p>
</div>
<div class="pic" style="background-image:url(assets/stradanus.jpg)"></div>

<style>
.reveal .shape { font-size: 0.8em; }
.reveal .inwords { font-size: 0.72em; }
.reveal .rowlabel { font-size: 0.7em; }
.reveal .caption { font-size: 0.6em; }
.reveal img.fig.lent { border: 1px solid var(--rule); }
.reveal pre { flex-shrink: 0; }
.reveal pre code { max-height: none; }
.reveal img.fig.paper, .reveal img.fig.shot { max-height: 76vh !important; border: 1px solid var(--rule); box-shadow: 8px 10px 0 rgba(38, 34, 30, 0.10); }
.reveal .tokline.block { gap: 0.18em 0.14em; margin: 0.5em 0; }
.reveal .tokline.block .tok { font-size: 0.74em; padding: 0.26em 0.42em; }
</style>

---

<p class="kicker">Last week</p>

## Nothing was trained.

<img class="fig" src="figures/05_t_bar.png" alt="last week's model gives cat a probability of 0.13 after the dog chased the">

<p class="shape">what last week's block says comes after "the dog chased the" · the text says cat</p>

Note: The six numbers are the hand-written model of lecture 4 (figures/hand.py) reading the first four words. We chose every weight by hand so the pictures would read. Nobody chose them to predict well, and it shows: cat gets 0.13, a fair six-sided die would give 0.17. Today the text chooses the weights.

---

<p class="kicker">Three stages</p>

## Today: pretraining.

<img class="fig" src="figures/05_t_stages.png" alt="three boxes: pretraining in colour, mid-training and post-training greyed out">

Note: Pretraining: random weights, a large pile of raw text, one objective. Mid-training is a newer word for a second pass on more targeted data before the chat stage; mention and move on. Post-training is lectures 10 and 11. Everything today happens in the first box.

---

<p class="kicker">Why pretrain · January 2018</p>

## ULMFiT.

<img class="fig paper" src="assets/papers/howard2018_abstract.png" alt="Title and abstract of Howard and Ruder 2018, Universal Language Model Fine-tuning for Text Classification">

Note: Howard and Ruder, arXiv 1801.06146. The paper that made pretraining a language model the default first step in NLP. Next slide is its recipe.

---

<p class="kicker">Why pretrain · ULMFiT, 2018</p>

## Pretrain once, then teach it any task.

<img class="fig" src="figures/05_t_ulmfit.png" alt="Wikipedia, then a language model, then three tasks">

<p class="shape">Howard and Ruder 2018 · 100 labelled examples matched training from scratch on 100 times more</p>

Note: Universal Language Model Fine-tuning. One language model, pretrained on 103 million words of Wikipedia by guessing the next word, then fine-tuned for six text classification tasks, beat the models built for each of them. The claim on the slide is from the paper's abstract. The model was an LSTM, not a transformer; what stuck was the recipe. After this paper the question became how far pretraining scales, and BERT and GPT are the two answers of the same year.

---

<!-- .slide: data-auto-animate -->

<p class="kicker">The data · 1</p>

## A corpus is every document, joined end to end.

<div class="tokline block">
<span class="tok" data-id="t0"> -</span><span class="tok" data-id="t1"> to</span><span class="tok" data-id="t2"> always</span><span class="tok" data-id="t3"> listen</span><span class="tok" data-id="t4"> to</span><span class="tok" data-id="t5"> his</span><span class="tok" data-id="t6"> mom</span><span class="tok" data-id="t7"> and</span><span class="tok" data-id="t8"> be</span><span class="tok" data-id="t9"> careful</span><span class="tok" data-id="t10"> when</span><span class="tok" data-id="t11"> playing</span><span class="tok" data-id="t12"> outside</span><span class="tok" data-id="t13">.</span><span class="tok hole" data-id="t14">&lt;|endoftext|&gt;</span><span class="tok" data-id="t15">Once</span><span class="tok" data-id="t16"> upon</span><span class="tok" data-id="t17"> a</span><span class="tok" data-id="t18"> time</span><span class="tok" data-id="t19">,</span><span class="tok" data-id="t20"> there</span><span class="tok" data-id="t21"> was</span><span class="tok" data-id="t22"> an</span><span class="tok" data-id="t23"> old</span><span class="tok" data-id="t24"> man</span><span class="tok" data-id="t25">.</span><span class="tok" data-id="t26"> He</span><span class="tok" data-id="t27"> liked</span><span class="tok" data-id="t28"> to</span><span class="tok" data-id="t29"> read</span><span class="tok" data-id="t30"> ma</span><span class="tok" data-id="t31">g</span><span class="tok" data-id="t32">a</span><span class="tok" data-id="t33">z</span><span class="tok" data-id="t34">in</span><span class="tok" data-id="t35">es</span><span class="tok" data-id="t36">.</span><span class="tok" data-id="t37"> One</span><span class="tok" data-id="t38"> day</span><span class="tok" data-id="t39">,</span><span class="tok" data-id="t40"> he</span><span class="tok" data-id="t41"> saw</span><span class="tok" data-id="t42"> two</span><span class="tok" data-id="t43"> birds</span><span class="tok" data-id="t44"> fight</span><span class="tok" data-id="t45">ing</span><span class="tok" data-id="t46"> outside</span><span class="tok" data-id="t47"> his</span><span class="tok" data-id="t48"> window</span><span class="tok" data-id="t49">.</span><span class="tok" data-id="t50"> He</span><span class="tok" data-id="t51"> thought</span><span class="tok" data-id="t52"> it</span><span class="tok" data-id="t53"> was</span><span class="tok" data-id="t54"> not</span><span class="tok" data-id="t55"> nice</span><span class="tok" data-id="t56"> to</span><span class="tok" data-id="t57"> fight</span><span class="tok" data-id="t58">.</span><span class="tok" data-id="t59"> He</span><span class="tok" data-id="t60"> wished</span><span class="tok" data-id="t61"> they</span><span class="tok" data-id="t62"> could</span><span class="tok" data-id="t63"> be</span><span class="tok" data-id="t64"> friends</span><span class="tok" data-id="t65"> instead</span><span class="tok" data-id="t66">.</span><span class="tok" data-id="t67"> The</span><span class="tok" data-id="t68"> end</span><span class="tok" data-id="t69">.</span><span class="tok hole" data-id="t70">&lt;|endoftext|&gt;</span><span class="tok" data-id="t71">Once</span><span class="tok" data-id="t72"> upon</span><span class="tok" data-id="t73"> a</span><span class="tok" data-id="t74"> time</span><span class="tok" data-id="t75">,</span><span class="tok" data-id="t76"> there</span><span class="tok" data-id="t77"> was</span><span class="tok" data-id="t78"> a</span><span class="tok" data-id="t79"> little</span><span class="tok" data-id="t80"> boy</span><span class="tok" data-id="t81"> named</span><span class="tok" data-id="t82"> Timmy</span><span class="tok" data-id="t83">.</span><span class="tok" data-id="t84"> Timmy</span><span class="tok" data-id="t85"> had</span><span class="tok" data-id="t86"> a</span><span class="tok" data-id="t87"> bl</span>
</div>

<p class="shape">three real stories from the training set · 1,133 in all · 239,791 tokens · an end token after each</p>

Note: Real tokens from code/data/train.bin, cut with the tokenizer from lecture 2 (2,048 tokens). The end of one story, the shortest story in full (55 tokens, an old man and two birds), and the start of the next. data.py encodes each document, appends the end token, and writes one flat array of ids. The model has to learn that a story ends, so the end token is a target like any other.

---

<!-- .slide: data-auto-animate -->

<p class="kicker">The data · 2</p>

## Pick a random window.

<div class="tokline block">
<span class="tok off" data-id="t0"> -</span><span class="tok off" data-id="t1"> to</span><span class="tok off" data-id="t2"> always</span><span class="tok off" data-id="t3"> listen</span><span class="tok off" data-id="t4"> to</span><span class="tok off" data-id="t5"> his</span><span class="tok off" data-id="t6"> mom</span><span class="tok off" data-id="t7"> and</span><span class="tok off" data-id="t8"> be</span><span class="tok off" data-id="t9"> careful</span><span class="tok off" data-id="t10"> when</span><span class="tok off" data-id="t11"> playing</span><span class="tok off" data-id="t12"> outside</span><span class="tok off" data-id="t13">.</span><span class="tok hole off" data-id="t14">&lt;|endoftext|&gt;</span><span class="tok off" data-id="t15">Once</span><span class="tok off" data-id="t16"> upon</span><span class="tok off" data-id="t17"> a</span><span class="tok off" data-id="t18"> time</span><span class="tok off" data-id="t19">,</span><span class="tok off" data-id="t20"> there</span><span class="tok off" data-id="t21"> was</span><span class="tok off" data-id="t22"> an</span><span class="tok off" data-id="t23"> old</span><span class="tok off" data-id="t24"> man</span><span class="tok off" data-id="t25">.</span><span class="tok off" data-id="t26"> He</span><span class="tok off" data-id="t27"> liked</span><span class="tok off" data-id="t28"> to</span><span class="tok off" data-id="t29"> read</span><span class="tok off" data-id="t30"> ma</span><span class="tok off" data-id="t31">g</span><span class="tok off" data-id="t32">a</span><span class="tok off" data-id="t33">z</span><span class="tok off" data-id="t34">in</span><span class="tok off" data-id="t35">es</span><span class="tok off" data-id="t36">.</span><span class="tok off" data-id="t37"> One</span><span class="tok off" data-id="t38"> day</span><span class="tok off" data-id="t39">,</span><span class="tok off" data-id="t40"> he</span><span class="tok off" data-id="t41"> saw</span><span class="tok off" data-id="t42"> two</span><span class="tok off" data-id="t43"> birds</span><span class="tok off" data-id="t44"> fight</span><span class="tok off" data-id="t45">ing</span><span class="tok off" data-id="t46"> outside</span><span class="tok off" data-id="t47"> his</span><span class="tok off" data-id="t48"> window</span><span class="tok off" data-id="t49">.</span><span class="tok off" data-id="t50"> He</span><span class="tok off" data-id="t51"> thought</span><span class="tok off" data-id="t52"> it</span><span class="tok off" data-id="t53"> was</span><span class="tok off" data-id="t54"> not</span><span class="tok off" data-id="t55"> nice</span><span class="tok off" data-id="t56"> to</span><span class="tok off" data-id="t57"> fight</span><span class="tok off" data-id="t58">.</span><span class="tok off" data-id="t59"> He</span><span class="tok off" data-id="t60"> wished</span><span class="tok off" data-id="t61"> they</span><span class="tok off" data-id="t62"> could</span><span class="tok off" data-id="t63"> be</span><span class="tok off" data-id="t64"> friends</span><span class="tok" data-id="t65"> instead</span><span class="tok" data-id="t66">.</span><span class="tok" data-id="t67"> The</span><span class="tok" data-id="t68"> end</span><span class="tok" data-id="t69">.</span><span class="tok hole" data-id="t70">&lt;|endoftext|&gt;</span><span class="tok" data-id="t71">Once</span><span class="tok" data-id="t72"> upon</span><span class="tok off" data-id="t73"> a</span><span class="tok off" data-id="t74"> time</span><span class="tok off" data-id="t75">,</span><span class="tok off" data-id="t76"> there</span><span class="tok off" data-id="t77"> was</span><span class="tok off" data-id="t78"> a</span><span class="tok off" data-id="t79"> little</span><span class="tok off" data-id="t80"> boy</span><span class="tok off" data-id="t81"> named</span><span class="tok off" data-id="t82"> Timmy</span><span class="tok off" data-id="t83">.</span><span class="tok off" data-id="t84"> Timmy</span><span class="tok off" data-id="t85"> had</span><span class="tok off" data-id="t86"> a</span><span class="tok off" data-id="t87"> bl</span>
</div>

<p class="shape">8 tokens here · 128 in our model · it may cross the end of a story</p>

Note: A random start, then a fixed number of tokens. No sentence or story boundaries are respected: this window starts in the last sentence of one story and ends in the first words of the next. One click from the last slide: the same block, eight tokens stay lit. One more click and those eight move out to become the example.

---

<!-- .slide: data-auto-animate -->

<p class="kicker">The data · 3</p>

## A training example is a random window.

<img class="sticker tr sm" src="stickers/dice.png">

<p class="rowlabel">the model reads</p>
<div class="tokline lg">
<span class="tok" data-id="t65"> instead</span><span class="tok" data-id="t66">.</span><span class="tok" data-id="t67"> The</span><span class="tok" data-id="t68"> end</span><span class="tok" data-id="t69">.</span><span class="tok hole" data-id="t70">&lt;|endoftext|&gt;</span><span class="tok" data-id="t71">Once</span><span class="tok" data-id="t72"> upon</span>
</div>
<p class="rowlabel">it has to guess</p>
<div class="tokline lg">
<span class="tok hit">.</span><span class="tok hit"> The</span><span class="tok hit"> end</span><span class="tok hit">.</span><span class="tok hit">&lt;|endoftext|&gt;</span><span class="tok hit">Once</span><span class="tok hit"> upon</span><span class="tok hit"> a</span>
</div>

<p class="shape">one example · the same tokens, moved one to the left · 8 tokens, 8 guesses</p>

Note: Take 8 tokens as the input and the 8 tokens starting one later as the targets. Position 1 sees " instead" and guesses "."; position 6 sees the end token and has to guess how a new story starts; position 8 sees all eight and guesses " a". The causal mask from lecture 3 is what makes this honest: no position can look at its own answer. One window gives as many examples of next-token guessing as it has tokens. data.get_batch does exactly this.

---

<p class="kicker">The objective · October 2018</p>

## BERT.

<img class="fig paper" src="assets/papers/devlin2018_abstract.png" alt="Title and abstract of Devlin et al. 2018, BERT">

Note: Devlin, Chang, Lee and Toutanova, arXiv 1810.04805. The B is for bidirectional, the E for encoder.

---

<p class="kicker">The objective · BERT, 2018</p>

## BERT hides words in the middle.

<img class="fig" src="figures/05_t_mlm.png" alt="five tokens, one masked, an encoder, and the guess chased above the mask">

<p class="shape">an encoder · sees both sides · guesses the hidden tokens · about 15% of the text</p>

Note: Masked language modelling, Devlin et al. 2018. 15% of the tokens are picked and the model is scored only on those. It reads left and right at once, so it is good at understanding a text and cannot write one from left to right. For the first years this was the model everybody used; encoders still run most classification and search.

---

<p class="kicker">The objective · June 2018</p>

## GPT.

<img class="fig paper" src="assets/papers/radford2018_abstract.png" alt="Title and abstract of Radford et al. 2018, Improving Language Understanding by Generative Pre-Training">

Note: Radford, Narasimhan, Salimans and Sutskever, OpenAI. Generative pre-training: the G and the P. Published four months before BERT and far less noticed at the time.

---

<p class="kicker">The objective · GPT, 2018</p>

## GPT hides everything to the right.

<img class="fig" src="figures/05_t_clm.png" alt="four tokens, a decoder, and the next token above each">

<p class="shape">a decoder · sees only the left · guesses every next token</p>

Note: Next-token prediction, Radford et al. 2018. GPT-1 and BERT are from the same year and few people cared about GPT at the time; predicting the next word is much older (lecture 1). Every position is scored, so this sentence teaches four guesses where BERT's teaches one. This is the objective of every model in this course. Both objectives are scored the same way, and that score is the next six slides.

---

<p class="kicker">The loss · 1</p>

## Compare the guess with the token that came.

<img class="fig lent" src="https://substack-post-media.s3.amazonaws.com/public/images/d162da5e-a14f-42ba-bf51-9425b199fd35_1242x1188.png" alt="a decoder-only transformer turns tokens into next-token probabilities; the loss compares them with the actual next token">

<p class="caption">Figure: Cameron R. Wolfe, <a href="https://cameronrwolfe.substack.com/p/language-model-training-and-inference">Language Model Training and Inference: From Concept to Code</a>.</p>

Note: Bottom to top is last week: tokens, embeddings, blocks, a linear layer, a softmax, and out come probabilities for the next token. The red path is today. Take the probabilities the model gave, take the token that really came next, and compute one number that says how far apart they are. The image loads from the author's site, so it needs the network.

---

<p class="kicker">The loss · 2</p>

## The loss is the surprise at the right token.

<img class="fig" src="figures/05_t_neglog.png" alt="minus log p falls from high to zero as p goes to one">

<p class="inwords">In words: loss = −ln p(right token). A high probability on the right token gives a loss near zero; a low one gives a large loss.</p>

Note: The model gave six probabilities (slide 2); the loss looks at one of them, the one of the token that really came next. This is lecture 1's surprise with one change: natural log instead of log base 2, because torch uses it. The unit is nats; bits = nats ÷ 0.693. Read the three dots: p = 0.90 costs 0.11, the fair die costs 1.79, last week's model costs 2.03. The curve is steep near zero: giving the truth a probability of 0.01 costs 4.6.

---

<p class="kicker">The loss · 3</p>

## One line of numpy.

```python run
import numpy as np
words = ["the", "dog", "cat", "chased", "sat", "old"]
p = np.array([0.2041, 0.1182, 0.1318, 0.2186, 0.1804, 0.1470])

print(round(-np.log(p[words.index("cat")]), 2))
print(round(-np.log(1 / 6), 2))
```

<pre class="output fragment">2.03
1.79</pre>

<p class="shape">last week's model: 2.03 nats · a fair die: 1.79 nats · lower is better</p>

Note: Run it in the slide (the run button), or click once to show the output: 2.03, then 1.79. The six numbers are the bars of slide 2.

---

<p class="kicker">The loss · 4</p>

## Average it over every position.

<div class="tokline lg">
<span class="tok">dog</span><span class="tok">chased</span><span class="tok">the</span><span class="tok">cat</span>
</div>
<div class="tokline lg">
<span class="tok hit">2.77</span><span class="tok hit">2.17</span><span class="tok hit">1.82</span><span class="tok hit">2.03</span>
</div>

<p class="inwords">In words: (2.77 + 2.17 + 1.82 + 2.03) ÷ 4 = 2.20 nats. That one number is the cross-entropy loss.</p>

Note: Four guesses, four surprises, one mean; the numbers are last week's model on its own sentence. The name comes after the thing: cross-entropy is the average surprise of the model at the true next token. In torch it is F.cross_entropy(logits, targets), which takes the raw scores, does the softmax and the −ln, and averages over every position of every example in the batch. BERT uses the same function on its masked positions.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">The loss · 5</p>

## An untrained model starts at ln(vocabulary).

| vocabulary | guessing evenly | loss (nats) |
|---|---|---|
| 6 words | p = 1/6 | 1.79 |
| 2,048 tokens | p = 1/2,048 | 7.62 |
| 50,257 tokens (GPT-2) | p = 1/50,257 | 10.82 |

<p class="inwords">In words: guessing evenly gives the right token p = 1/V, and −ln(1/V) = ln V. Our model before its first step: 7.66.</p>

Note: Random weights give nearly even probabilities, so the first loss you print should be close to ln(vocabulary size). It is the cheapest check there is. If your first loss is 30, something is wrong before training started. The 7.66 is code/loop.py at step 0, on held-out text.

---

<p class="kicker">The budget · 1 · March 2022</p>

## Chinchilla.

<img class="fig paper" src="assets/papers/hoffmann2022_abstract.png" alt="Title and abstract of Hoffmann et al. 2022, Training Compute-Optimal Large Language Models">

Note: Hoffmann et al., DeepMind, arXiv 2203.15556. Over 400 models trained to answer one question: for a fixed amount of compute, how big should the model be and how much text should it read? Chinchilla is the name of the model they trained to test the answer.

---

<p class="kicker">The budget · 2</p>

## For one budget there is a best model size.

<img class="fig" src="figures/05_chinchilla_isoflop.png" alt="loss against parameters for four compute budgets, each a valley">

<p class="shape">Hoffmann et al. 2022, "Chinchilla" · the bottom of each valley sits near 20 tokens per parameter</p>

Note: Each curve is one compute budget, spent in different ways: left is a small model reading a lot, right is a big model reading little. Both ends lose. The curves are the fit from the paper, drawn by figures/05_chinchilla_isoflop.py. The valley is flat, so being off by a factor of two costs little. Many recent models train well past 20 tokens per parameter, because a smaller model is cheaper to run afterwards.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">The budget · 3</p>

## Training costs about 6 × parameters × tokens.

<img class="sticker tr sm" src="stickers/hourglass.png">

| run | parameters N | tokens D | cost in FLOPs: 6ND |
|---|---|---|---|
| ours, today | 1.1 million | 0.7 million read | 4.8 × 10¹² |
| GPT-3 | 175 billion | 300 billion | 3.15 × 10²³ |

<p class="inwords">In words: a FLOP is one multiplication or one addition of two numbers. Per token, each weight costs about 2 going forward and 4 going backward.</p>

Note: FLOPs, floating-point operations: one multiply or one add. Forward is about 2 per weight per token, backward about 4, so 6ND. GPT-3: 6 × 175e9 × 300e9 = 3.15e23; the paper's table reports 3.14e23 (Brown et al. 2020, appendix D). Ours: 6 × 1,115,264 × 718,848 = 4.8e12, which a laptop does in about 35 seconds. Source for the rule: CS336 lecture 2.

---

<p class="kicker">The slope</p>

## The gradient says which way is downhill.

<img class="fig" src="figures/05_gradient_descent.png" alt="a parabola with dots walking down to its minimum">

<p class="inwords">In words: new weight = old weight − learning rate × slope. Here: 1 − 0.05 × (−16) = 1.8.</p>

Note: One weight, one loss, loss = (2w − 6)². The slope at w = 1 is −16; with a learning rate of 0.05 the step is +0.8. Repeat. A real model has a million weights and one slope per weight; the list of slopes is the gradient. You never compute it by hand: torch records every operation of the forward pass and loss.backward() walks them in reverse. That is all we say about backpropagation today.

---

<p class="kicker">The optimizer · 1</p>

## The optimizer moves the weights.

<img class="fig" src="figures/05_t_optimizers.png" alt="SGD, momentum, RMSprop, Adam, AdamW in a row, each with the idea it adds">

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
```

Note: You instantiate an optimizer and give it the model's weights; you do not write it. lr is the learning rate, the step size from the last slide. Each one in the row adds an idea to the one before. SGD, stochastic gradient descent (Robbins and Monro 1951): the rule from the slope slide, with the slope measured on one batch. Momentum (Polyak 1964): follow the average of the recent slopes, so one noisy batch does not throw you. RMSprop (Hinton's 2012 lecture notes): weights whose slopes are always large take smaller steps. Adam (Kingma and Ba 2014) does both. AdamW (Loshchilov and Hutter 2017) adds weight decay, a small pull of every weight towards zero, done separately from the slope. AdamW trains nearly every LLM; SGD with momentum is still common in vision. You do not need the formulas today. If someone asks for the formulas: the Stanford CS336 handout, section 4.3.

---

<p class="kicker">The optimizer · 2 · November 2017</p>

## AdamW.

<img class="fig paper" src="assets/papers/loshchilov2017_abstract.png" alt="Title and abstract of Loshchilov and Hutter, Decoupled Weight Decay Regularization">

Note: Loshchilov and Hutter, arXiv 1711.05101, published at ICLR 2019. The W is weight decay. The paper's point: in Adam, adding the decay to the gradient is not the same as shrinking the weights directly, and shrinking them directly works better. Skippable if time is short; nothing later depends on it.

---

<p class="kicker">The schedule · 1</p>

## The learning rate changes during the run.

<img class="fig" src="figures/05_t_schedules.png" alt="five learning rate shapes: constant, inverse square root, linear, cosine, warmup-stable-decay">

<p class="shape">all start with a warmup, a short ramp up from near zero · they differ in how they come down</p>

Note: Warmup: the first steps use a small learning rate because the gradients of a random model are large and badly scaled. After it, the common shapes. Constant: simplest, rarely the best. Inverse square root: the original Transformer (Vaswani et al. 2017). Linear to zero: BERT, and the default in Hugging Face's course. Cosine: GPT-3 (down to 10% of the peak) and most course code. Warmup-stable-decay, WSD: stay at the peak, cool down at the end. The panels are schematic, same peak and same length.

---

<p class="kicker">The schedule · 2 · May 2024</p>

## Warmup, stable, decay.

<img class="fig paper" src="assets/papers/hagele2024_abstract.png" alt="Title and abstract of Hägele et al. 2024, Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations">

Note: Hägele et al., EPFL and Hugging Face, arXiv 2405.18392. Their argument: a cosine schedule ties each run to one length. A constant learning rate with a short cooldown does as well, and one run can serve several lengths, since you can cool down whenever you want a checkpoint. SmolLM3 was trained this way.

---

<p class="kicker">The schedule · 3</p>

## Ours: 15 steps up, flat, down in the last 20%.

<img class="fig" src="figures/05_t_schedule.png" alt="the learning rate of loop.py over 351 steps, with a cosine for comparison">

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
```

Note: lr_factor is the red line as a function: it takes the step and returns a number between 0 and 1, and the scheduler multiplies the peak learning rate by it. Like the optimizer, you instantiate it once and call it once per step.

---

<p class="kicker">The logbook · 1</p>

## Log every run.

<img class="sticker tr sm" src="stickers/book.png">

```python
import wandb
wandb.init(project="fundamentals-of-llms", config={"lr": 3e-3, "batch_size": 16})

wandb.log({"train_loss": 5.33, "val_loss": 5.36, "lr": 0.003}, step=50)
```

<p class="shape">Weights & Biases · every number you log becomes a curve · every run keeps its settings</p>

Note: Two calls. init opens a run and stores the settings; log sends a row of numbers. The website draws the curves, overlays runs, and lets you write a report with the plots embedded. Free for students. A text file of JSON lines does the same job with less comfort; what matters is that the settings and the curve are saved together, by the run itself.

---

<p class="kicker">The logbook · 2</p>

## Each number you log becomes a curve.

<img class="fig shot" src="assets/wandb_runs.png" alt="two Weights & Biases panels: validation loss curves of three runs, and a bar chart of how long each run took">

<p class="caption">Screenshot: three runs in a public W&B report, <a href="https://wandb.ai/ai-team-articles/muon-nanogpt/reports/Understanding-the-Muon-Optimizer-Explained-and-Tested-against-AdamW-with-NanoGPT--VmlldzoxNzExOTE4MQ">Understanding the Muon Optimizer</a> (W&B articles).</p>

Note: Somebody else's runs, not ours: a small GPT trained three times with different optimizers. Each coloured line is one run; the bars on the right are how long each took. Every run keeps the settings it was started with, so two curves can always be traced back to what differed. Skippable if time is short; nothing later depends on it.

---

<p class="kicker">The logbook · 3</p>

## A report is the curves with your words around them.

<img class="fig shot" src="assets/wandb_report.png" alt="a Weights & Biases report: paragraphs of text followed by live panels for batch loss, gradient norm and training loss">

<p class="caption">Screenshot: the same public W&B report.</p>

Note: A report is a page where the charts are live panels and the text is yours. Read the second paragraph aloud: it states what was held fixed and what changed, then reads numbers off the curves. That is the form of every comparison you will write. Skippable if time is short; nothing later depends on it.

---

<p class="kicker">Words · batch</p>

## A batch is a stack of examples.

<img class="fig" src="figures/05_t_batch.png" alt="sixteen rows of token cells, each row one example of 128 tokens">

<p class="shape">shape (16, 128) · batch and mini-batch mean the same · the model reads all 16 at once</p>

Note: From here on a window is called an example: one data point. The model never sees one example at a time; it sees a batch, a table with one example per row. Batch size: the number of rows, 16. Context, or block size: the number of tokens per row, 128. Batch and mini-batch mean the same thing today.

---

<p class="kicker">Words · step, epoch</p>

## A step is one batch. An epoch is the whole corpus once.

<img class="fig" src="figures/05_t_units.png" alt="a strip of 117 cells, the first one red">

Note: One step: read one batch, update the weights once. It reads 16 × 128 = 2,048 tokens. One epoch: as many steps as it takes to read the training set once, 239,791 ÷ 2,048 ≈ 117. This is the sum you do when you decide how long to train. Our examples are cut at random places, so an epoch here means "as many tokens as the corpus has", not "every token exactly once". LLM pretraining is usually counted in steps or tokens and often stays under one epoch, because the corpus is larger than the budget.

---

<p class="kicker">The loop</p>

## One step, five boxes.

<img class="fig" src="figures/05_t_loop.png" alt="batch, model, loss, gradients, update, and an arrow back to the start">

Note: The map for the next slides. Each box becomes one or two lines of torch. Everything before this slide was one of the boxes: the examples make the batch, last week is the model, cross-entropy is the loss, backward gives the gradients, the optimizer does the update.

---

<p class="kicker">The loop · setup</p>

## A tokenizer, the data, a model.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
merges, specials = tokenizer.load("data/tokenizer.json")
train_ids = data.load_split("train")

model = GPT(vocab_size=2048, n_layer=4, n_head=4, d_model=128, block_size=128)
model.to(device)
```

<pre class="output fragment">1,115,264 parameters on cpu</pre>

<p class="shape">the tokenizer is already trained · the model's weights are random</p>

Note: tokenizer.py is lecture 2, model.py is lectures 3 and 4: four blocks, four heads, width 128, examples of 128 tokens. device: the graphics card if there is one, else the processor; model.to(device) moves the weights there. The file is code/loop.py; python loop.py takes about 35 seconds on a laptop.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">The loop · the numbers you choose</p>

## Three epochs of 117 steps.

```python
epochs, batch_size, context = 3, 16, 128
peak_lr, warmup = 3e-3, 15

steps_per_epoch = len(train_ids) // (batch_size * context)     # 117
total_steps = epochs * steps_per_epoch                         # 351
```

<p class="shape">these are the hyperparameters: you choose them · the weights are the parameters: training chooses them</p>

Note: The sum from the epoch slide, in code. In loop.py the first of the two computed lines also multiplies by accumulate, which is 1 until the gradient accumulation slide.

---

<p class="kicker">The loop · batch</p>

## Sixteen examples at once.

```python
x, y = data.get_batch(train_ids, batch_size, context)
x, y = x.to(device), y.to(device)
print(x.shape, y.shape)
```

<pre class="output fragment">torch.Size([16, 128]) torch.Size([16, 128])</pre>

<p class="shape">x · the ids the model reads · y · the same ids moved by one</p>

Note: Click once to show the shapes: 16 rows of 128 token ids for x, the same for y.

---

<p class="kicker">The loop · model and loss</p>

## Forward, then one number.

```python
logits, _ = model(x)
loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
print(logits.shape, loss.item())
```

<pre class="output fragment">torch.Size([16, 128, 2048]) 7.666</pre>

<p class="shape">logits: 2,048 raw scores for each of 16 × 128 positions · the loss: their average surprise</p>

Note: Logits: the raw scores before the softmax. flatten turns (16, 128, 2048) into (2048, 2048): one row of scores per position, and y into 2,048 right answers; that the two numbers are both 2,048 is a coincidence of 16 × 128. cross_entropy does softmax, −ln and the mean. 7.666 against 7.62: the check from the loss slides passes.

---

<p class="kicker">The loop · gradients and update</p>

## Four lines update every weight.

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

- `zero_grad` clears the gradients of the last step.
- `backward` computes a new gradient for every weight.
- `clip_grad_norm_` shortens the gradient if it is longer than 1.0.
- `step` lets AdamW move every weight.

Note: The gradient is the list of slopes, one per weight. Gradients add up in torch until you clear them, so zero_grad comes before backward. Forget it and the loop still runs and still learns, only worse (you will try it in a few minutes). Clipping: if the gradient as a whole is longer than 1.0, scale it down to 1.0, so one strange batch cannot throw the weights far. Bugs in training loops rarely crash. Break after this slide: about 45 minutes in, and the whole loop comes right after it.

---

<p class="kicker">The loop · all of it</p>

## The training loop.

```python
step = 0
model.train()
for epoch in range(epochs):
    for _ in range(steps_per_epoch):
        optimizer.zero_grad()
        x, y = data.get_batch(train_ids, batch_size, context)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % log_every == 0:
            log(step, loss.item())
```

Note: Two loops: epochs outside, steps inside. Inside: clear the gradients, batch, forward, loss, then backward, clip and step from the last slide. scheduler.step(): set the learning rate for the next step. log: print the losses and send them to W&B. model.train() says "we are training"; some layers behave differently when evaluating. This is the loop of code/loop.py with accumulate = 1; the Hugging Face course's loop has the same lines with a DataLoader in place of get_batch.

---

<p class="kicker">The loop · run it</p>

## 351 steps, about 35 seconds.

<pre class="output visible">3 epochs x 117 steps = 351 steps
step 0    lr 0.0002  val_loss 7.6592
step 50   lr 0.0030  val_loss 5.3558  train_loss 5.3281
step 100  lr 0.0030  val_loss 4.4031  train_loss 4.4239
-- epoch 1 of 3 done
...
step 351  lr 0.0000  val_loss 3.4155  train_loss 3.3723
-- epoch 3 of 3 done</pre>

<div class="quote" style="font-size:1.05em">One day, a big dog and the puppy was very nice.</div>

<p class="shape">val_loss: the loss on stories kept aside · 3.42 nats per token = 1.31 bits per byte</p>

Note: Run it live: cd code && python loop.py. The output is the real one (one run, seed 0, laptop CPU), with the rows between step 100 and step 351 and the closing timing line cut. The run takes 31 to 38 seconds on this laptop, depending on what else it is doing. val_loss is the loss on held-out text, next slide. e^3.42 ≈ 30: the model is now as unsure as a fair 30-sided die, down from 2,048 sides; e^loss is called perplexity. Bits per byte: 3.4155 ÷ 0.693 × 0.266 tokens per byte = 1.31; it lets you compare models with different tokenizers. The quote is how its continuation of "One day" begins.

---

<p class="kicker">The loop · check it</p>

## Measure on text the model never trained on.

<img class="sticker tr sm" src="stickers/scales.png">

<img class="fig" src="figures/05_t_run.png" alt="held-out and training loss falling together over 351 steps">

<p class="shape">both fall together: it is learning · only the training loss falls: it is memorising</p>

Note: The last 10% of the stories never enter training (data.py). Every 50 steps we stop, switch off gradients, and average the loss over the same 20 batches of them. This is the val_loss column; held-out, validation and development set are three names for the same thing. While the two curves fall together the model is learning something general. When training loss keeps falling and held-out loss stops, it is memorising (overfitting). In the longer run of train.py (976 steps, code/README.md) that starts at about step 500, because 240k training tokens is very little text. One run, one seed.

---

<p class="kicker">The loop · use it</p>

## To write, the model guesses one token at a time.

<img class="fig lent" src="https://substack-post-media.s3.amazonaws.com/public/images/08729f45-ace9-419d-80dd-4520c878cfac_2300x1164.png" alt="three time steps: the model reads the text so far and adds one token each time">

<p class="caption">Figure: Cameron R. Wolfe, <a href="https://cameronrwolfe.substack.com/p/language-model-training-and-inference">Language Model Training and Inference: From Concept to Code</a>.</p>

Note: The sentence two slides back was written this way: feed "One day", take the probabilities for the next token, pick one, append it, feed the longer text again. Training guesses every position at once because the answers are in the text; writing has to go one token at a time because the answer is what it just wrote. Decoding is lecture 8. The image loads from the author's site, so it needs the network.

---

<p class="kicker">The loop · save it</p>

## Save the run as you go.

```python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step}, "ckpt.pt")
```

<p class="shape">weights · optimizer state · step · enough to continue after a crash</p>

Note: A checkpoint. Jobs on a shared cluster get stopped; a run that saves every few hundred steps loses minutes, one that does not loses everything. The optimizer state matters: AdamW's running averages are part of where the run is. train.py does this every time it evaluates and resumes with --resume; loop.py leaves it out to stay short.

---

<p class="kicker">The loop · share it · 1</p>

## Models are shared on the Hugging Face Hub.

<img class="fig shot" src="assets/hf_gpt2.png" alt="the Hugging Face page of GPT-2: model card, files tab, downloads last month, model size">

<p class="caption">Screenshot: huggingface.co/openai-community/gpt2.</p>

Note: The Hub hosts most open models, datasets and demos. Each model gets a repository with its weights, a model card (what it is, how it was trained) and a download counter. This is GPT-2's page. When a paper says "we release our model", this is usually where.

---

<p class="kicker">The loop · share it · 2</p>

## Uploading a checkpoint takes three lines.

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-name/tiny-gpt", exist_ok=True)
api.upload_file(path_or_fileobj="runs/example/ckpt.pt",
                path_in_repo="ckpt.pt", repo_id="your-name/tiny-gpt")
```

<p class="shape">log in once with a token · a repository per model · add a model card</p>

Note: Not run for this deck: we did not upload anything. huggingface_hub is the Hub's Python library; you log in once (huggingface-cli login) with a token from your account. create_repo makes the repository, upload_file puts the checkpoint in it. A raw checkpoint like ours is only loadable with our own model.py; models written with the transformers library use save_pretrained and push_to_hub, which also store the configuration. Write the model card: what data, how many tokens, which loss. Skippable if time is short; nothing later depends on it.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">Now you · why each part is there</p>

## Change one thing in loop.py and see what it costs.

| change | what goes wrong |
|---|---|
| `peak_lr = 3.0` | the steps are too big |
| `peak_lr = 0.00003` | the steps are too small |
| delete `optimizer.zero_grad()` | old gradients pile up |

<p class="shape">one change per run · first write down the held-out loss you expect at the end</p>

<img class="sticker br sm" src="stickers/manicule-right.png">

Note: Ten minutes, pairs, laptops, code/loop.py, about 40 seconds per run. Three runs, one change each, and put the line back before the next. The point is to see why the learning rate has to be in a narrow range and why zero_grad is in the loop: neither mistake gives an error message, the run just ends worse. Ask for the guess before each run. Then ask which of the three would be hardest to notice in a real project.

---

<p class="kicker">Knobs · learning rate</p>

## None of the three crashes; each ends with a worse loss.

<img class="fig" src="figures/05_t_broken.png" alt="held-out loss: unchanged 3.42, no zero_grad 5.25, learning rate 0.00003 6.43, learning rate 3.0 7.41">

<p class="shape">loop.py with one line changed · seed 0, one run each</p>

Note: Measured with loop.py as it is in the repo, changing only the one line. Learning rate 3.0: the steps overshoot, the loss blows up and ends near random. 0.00003: it learns, a hundred times too slowly. No zero_grad: every step adds the new gradient to all the old ones; it still learns, badly. The learning rate is the first thing to check when a run misbehaves, and its useful range is narrow.

---

<p class="kicker">Knobs · gradient accumulation</p>

## Gradient accumulation builds a big batch from small ones.

<img class="fig" src="figures/05_t_accum.png" alt="four small batches, four backward calls, one update">

Note: A larger batch gives a steadier gradient but has to fit in memory. Gradient accumulation: run backward on several small batches without clearing the gradients, then step once. It costs the same compute and less memory. Divide each loss by the number of batches so the sum is an average. This is why zero_grad is a separate call. In loop.py: set batch_size to 4 and accumulate to 4.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">Knobs · effective batch size</p>

## The effective batch is the number of examples per update.

| batch | accumulate | effective batch | updates per epoch | held-out loss |
|---|---|---|---|---|
| 16 | 1 | 16 | 117 | 3.42 |
| 4 | 4 | 16 | 117 | 3.43 |
| 4 | 1 | 4 | 468 | 3.30 |

<p class="inwords">In words: effective batch = batch × accumulation steps. Updates per epoch = tokens ÷ (effective batch × context).</p>

Note: Batch (or mini-batch): what goes through the model in one forward pass, limited by memory. Effective batch: what one update is based on. Rows 1 and 2 are the same training run paid differently, and they end within 0.01 of each other (one seed each); row 2 needs a quarter of the memory. Row 3 is a different run: four times as many updates for the same tokens. 239,791 ÷ (16 × 128) = 117; ÷ (4 × 128) = 468. All three: loop.py, 3 epochs, seed 0, one run each. When you read a paper's batch size, it is the effective one.

---

<p class="kicker">On a GPU · 1</p>

## Watch two numbers while it trains.

<img class="fig" src="figures/05_t_gpu.png" alt="two gauges, memory and utilisation, both nearly full">

<p class="shape">nvidia-smi shows both · a schematic, not a measurement</p>

Note: nvidia-smi prints, per card, the memory in use and the utilisation, the share of time the card was computing. Memory far below the limit: the batch could be bigger. Utilisation far below 100%: the card is waiting, usually for data loading or logging, and a faster model will not help. General guidance, nothing we measured for this slide.

---

<p class="kicker">On a GPU · 2</p>

## Doubling finds the largest batch that fits.

<img class="fig" src="figures/05_t_doubling.png" alt="16, 32, 64, 128, then 256 runs out of memory">

<p class="shape">an illustration: where it breaks depends on the model, the context and the card</p>

Note: The empirical way to find the memory limit: double the mini-batch until the card reports out of memory, step back one. The largest batch that fits gives the most tokens per second. The next slide checks whether it also gives the best model. Skippable if time is short; nothing later depends on it.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">On a GPU · 3</p>

## On a laptop CPU, batch 64 was fastest and had the worst loss.

<img class="fig" src="figures/05_t_batchsize.png" alt="batch 4: 57 s, loss 3.30; batch 16: 36 s, loss 3.42; batch 64: 32 s, loss 4.10">

<p class="shape">the same 3 epochs of tokens · loop.py on a laptop CPU, seed 0, one run each</p>

Note: Same model, same tokens, same learning rate. Tokens per second rose with the batch size, from about 12,500 to about 22,500. The big batch is the fastest per token and ends with the worst loss, because it made only 87 updates; the small batch made 1,404 and wins on loss while taking the longest. Filling the memory gives the most tokens per second. It did not give the lowest loss here, so measure both: the time it takes to reach a given held-out loss. Caveats: a laptop CPU and a tiny model, where big batches gain little speed (on a GPU they gain far more). The learning rate was not retuned per batch size. Timings move by a few seconds; a second set of runs gave 55, 33 and 32 seconds, same order. Optional exercise: try batch 64 with a higher learning rate.

---

<!-- .slide: class="bigtable" -->

<p class="kicker">Knobs · the rest</p>

## Write the knobs down with the result.

| knob | ours today | what it does |
|---|---|---|
| epochs | 3 | how long you train |
| warmup | 15 steps | how long the learning rate ramps up |
| weight decay | 0.1 | how hard weights are pulled to zero |
| gradient clip | 1.0 | the longest gradient allowed |
| seed | 0 | fixes the random start and the random examples |

<p class="shape">change one knob at a time · a number without its settings cannot be compared</p>

Note: Seed: the starting point of the random number generator; two seeds give two slightly different runs, so a difference between two runs that is smaller than the difference between two seeds means nothing. When you compare two versions of a model, everything in this table stays fixed except the one thing you are testing.

---

<p class="kicker">Today</p>

## One loss, one loop, a few knobs.

<img class="fig" src="figures/05_t_loop.png" alt="batch, model, loss, gradients, update, and an arrow back to the start">

<p class="shape">this is pretraining · post-training keeps the loop and changes the data and the loss: lecture 10</p>

Note: The whole lecture on one picture. Examples stacked into a batch; the model from last week; cross-entropy as the one number; backward for the gradients; AdamW and a schedule for the update; and around it the knobs: learning rate, batch size, epochs. Everything later in the course reuses this loop.

---

<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Next week · 19 Nov</p>

# Cleaning the web

<p class="tagline">Filtered, deduplicated web is the pretraining corpus. We clean some in class.</p>
<p class="kicker">Thursdays 15:15 · SR 127.11 · Leechgasse 34</p>
</div>
<div class="pic" style="background-image:url(assets/luyken.jpg)"></div>

---

<p class="kicker">Reading</p>

## Recommended reading.

<img class="sticker br sm" src="stickers/owl.png">

- Hugging Face LLM course, [A full training loop](https://huggingface.co/learn/llm-course/chapter3/4). The same loop, written for fine-tuning.
- Hugging Face LLM course, [Training a causal language model from scratch](https://huggingface.co/learn/llm-course/chapter7/6).
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY). The second half is this lecture.
- Stanford CS336, [lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py), for where 6ND comes from.

Note: Optional: Hoffmann et al. 2022 (Chinchilla, arXiv 2203.15556); Hägele et al. 2024 (WSD, arXiv 2405.18392); Loshchilov and Hutter 2019 (AdamW, arXiv 1711.05101); Devlin et al. 2018 (BERT, arXiv 1810.04805); Radford et al. 2018 (GPT-1).

---

<p class="kicker">Reading</p>

## Extra material.

- Hoffmann et al. 2022, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). The Chinchilla paper.
- Hägele et al. 2024, [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392). The WSD paper.
- Sebastian Raschka, [Build a Large Language Model From Scratch, chapter 5](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05). Pretraining on unlabeled data, as notebooks.
- Weights & Biases, [quickstart](https://docs.wandb.ai/quickstart).
