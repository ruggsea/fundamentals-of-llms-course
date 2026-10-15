<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Advanced Machine Learning · Lecture 2 · 15 October 2026</p>

# Tokenizers, from letters to bytes

<p class="tagline">LLMs do not read letters. We follow the unit they read, from RNNs to BPE to your own.</p>
<p class="kicker">Jana Lasser · Ruggero Marino Lazzaroni · Lorenz Prattes<br>Uni Graz · Winter 2026/27 · DAT.C3111UB</p>
</div>
<div class="pic" style="background-image:url(assets/diderot.jpg)"></div>

---

<p class="kicker">Last week</p>

## Last week we counted letters. The models moved on.

- Entropy is average surprise. Our letter model scored 2.10 bits per letter on Frankenstein.
- More context helped until the tables grew faster than the data: 82 % of word triples in a new book were never seen.
- So we said we would learn the counts. First the models had to change what they count.

Note: Recap of lecture 1 in one minute, from the deck-01 slides: the guessing game, entropy 2.10 bits/letter, the n-gram table blow-up, "so we will learn the counts". Set up the history slide next.

---

<p class="kicker">Last week</p>

## From counting to LLMs in five steps.

| year | what happened |
|---|---|
| 2003 | Bengio's neural LM swaps the count table for a learned function. |
| 2010s | RNNs generate text one letter at a time. Translation is the big task. |
| 2015 | Karpathy's character RNN writes Shakespeare that was never written. |
| 2016 | Sennrich brings BPE into translation: the rare-word problem. |
| 2017 | Transformers. Stack them, train them on the web: the LLM. |

<p class="source"><a target="_blank" href="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">Bengio et al. 2003</a> · <a target="_blank" href="https://arxiv.org/abs/1508.07909">Sennrich, Haddow, Birch 2016</a> · <a target="_blank" href="https://arxiv.org/abs/1706.03762">Vaswani et al. 2017</a></p>

Note: One breath, not a history lecture. Karpathy 2015: "The Unreasonable Effectiveness of Recurrent Neural Networks" (char-rnn), a character-level LM. RNNs were used mostly for translation; that is where the open-vocabulary problem bites, and that is the paper BPE comes from (next part). Transformer = "Attention Is All You Need". We build a transformer block in lectures 3 and 4.

---

<p class="kicker">The problem</p>

## The LLM never sees your text. It sees tokens.

- Bad spelling, bad string reversal, bad arithmetic, a model that halts on one special string: Karpathy calls tokenization "the root of all suffering" in LLMs.
- All of it traces to one place: what the model calls a unit of text.
- Today we choose that unit, on purpose, and build it.

<p class="source"><a target="_blank" href="https://www.youtube.com/watch?v=zduSFxRajkE">Karpathy · Let's build the GPT Tokenizer (2024)</a></p>

Note: The famous list from the video: "Why do LLMs struggle with spelling? Reversing strings? Non-English? Arithmetic?" All are tokenization artifacts. Say two of them, let the third be the tease. fast.ai text+code re-rendering of the video: https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html

---

<p class="kicker">The interface</p>

## GPT-4o's tokenizer cuts this line into 8 pieces, and the globe into two.

<p style="font-family: var(--r-code-font); font-size: 1.1em; line-height: 2.2;"><code>Hello</code> <code>,</code> <code>␣F0 9F 8C</code> <code>8D</code> <code>!</code> <code>␣</code> <code>你好</code> <code>!</code></p>

<p class="caption">"Hello, 🌍! 你好!" in o200k_base, the tokenizer of GPT-4o. 13 characters, 20 bytes, 8 pieces. ␣ is a space. Hex numbers are raw bytes: the globe is stored as 4 bytes, and Part 1 shows why.</p>

<div class="fragment">

- Each piece is a *token*. The model only ever sees tokens.
- `你好` ("hello") is one token. The globe is not a token at all: three of its bytes ride with the space, the fourth stands alone.

</div>

Note: Let the room guess the piece count before the fragment. Pieces from tiktoken, decode_single_token_bytes per id: [b'Hello', b',', b' \xf0\x9f\x8c', b'\x8d', b'!', b' ', b'\xe4\xbd\xa0\xe5\xa5\xbd', b'!']. o200k_base is OpenAI's 2024 encoding for GPT-4o. TODO: decide whether to also show the id list today.

---

<p class="kicker">The interface</p>

## A tokenizer is two functions: text to integers, and back.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
ids = enc.encode("Hello, 🌍! 你好!")        # text -> list of integers
print(ids)
print(enc.decode(ids))                      # integers -> text
```

<pre class="output fragment">[13225, 11, 130321, 235, 0, 220, 177519, 0]
Hello, 🌍! 你好!</pre>

`decode(encode(s)) == s` must hold for every string.

Note: Not live (tiktoken does not run in the browser); output from the figures venv. The integer is the token's row number in the vocabulary, next slide but one. CS336 lecture 1 defines this abstract Tokenizer interface; we keep it.

---

<!-- .slide: class="divider" data-background-image="assets/stradanus.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 1</p>

## Text becomes bytes.
</div>

---

<p class="kicker">Characters, bytes, words · 1</p>

## Characters give a huge vocabulary and almost no compression.

```python run
for ch in ["a", "è", "你", "🌍"]:
    print(ch, ord(ch))
```

- About 150,000 code points, most of them rare. One token per character: our line is 13 tokens.
- CS336 calls it "the worst of both worlds": large vocabulary, low compression.

Note: Compression ratio = bytes of text per token. 150K code points: CS336 lecture 1, citing Wikipedia. "È" matters: Italian is written with it, and it costs 2 bytes in UTF-8, next slide.

---

<p class="kicker">Characters, bytes, words · 2</p>

## Bytes give a vocabulary of 256 and no compression at all.

```python run
for ch in ["a", "è", "你", "🌍"]:
    print(ch, list(ch.encode("utf-8")))
```

- UTF-8 stores common characters in fewer bytes: a is 1, è is 2, 你 is 3, 🌍 is 4.
- Every string in every language is a byte sequence. Nothing is out of vocabulary.
- One token per byte: our line is 20 tokens. Long texts get very long.

Note: Same loop as the character slide, now over bytes. Why length hurts: attention compares every token with every other (next week). Byte-level BPE keeps this "nothing is unknown" property, Part 3.

---

<p class="kicker">Characters, bytes, words · 3</p>

## Words compress well, but the vocabulary never closes.

| unit | vocabulary | tokens for "Hello, 🌍! 你好!" | a new word? |
|---|---|---|---|
| characters | ~150,000 | 13 | fine |
| bytes | 256 | 20 | fine |
| words | open-ended | 8 | `UNK` |
| BPE (o200k_base) | 200,019 | 8 | fine, falls back to bytes |

- Our four novels have 21,422 word types; 8,221 occur once. A word the model never saw becomes `UNK`, and the model learns nothing about it.
- BPE is the unit in between: byte-sized floor, word-sized ceiling. We build it now.

Note: Word counts: four Gutenberg novels from lecture 1, lowercased runs of letters. Last week: 2.7 % of Frankenstein's running words never occur in those four novels. Table is the whole lecture in one picture; point at the last row.

---

<p class="kicker">Characters, bytes, words · 4</p>

## BPE is a 1994 compression scheme. Sennrich made it a tokenizer.

- Gage, 1994: replace the most frequent pair of bytes with one new symbol, repeat. A compression trick for text.
- Sennrich, Haddow, Birch 2016: use it on translation, where rare words break the open vocabulary. It sets the standard for neural language modelling.
- The tokenizer is trained once, before the model, and it is just data: a list of merges.

<p class="source"><a target="_blank" href="https://arxiv.org/abs/1508.07909">Sennrich, Haddow, Birch · NMT of Rare Words with Subword Units, ACL 2016</a></p>

Note: Gage 1994, "A New Algorithm for Data Compression", C Users Journal. Sennrich et al.'s own framing: NMT forces a fixed vocabulary; back-off to "UNK" for rare words; BPE works on the symbol (here: byte) sequence, so "the final symbol vocabulary size is equal to the initial vocabulary plus the number of merge operations". Train it as its own stage, on its own data. This lecture follows that arc: paper version first, then GPT-2's version, then today's.

---

<!-- .slide: class="divider" data-background-image="assets/babbage.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 2</p>

## BPE by hand.
</div>

---

<p class="kicker">BPE by hand · 1</p>

## BPE merges the most frequent pair, again and again.

<div class="bpe-board" data-id="bpe">

`a a a b d a a a b a c`

<div class="fragment">

**merge 1.** `aa` → `Z`, it occurs 3 times:

`Z a b d Z a b a c`

</div>

<div class="fragment">

**merge 2.** `ab` → `Y`, 2 times, tied with `Za`; the tie goes to the pair that sorts later:

`Z Y d Z Y a c`

</div>

<div class="fragment">

**merge 3.** `ZY` → `X`, 2 times:

`X d X a c`

</div>

</div>

Note: The example is Wikipedia's BPE article, after Gage 1994. One step per click, and every step stays on screen. The tie at merge 2: "Za" and "ab" both occur twice; capitals sort before lowercase, so "ab" > "Za" and "ab" wins. After merge 3 every pair occurs once: we stop. Eleven symbols became five, and the vocabulary grew by three. The three merges ARE the tokenizer; ask the room what happens to a new string, answer: replay the same merges in the same order (the recipe, next slide).

---

<p class="kicker">BPE by hand · 2</p>

## The recipe, on bytes.

1. Start with the 256 byte values as the vocabulary.
2. Count every adjacent pair of symbols in the training text.
3. Merge the most frequent pair into a new symbol, number 256, 257, 258, ...
4. Repeat until the vocabulary has the size you chose.

The vocabulary is 256 + one entry per merge + the special tokens.

Note: Same procedure as the previous slide, one abstraction level up: bytes, not letters. This is GPT-2's byte-level BPE (Radford et al. 2019). Special tokens next part. Lorenz does the by-hand part now.

---

<!-- .slide: class="statement" -->

<div>

# Now you: find the first six merges.

`gatto gatto gatto gatti gatti matto matto matti gattino`

<p class="caption">Split on spaces. Merge the most frequent pair. A tie goes to the pair that sorts later.</p>
</div>

Note: Lorenz runs this block, pen and paper in pairs, about 10 minutes. Same rules as the toy corpus. Afterwards change the vocab line of the training-loop code (next slide) to tuple("gatto"): 3 etc. and let them check.
Answer (figures/02_toy_merges.py): 1. t+t (9, tied with a+t) 2. a+tt (9) 3. g+att (6) 4. m+att (3, three-way tie) 5. gatt+o (3) 6. gatt+i (3). After six merges: gatto, gatti, matt·o, matt·i, gatti·n·o. Point to make: "gattino" is built from pieces learned from other words.

<img class="sticker br" src="stickers/manicule-right.png">

---

<p class="kicker">BPE by hand · 3</p>

## Counting the pairs is a Counter over neighbours.

```python run
from collections import Counter
words = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
pairs = Counter()
for word, freq in words.items():
    for pair in zip(word, word[1:]):
        pairs[pair] += freq
print(pairs.most_common(4))
print("the rule picks", max(pairs, key=lambda p: (pairs[p], p)))
```

Note: Sennrich et al.'s stylized corpus. zip(word, word[1:]) walks the neighbours: ('l','o'), ('o','w'). A tie goes to the lexicographically greater pair (CS336 A1 2.4): e+s and s+t tie at 9, s+t wins. most_common shows e+s first because insertion order; the max line applies the rule.

---

<p class="kicker">BPE by hand · 4</p>

## The whole training loop fits on one slide. Press run.

```python run [1-6|7-11|12-13]
def merge(word, a, b):                        # every neighbouring a, b becomes one symbol ab
    out = []
    for sym in word:
        if out and out[-1] == a and sym == b: out[-1] = a + b
        else: out.append(sym)
    return tuple(out)
vocab = {tuple("low"): 5, tuple("lower"): 2, tuple("widest"): 3, tuple("newest"): 6}
for step in range(1, 8):
    pairs = {}
    for word, freq in vocab.items():
        for pair in zip(word, word[1:]): pairs[pair] = pairs.get(pair, 0) + freq
    a, b = max(pairs, key=lambda p: (pairs[p], p))        # most frequent; ties: greater pair
    vocab = {merge(w, a, b): f for w, f in vocab.items()}
    print(step, a, "+", b, pairs[a, b], ["·".join(w) for w in vocab])
```

Note: Three highlights: merge one pair inside a word (1-6), count the pairs (7-10), pick and merge everywhere (11-13). It prints the seven merges that turn "newest" into one token (the table is in the repo deck if anyone wants it). After Sennrich et al. 2016 Algorithm 1 with the A1 tie-break. It recounts everything after each merge: fine for a slide, too slow for real data; code/tokenizer.py in the course repo only recounts the words that changed. Say so, it is the A1 hint.

---

<!-- .slide: class="divider" data-background-image="assets/luyken.jpg" data-background-size="cover" data-background-position="center" -->

<div class="band">
<p class="kicker">Part 3</p>

## The real deal: GPT-2 and after.
</div>

---

<p class="kicker">GPT-2's version · 1</p>

## GPT-2 runs BPE on bytes, displayed as characters.

- Text becomes UTF-8 bytes. Each byte is mapped to one printable character for a clean vocabulary file, and mapped back after merging.
- Nothing is ever unknown: emoji, Chinese, a typo, garbage, all decompose into bytes.
- The merges never cross a character boundary, because the training text was cut first. Next slide.

<p class="source"><a target="_blank" href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">Radford et al. 2019 · GPT-2</a>, section 2.2</p>

Note: fast.ai chapter: "Bytes, not letters. Production tokenizers start from the 256 possible bytes... each byte mapped to a printable Unicode character for clean display" (that mapping is GPT-2's bytes_to_unicode). GPT-2: 50,257 vocabulary = 256 + 50,000 merges + 1 special. Radford et al. 2019, "Language Models are Unsupervised Multitask Learners", section 2.2.

---

<p class="kicker">GPT-2's version · 2</p>

## A regex cuts the text first, so merges never cross word boundaries.

```text
PAT = '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

| alternative | matches | example |
|---|---|---|
| `'(?:[sdmt]|ll|ve|re)` | a contraction | `it's` → `it` + `'s` |
| ` ?\p{L}+` | a word, space glued to its front | ` the` ` cat` |
| ` ?\p{N}+` | a number, front space | ` 42` |
| ` ?[^\s\p{L}\p{N}]+` | a punctuation run | `!` `...` |
| `\s+(?!\S)` | spaces at a line's end | |
| `\s+` | the rest: newlines | |

<div class="fragment">

`regex.findall(PAT, "The dog ran. The dog!")` → `['The', ' dog', ' ran', '.', ' The', ' dog', '!']`

The space sticks to the *front* of the next word. `dog.` and `dog!` stay cut, so they never become tokens.

</div>

Note: One row per click if the room is slow; the pattern is GPT-2's, in the tidier form from tiktoken PR 234. \p{L} is any letter in any alphabet. Read the bottom line with the room: same string, two cuts. Sennrich's original split on whitespace only. The table replaces the old unreadable one-line regex slide.

---

<p class="kicker">GPT-2's version · 3</p>

## GPT-4's tokenizer fixes the regex and doubles the vocabulary.

| | GPT-2 | GPT-4 (o200k_base) |
|---|---|---|
| contractions | lowercase only, `HOW'S` breaks | case-insensitive `(?i:'s|'S...)` |
| numbers | ` 42`, ` 12345` one run | runs of at most 3 digits, ` 123` `456` |
| vocabulary | 50,257 | 200,019 |
| non-English | fertile | ~12 % fewer tokens for Italian (1.59 → 1.41) |

- Number chunks of fixed width make arithmetic easier to learn: the ones digit stops moving.
- Bigger vocabulary: shorter sequences, and more rows of parameters, Part 4.

<p class="source"><a target="_blank" href="https://www.youtube.com/watch?v=zduSFxRajkE">Karpathy · Let's build the GPT Tokenizer</a>, tiktoken section</p>

Note: fast.ai/Karpathy: o200k_base adds (?i:...) for contractions, digit-grouping (he says runs of 3), tightened whitespace/newline handling, 100,000 merges target ("200k vocab"). GPT-2 numbers: 50,257. The 12 % is our own figure two slides on: Pinocchio bytes/token 1.59 on GPT-2, 1.54 on cl100k, 1.41 on o200k (figures/02_tiktoken_languages.py). TODO: verify the digit-run width (2 or 3) before stating it as fact.

---

<p class="kicker">One other family</p>

## SentencePiece: Llama and Mistral tokenize differently, same idea.

- tiktoken (GPT-2, 4o): text → UTF-8 bytes → BPE on bytes.
- SentencePiece (Llama 1, 2; Mistral): BPE on Unicode characters, with a byte fallback for the rare ones.
- Both keep "nothing is unknown". Both keep a special `unk` token as a last resort.

<p class="caption">One slide, for completeness. The A1 tokenizer is tiktoken-style: bytes.</p>

Note: SentencePiece trains on characters above a coverage threshold (99.95 % in Llama 2) and falls back to bytes for the rest; add_dummy_prefix glues a space to the start of every input, same idea as the regex's front space. From the fast.ai chapter and the Llama 2 trainer_spec. SentencePiece also implements unigram LM tokenization; we do not cover it. Keep to 60 seconds.

---

<p class="kicker">Special tokens</p>

## Special tokens bypass BPE entirely.

- A registry of strings such as ‹|endoftext|› that are split off before merging and never split.
- They get their own ids after the merges, and mark document boundaries in training data.
- GPT-4 adds more: fill-in-the-middle markers for code, an end-of-prompt marker.

<p class="caption">Adding a special token later means surgery on the model: one new row of parameters.</p>

Note: Written in plain text, not inline code, so the ligature bug (design.md, deck 02) does not bite. <|endoftext|> id 50256 in GPT-2. Adding tokens later: extend the embedding table and lm_head (fast.ai "model surgery"). Our A1 specials: only endoftext; sft.py will add <|user|> and <|assistant|>.

---

<p class="kicker">Live</p>

## Our own tokenizer, trained on TinyStories. Type into it.

<div id="tok-demo">

<input id="tok-input" value="Once upon a time, there was a little dog." aria-label="text to tokenize">

<button id="tok-btn">tokenize ▸</button>

<div id="tok-out"></div>

<p class="caption" id="tok-count"></p>

<p class="caption">Our byte-level BPE from code/tokenizer.py: 256 bytes + 1,791 merges + 1 special, GPT-2 regex, trained in 4 s on the A1 corpus. Edit the text and press tokenize.</p>

</div>

Note: Interactive slide, the "real model surprises the room" moment of this deck. The engine loads slides/assets/tokenizer_merges.json (exported from code/data/tokenizer.json), replays the merges in order, and shows the pieces. Good things to try: "Ciao, come stai?" (Italian splits), "2 + 2 = 4", a spelling mistake, the longest German word you know. The old slides had figures for all of this; now the room makes their own. TODO: if the projector machine is offline, this slide is still fine (no CDN needed, the json is local; but the run-blocks above need Pyodide).

---

<p class="kicker">Lab</p>

## Your tokenizer now encodes a corpus into training windows.

<img class="sticker tr sm" src="stickers/gear.png">

```python
def train(text, vocab_size, specials=SPECIALS):                     # tokenizer.py
    parts = [part for part in split_specials(text, specials) if part not in specials]
    counts = Counter(piece for part in parts for piece in PRETOKEN.findall(part))
    return train_on_counts(counts, vocab_size - 256 - len(specials))

def get_batch(ids, batch_size, block_size):                         # data.py
    starts = np.random.randint(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + block_size + 1] for s in starts])
```

<pre class="output fragment">train: 1133 documents, 239,791 tokens, 890,183 bytes, 3.71 bytes per token
val:   125 documents,  23,934 tokens,  89,916 bytes, 3.76 bytes per token</pre>

Note: Numbers from code/README.md (tokenizer.py trains 1,791 merges on data/tiny.txt in ~4 s; data.py encodes and writes train.bin/val.bin in ~3 s). Commands from the repo root: cd code && python tokenizer.py && python data.py, run tests.py first. get_batch is lecture 5's; today it only shows where the ids go.

---

<!-- .slide: class="divider" data-background-image="assets/diderot.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 4</p>

## What a vocabulary costs.
</div>

---

<p class="kicker">Costs · 1</p>

## Every entry costs a row of parameters, and rare tokens train less.

- Each token gets a row of 768 learned numbers. GPT-2: 50,257 rows, about 38.6 M parameters, a third of the model.
- A bigger vocabulary buys shorter sequences. Italian costs 1.4 to 1.6 times the tokens of English in the same OpenAI tokenizers, and more in GPT-2.
- In a fixed context window, 1.4× the tokens means 30 % less story fits.

![Italian tokens divided by English tokens for the same story, three OpenAI tokenizers](figures/02_tiktoken_languages.png)

<p class="caption">Pinocchio, Collodi's Italian and Della Chiesa's English translation, 36 chapters each, three tokenizers. 1 would mean equal cost.</p>

Note: 768 = GPT-2 small's width. Figure: figures/02_tiktoken_languages.py (bytes per token per language per tokenizer; the bars are bytes per token, read the Italian/English comparison from the note). The 1.4–1.6 is the ratio range over the three tokenizers on the same story. 1 − 1/1.41 = 0.29 ≈ 30 %. The figure plots bytes/token; if asked, Italian bar higher = more bytes per token = worse compression.

---

<p class="kicker">Costs · 2</p>

## Compare tokenizers by bits per byte, never by bits per token.

`"the cat sat."` — 12 bytes. Two models, one sentence:

<div class="fragment">

**A**, 3 tokens: 9 bits paid for the whole sentence. **B**, 12 byte-tokens: 12 bits paid.

</div>

<div class="fragment">

Per token: A pays 3.0, B pays 1.0. B looks three times better.

</div>

<div class="fragment">

$$ \text{bits per byte} = \frac{-\sum_t \log_2 p(x_t \mid x_{<t})}{\text{UTF-8 bytes of the text}} $$

<p class="inwords">Divide the same total bits by the bytes, which no tokenizer can change. A: 9/12 = 0.75. B: 12/12 = 1.0. A wins.</p>

</div>

The A1 leaderboard ranks by bits per byte on held-out text. Your perplexity is shown next to it.

Note: Toy numbers, made up to show the arithmetic. A has the bigger vocabulary, so each prediction is harder, but it makes fewer of them. Every student brings their own tokenizer to A1, so per-token numbers cannot be compared; bits per byte is tokenizer-fair. Same story as deck 1's last counting slide: 𝓛 is the same chain, the unit changed.

---

<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Next week · 22 October</p>

# Attention, by hand

<p class="tagline">Tokens become vectors, and every vector looks at the others. Four tokens, real numbers, a pen.</p>
<p class="kicker">Thursdays 15:15 · SR 127.11 · Leechgasse 34</p>
</div>
<div class="pic" style="background-image:url(assets/babbage.jpg)"></div>

---

<p class="kicker">Reading</p>

## Read these before next Thursday.

<img class="sticker br sm" src="stickers/owl.png">

- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). Builds a production-style tokenizer from scratch; the first hour covers the material of this lecture. A [text version](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html) by fast.ai covers the same content.
- Sennrich, Haddow, Birch 2016, [Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), sections 1–3.
- Sebastian Raschka, [BPE tokenizer from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb). The same algorithm, implemented step by step in a notebook.
- Radford et al. 2019, [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), section 2.2. The byte-level BPE that GPT-2 uses.
- Stanford CS336, [assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics), section 2. Assignment 1 is to train a small language model; this is the tokenizer part of it.

Note: Karpathy's video is the one this deck's arc follows (fast.ai text version: https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html). Figures in this deck: slides/figures/02_*.py; tiktoken numbers need the figures venv.

---

<p class="kicker">Reading</p>

## Extra material.

<img class="sticker br sm" src="stickers/owl.png">

- Gage 1994, [A New Algorithm for Data Compression](https://www.cs.cmu.edu/~wcohen/10-605/papers/bpe.pdf). The data-compression paper BPE started as.
- janus, [SolidGoldMagikarp (plus prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLYq/solidgoldmagikarp-plus-prompt-generation-or-glossary), 2023. What happens when a vocabulary token is never activated during training.
- Beren Millidge, [Integer tokenization is insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/), 2023. How the GPT-2/3 tokenizers split integers, and why that hurts arithmetic.
- Yu et al. 2023, [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185). Modelling raw bytes directly, without a tokenizer.
- Andrej Karpathy, [minbpe](https://github.com/karpathy/minbpe), 2024. A minimal reference implementation of the algorithm, with exercises.
- Tao et al. 2024, [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies](https://arxiv.org/abs/2407.13623). How to choose the vocabulary size jointly with model and data size.
- [tiktokenizer](https://tiktokenizer.vercel.app/). Type text, see how different OpenAI tokenizers cut it.

Note: Synced to the course README under the lecture 2 section, same list as the Discord message.