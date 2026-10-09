<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Advanced Machine Learning · Lecture 1 · 8 October 2026</p>

# Shannon's guessing game

<p class="tagline">Language modelling is guessing the next letter. An n-gram model does it by counting.</p>
<p class="kicker">Jana Lasser · Ruggero Marino Lazzaroni · Lorenz Prattes<br>Uni Graz · Winter 2026/27 · DAT.C3111UB</p>
</div>
<div class="pic" style="background-image:url(assets/press.jpg)"></div>

Note: Part 1 is the admin, about 25 minutes. Then the live game restarts the room.

---

<!-- .slide: class="divider" data-background-image="assets/luyken.jpg" data-background-size="cover" data-background-position="center" -->

<div class="band">
<p class="kicker">Part 1</p>

## This is the plan for the course.
</div>

---

<p class="kicker">This course · 1</p>

## Our tentative division of the course by topic.

| block | dates | what you build or learn |
|---|---|---|
| Build | 8 Oct – 5 Nov | n-grams, tokenizers, attention, a transformer block, training |
| Data | 12 – 19 Nov | clean a Common Crawl sample, mix data |
| Use and measure | 26 Nov – 3 Dec | inference and quantization, evaluation |
| Post-train | 10 Dec – 7 Jan | SFT, RLHF, DPO, GRPO, reasoning |
| Agents and tools | 14 – 28 Jan | tool use, agents, reading a 2026 tech report, project discussion |

<p class="caption">SFT: supervised fine-tuning. RLHF: reinforcement learning from human feedback. DPO: direct preference optimization. GRPO: group relative policy optimization.</p>

Note: From PLAN.md and the course page. The last three Thursdays are tools, reading and the project. By January you should be able to read an LLM (large language model) tech report as documentation for things you have built. The acronyms in the post-train row are names for now; each gets its own lecture. Tentative: topics can move, the Thursdays stay.

---

<p class="kicker">This course · 2</p>

## We meet Thursdays 15:15–17:00 in SR 127.11, Leechgasse 34.

- The exception is 3 December, which is in SR 15.05.
- Active attendance is required.
- Attending 80 % of the classes is also required, and you can have up to two unjustified absences.
- General questions about the course, progress details, whatever, go to Lorenz.
- Other questions go to Ruggero, and when in doubt, cc us both.
- Solid Python is assumed. If you are short on it, the first two weeks are the time to catch up.
- Slides, schedule and reading lists live in the course repository: [github.com/ruggsea/fundamentals-of-llms-course](https://github.com/ruggsea/fundamentals-of-llms-course).
- The rules of the course are on the course page: [cs2.uni-graz.at/teaching/fundamentals-of-llms](https://cs2.uni-graz.at/teaching/fundamentals-of-llms/).

Note: DAT.C3111UB, 3 ECTS, elective in the Data Science and Mathematics masters; TU Graz students welcome. Present yourselves by name here. The communication channel is on the Discord slide below.

---

<p class="kicker">This course · 3</p>

## Two individual assignments.

<p class="caption">GPT: generative pretrained transformer.</p>

| | weight | date |
|---|---|---|
| Assignment 1: your tokenizer and a small GPT, fixed training budget | 30 % | out 5 Nov, due Wed 9 Dec |
| A1 oral discussion, ten minutes | 10 % | 16–18 Dec, bookable slots |
| Assignment 2: post-train your A1 model, build an eval, write it up | 40 % | report due Mon 15 Feb, proposed |
| A2 oral discussion, ten minutes | 20 % | 3–5 Feb, bookable slots |

- We are thinking of having a leaderboard for A1: lowest bits per byte on held-out text at the same training budget wins. It will not impact your grade, but it could be a fun competition.
- There is no final exam.

Note: Weights and dates from the course page. A2 comes out on 7 Jan, and the presentations on 3–5 Feb come before the report. Bits per byte, not perplexity: every student trains their own tokenizer, and per-token numbers do not compare across tokenizers. Lecture 2 argues it, deck 09 measures it.

---

<p class="kicker">This course · 4</p>

## Grading.

| weighted total | grade |
|---|---|
| 87.5 – 100 % | 1 |
| 75 – 87.5 % | 2 |
| 62.5 – 75 % | 3 |
| 50 – 62.5 % | 4 |
| below 50 % | 5 |

- You pass at 50 % of the weighted total.
- If you don't turn in an assignment, you cannot show up for its defence. So it's an automatic fail for both.
- Insufficient understanding of your submitted work in an oral discussion can lead to an overall negative grade.

Note: Wording from the course page. Four equal bands above the pass line, a uniform scale. If either this table or the course page changes, change both. The not-turned-in rule: a Teilleistung never attempted stops the whole assessment (`grading.md` § 4.4), so on the slide it reads as an automatic fail for both.

---

<p class="kicker">This course · 5</p>

## Finalizing course attendance.

- After this lecture we finalize who is taking the course: everyone here today, and everyone who emailed us about skipping.
- We send the list to university IT, who give you access to GSC1, the university's computing cluster.
- You get an email with an invite to our Discord server, the course's main place for questions, materials and extras.

Note: Deregistration stays open in UNIGRAZonline for anyone who wants out. The Discord invite goes to the finalized list only.

---

<p class="kicker">This course · 6</p>

## Compute.

- GSC1 is the university's computing cluster; the course runs on its GPU node, eight H200s.
- You get access through the participant list from the previous slide.
- Your jobs wait in a queue, sometimes for days. Treat every deadline as a queue deadline and start early.
- The allowance for graded runs will be announced.
- Please do things on time.

Note: The node is Hydra, partition idealab_cs2; accounts come from the list on the previous slide. The allowance waits on the HPC team's answer. TODO: chase the GPU cap before A1 opens.

---

<p class="kicker">This course · 7</p>

## Use of AI.

- Working with generative AI, especially coding agents, is encouraged in this course.
- It requires engaging more deeply with the techniques and systems you work with.
- In the oral discussions, you will need to demonstrate your understanding to us.
- Briefly document which tools you used and for what.
- We are planning an AI coding workshop next week, hosted by Lorenz.

Note: Wording from the course page. The oral discussions run for everyone, which is what makes an open AI policy safe to give. The last line is the pitch: ask the room who already does agentic coding, then announce the workshop. Details are on the next slide.

---

<p class="kicker">This course · 8</p>

## Agentic workshop.

- Things to apply to to have a free coding agent.
- If you have the budget for it, an OpenCode Go subscription, about 10 € a month ([opencode.ai/go](https://opencode.ai/go)).
- For the workshop we would ask you to bring, or to have by then, at least one subscription and one installed harness.
- If you want a recommendation for a harness, it is still pi: `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` <img src="assets/pi-logo.svg" alt="pi logo" style="height:1.2em;vertical-align:-0.25em"> · [pi.dev](https://pi.dev)

Note: Fill the first bullet live from the links: GitHub Copilot, verify as a student ([education.github.com/pack](https://education.github.com/pack)); or a year of Google AI Pro ([gemini.google/students](https://gemini.google/students)). Prices checked 8 October 2026: OpenCode Go 10 $ a month; Copilot and a year of Google AI Pro free for verified students. Any harness works in the workshop: Copilot, Claude Code, OpenCode, Cursor, pi. Lorenz hosts the workshop, announced on the Use of AI slide. We explain what a coding harness is at the workshop.

---

<p class="kicker">This course · 9</p>

## Follow us!

<div class="cols">
<div>

<img src="assets/avatars/ruggsea.webp" alt="Ruggero's Bluesky avatar" style="height:96px;width:96px;object-fit:cover;border-radius:48px">

**Ruggero** · [@ruggsea.eurosky.social](https://bsky.app/profile/ruggsea.eurosky.social)

</div>
<div>

<img src="assets/avatars/lorrenz.webp" alt="Lorenz's Bluesky avatar" style="height:96px;width:96px;object-fit:cover;border-radius:48px">

**Lorenz** · [@lorrenz.bsky.social](https://bsky.app/profile/lorrenz.bsky.social)

</div>
</div>

- You don't need to follow us.
- This is a general encouragement to get a Bluesky account.
- Follow us, or researchers in general.

Note: Say by voice: Twitter used to be the place to be for this field, and it is not any more; Bluesky is where the AI researchers went. If they ask, we can direct them to a good starting pack on Bluesky to follow researchers from this field (starter packs exist for ML/NLP; pick one before next Thursday). The honest pitch, in Ruggero's words: doom-scrolling on a website whose main reason you scroll is to find the AI papers. Transmits the passion. Handles verified against the Bluesky API this afternoon: @ruggsea.eurosky.social, @lorrenz.bsky.social. Avatars from the same API, licences in assets/README.md. The account costs nothing; encourage them to make it tonight.

---

<!-- .slide: class="statement" -->

<div>

# Guess the next letter.

<div id="gg-board"></div>
<div id="gg-keys"></div>
<p id="gg-count" class="caption"></p>
<p id="gg-result" class="caption"></p>

<p class="caption">27 symbols: the letters A to Z and the space (␣).</p>
</div>

Note: Live game. Click a key or type: a wrong letter counts an attempt, the right letter commits and the cursor moves on; the number under each letter is the attempts it took, and the big number is the running average of guesses per letter. Entropy averages log2 of the attempts (Shannon 1951, section 6 style; an upper-bound estimate, say it is rough). The first letter is given: one attempt, zero bits. The target text is the Frankenstein line by default; set your own by adding &gg=YOUR+TEXT to the deck URL before class, the room never sees it. TODO: pick a passage of about 100 letters the students will not know. Backspace undoes the last letter; while this slide is up, letters and space type guesses and the arrows still move slides.

<img class="sticker br" src="stickers/manicule-right.png">

---

<!-- .slide: class="divider" data-background-image="assets/stradanus.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 2</p>

## Shannon played a guessing game.
</div>

---

<p class="kicker">The guessing game · 1</p>

## Shannon's subject needed one guess for 79 of 102 letters.

| guess needed | 1st | 2nd | 3rd | 4th | 5th | more |
|---|---|---|---|---|---|---|
| letters | 79 | 8 | 3 | 2 | 2 | 8 |

Most letters are easy to guess. But some are not.

<p class="caption">Text: "THERE IS NO REVERSE ON A MOTORCYCLE A FRIEND OF MINE FOUND THIS OUT RATHER DRAMATICALLY THE OTHER DAY". Shannon 1951, p. 56.</p>

<p class="source"><a target="_blank" href="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf">Shannon 1951 · Prediction and Entropy of Printed English</a>, §3</p>

Note: Shannon 1951, "Prediction and Entropy of Printed English", Bell System Technical Journal 30, pp. 50–64, section 3, example (9), p. 56. Shannon calls results of this order "typical of prediction by a good subject". In the first experiment (one guess only, p. 55), 89 of 129 letters, 69%, were right first time. Errors cluster at the start of words. Compare with the tally the room just made.

---

<p class="kicker">The guessing game · 2</p>

## The list of guess counts is a compressed copy of the text.

<div class="cols">
<div>

- Replace each letter by the number of guesses it took.
- The list is mostly 1s, which is easy to compress: to say it with fewer symbols than the text.
- An identical twin who guesses the same way can rebuild the text from the numbers.

</div>
<div>

`THERE IS NO REVERSE`<br>
`1 1 1 5 1 1 2 1 1 2 1 1 ...`

</div>
</div>

This is actually how we score every model in this course, even if it is this simple thing.

<p class="source"><a target="_blank" href="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf">Shannon 1951 · Prediction and Entropy of Printed English</a>, §3</p>

Note: Shannon 1951, section 3 and Fig. 2 ("communication system using reduced text"): two identical predictors, only the guess counts are transmitted. The digit string on the right is the start of example (9), checked against the scan of p. 56. Next week this comes back: a tokenizer is a compressor too.

---

<p class="kicker">The guessing game · 3</p>

## Making a simple language model of English.

- Take text as training data.
- Count, for every letter and every group of two letters, how often each other letter follows: the co-occurrences.
- Turn the counts into probabilities: the share each letter has of what follows.

After T and H, English picks E six times out of ten:

| next letter | E | A | space | I | O | R | other |
|---|---|---|---|---|---|---|---|
| share after TH | 0.62 | 0.13 | 0.10 | 0.08 | 0.04 | 0.02 | 0.01 |

A language model gives a probability to every possible next letter, and they add up to 1.

<p class="caption">Training data: four novels, Alice, Pride and Prejudice, Moby Dick, Sherlock Holmes (Project Gutenberg), 2.5 million letters; the row is all 62,194 places where T is followed by H.</p>

Note: Say the steps in order: training data, co-occurrence counts, shares. This is the first model of the course: counting plus normalization, nothing else; every later model changes only where the probabilities come from. Script: figures/01_model_guesses.py ("after TH" line); "other" is 1 minus the six shown, rounded. One row, 27 columns: a language model is a machine that fills in this row for any context, from a table of counts to a reasoning model. Wolfe ("Language Model Training and Inference: From Concept to Code") calls it the autoregressive mechanism "which makes token-level decisions one by one". Symbols are letters today, tokens next week.

---

<p class="kicker">The guessing game · 4</p>

## The probability of a whole text is a chain of next-letter guesses.

$$ p(\text{THE}) = p(\text{T}) \cdot p(\text{H} \mid \text{T}) $$

<p class="inwords">7.4% of all symbols in our four novels are a T, and a third of those Ts are followed by H: 0.074 × 0.332.</p>

<div class="fragment">

$$ p(\text{THE}) = p(\text{T}) \cdot p(\text{H} \mid \text{T}) \cdot p(\text{E} \mid \text{TH}) = 0.015 $$

<p class="caption">about 1.5% of three-letter windows in the novels</p>

</div>

<div class="fragment">

$$ p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}) $$

<p class="caption">the general formula, for any text of any length</p>

</div>

Note: One formula per click. All three numbers from figures/01_model_guesses.py; check it the other way, 38,252 THEs among 2.54M windows, 1.5%. The chain rule is exact. Say the last one by hand: x_t is the symbol at position t, T is the length (the length, not the letter), and ∏ means multiply all of these. From next week on, the symbols are called tokens. Everything later measures this chain: the loss, entropy, perplexity.

---

<p class="kicker">Surprise · 1</p>

## Probability and surprisal.

| what happened | probability | how surprised |
|---|---|---|
| a coin lands heads | 1/2 | a little |
| a die shows a six | 1/6 | more |
| you draw the ace of spades | 1/52 | a lot |
| the letter after TH is E | 0.62 | barely |
| the letter after TH is a space | 0.10 | more |

We want to replace the last column with a number, and the number has to behave well when outcomes combine.

Note: The two letter rows are the real shares from the previous slide. Ask the room to order the rows before showing the middle column. The point of the next four slides is that there is only one sensible way to turn the middle column into the right column.

---

<p class="kicker">Surprise · 2</p>

## Two dice together: the probabilities multiply.

<div class="cols pic-left">
<div>

![the 36 outcomes of two dice, double six marked](figures/01_dice_grid.png)

</div>
<div>

Rolling a six has probability one out of six. Rolling another six: we have to do it twice.

$$ p(6, 6) = \tfrac{1}{6} \cdot \tfrac{1}{6} = \tfrac{1}{36} $$

<p class="inwords">Independent outcomes multiply with each other, and the probability of both of them is the product.</p>

</div>
</div>

Note: Script: figures/01_dice_grid.py. Independence is the assumption here; the chain rule from the previous slides is the general case, where the second factor is conditional on the first. Either way, probabilities of a sequence multiply.

---

<p class="kicker">Surprise · 3</p>

## Surprise should add up when probabilities multiply.

The surprisal should transform multiplying probabilities into a summation.

$$ \text{surprise}(6, 6) = \text{surprise}(6) + \text{surprise}(6) $$

<p class="inwords">The surprisal of rolling a six and rolling a six is the surprise of rolling one six plus the surprise of rolling another six.</p>

<div class="fragment"><p class="inwords">The logarithm is a function that turns multiplying into adding.</p></div>

<div class="fragment">$$ \log(a \cdot b) = \log a + \log b $$</div>

<div class="fragment">$$ \log_2 36 = \log_2 6 + \log_2 6 = 2.58 + 2.58 = 5.17 $$</div>

<p class="source"><a target="_blank" href="https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf">Shannon 1948 · A Mathematical Theory of Communication</a>, §1</p>

Note: This is the requirement, stated before the answer: let the room propose functions, squaring does not work, subtracting from one does not work. The log does exactly this, and one line of school maths carries the whole of information theory. Shannon 1948, section 1, gives this justification: parameters of engineering importance vary linearly with the logarithm.

---

<p class="kicker">Surprise · 4</p>

## Surprisal.

<div class="bigformula">$$ \text{surprise}(x) = \log_2 \frac{1}{p(x)} = -\log_2 p(x) $$</div>

- The logarithm lets us add.
- The minus sign is there because we invert the probability: lower probability, higher surprise.

<p class="source"><a target="_blank" href="https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf">Shannon 1948 · A Mathematical Theory of Communication</a>, §1</p>

Note: Say by hand: one over the probability is how many equally likely things this was one of, and the log turns that count into something that adds. A six is one of 6, so log₂ 6 = 2.58; double six is one of 36, so log₂ 36 = 5.17, which is 2.58 + 2.58. E after TH costs 0.69 bits: barely surprised. A space after TH costs 3.32 bits. Fill the last column of the Probability and surprisal table live.

---

<p class="kicker">Surprise · 5</p>

## Probability and surprisal.

<div>

<img class="bigfig" src="figures/01_surprise_curve.png?v=2" alt="surprise against probability, with a coin, a die, a card and two letter probabilities marked">

</div>

<p class="caption">The curve is log₂(1/p). It is 0 for something certain and climbs without limit as p goes to 0. The two letter points are real shares from the four novels.</p>

Note: Script: figures/01_surprise_curve.py. Read the plot with the room: p on the horizontal axis, surprise in bits on the vertical. A certain event (p = 1) has surprise 0. Halving the probability adds exactly one bit, wherever you are on the curve: that is the "adds up" property drawn.

---

<p class="kicker">Surprise · 6</p>

## The unit depends on the base of the logarithm.

| base of the log | unit | one unit is | where you meet it |
|---|---|---|---|
| 2 | bit | one fair coin, one yes/no question | Shannon, compression, this lecture |
| e ≈ 2.718 | nat | an "e-sided die" | every training loop, torch, the loss curves |
| 10 | hartley | one decimal digit | almost nowhere in this course |

```python run
import math
six = 1/6
print("a six:", round(math.log2(1/six), 2), "bits =", round(math.log(1/six), 2), "nats")
print("1 nat =", round(1/math.log(2), 3), "bits;  1 bit =", round(math.log(2), 3), "nats")
```

<p class="inwords">The unit changes the numbers by a constant factor and nothing else. 2.58 bits and 1.79 nats are the same surprise. Divide nats by 0.693 to get bits.</p>

Note: Code uses the natural log because that is what the maths library and the gradients prefer, so every loss number you will see in this course is in nats. We report bits when we talk to people. The conversion is the only thing to remember: bits = nats / ln 2.

---

<p class="kicker">Surprise · 7</p>

## Thinking about a bit.

<div class="cols">
<div>

**Eight doors, one prize, all equally likely.**

1. "Is it in the left half?"
2. "Is it in the left half of that?"
3. "Left or right?"

Three questions always find it, and log₂ 8 = 3. A bit is a yes/no question.

</div>
<div>

**Four letters, but A comes half the time.**

| letter | A | B | C | D |
|---|---|---|---|---|
| share | 1/2 | 1/4 | 1/8 | 1/8 |
| questions | 1 | 2 | 3 | 3 |

Ask "A?" first. On average you need 1.75 questions.

</div>
</div>

Note: Do this aloud with the room. With skewed shares, a smart questioner asks about the likely letters first and saves questions on average: ½·1 + ¼·2 + ⅛·3 + ⅛·3 = 1.75. This "average number of questions" is what the next slide names.

---

<p class="kicker">Surprise · 8</p>

## Entropy is the average surprise, weighted by how often each outcome shows up.

**A fair die.** Every face has probability 1/6 and surprise 2.58 bits, so the average is 2.58 bits.

<div class="fragment">

**A loaded die.** Imagine a die where six comes up half the time, so the surprise is 1 bit; each other face comes one time in ten (surprise 3.32 bits).

$$ \tfrac{1}{2} \cdot 1 + 5 \cdot \tfrac{1}{10} \cdot 3.32 = 0.5 + 1.66 = 2.16 \text{ bits} $$

</div>

<div class="fragment">$$ H = \sum_{x} p(x) \log_2 \frac{1}{p(x)} = -\sum_{x} p(x) \log_2 p(x) \quad \text{bits} $$</div>

<p class="source"><a target="_blank" href="https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf">Shannon 1948 · A Mathematical Theory of Communication</a>, §6</p>

Note: One step per click: fair die, loaded die, the formula. Shannon 1948, section 6, eq. (4). H is the letter Shannon used originally for entropy. Real shares are not powers of 1/2, so the average number of yes/no questions is fractional; the formula still holds.

---

<p class="kicker">Surprise · 9</p>

## A loaded die is easier to guess, so its entropy is lower.

```python run
import math
def entropy(shares):
    return sum(p * math.log2(1/p) for p in shares if p > 0)
print("fair die      ", round(entropy([1/6]*6), 2), "bits")
print("loaded die    ", round(entropy([1/2] + [1/10]*5), 2), "bits")
print("fair coin     ", round(entropy([1/2, 1/2]), 2), "bits")
print("two-headed    ", round(entropy([1.0, 0.0]), 2), "bits")
print("27 equal letters", round(entropy([1/27]*27), 2), "bits")
```

<p class="inwords">The more predictable the source, the lower its entropy. A two-headed coin never surprises you, so its entropy is 0. Twenty-seven equally likely symbols cost 4.75 bits, which is where English would be if every letter were equally likely.</p>

Note: Press run. The last line is Shannon's F₀ = log₂ 27 = 4.76 (Shannon 1951, section 2). His F₁ uses letter frequencies, F_N the previous N−1 letters; the next slides walk down that ladder with our own counts.

---

<p class="kicker">Surprise · 10</p>

## The entropy of a coin peaks when the coin is fair.

<div>

<img class="bigfig" src="figures/01_binary_entropy.png" alt="entropy of a biased coin against the probability of heads">

</div>

<p class="caption">One bit at p = 0.5, 0.47 bits at 9 heads in 10, 0 bits for a two-headed coin. The curve is symmetric: a coin that almost always lands tails is just as predictable.</p>

Note: Script: figures/01_binary_entropy.py. Read the plot: probability of heads on the horizontal axis, average surprise per flip on the vertical. This is the whole idea of language modelling in one picture: a model that knows the shares of the next letter is a coin it can call, and the better it knows them, the further left or right on this curve it sits.

---

<p class="kicker">Surprise · 11</p>

## Entropy of English.

We can work out the upper bound of the entropy of English.

<div class="fragment">

Assuming every letter is equally likely, this would be the entropy of English:

$$ H = \log_2 27 = 4.75 \text{ bits} $$

</div>

<div class="fragment">

But we can do better than this: weighting the surprisal by the letter frequencies, the entropy of English goes down to 4.09 bits.

$$ H = \sum_{x} p(x) \log_2 \frac{1}{p(x)} = 4.09 \text{ bits} $$

</div>

<div>

<img class="bigfig" style="max-height: 10.5em" src="figures/01_letter_freq.png?v=2" alt="Frequency of the 27 symbols in four English novels">

</div>

<div class="source">

<p>2.5M symbols from Alice, Pride and Prejudice, Moby Dick and Sherlock Holmes (Project Gutenberg). Shannon 1951 reports 4.03 bits for the same 27 symbols.</p>

<p><a target="_blank" href="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf">Shannon 1951 · Prediction and Entropy of Printed English</a>, §2</p>

</div>

Note: Script: figures/01_letter_freq.py. The 4.75 is Shannon's F0, log₂ 27 (he reports 4.76); the 4.09 is his F1, letter frequencies (he reports 4.03, on different text). Every step is an upper bound on the true entropy of English, and each piece of context added lowers it; the next slide pushes it to about one bit. Space is 18.9% of symbols (Shannon: 18.2%), E is 10.1% (Shannon: 10.7%). Read the plot: one bar per symbol, sorted, height is its share.

---

<p class="kicker">Surprise · 12</p>

## With 100 letters of context, English costs about one bit per letter.

| letters known | 0 | 1 | 2 | 3 | 7 | 99 |
|---|---|---|---|---|---|---|
| upper bound, bits | 4.03 | 3.42 | 3.0 | 2.6 | 1.8 | 1.3 |
| lower bound, bits | 3.19 | 2.50 | 2.1 | 1.7 | 1.0 | 0.6 |

Shannon observed only the guesses needed by a person, never the probabilities in their head. So we can use the guess counts to bound the surprisal.

<p class="caption">Shannon 1951, section 6, bounds from one subject's guesses on 100 samples. Columns are the number of letters the guesser saw before the one to guess.</p>

<p class="source"><a target="_blank" href="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf">Shannon 1951 · Prediction and Entropy of Printed English</a>, §6</p>

Note: Shannon 1951, section 6. The table is noisy: the column for 14 letters has a higher upper bound (2.1) than for 7 (1.8). He says so himself ("considerable sampling error"). One subject, n=1. The acknowledgment thanks Mrs. Mary E. Shannon for help with the experimental work.

---

<p class="kicker">Surprise · 13</p>

## Perplexity turns bits back into "how many equally likely options".

Perplexity is 2 to the power of the entropy.

```python run
import math
for name, bits in [("fair die", 2.58), ("loaded die", 2.16), ("27 equal letters", 4.75), ("English letters, frequencies only", 4.09)]:
    print(f"{name:33s} {bits:.2f} bits -> perplexity {2**bits:.1f}")
```

<div class="fragment">

<p class="inwords">A fair die has perplexity 6: you are choosing among 6. The loaded die has perplexity 4.5: it is as hard to guess as a fair 4.5-sided die. English letters with only frequencies known: as hard as 17 equally likely letters, not 27.</p>

</div>

Note: Press run, then click for the reading. Perplexity is the entropy undone: it says how many equally likely options the surprise is worth. People quote it because "as hard as a 4.5-sided die" is easier to feel than "2.16 bits". Same information, different unit. Lower is better, and 1 means certainty.

---

<p class="kicker">Surprise · 14</p>

## Every training loop reports nats.

```python run
import math
for step, nats in [(0, 11.42), (5720, 3.31)]:
    bits = nats / math.log(2)
    print(f"step {step}: {nats:.2f} nats = {bits:.2f} bits, perplexity {math.exp(nats):,.0f}")
```

<p class="inwords">At step 0 my 125M-parameter model was as unsure as choosing among 91,000 options, more than its whole vocabulary, because a freshly initialised model is not even uniform. After 5,720 steps it was as unsure as choosing among 27. Same numbers as the loss curve on the course cover.</p>

Note: Press run. The two loss values are the first and last points of my FineWeb run (slides/assets/loss_curves.json, run fineweb_r000_seed1_125m). TODO: confirm the vocabulary size of that run so the "more than its whole vocabulary" line has the exact number. Perplexity in nats is e to the loss; in bits it is 2 to the loss; they agree because the unit conversion is built in. Assignment 1 will rank you by a close relative of this number, bits per byte, for a reason lecture 2 explains: your tokenizers will differ, and a bit per token is not comparable across tokenizers.

---

<!-- .slide: class="divider" data-background-image="assets/diderot.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 3</p>

## A machine can play the game by counting.
</div>

---

<!-- .slide: data-auto-animate -->

<p class="kicker">Counting · 1</p>

## An n-gram model guesses by counting what followed.

An n-gram model looks at the previous n − 1 words. Here n = 2: one word back. That makes it a Markov chain: the next word depends only on those, not on the whole past.

`the cat sat . the cat ran . the dog sat . the cat ate .`

<div class="grid-example" data-id="counts">
the → cat  ·  count 3
</div>

Note: Build-up. Ask the room: after "the", what came next, and how often? Count on the slide with them. An n-gram model is a Markov chain of order n − 1 (a bigram is a first-order chain): the next symbol's distribution depends only on the previous n − 1 symbols. Shannon 1948, section 3, builds his approximations to English exactly this way, as Markov processes on letters and words.

---

<!-- .slide: data-auto-animate -->

<p class="kicker">Counting · 1</p>

## An n-gram model guesses by counting what followed.

`the cat sat . the cat ran . the dog sat . the cat ate .`

<div class="grid-example" data-id="counts">
the → cat  ·  count 3<br>
the → dog  ·  count 1
</div>

$$ p(\text{cat} \mid \text{the}) = \frac{3}{3+1} = 0.75 $$

<p class="inwords">Of the four times we saw "the", three were followed by "cat".</p>

---

<!-- .slide: data-auto-animate -->

<p class="kicker">Counting · 1</p>

## An n-gram model guesses by counting what followed.

`the cat sat . the cat ran . the dog sat . the cat ate .`

<div class="grid-example" data-id="counts">
the → cat  ·  count 3<br>
the → dog  ·  count 1
</div>

<div class="fragment">

$$ p(\text{cat} \mid \text{the}) = \frac{3}{3+1} = 0.75 $$

<p class="inwords">Count in the training text: c(the, cat) = 3 over c(the) = 4.</p>

</div>

<div class="fragment">

$$ p(x_t \mid \underbrace{x_1, \dots, x_{t-1}}_{\text{the previous } n-1}) = \frac{c(x_1, \dots, x_{t-1}, x_t)}{c(x_1, \dots, x_{t-1})} $$

<p class="inwords">The same fraction, with names instead of words.</p>

</div>

Note: Maximum-likelihood estimate. n = 2 is a bigram, n = 3 a trigram. "Training" is one pass over the text with a dictionary of counts; the model is the table. The Q count is from the four training novels (the other 5 times a space follows). CS336 lecture 1 lists n-gram models as the pre-neural state of the art for machine translation and speech recognition (Brants et al. 2007, "Large language models in machine translation", trained on up to 2 trillion tokens, checked in the paper).

---

<p class="kicker">Counting · 2</p>

## Sampling from the table writes new text. Press run twice.

```python run
import random
words = "the cat sat on the mat . the dog sat on the cat . the cat ran .".split()
followers = {w: [] for w in words}
for a, b in zip(words, words[1:]):     # each word and the word after it
    followers[a].append(b)
word, out = "the", ["the"]
for _ in range(12):
    word = random.choice(followers[word])
    out.append(word)
print(" ".join(out))
```

Note: Training is lines 3 to 5: one pass over the text, each word's followers in a list. zip(words, words[1:]) walks the pairs (the, cat), (cat, sat), and so on. The text adds "on the mat" to the one on the counting slide, so there are more branches. followers["the"] is ["cat", "mat", "dog", "cat", "cat"]: repeats in the list are the counts, so random.choice picks "cat" 3 times in 5. Most runs give a sentence that never appeared in the text; with so few branches, some runs copy it. This is generation: pick a next word, append it, repeat.

---

<p class="kicker">Counting · 3</p>

## Shannon generated English by hand in 1948, one lookup at a time.

```text
random letters: XFOML RXKHRJFFJUJ ZLPWCFWKCYJ FFJEYVKCQSGHYD QPAAMKBZAACIBZLHJQD.
letter pairs:   ON IE ANTSOUTINYS ARE T INCTORE ST BE S DEAMY ACHIN D ILONASIVE
                TUCOOWE AT TEASONARE FUSO TIZIN ANDY TOBE SEACE CTISBE.
word pairs:     THE HEAD AND IN FRONTAL ATTACK ON AN ENGLISH WRITER THAT THE
                CHARACTER OF THIS POINT IS THEREFORE ANOTHER METHOD FOR THE LETTERS THAT
                THE TIME OF WHO EVER TOLD THE PROBLEM FOR AN UNEXPECTED.
```

His sampler: open a book at a random page, read until the current letter appears, write down the letter after it.

<p class="source"><a target="_blank" href="https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf">Shannon 1948 · A Mathematical Theory of Communication</a>, §3</p>

Note: Shannon 1948, "A Mathematical Theory of Communication", section 3, "The series of approximations to English", samples 1, 3 and 6 of his six (the others are letter frequencies, letter triples and word frequencies). "Random letters" is his "zero-order approximation": every symbol equally likely. The book-opening procedure is his, same section. He notes the samples "have reasonably good structure out to about twice the range that is taken into account in their construction".

---

<p class="kicker">Counting · 4</p>

## A longer prefix makes better text, until it starts copying the book.

| prefix | sample from Alice in Wonderland | 8-word runs copied |
|---|---|---|
| 1 word | "she found she remembered that it would," said Alice: "I didn't think of me? | 0% |
| 2 words | And Alice was very fond of pretending to be an old conger-eel | 15% |
| 3 words | the Duchess said in a low voice, to the company generally | 55% |

Hoyt puts it this way: "Making the prefix shorter tends to produce less coherent prose; making it longer tends to reproduce the input text verbatim."

<p class="source"><a target="_blank" href="https://benhoyt.com/writings/markov-chain/">Hoyt · Markov chains in 20 lines of Python</a></p>

Note: Same loop as the cat-and-dog sampler, keyed on the last one, two or three words: Ben Hoyt, "Using a Markov chain to generate readable nonsense with 20 lines of Python", benhoyt.com/writings/markov-chain/, on the reading list. Script: figures/01_markov_samples.py (seed 8, 40 words per sample, excerpts shown). "Copied" is the share of 8-word windows in the sample that occur word for word in the book. n=1 seed per row, so read the trend, not the exact percentages. Hoyt settles on two words as the compromise for English.

---

<p class="kicker">Counting · 5</p>

## Playing your game, a counting model got 24 of 48 letters first time.

| guesser | text | right first time |
|---|---|---|
| you | the Frankenstein line | ___ of 48, from your tally |
| counting model, 4 letters of context | the Frankenstein line | 24 of 48 |
| counting model, 4 letters of context | Shannon's motorcycle line | 45 of 101 |
| Shannon's subject | Shannon's motorcycle line | 79 of 102 |

Humans still win: 79 of 102 against 45 of 101.

<p class="source"><a target="_blank" href="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf">Shannon 1951 · Prediction and Entropy of Printed English</a>, §3</p>

Note: Script: figures/01_model_guesses.py. Shannon counts 102 symbols in the motorcycle line; our copy of it has 101. TODO: verify which symbol his count includes (probably a final space). The model ranks the 27 symbols by how often they followed the previous four letters in the four training novels and guesses in that order. With 2 letters of context it gets 13 of 48 and 37 of 101. Its misses are at word starts: the B of BENEVOLENT took it 14 guesses, the G of GOOD 19, the M of MISERY 16. Fill the first row from the board.

---

<!-- .slide: data-auto-animate -->

<p class="kicker">Counting · 6</p>

## A model's score is how surprised it is by text it has not seen.

<div data-id="loss">

$$ \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log_2 p(x_t \mid x_{<t}) $$

</div>

<p class="inwords">The surprise from Part 1, computed with the model's probabilities instead of the true shares, averaged over all T letters of a text the model never saw (x<sub>&lt;t</sub> is everything before letter t). Four letters given 1/2, 1/4, 1/8, 1/2 cost 1 + 2 + 3 + 1 = 7 bits, so 𝓛 = 1.75 bits per letter. This is the loss, also called cross-entropy: the entropy formula with the model's guesses inside the log. Lower is better.</p>

---

<!-- .slide: data-auto-animate -->

<p class="kicker">Counting · 6</p>

## A model's score is how surprised it is by text it has not seen.

<div data-id="loss">

$$ \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log_2 p(x_t \mid x_{<t}) \qquad \text{perplexity} = 2^{\mathcal{L}} $$

</div>

<p class="inwords">2<sup>1.75</sup> = 3.4, the perplexity from Part 1 again. The model is as unsure as someone choosing among 3.4 equally likely letters.</p>

Note: 𝓛 is Shannon's bits per letter, measured on held-out text. Every later lecture calls it the loss. Training uses natural log (nats); bits = nats / ln 2.

---

<p class="kicker">Counting · 7</p>

## Our best counting model scores 2.10 bits per letter: perplexity 4.3.

```python run
import math
probs = [1/2, 1/4, 1/8, 1/2]          # probability the model gave each true letter
bits = [-math.log2(p) for p in probs]
loss = sum(bits) / len(bits)
print("bits:", bits, " average:", loss, " perplexity:", round(2 ** loss, 2))
print("best counting model on Frankenstein (Part 3): perplexity", round(2 ** 2.10, 1))
```

Note: The 2.10 is the n=5 letter model on Frankenstein; the plot in Part 3 shows where it comes from. Assignment 1's leaderboard ranks by a close relative, bits per byte (end of today, and lecture 2).

---

<!-- .slide: class="divider" data-background-image="assets/babbage.jpg" data-background-size="cover" data-background-position="top" -->

<div class="band">
<p class="kicker">Part 4</p>

## Counting breaks down.
</div>

---

<p class="kicker">Where counting breaks · 1</p>

## Past n = 5, more counting makes the model worse on a new book.

![Bits per letter of letter n-gram models, training books versus Frankenstein](figures/01_ngram_entropy.png)

<p class="caption">Trained on four novels, scored on Frankenstein. Unseen letter sequences get a small share instead of zero (smoothing, three slides on). With one letter of context our tables give 3.32 bits, exactly Shannon's figure.</p>

Note: Script: figures/01_ngram_entropy.py. Explain the plot: x is n, y is bits per letter, lower is better. Red: entropy of the counts on the training books (Shannon's F_N estimated from our tables); it keeps falling because the table memorises. Black: the same model scored on a book it never saw; best at n=5 (2.10 bits), 3.80 bits by n=10, worse than n=2. Grey band: Shannon's humans at 100 letters. Frankenstein held out as a whole book on purpose: a random split would leak repeated phrases.

---

<p class="kicker">Where counting breaks · 2</p>

## 82% of Frankenstein's word triples never occur in four other novels.

![Share of held-out n-grams never seen in training](figures/01_unseen_ngrams.png)

<p class="caption">Same books. Word n-grams: 2.7% of single words, 38.7% of pairs, 81.8% of triples, 97.0% of 4-grams unseen.</p>

Note: Script: figures/01_ngram_entropy.py (second figure). An unseen n-gram gets count zero, so without smoothing its probability is zero and the held-out loss is infinite. Letters stay dense much longer than words because there are only 27 of them.

---

<p class="kicker">Where counting breaks · 3</p>

## The table grows as V to the power n, and the text does not.

```python run
for n in [1, 2, 3, 5, 10]:
    print(f"letter {n}-grams possible: {27 ** n:,}")
print(f"word trigrams, 50,000 words: {50_000 ** 3:,}")
```

<p class="caption">V is the number of symbols: 27 letters, or 50,000 words. Our four novels contain 2.5M letters and 101,145 distinct letter 5-grams.</p>

Note: Seen counts from figures/01_ngram_entropy.py output: 5,619 distinct 3-grams of 19,683 possible, 101,145 5-grams of 14.3M, 1,449,664 10-grams of 2.1e14. Every extra symbol of context multiplies the table by V, while the data grows by nothing.

---

<p class="kicker">Where counting breaks · 4</p>

## Smoothing gives unseen pairs a small share instead of zero.

`the cat sat . the cat ran . the dog sat . the cat ate .`

$$ p(\text{sat} \mid \text{the}) = \frac{0}{4} = 0 $$

<div class="fragment">

$$ p_{\text{add-one}}(x \mid \text{ctx}) = \frac{c(\text{ctx}, x) + 1}{c(\text{ctx}) + V} $$

<p class="inwords">Pretend every pair was seen once more. The toy text has V = 7 word types: p(sat | the) = 1/11 = 0.09, and p(cat | the) drops from 0.75 to 4/11 = 0.36.</p>

</div>

Note: "The sat" never occurred, so a raw count model says it is impossible; if a test text contains one such pair, the loss is infinite. Add-one (Laplace) is what the plots use; ctx is the context, the previous n − 1 symbols. The 7 types: the, cat, sat, ran, dog, ate, and the full stop. Better smoothing exists (Kneser–Ney, backoff) and bought decades of machine translation.

---

<p class="kicker">Where counting breaks · 5</p>

## Counts never learn that two words are alike.

- "walked to the" and "went to the" are two unrelated rows of the table.
- Seeing one teaches the model nothing about the other.
- 8,221 of the 21,422 word types in our four novels occur exactly once.

Smoothing redistributes shares. It does not share evidence between similar contexts.

Note: Word counts from the four training novels, lowercased. This is the failure lecture 3 starts from: vectors, where "walked" and "went" end up close.

---

<!-- .slide: class="statement" -->

<div>

# So we will learn the counts.
</div>

Note: The turn of the lecture. Keep the objective, replace the table by a function with parameters.

---

<p class="kicker">Learning the counts · 1</p>

## A neural model swaps the table for a function and keeps the score.

<table>
<tr><th></th><th>n-gram model</th><th>neural language model</th></tr>
<tr class="fragment"><td>context</td><td>the last n − 1 symbols</td><td>up to thousands of tokens</td></tr>
<tr class="fragment"><td>where p(x | ctx) comes from</td><td>a count table</td><td>a function with adjustable numbers</td></tr>
<tr class="fragment"><td>size</td><td>grows as Vⁿ</td><td>fixed when you design it</td></tr>
<tr class="fragment"><td>similar contexts</td><td>share nothing</td><td>get similar numbers</td></tr>
</table>

<p class="caption">A token is a piece of text, often part of a word (next week). The adjustable numbers are called parameters. "Neural" means the function is built from many small steps of arithmetic (lectures 3 and 4).</p>

<p class="source"><a target="_blank" href="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">Bengio et al. 2003 · A Neural Probabilistic Language Model</a></p>

Note: GPT-2 small has 124M parameters and reads 1,024 tokens back. Bengio et al. 2003, "A Neural Probabilistic Language Model", is the first neural LM in CS336 lecture 1's timeline. GPT-2 small config from Wolfe (NanoGPT, 117M parameters as reported in the GPT-2 paper, 124M when counted; 12 blocks, 1,024 context, 768-dimensional embeddings). Same loss as Part 2: cross-entropy of the next token.

---

<p class="kicker">Learning the counts · 2</p>

## Training is the guessing game played billions of times, adjusting after each guess.

- Take a stretch of text. At every position, the model guesses the next token.
- The right answer is already in the text, so no labels are needed.
- Score the guess in bits, as in Part 2. Nudge every number so the right answer gets a bit more probability (how: lecture 5).
- To generate, sample a token, append it and guess again. It is the same loop as the cat-and-dog sampler.

<p class="source"><a target="_blank" href="https://cameronrwolfe.substack.com/p/language-model-training-and-inference">Wolfe · Language Model Training and Inference: From Concept to Code</a></p>

Note: The nudge is gradient descent; lecture 5. Wolfe, "Language Model Training and Inference": "the ground truth next token is already present within the corpus itself"; generation loop: forward pass, scale logits by temperature, optional top-k, softmax, sample. We build every piece of this in lectures 3 to 5.

---

<!-- .slide: class="title" -->

<div class="text">
<p class="kicker">Next week · 15 October</p>

# Tokenizers from scratch

<p class="tagline">Before the model can guess, text has to become integers. We write BPE by hand.</p>
<p class="kicker">Thursdays 15:15 · SR 127.11 · Leechgasse 34</p>
</div>
<div class="pic" style="background-image:url(assets/semaphore.jpg)"></div>

Note: Ask them to bring a laptop with Python 3.11 or newer. TODO: confirm the Python version we support.

---

<p class="kicker">Reading</p>

## Read these before next Thursday.

<img class="sticker br sm" src="stickers/owl.png">

- Shannon 1951, [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), sections 1–3.
- Shannon 1948, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), section 3 only.
- Ben Hoyt, [Markov chain in 20 lines of Python](https://benhoyt.com/writings/markov-chain/). Run it on a book you like.
- Cameron Wolfe, [Language Model Training and Inference: From Concept to Code](https://cameronrwolfe.substack.com/p/language-model-training-and-inference).
- Stanford CS336, [lecture 1](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py), first half.

Note: All figures in this deck are reproducible from slides/figures/01_*.py (texts from Project Gutenberg, cached in figures/data/).
