# PRODIGY_GA_03 — Text Generation with Markov Chains

## Task
Implement a simple text generation algorithm using Markov chains. This task
involves creating a statistical model that predicts the probability of a
character or word based on the previous one(s).

## Approach
This project implements a Markov chain text generator **from scratch** — no
ML libraries, just the Python standard library — since the task is about
understanding the statistical model itself rather than using a pre-built one.

The model is trained on **Alice's Adventures in Wonderland** (public domain,
downloaded automatically from Project Gutenberg). Three things were built:

1. **A configurable `MarkovChain` class** — works at either **word level**
   or **character level**, with an adjustable **order** *n* (how many
   previous tokens the next-token prediction is conditioned on).
2. **Word-level comparison** — the same corpus modelled at order 1, 2, and 3
   to show how a longer context window produces more coherent text.
3. **Character-level comparison** — the same corpus modelled at order 2, 4,
   and 8 to show the progression from letter-soup to near-English.

## How Markov Chains Work (Concept Summary)
A Markov chain assumes the next token depends **only on the current state**
(the last *n* tokens), not the full history. Training is just counting: slide
a window over the corpus and record, for every state, how often each token
follows it. Generation then walks the chain — start from a state, sample the
next token **proportionally to its observed frequency**, shift the window,
repeat. For example, if "Alice was" was followed by "beginning" 3 times and
"not" 1 time in the corpus, the model picks "beginning" with probability 75%.

This is fundamentally the same *idea* as GPT-2's next-token prediction (see
[`PRODIGY_GA_01`](https://github.com/anurita-bose/PRODIGY_GA_01)) — but where GPT-2 learns dense
representations that generalize to unseen contexts, a Markov chain can only
replay exact n-gram statistics it has counted, which is why it needs so
little compute and why its coherence collapses beyond short spans.

## Results

### Word-Level: Effect of Order
| Order | Behaviour |
|---|---|
| 1 | Grammatical fragments that derail every few words — each word only "knows" the one before it |
| 2 | Locally fluent phrases and clauses; sentences drift but read naturally in short spans |
| 3 | Long verbatim-feeling passages — states become so specific they usually have only one continuation, so the model mostly quotes the corpus |

Example output (samples vary per run — full samples are written to
`outputs/word_level_samples.txt`):

```
[order=1] Rabbit with pink eyes ran close by her sister on the pleasure of having nothing so very remarkable in it...
[order=2] So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy...
```

### Character-Level: Effect of Order
| Order | Behaviour |
|---|---|
| 2 | Pronounceable pseudo-words ("Alicket slear do. The use by sied pice trought") — captures letter pairs, not vocabulary |
| 4 | Mostly real words with occasional inventions; word boundaries and spelling largely correct |
| 8 | Near-perfect English at the phrase level — approaching word-level order-2 quality |

Full samples are written to `outputs/char_level_samples.txt`.

### Key Observation
Order controls a **coherence vs. novelty trade-off**: low orders generate
novel but incoherent text, high orders generate coherent text that is
increasingly just memorized corpus. The sweet spot (word-level order 2, or
char-level order ~8 for this corpus size) balances the two — and no order
setting fixes the core limitation that the model has no memory beyond its
window and no notion of meaning.

## How to Run
```bash
pip install -r requirements.txt
python text_generation.py
```
Downloads the corpus, trains all six models, prints samples, writes them to
`outputs/word_level_samples.txt` and `outputs/char_level_samples.txt`, and
bundles them into `markov_outputs.zip` (auto-downloaded when run in Colab).
(No third-party dependencies — `requirements.txt` is included for
consistency; only the standard library is used.)

Or open `PRODIGY_GA_03.ipynb` in Google Colab and run all cells.

## What I Learned
Building the model from scratch made the mechanics of statistical language
modelling concrete: "training" a Markov chain is literally frequency
counting, and "generation" is weighted random sampling — there is no
optimization, no loss function, no parameters in the neural sense. The most
instructive part was varying the order and watching the model slide from
noise, through creative-but-broken text, into plain memorization. It made
clear *why* neural language models were needed: a Markov chain's state space
explodes exponentially with context length, while a transformer compresses
context into representations that generalize instead of memorize.

## Tech Stack
- Python 3 (standard library only: `collections`, `random`, `re`, `urllib`)
- Corpus: *Alice's Adventures in Wonderland* via Project Gutenberg
- Google Colab
