"""Byte-level BPE from scratch (lecture 2). A tokenizer is a list of merges: ids 0-255 are the
256 bytes, merge number i makes id 256 + i, and the special tokens come after the merges.
    python tokenizer.py      # fetches data/tiny.txt if missing, trains 2048 tokens, writes data/tokenizer.json
"""
import json
import os
import re
import sys
from collections import Counter

SPECIALS = ["<|endoftext|>"]               # sft.py appends "<|user|>" and "<|assistant|>"
CHAT_SPECIALS = SPECIALS + ["<|user|>", "<|assistant|>"]

# GPT-2's pre-tokenizer in plain re: contraction, word with its leading space, number, punctuation, spaces.
PRETOKEN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def merge_pair(word, pair, new_id):
    """Replace every (a, b) in the tuple `word` with new_id."""
    out = []
    for token in word:
        if out and (out[-1], token) == pair:
            out[-1] = new_id
        else:
            out.append(token)
    return tuple(out)


def train_on_counts(word_counts, n_merges):
    """BPE on {pre-token string: count}. Returns merges as a list of (id, id) pairs."""
    vocab = [bytes([b]) for b in range(256)]                  # id -> the bytes it stands for
    words = {tuple(w.encode("utf-8")): c for w, c in word_counts.items()}
    pairs = Counter()
    for word, count in words.items():
        for pair in zip(word, word[1:]):
            pairs[pair] += count
    merges = []
    for _ in range(n_merges):
        # most frequent pair; ties go to the lexicographically greater pair of byte strings
        best = max(pairs, key=lambda p: (pairs[p], vocab[p[0]], vocab[p[1]]))
        if pairs[best] == 0:                                  # every word is a single token already
            break
        new_id = len(vocab)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        merges.append(best)
        # only the words that contain the pair change: take their old pairs out, put the new ones in
        for word in [w for w in words if best in zip(w, w[1:])]:
            count = words.pop(word)
            for pair in zip(word, word[1:]):
                pairs[pair] -= count
            merged = merge_pair(word, best, new_id)
            words[merged] = count
            for pair in zip(merged, merged[1:]):
                pairs[pair] += count
    return merges


def split_specials(text, specials):
    """'a<|endoftext|>b' -> ['a', '<|endoftext|>', 'b']. Special tokens are never split or merged."""
    pattern = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
    return [part for part in re.split(f"({pattern})", text) if part] if specials else [text]


def train(text, vocab_size, specials=SPECIALS):
    parts = [part for part in split_specials(text, specials) if part not in specials]
    counts = Counter(piece for part in parts for piece in PRETOKEN.findall(part))
    return train_on_counts(counts, vocab_size - 256 - len(specials))


def encode(text, merges, specials=SPECIALS):
    ranks = {pair: i for i, pair in enumerate(merges)}        # earlier merge = lower rank
    special_id = {s: 256 + len(merges) + i for i, s in enumerate(specials)}
    ids = []
    for part in split_specials(text, specials):
        if part in special_id:
            ids.append(special_id[part])
            continue
        for piece in PRETOKEN.findall(part):
            word = tuple(piece.encode("utf-8"))
            # replay the merges in the order they were learned: always the earliest-learned pair first
            while len(word) > 1:
                pair = min(zip(word, word[1:]), key=lambda p: ranks.get(p, len(merges)))
                if pair not in ranks:
                    break
                word = merge_pair(word, pair, 256 + ranks[pair])
            ids.extend(word)
    return ids


def token_bytes(merges, specials=SPECIALS):
    """id -> the bytes it stands for."""
    vocab = [bytes([b]) for b in range(256)]
    for a, b in merges:
        vocab.append(vocab[a] + vocab[b])
    return vocab + [s.encode("utf-8") for s in specials]


def decode(ids, merges, specials=SPECIALS):
    return b"".join(token_bytes(merges, specials)[i] for i in ids).decode("utf-8", errors="replace")


def save(path, merges, specials=SPECIALS):
    json.dump({"merges": [list(p) for p in merges], "specials": specials}, open(path, "w"))


def load(path):
    saved = json.load(open(path))
    return [tuple(p) for p in saved["merges"]], saved["specials"]


if __name__ == "__main__":
    if not os.path.exists("data/tiny.txt"):
        import data                                           # only to fetch the corpus
        data.download()
    merges = train(open("data/tiny.txt", encoding="utf-8").read(), int(sys.argv[1]) if sys.argv[1:] else 2048)
    save("data/tokenizer.json", merges)
    vocab = token_bytes(merges)
    print(f"{len(merges)} merges. The first ten:", [vocab[256 + i] for i in range(10)])
    ids = encode("Once upon a time, there was a little dog.", merges)
    print(f"41 bytes, {len(ids)} tokens:", [vocab[i].decode() for i in ids])
