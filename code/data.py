"""From text to training batches (lectures 5 and 6).

    python data.py      # encodes data/tiny.txt to data/train.bin and data/val.bin

Every document becomes token ids, documents are joined with <|endoftext|>, and the ids are
stored as uint16 (2 bytes each, fine while the vocabulary is under 65,536).
"""
import json
import os
import urllib.request

import numpy as np
import torch

import tokenizer

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
SEPARATOR = "<|endoftext|>"


def download(path="data/tiny.txt", n_bytes=1_000_000):
    """The first ~1 MB of TinyStories (Eldan and Li 2023): about 1,200 short stories."""
    request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{n_bytes}"})
    raw = urllib.request.urlopen(request).read().decode("utf-8", errors="ignore")
    stories = [s.strip() for s in raw.split(SEPARATOR)][1:-1]    # drop the cut-off first and last
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"\n{SEPARATOR}\n".join(stories))


def read_documents(path="data/tiny.txt"):
    return [d.strip() for d in open(path, encoding="utf-8").read().split(SEPARATOR) if d.strip()]


def encode_documents(documents, merges):
    """Documents -> one uint16 array, with <|endoftext|> after every document."""
    ids = []
    for doc in documents:
        ids.extend(tokenizer.encode(doc, merges))
        ids.append(256 + len(merges))                      # the id of <|endoftext|>
    return np.array(ids, dtype=np.uint16)


def get_batch(ids, batch_size, block_size):
    """batch_size random windows. y is x shifted by one: position t has to guess token t + 1."""
    starts = np.random.randint(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + block_size + 1] for s in starts])
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def load_split(split, data_dir="data"):
    return np.fromfile(f"{data_dir}/{split}.bin", dtype=np.uint16)


if __name__ == "__main__":
    if not os.path.exists("data/tokenizer.json"):
        raise SystemExit("Train the tokenizer first: python tokenizer.py")
    merges, _ = tokenizer.load("data/tokenizer.json")
    documents = read_documents()
    n_val = len(documents) // 10                                   # the last 10% of stories are held out
    splits = {"train": documents[:-n_val], "val": documents[-n_val:]}
    meta = {}
    for split, docs in splits.items():
        ids = encode_documents(docs, merges)
        ids.tofile(f"data/{split}.bin")
        n_bytes = sum(len(d.encode("utf-8")) for d in docs)
        meta[split] = {"documents": len(docs), "tokens": len(ids), "bytes": n_bytes}
        print(f"{split}: {len(docs)} documents, {len(ids):,} tokens, {n_bytes:,} bytes, "
              f"{n_bytes / len(ids):.2f} bytes per token")
    json.dump(meta, open("data/meta.json", "w"), indent=1)
    x, y = get_batch(load_split("train"), batch_size=2, block_size=8)
    print("x[0] =", x[0].tolist(), "\ny[0] =", y[0].tolist())
