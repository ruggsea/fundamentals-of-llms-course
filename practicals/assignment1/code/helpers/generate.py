"""Generate text using the tokenizer embedded in a saved checkpoint."""
import argparse
from pathlib import Path

import torch
from helpers.tokenizer import Tokenizer

from model import Config, GPT
from helpers.infrastructure import select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
    model = GPT(Config(**checkpoint["model_config"])).to(select_device(args.device))
    model.load_state_dict(checkpoint["model"])
    ids = [checkpoint["eot_id"]] + tokenizer.encode(args.prompt).ids
    torch.manual_seed(args.seed)
    generated = model.generate(ids, args.tokens, args.temperature, args.top_k, checkpoint["eot_id"])
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
