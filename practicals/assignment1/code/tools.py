"""Run the supplied utilities; see assignment.md for commands."""
import argparse
from importlib import import_module
import sys

DATA_COMMANDS = ("split-data", "train-tokenizer", "prepare", "tokenize")
COMMANDS = DATA_COMMANDS + ("generate", "evaluate", "plot-runs", "submission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.argv = [f"{sys.argv[0]} {args.command}", *args.arguments]
    if args.command in DATA_COMMANDS:
        import_module("helpers.data").main(args.command)
    else:
        import_module("helpers." + args.command.replace("-", "_")).main()


if __name__ == "__main__":
    main()
