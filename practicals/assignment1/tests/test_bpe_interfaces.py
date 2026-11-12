"""Supplied BPE checks: return types and one simple text round trip."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from helpers.tokenizer import Tokenizer


def test_bpe_interface_and_simple_round_trip():
    tokenizer = Tokenizer.train(["banana banana"], vocab_size=260)
    ids = tokenizer.encode("banana").ids
    assert isinstance(ids, list) and all(type(i) is int for i in ids)
    assert tokenizer.decode(ids) == "banana"
