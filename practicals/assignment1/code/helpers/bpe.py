"""Supplied Rust byte-level BPE backend; no tokenizer implementation is required."""
import json
import os

EOT = '<|endoftext|>'


def byte_decoder():
    """Invert the reversible byte alphabet used by Hugging Face ByteLevel."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {chr(value): value for value in visible}
    extra = 256
    for value in range(256):
        if value not in visible:
            mapping[chr(extra)] = value
            extra += 1
    return mapping


def train_backend(documents, vocab_size):
    if type(vocab_size) is not int or vocab_size < 257:
        raise ValueError('Vocabulary must fit all 256 bytes plus the end token')
    # Bound the Rust worker pool unless the operator explicitly set its size.
    os.environ.setdefault('RAYON_NUM_THREADS', '4')
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.encode_special_tokens = True
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=[EOT], show_progress=False)
    backend.train_from_iterator(documents, trainer=trainer)
    return backend, tokenizers.__version__


def load_backend(data):
    from tokenizers import Tokenizer
    if (data.get('normalizer') is not None or data.get('post_processor') is not None
            or data.get('model', {}).get('type') != 'BPE'
            or data.get('pre_tokenizer', {}).get('type') != 'ByteLevel'
            or data['pre_tokenizer'].get('add_prefix_space') is not False
            or data.get('truncation') is not None or data.get('padding') is not None):
        raise ValueError('Expected an unnormalized, unpadded course byte-level BPE tokenizer')
    backend = Tokenizer.from_str(json.dumps(data))
    # False would recognize a literal marker in user text as the special token.
    backend.encode_special_tokens = True
    return backend
