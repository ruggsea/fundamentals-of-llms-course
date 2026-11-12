"""Course tokenizer interface over the supplied Rust BPE backend."""
from dataclasses import dataclass
import json
from pathlib import Path

from helpers.bpe import EOT, byte_decoder, load_backend, train_backend

FORMAT = 'course-hf-byte-bpe-v1'


@dataclass(frozen=True)
class Encoding:
    ids: list[int]


class Tokenizer:
    def __init__(self, backend, version):
        self._backend, self._backend_version = backend, version
        self.eot_id = backend.token_to_id(EOT)
        if self.eot_id is None:
            raise ValueError('Tokenizer is missing the end-of-document token')
        decoder = byte_decoder()
        self.vocabulary = []
        for token_id in range(backend.get_vocab_size()):
            token = backend.id_to_token(token_id)
            if token is None:
                raise ValueError('Tokenizer vocabulary IDs must be contiguous')
            try:
                piece = b'' if token_id == self.eot_id else bytes(decoder[c] for c in token)
            except KeyError as error:
                raise ValueError('Tokenizer contains a non-byte-level vocabulary entry') from error
            self.vocabulary.append(piece)
        if not set(bytes([i]) for i in range(256)) <= set(self.vocabulary):
            raise ValueError('Tokenizer must retain all 256 base bytes')

    @classmethod
    def train(cls, documents, vocab_size):
        backend, version = train_backend(documents, vocab_size)
        return cls(backend, version)

    def encode(self, text):
        raw = text.encode('utf-8')
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if self.eot_id in ids or b''.join(self.vocabulary[i] for i in ids) != raw:
            raise ValueError('Encoded tokens do not preserve the original UTF-8 bytes')
        return Encoding(ids)

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for token_id in ids:
            self._validate_id(token_id)
            if token_id == self.eot_id:
                if not skip_special_tokens:
                    pieces.append(EOT.encode('utf-8'))
            else:
                pieces.append(self.vocabulary[token_id])
        # Samples may end partway through a UTF-8 character.
        return b''.join(pieces).decode('utf-8', errors='replace')

    def _validate_id(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < len(self.vocabulary):
            raise ValueError('Invalid token ID')

    def token_bytes(self, token_id):
        self._validate_id(token_id)
        return self.vocabulary[token_id]

    def token_to_id(self, text):
        return self.eot_id if text == EOT else None

    def get_vocab_size(self):
        return len(self.vocabulary)

    def to_str(self):
        data = {'format': FORMAT, 'backend_version': self._backend_version,
                'tokenizer': json.loads(self._backend.to_str())}
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_str(cls, value):
        data = json.loads(value)
        if data.get('format') == FORMAT:
            return cls(load_backend(data['tokenizer']), data['backend_version'])
        raise ValueError('Expected a versioned course tokenizer file')

    def save(self, path):
        Path(path).write_text(self.to_str(), encoding='utf-8')

    @classmethod
    def from_file(cls, path):
        return cls.from_str(Path(path).read_text(encoding='utf-8'))
