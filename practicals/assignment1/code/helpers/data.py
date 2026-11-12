"""Split documents, train one tokenizer, encode data and inspect token IDs."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
from helpers.tokenizer import EOT, Tokenizer

DEFAULT_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_format(path, requested='auto'):
    if requested not in ('auto', 'text', 'jsonl'):
        raise ValueError('format must be auto, text or jsonl')
    return ('jsonl' if Path(path).suffix.lower() == '.jsonl' else 'text') if requested == 'auto' else requested


def iter_documents(path, separator=EOT, input_format='auto', text_field='text',
                   max_document_bytes=DEFAULT_MAX_DOCUMENT_BYTES):
    """Yield verbatim nonempty documents; memory is bounded by one document."""
    if max_document_bytes < 1 or not separator:
        raise ValueError('Need a positive document limit and a nonempty separator')
    kind = resolve_format(path, input_format)
    with Path(path).open('rb') as stream:
        if kind == 'jsonl':
            line_number = 0
            while raw := stream.readline(max_document_bytes * 8 + 1):
                line_number += 1
                if len(raw) > max_document_bytes * 8:
                    raise ValueError(f'JSONL line {line_number} exceeds the input limit')
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                    text = record[text_field]
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f'Invalid JSONL text record at line {line_number}') from error
                if not isinstance(text, str):
                    raise ValueError(f'JSONL field {text_field!r} must be text at line {line_number}')
                if len(text.encode('utf-8')) > max_document_bytes:
                    raise ValueError(f'Document at line {line_number} exceeds max_document_bytes')
                if text.strip():
                    yield text
        else:
            delimiter, pending = separator.encode('utf-8'), b''
            for chunk in iter(lambda: stream.read(65536), b''):
                pieces = (pending + chunk).split(delimiter)
                pending = pieces.pop()
                for piece in pieces:
                    if len(piece) > max_document_bytes:
                        raise ValueError('Document exceeds max_document_bytes')
                    text = piece.decode('utf-8')
                    if text.strip():
                        yield text
                # The suffix may contain the first few bytes of a delimiter.
                if len(pending) > max_document_bytes + len(delimiter) - 1:
                    raise ValueError('Document exceeds max_document_bytes; check the separator')
            if len(pending) > max_document_bytes:
                raise ValueError('Document exceeds max_document_bytes')
            text = pending.decode('utf-8')
            if text.strip():
                yield text


def validation_member(text, seed, fraction):
    if not 0 < fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    digest = hashlib.sha256(str(seed).encode('ascii') + b'\0' + text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') < int(fraction * (1 << 64))


class DocumentStore:
    """Temporary indexed spool; duplicate hashes are checked against exact text.

    The SQLite index stores hashes and file offsets, not corpus text. Text is
    written once to a JSON-string spool. Both are deleted by staged_output().
    """
    def __init__(self, directory):
        directory = Path(directory)
        self.spool_path = directory / 'documents.jsonl'
        self.spool = self.spool_path.open('w+b', buffering=0)
        self.database = sqlite3.connect(directory / 'documents.sqlite')
        self.database.execute('PRAGMA journal_mode=OFF')
        self.database.execute('PRAGMA synchronous=OFF')
        self.database.execute('PRAGMA cache_size=-8192')
        self.database.execute('PRAGMA temp_store=FILE')
        self.database.execute('CREATE TABLE documents (digest BLOB PRIMARY KEY, split TEXT, offset INTEGER, size INTEGER) WITHOUT ROWID')
        self.database.execute('CREATE INDEX split_order ON documents(split, offset)')

    def add(self, text, split):
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        previous = self.database.execute('SELECT split, offset, size FROM documents WHERE digest=?', (digest,)).fetchone()
        if previous is not None:
            existing_split, offset, size = previous
            self.spool.seek(offset)
            if json.loads(self.spool.read(size)) != text:
                raise RuntimeError('SHA-256 collision between different documents')
            return False, existing_split
        record = (json.dumps(text, ensure_ascii=False) + '\n').encode('utf-8')
        offset = self.spool.seek(0, 2)
        self.spool.write(record)
        self.database.execute('INSERT INTO documents VALUES (?, ?, ?, ?)', (digest, split, offset, len(record)))
        return True, None

    def documents(self, split, hash_order=False):
        # Hash order gives a reproducible mix independent of input-file order.
        order = 'digest' if hash_order else 'offset'
        with self.spool_path.open('rb') as stream:
            for offset, size in self.database.execute(f'SELECT offset, size FROM documents WHERE split=? ORDER BY {order}', (split,)):
                stream.seek(offset)
                yield json.loads(stream.read(size))

    def close(self):
        self.spool.close()
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


@contextmanager
def staged_output(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to replace {out}; choose a new output directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.a1-prepare-', dir=out.parent) as temporary:
        workspace = Path(temporary)
        stage = workspace / 'output'
        stage.mkdir()
        yield stage, workspace
        if out.exists():
            raise FileExistsError(f'Output appeared while preparing: {out}')
        stage.rename(out)


def write_documents(path, documents, output_format='text', cap=None, separator=EOT):
    """Write a deterministic in-order whole-document subset within a byte cap."""
    if cap is not None and cap < 1:
        raise ValueError('Document byte caps must be positive')
    count, total, omitted = 0, 0, 0
    with Path(path).open('wb') as stream:
        for text in documents:
            raw = text.encode('utf-8')
            if cap is not None and total + len(raw) > cap:
                omitted += 1
                continue
            if output_format == 'jsonl':
                stream.write((json.dumps({'text': text}, ensure_ascii=False) + '\n').encode('utf-8'))
            else:
                if separator in text:
                    raise ValueError('A document contains the output separator; use JSONL output')
                stream.write(raw + separator.encode('utf-8'))
            total += len(raw)
            count += 1
    if not count:
        raise ValueError(f'No whole document fits in {path.name}; increase its byte cap or supply more data')
    return {'documents': count, 'bytes': total, 'omitted_for_cap': omitted,
            'file': Path(path).name, 'sha256': sha256(path)}


def split_data(source, out, seed=0, val_fraction=0.01, val_bytes=None,
               laptop_train_bytes=2097152, laptop_val_bytes=65536, train_bytes=None,
               input_format='auto', output_format='auto', text_field='text', separator=EOT,
               max_document_bytes=DEFAULT_MAX_DOCUMENT_BYTES):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    if any(value is not None and value < 1 for value in
           (val_bytes, laptop_train_bytes, laptop_val_bytes, train_bytes)):
        raise ValueError('Byte caps must be positive')
    source = Path(source).resolve()
    kind = resolve_format(source, input_format)
    output_kind = kind if output_format == 'auto' else output_format
    if output_kind not in ('text', 'jsonl'):
        raise ValueError('Output format must be auto, text or jsonl')
    extension = 'jsonl' if output_kind == 'jsonl' else 'txt'
    with staged_output(out) as (stage, temporary), DocumentStore(temporary) as store:
        total, duplicates, counts = 0, 0, {'train': 0, 'val': 0}
        for text in iter_documents(source, separator, kind, text_field, max_document_bytes):
            total += 1
            split = 'val' if validation_member(text, seed, val_fraction) else 'train'
            fresh, _ = store.add(text, split)
            if not fresh:
                duplicates += 1
            else:
                counts[split] += 1
        manifest = {'format': 'course-document-split-v1',
                    'source': {'path': str(source), 'sha256': sha256(source), 'format': kind},
                    'seed': seed, 'val_fraction': val_fraction,
                    'assignment': 'sha256(ascii(seed) + NUL + UTF8(text)), first 64 bits, big endian',
                    'subset_selection': 'in-order whole documents; skip documents exceeding the remaining cap',
                    'caps': {'train': train_bytes, 'val': val_bytes,
                             'laptop_train': laptop_train_bytes, 'laptop_val': laptop_val_bytes},
                    'input_documents': total, 'unique_documents': total - duplicates,
                    'duplicates_removed': duplicates, 'assigned_documents': counts,
                    'disjoint': True, 'laptop_subsets_nested': True, 'splits': {}}
        for split, cap in [('train', train_bytes), ('val', val_bytes)]:
            path = stage / f'{split}.{extension}'
            manifest['splits'][split] = write_documents(path, store.documents(split), output_kind, cap, separator)
            laptop_cap = laptop_train_bytes if split == 'train' else laptop_val_bytes
            laptop_path = stage / f'laptop-{split}.{extension}'
            # Read only the selected main split: omitted holdout docs never leak
            # into training, and the laptop data is an exact document subset.
            manifest['splits'][f'laptop_{split}'] = write_documents(
                laptop_path, iter_documents(path, separator, output_kind, 'text', max_document_bytes),
                output_kind, laptop_cap, separator)
        (stage / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def fitting_prefix(documents, cap):
    if cap is not None and cap < 1:
        raise ValueError('fit_bytes must be positive')
    selected, total, seen = [], 0, set()
    digest = hashlib.sha256()
    for text in documents:
        if text in seen:
            continue
        raw = text.encode('utf-8')
        if cap is not None and total + len(raw) > cap:
            break
        selected.append(text)
        seen.add(text)
        total += len(raw)
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    if not selected:
        raise ValueError('No whole training document fits --fit-bytes; increase the cap')
    return selected, {'selection': 'training_prefix_whole_documents', 'max_bytes': cap,
                      'documents': len(selected), 'bytes': total,
                      'framed_text_sha256': digest.hexdigest()}


def write_tokens(path, documents, tokenizer):
    eot = tokenizer.token_to_id(EOT)
    count, byte_count, token_count = 0, 0, 1
    with Path(path).open('wb') as stream:
        stream.write(np.asarray([eot], dtype='<u4').tobytes())
        for text in documents:
            ids = tokenizer.encode(text).ids
            ids.append(eot)
            stream.write(np.asarray(ids, dtype='<u4').tobytes())
            token_count += len(ids)
            byte_count += len(text.encode('utf-8'))
            count += 1
    if not count:
        raise ValueError(f'{path.name} has no documents; provide a nonempty disjoint split')
    return {'documents': count, 'tokens': token_count, 'bytes': byte_count}


def train_tokenizer(source, out, vocab_size=4096, fit_bytes=33554432,
                    input_format='auto', text_field='text', separator=EOT,
                    max_document_bytes=DEFAULT_MAX_DOCUMENT_BYTES):
    """Fit a bounded, reproducible document sample from across the training file."""
    if fit_bytes < 1:
        raise ValueError('fit_bytes must be positive')
    source = Path(source).resolve()
    with staged_output(out) as (stage, temporary), DocumentStore(temporary) as store:
        unique, duplicates = 0, 0
        for text in iter_documents(source, separator, input_format, text_field, max_document_bytes):
            fresh, _ = store.add(text, 'train')
            unique += fresh
            duplicates += not fresh
        # Select by document hash, not position: a sorted source should not make
        # the tokenizer see only its first topic, website or crawl.
        eligible = (text for text in store.documents('train', hash_order=True)
                    if len(text.encode('utf-8')) <= fit_bytes)
        documents, fitting = fitting_prefix(eligible, fit_bytes)
        fitting.update(selection='training_sha256_order_whole_documents',
                       source_documents=unique, duplicates_removed=duplicates)
        tokenizer = Tokenizer.train(documents, vocab_size)
        tokenizer.save(stage / 'tokenizer.json')
        metadata = {'format': 'course-tokenizer-training-v1',
                    'source': {'path': str(source), 'sha256': sha256(source),
                               'format': resolve_format(source, input_format),
                               'text_field': text_field, 'separator': separator},
                    'vocab_size': tokenizer.get_vocab_size(), 'eot_id': tokenizer.token_to_id(EOT),
                    'tokenizer_training': fitting,
                    'tokenizer_sha256': sha256(stage / 'tokenizer.json')}
        split_manifest = source.parent / 'split.json'
        if split_manifest.exists():
            metadata['document_split'] = {'path': str(split_manifest.resolve()),
                                          'sha256': sha256(split_manifest)}
        (stage / 'tokenizer-meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def prepare(source, out, *, val_input, tokenizer_path, input_format='auto',
            text_field='text', separator=EOT, max_document_bytes=DEFAULT_MAX_DOCUMENT_BYTES):
    """Encode supplied train/validation splits with the already trained tokenizer."""
    source, val_input, tokenizer_path = map(lambda p: Path(p).resolve(),
                                          (source, val_input, tokenizer_path))
    tokenizer = Tokenizer.from_file(tokenizer_path)
    fitting = {'selection': 'reused_tokenizer', 'source': str(tokenizer_path),
               'source_sha256': sha256(tokenizer_path)}
    tokenizer_metadata = tokenizer_path.parent / 'tokenizer-meta.json'
    if tokenizer_metadata.exists():
        provenance = json.loads(tokenizer_metadata.read_text())
        if (provenance.get('format') != 'course-tokenizer-training-v1'
                or provenance.get('tokenizer_sha256') != fitting['source_sha256']):
            raise ValueError('Tokenizer fitting metadata does not match the tokenizer artifact')
        fitting['provenance'] = provenance
    with staged_output(out) as (stage, temporary), DocumentStore(temporary) as store:
        duplicates = {'train': 0, 'val': 0}
        for split, path in (('train', source), ('val', val_input)):
            for text in iter_documents(path, separator, input_format, text_field, max_document_bytes):
                fresh, existing_split = store.add(text, split)
                if existing_split is not None and existing_split != split:
                    raise ValueError('Train and validation have exact document overlap')
                duplicates[split] += not fresh
        shutil.copyfile(tokenizer_path, stage / 'tokenizer.json')
        meta = {'format': 1, 'dtype': '<u4', 'separator': separator,
                'vocab_size': tokenizer.get_vocab_size(), 'eot_id': tokenizer.token_to_id(EOT),
                'source': {'path': str(source), 'sha256': sha256(source)},
                'val_source': {'path': str(val_input), 'sha256': sha256(val_input)},
                'split_method': 'provided_exact_deduplicated_documents',
                'tokenizer_training': fitting, 'duplicates_removed': duplicates,
                'document_format': input_format, 'text_field': text_field}
        provenance = source.parent / 'source.json'
        if provenance.exists():
            meta['upstream'] = json.loads(provenance.read_text())
        split_manifest = source.parent / 'split.json'
        if split_manifest.exists():
            meta['document_split'] = {'path': str(split_manifest.resolve()), 'sha256': sha256(split_manifest)}
        for split in ('train', 'val'):
            meta[split] = write_tokens(stage / f'{split}.bin', store.documents(split), tokenizer)
        meta['files'] = {name: sha256(stage / name) for name in ('tokenizer.json', 'train.bin', 'val.bin')}
        (stage / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def main(command, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if command == 'tokenize':
        parser.add_argument('--tokenizer', required=True, type=Path)
        parser.add_argument('--text', required=True)
        args = parser.parse_args(argv)
        tokenizer = Tokenizer.from_file(args.tokenizer)
        ids = tokenizer.encode(args.text).ids
        print('ID\tBytes')
        for token_id in ids:
            print(f'{token_id}\t{tokenizer.token_bytes(token_id)!r}')
        print(f'Decoded: {tokenizer.decode(ids)}')
        return

    parser.add_argument('--input', dest='source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--format', dest='input_format', choices=('auto', 'text', 'jsonl'), default='auto')
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--separator', default=EOT)
    parser.add_argument('--max-document-bytes', type=int, default=DEFAULT_MAX_DOCUMENT_BYTES)
    if command == 'split-data':
        parser.add_argument('--output-format', choices=('auto', 'text', 'jsonl'), default='auto')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--val-fraction', type=float, default=0.01,
                            help='Fraction of documents reserved for validation: 0.01 means 1%% (default)')
        parser.add_argument('--val-bytes', type=int, help='Optional cap; otherwise keep all validation documents')
        parser.add_argument('--train-bytes', type=int)
        parser.add_argument('--laptop-train-bytes', type=int, default=2097152)
        parser.add_argument('--laptop-val-bytes', type=int, default=65536)
        action = split_data
    elif command == 'train-tokenizer':
        parser.add_argument('--vocab-size', type=int, default=4096)
        parser.add_argument('--fit-bytes', type=int, default=33554432,
                            help='Maximum UTF-8 bytes in the sampled training documents')
        action = train_tokenizer
    elif command == 'prepare':
        parser.add_argument('--val-input', required=True, type=Path)
        parser.add_argument('--tokenizer', dest='tokenizer_path', required=True, type=Path)
        action = prepare
    else:
        raise ValueError(f'Unknown data command: {command}')
    result = action(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
