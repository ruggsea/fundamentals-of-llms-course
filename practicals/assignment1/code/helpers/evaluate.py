"""Score text with the checkpoint's tokenizer and the fixed course protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from helpers.tokenizer import Tokenizer

from helpers.infrastructure import evaluate, select_device
from model import Config, GPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--text', type=Path, required=True,
                        help='UTF-8 documents separated by <|endoftext|>')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps', 'auto'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_str(saved['tokenizer'])
    raw = args.text.read_bytes()
    documents = [doc for doc in raw.decode('utf-8').split('<|endoftext|>') if doc.strip()]
    if not documents:
        raise ValueError('Evaluation text contains no nonempty documents')
    ids = [saved['eot_id']]
    for doc in documents:
        ids.extend(tokenizer.encode(doc).ids)
        ids.append(saved['eot_id'])
    lengths = np.array([0 if i == saved['eot_id'] else len(tokenizer.token_bytes(i))
                        for i in range(saved['model_config']['vocab_size'])])
    device = select_device(args.device)
    model = GPT(Config(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    result = evaluate(model, np.asarray(ids, dtype=np.int64), lengths, device, 'float32', args.batch_size)
    print(json.dumps({
        'protocol': 'packed-nonoverlapping-v1', 'checkpoint': str(args.checkpoint),
        'text_sha256': hashlib.sha256(raw).hexdigest(),
        'tokenizer_sha256': hashlib.sha256(saved['tokenizer'].encode()).hexdigest(),
        'context': model.config.context, 'precision': 'float32',
        'loss_nats_per_token': result['val_loss'], 'bits_per_byte': result['val_bpb'],
        'scored_tokens': result['val_tokens'], 'text_bytes': result['val_bytes'],
    }, indent=2))


if __name__ == '__main__':
    main()
