"""Portable optional byte-level BPE for the isolated PyTorch experiment.

No network, external pretrained vocabulary, normalization or persona defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPECIAL = ('<|bos|>', '<|user|>', '<|assistant|>', '<|eos|>')
MAX_JSON = 2 * 1024 * 1024
MAX_CORPUS = 512 * 1024 * 1024
MAX_LINE = 1024 * 1024


class ByteTokenizer:
    kind = 'byte260'
    vocab_size = 260
    bos, user, assistant, eos = 256, 257, 258, 259
    special_ids = frozenset((256, 257, 258, 259))

    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='replace')

    def metadata(self):
        return 'byte260'


def _library():
    try:
        import tokenizers
    except ImportError as exc:
        raise ValueError('BPE requires optional requirements-tokenizer.txt dependency') from exc
    return tokenizers


class BpeTokenizer:
    kind = 'bpe4096'

    def __init__(self, raw):
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_JSON:
            raise ValueError('invalid tokenizer JSON size')
        library = _library()
        try:
            spec = json.loads(raw)
            model = spec['model']
            pre = spec['pre_tokenizer']
            if (model['type'] != 'BPE' or model.get('dropout') is not None
                    or model.get('unk_token') is not None
                    or model.get('byte_fallback', False)
                    or model.get('continuing_subword_prefix') is not None
                    or model.get('end_of_word_suffix') is not None
                    or spec.get('normalizer') is not None
                    or spec.get('post_processor') is not None
                    or spec.get('padding') is not None or spec.get('truncation') is not None
                    or pre['type'] != 'ByteLevel' or pre['add_prefix_space']
                    or spec['decoder']['type'] != 'ByteLevel'):
                raise ValueError('unsupported tokenizer pipeline')
            try:
                self.backend = library.Tokenizer.from_str(raw)
            except Exception as exc:
                raise ValueError('invalid tokenizer model JSON') from exc
            self.vocab_size = self.backend.get_vocab_size()
            vocab = self.backend.get_vocab()
            if (not 260 <= self.vocab_size <= 4096
                    or set(vocab.values()) != set(range(self.vocab_size))
                    or not set(library.pre_tokenizers.ByteLevel.alphabet()) <= set(vocab)
                    or len(spec['added_tokens']) != 4
                    or any(not token['special'] for token in spec['added_tokens'])):
                raise ValueError('invalid BPE vocabulary')
            ids = [self.backend.token_to_id(s) for s in SPECIAL]
            if ids != [0, 1, 2, 3]:
                raise ValueError('unexpected special token IDs')
            self.bos, self.user, self.assistant, self.eos = ids
            self.special_ids = frozenset(ids)
            # Literal delimiter spellings are user text, never structural IDs.
            self.backend.encode_special_tokens = True
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError('malformed tokenizer JSON') from exc
        self.raw = raw
        self.digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def encode(self, text):
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if self.special_ids & set(ids):
            raise ValueError('text encoded as structural special token')
        return ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)

    def metadata(self):
        return {'kind': self.kind, 'vocab_size': self.vocab_size,
                'sha256': self.digest, 'json': self.raw}


def load_tokenizer(metadata):
    if metadata == 'byte260':
        return ByteTokenizer()
    if not isinstance(metadata, dict) or metadata.get('kind') != 'bpe4096':
        raise ValueError('unknown checkpoint tokenizer')
    raw = metadata.get('json')
    if (not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_JSON
            or hashlib.sha256(raw.encode('utf-8')).hexdigest() != metadata.get('sha256')):
        raise ValueError('tokenizer digest mismatch')
    tokenizer = BpeTokenizer(raw)
    if tokenizer.vocab_size != metadata.get('vocab_size'):
        raise ValueError('tokenizer vocabulary mismatch')
    return tokenizer


def tokenizer_file(path):
    if not path.is_file() or path.stat().st_size > MAX_JSON:
        raise ValueError('tokenizer JSON missing or too big')
    return BpeTokenizer(path.read_text(encoding='utf-8'))


def train_tokenizer(paths, output, vocab_size=4096):
    if not 260 <= vocab_size <= 4096:
        raise ValueError('vocabulary limit must be 260..4096')
    if output.exists() or output.is_symlink():
        raise ValueError('refusing tokenizer overwrite')
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('missing explicit UTF-8 training files')
    if sum(p.stat().st_size for p in paths) > MAX_CORPUS:
        raise ValueError('tokenizer corpus exceeds 512 MiB bound')

    def lines():
        count = 0
        for path in paths:
            with path.open('rb') as stream:
                while line := stream.readline(MAX_LINE + 1):
                    if len(line) > MAX_LINE:
                        raise ValueError('tokenizer corpus line exceeds 1 MiB')
                    count += len(line)
                    if count > MAX_CORPUS:
                        raise ValueError('tokenizer corpus exceeds 512 MiB bound')
                    yield line.decode('utf-8')
        if not count:
            raise ValueError('empty tokenizer corpus')

    library = _library()
    backend = library.Tokenizer(library.models.BPE())
    backend.pre_tokenizer = library.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = library.decoders.ByteLevel()
    backend.train_from_iterator(lines(), library.trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=library.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL), show_progress=False))
    tokenizer = BpeTokenizer(backend.to_str())
    with output.open('x', encoding='utf-8') as stream:
        stream.write(tokenizer.raw)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train-tokenizer'])
    parser.add_argument('--input', type=Path, nargs='+', required=True,
                        help='explicit plain UTF-8 files; train split only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocab-size', type=int, default=4096)
    args = parser.parse_args()
    try:
        tokenizer = train_tokenizer(args.input, args.output, args.vocab_size)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'TOKENIZER ERROR: {exc}\n')
    print(json.dumps({'kind':tokenizer.kind, 'vocab_size':tokenizer.vocab_size,
                      'sha256':tokenizer.digest, 'path':str(args.output)}))


if __name__ == '__main__':
    main()
