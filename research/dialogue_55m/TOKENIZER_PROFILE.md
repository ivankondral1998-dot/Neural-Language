# Optional BPE experiment

This is an owner-authorized extension of the isolated PyTorch trainer. The
product barrier remains expressing and efficiently executing neural programs
in the native Neural Language compiler. This experiment removes a research
training limitation: Russian text consumes two byte tokens per Cyrillic letter
and the old context is only 128 tokens. It does **not** add transformer operators
or a tokenizer to `language/compiler.py`, prove conversational quality, or prove
an Android advantage.

The established baseline is the unchanged `byte260` model. The optional baseline
technology is Hugging Face `tokenizers==0.22.2`, with ordinary ByteLevel BPE, a full
256-byte alphabet, no normalization, and four structural delimiter IDs. Literal
delimiter spellings inside messages encode as text. Tokenizer inference needs
no external service or pretrained vocabulary. The added Python dependency is
optional: byte260 still works without it.

An alternative with no new library is mapping a fixed Cyrillic Unicode block to
single tokens and retaining UTF-8 byte fallback. It is small and deterministic,
but only compresses those characters and learns no useful multi-character units.
This change uses established BPE to avoid implementing and maintaining a new BPE
trainer. Tokenizer JSON is portable; the current runtime dependency is Python/Rust
`tokenizers`, and there is no native C++/Android tokenizer implementation here.

`experiment.nir` retains its original profile: context128, width768, heads12,
blocks10, feedforward2048 and 55,439,360 tied-embedding parameters.
`experiment_bpe.nir` explicitly selects context1024, width768, heads12, blocks10,
feedforward1824. It uses the **actual** vocabulary size, at most4096, with no
padding weights. Its exact count is `52,684,608 + (vocab_size - 260) * 768`:
55,630,656 when the vocabulary contains4096 entries. Small synthetic training
corpora may yield substantially fewer entries and parameters. Reports include
the actual count and whether it lies in55–56million; such smaller test models
are not presented as55M models.

The longer context increases activation and attention costs; equal parameter
count does not imply equal training RAM, speed, or mobile suitability. Both
profiles remain single-turn causal training with answer-only loss, sequential
batch size1, no packing, no shuffle, and no KV cache. Generation rejects an
overlong initial prompt and then uses the existing sliding token window.
Comparisons of NLL or accuracy between different tokenizers are not meaningful
as direct quality comparisons. The assistant reports token metrics for BPE and
compares candidates using the same checkpoint tokenizer.

Commands from the repository root (use a virtual environment):

```bash
python -m pip install -r research/dialogue_55m/requirements-tokenizer.txt
python research/dialogue_55m/dialogue_tokenizer.py train-tokenizer \
  --input /path/to/train_only_utf8.txt --output /path/to/new-tokenizer.json
python research/dialogue_55m/train.py research/dialogue_55m/experiment_bpe.nir \
  --tokenizer /path/to/new-tokenizer.json --dry-run
python research/dialogue_55m/train.py research/dialogue_55m/experiment_bpe.nir \
  --tokenizer /path/to/new-tokenizer.json --output-dir /path/to/fresh-run
python research/dialogue_55m/chat.py /path/to/fresh-run/dialogue_bpe.pt 'Привет'
```

The checked-in BPE source names the existing toy file for a smoke example;
copy the source beside an explicitly chosen dataset for other experiments.
Train the tokenizer on the training split only. Do not use validation/test data
to choose its vocabulary. Tokenizer fitting uses explicit UTF-8 training sample
files bounded to 512 MiB total and 1 MiB per line; it need not fit on the complete
corpus. No persona data or existing weights are implicit inputs.

The actual training path streams JSONL up to 16 GiB, with a 1 MiB per-row bound,
validates every row before training,
and reopens the file in deterministic order at each epoch. Only one example's
tensors are retained. The compatibility `examples()` helper and assistant split
loader refuse inputs larger than 16 MiB before materializing their small
datasets; this is not a claim of streaming the assistant's split-leakage sets.
There is no plain-text pretraining path or evidence of training on the previously
requested multi-gigabyte corpus. Supporting its size as JSONL is only an input
bound, and a complete prevalidation pass adds startup I/O and tokenization cost.

New checkpoints embed the complete tokenizer JSON, its SHA256 and actual
vocabulary size. Every trainer/chat/assistant loader validates the digest and
embedding shape. Old byte260 checkpoints keep their original IDs and schema
support. There is no silent conversion of old weights into a new tokenizer.

Fresh training saves AdamW and torch CPU RNG state in addition to weights.
`--resume /path/to/checkpoint.pt` requires identical architecture, tokenizer,
dataset digest and learning rate; source `steps` specifies **additional** steps.
Choose a new output directory. Existing checkpoint paths and tokenizer files
are never overwritten. Optimizer-bearing checkpoints are approximately three
times the size of fp32 weights alone. Exact resume of historical weights-only
checkpoints is rejected; the assistant's existing fresh-optimizer continuation
still accepts them and clearly labels its behavior. Assistant candidates omit
the original optimizer state and cannot be passed off as exact resumes.

Acceptance evidence is deliberately bounded: synthetic Unicode roundtrip,
reserved-literal handling, digest/mismatch rejection, answer masking, context
rejection, streamed reading, old checkpoint inference, candidate metadata,
exclusive output creation, and tensor-identical split vs uninterrupted tiny CPU
training. `tokenizer_probe.py` produces a separate synthetic held-out compression
report and a tokenizer JSON, without training a language model:

```bash
cd research/dialogue_55m
python -m unittest -v test_tokenizer test_training_tokenizer test_assistant_dialogue
python tokenizer_probe.py --output-dir /tmp/new-synthetic-probe
```

Compression on this small synthetic probe does not establish general Russian
tokenization quality. A subsequent falsifiable experiment must train on a
properly licensed training-only corpus, evaluate held-out compression and answer
quality under matched compute, and measure real peak RAM/time. Full BPE model
training, useful conversation, native transformer compilation, and Android
execution remain unverified.

Recorded local evidence: [15 passing tests and source hashes](evidence/bpe_20261110/validation.json),
[test log](evidence/bpe_20261110/tests.log), and
[held-out compression report](evidence/bpe_20261110/compression.json).
The synthetic tokenizer has 462 vocabulary entries; three Russian held-out
sentences use 130 BPE tokens for 331 UTF-8 bytes (2.546 bytes/token), with exact
roundtrip. Its corresponding full architecture has 52,839,744 parameters, outside
the 55M band. The recorded maximum-profile 55,630,656 count applies only when
the actual vocabulary reaches 4096. Tests ran with
`/workspace/nl-dev-venv/bin/python -m unittest -v test_tokenizer test_training_tokenizer test_assistant_dialogue`
from this directory. This evidence contains no original checkpoint or persona
dialogue. Tokenizer fitting and model tests use only the visible synthetic data.
