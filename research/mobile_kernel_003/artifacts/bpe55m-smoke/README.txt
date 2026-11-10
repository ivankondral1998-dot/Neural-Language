Technical one-step smoke only; NOT a trained conversational model. All inputs are artificial. Full checkpoint contains fresh random-initialized 55,630,656-parameter BPE model after one optimizer step, AdamW state and CPU RNG. No original control weights or persona data used.
Source smoke.nir is preserved exactly; it requests new_random_bpe_smoke.pt. For resume use a NEW output directory to avoid overwriting this checkpoint.
Train command from repository root, with optional dependencies installed:
python research/dialogue_55m/train.py research/mobile_kernel_003/artifacts/bpe55m-smoke/smoke.nir --tokenizer research/mobile_kernel_003/artifacts/bpe55m-smoke/tokenizer.json --resume research/mobile_kernel_003/artifacts/bpe55m-smoke/new_random_bpe_smoke.pt --output-dir runs/NEW_resume_output
Checkpoint is stored via Git LFS: run git lfs pull after checkout. Existing installed binaries/weights are not overwritten.
