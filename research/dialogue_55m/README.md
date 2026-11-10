# NL-DIALOGUE-55M — isolated experimental training

**Scope:** separate owner-authorized experiment, NOT UNO and NOT the production \`language/compiler.py\`. The \`.nir\` file is parsed by a standalone PyTorch byte-level backend, not by the current C++17 compiler. Existing project files and main branch remain unchanged.

Base: \`opt/nl-mobile-opt-002-reuse-input-20261110\` at \`5cf5e5b9c96455903c8d898796389a965fd1cf66\`. That base retains earlier language, DAG, HAR-training and mobile research. **All 27 existing branches are NOT automatically merged**, because compiler and runtime changes conflict. See INTEGRATION_MATRIX.md.

The model is a 10-block causal transformer, width 768, 12 heads, FFN 2048, context 128, tied embedding, vocabulary 260 (UTF-8 bytes + separators). **55,439,360 trainable parameters**, not the normative 55,451,639-parameter UNO-B2 design.

Commands from repository root:

\`\`\`bash
python3 -m pip install torch
python3 research/dialogue_55m/train.py research/dialogue_55m/experiment.nir --dry-run
python3 research/dialogue_55m/train.py research/dialogue_55m/experiment.nir --output-dir runs/dialogue55m
python3 research/dialogue_55m/chat.py runs/dialogue55m/dialogue_55m.pt 'Привет!' --tokens 16
\`\`\`

Proof of small local CPU smoke, torch 2.10.0+cpu, four optimizer steps on **FOUR toy dialogues**:
- first_step_loss 5.639770030975342
- last_step_loss 3.9070489406585693 (different training sample; not a quality comparison)
- post_train_loss_first_sample 3.2774851322174072
- loop 10.028 seconds; process 18.26 seconds; peak RSS 1,457,244 KiB
- saved PyTorch checkpoint 221,799,185 bytes; SHA-256 \`a25b34910ad82a9d4efb30e375d1cd8c846880e58302a81c6285636c4fdc402d\`.
- loading and generating were run, but output was **nonsensical/invalid UTF-8**. The model is **NOT trained for useful conversation**.
- 32-step experiment was interrupted by execution time limit, and no such checkpoint exists.

The checkpoint is not committed to GitHub (over 100 MB); separately available in the conversation artifact. It is not an \`.nlw\`, not a GGUF, and not runnable with PocketPal. There is no realistic dialogue training dataset, long-horizon training, held-out evaluation or validated mobile inference. No CI/GitHub Actions used. Do not claim victory over other runtimes or conversational competence.
