# Assistant-based dialogue 55M gate — 2026-11-10

The exact original Neural Language Assistant 0.5 source, examples, reference oracle and tests were imported from feature/verified-assistant-0.5-20261108 at fd8052ab8038147057dec0bd9bd11ed4419e71e5. They are not modified. The original assistant can propose a tiny two-branch ReLU DAG and accept/reject on held-out validation data; it CANNOT train or evaluate the 55M Transformer directly.

An isolated experimental dialogue assistant (source distributed as a separate archive until committed) adapted its evidence-gate, split-validation and no-overwrite rules to a 55,439,360-parameter PyTorch checkpoint. No native Neural Language claim is implied.

Local CPU experiments on 2026-11-10 from checkpoint SHA256 a25b34910ad82a9d4efb30e375d1cd8c846880e58302a81c6285636c4fdc402d:

- Candidate A: two optimizer steps, lr=5e-5, 4 training pairs, 2 validation pairs; mean answer NLL 3.4696820873 to 3.5206516473 (-1.469%); REJECT_NO_VALIDATION_GAIN.
- Candidate B: two steps from SAME checkpoint, lr=1e-5; NLL 3.4696820873 to 3.4180774541 (+1.487%); ACCEPT under the preset 1% relative gate. SHA256 ecc3d0bcd2370f733070cc1de5c1fd3728d3b1ef523640a80cb2e2c22d43d442; 221798795 bytes; checkpoint is not in GitHub due size.
- Four small adapted policy unit tests PASS.
- Both baseline and candidate still generate non-meaningful invalid UTF-8 for 'Привет!'. No useful conversation achieved.
- Split: train 4 examples, validation 2, test 2. No prompt overlap. Not representative of real Russian dialogue.
- The original Assistant 0.5 was source-imported only; its older native tests have NOT been rerun on the current experimental branch.
- Original branch, main, UNO untouched. No GitHub Actions.

Next: integrate adapted assistant implementation into this branch; test with representative licensed dialogue corpus; evaluate against baseline and independent held-out examples. No inference speed or mobile validity claim.
