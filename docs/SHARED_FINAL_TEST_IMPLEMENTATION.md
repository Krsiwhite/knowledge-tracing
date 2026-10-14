# Shared final-test implementation scope

Implementation started 2026-10-15 in `research/shared-final-test`, based on
`1bc4111`. Main and the UKT research worktree remain unchanged.

This change adds a separately released final-test evaluator for checkpoints
selected by the shared KC200 development protocol. It does not change training,
model mathematics, existing development evaluation, or historical artifacts.

The release must bind the frozen analysis plan, a source-compatibility review,
the original training/configuration/model/checkpoint/validation identities, and
the new evaluation source and protocol. A persistent exclusive attempt record
must precede the first test-file read and survive failure. Test results are new
artifacts, never rewritten training metadata. Original held-out records retain
fold -1; training fold remains a separate run attribute.

Acceptance is limited to temporary synthetic files, JSON checkpoint fixtures,
and a fixture predictor. Tests must cover pre-access rejection, original-event
KC200 semantics, label isolation, old/new identity separation, complete output
receipts, source mutation, and one-attempt behavior. No real prepared data, runs,
weights, model forward, gradient, optimizer, GPU, or operational release record
will be accessed or created during this implementation.

The initial delivery prioritizes release, data, checkpoint, and evaluation
entry points. The final-statistics reader was deliberately deferred; statistical
computations themselves remain unchanged. The implemented interface, release
schema and remaining boundary are in [SHARED_FINAL_TEST.md](SHARED_FINAL_TEST.md).

## Acceptance completed

The final synthetic/pure-file run returned exit 0: **65 passed, 3 deselected
in 6.57s**. The three deselected cases are the optional existing-development-data
audits. Only one pure guard test from the engine suite was selected; tests that
construct real models were not run. CPU tensor collation verifies label
isolation, but no model forward occurred.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
/home/ml/.venvs/knowledge-tracing/bin/python -m pytest -q --tb=short \
  tests/test_shared_final_test.py tests/test_provenance.py \
  tests/test_shared_evaluation.py tests/test_shared_events.py \
  tests/test_shared_engine.py::test_shared_config_and_validation_only_guards \
  -k "not existing_development_data"
```

This command ran through `wsl -d Ubuntu-24.04 -- bash -lc` in the isolated
final-test worktree. Earlier fixture-only failures were resolved by stubbing
the source fingerprint consistently and supplying the full synthetic data
configuration to the tensor projection audit. Neither required model execution.

Coverage includes rejected control paths before data reads; original-vs-new
identity separation; epoch-zero best after later training; full KC200 event
histories; test fold -1 independent of training fold; duplicate KCs and repeated
question attempts; AS15 dummy mapping; label-independent input tensors;
single-class AUC; frozen-set multiple-run evaluation; active/failed attempt
reentry; changed source/receipts; and output receipt consistency. The source
diff retains existing development guards and original checkpoint validation.
