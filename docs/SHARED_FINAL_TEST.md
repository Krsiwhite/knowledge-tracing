# Released shared final-test entry

This implementation is separate from the frozen training implementations. It
does not turn a development result into a test result by changing its protocol
or source hash. The checkpoint remains labeled
`ktlab-v2-shared-event-context-kc200-dev`; the new evaluator and artifacts use
`ktlab-v2-shared-event-context-kc200-test-v1`.

Implementation acceptance used temporary synthetic files, JSON checkpoint
fixtures and fixed fixture probabilities. No real test release was generated,
and no real data, training weights, model forward, optimizer or GPU was used.

## Invocation after a reviewed freeze

The eventual invocation, after constructing and reviewing the records below,
is:

```sh
kt evaluate --run /absolute/path/to/frozen/run --split test --allow-test \
  --test-release /absolute/path/to/frozen/shared-test-release.json
```

The Python equivalents are
`evaluate_run(run_dir, split="test", allow_test=True, test_release_file=release)`
or `final_test.evaluate_shared_test(run_dir, release, allow_test=True)`.
The external argument does not modify `resolved_config.json`. The ordinary
`valid` path retains its original current-source identity checks. The direct
KTST reference evaluator still rejects test; shared KTST uses the common entry
above. There is no test-training or model-selection entry.

## Three frozen records

All records are JSON outside the prepared-data directory. Their SHA256 values
refer to actual file bytes. These are unsigned consistency receipts and a
recorded review/authorization basis, not a cryptographic signature or proof
that a training history or mathematical equivalence claim is true.

The **release** requires `schema_version: 1`, the final-test `protocol`,
`status: "approved"`, nonempty `user_confirmation` and `approved_at`, and:

| Field | Meaning |
| --- | --- |
| `source_sha256` | New evaluation implementation fingerprint, never the old training fingerprint |
| `analysis_plan_file`, `analysis_plan_sha256` | Frozen JSON plan path and file hash |
| `compatibility_review_file`, `compatibility_review_sha256` | Explicit old-training/new-evaluator compatibility review and file hash |
| `runs` | Complete frozen mapping from each original `run_id` to its candidate record |

Each candidate record contains `training_protocol` (the original shared-dev
protocol), `metadata_sha256`, `dataset_manifest_sha256`, `checkpoint_sha256`,
`config_sha256`, `implementation_sha256` (original training source),
`model_source_sha256`, `validation_source_sha256`, `validation_target_sha256`,
`validation_context_sha256`, `checkpoint_selection_policy`, `best_epoch`,
`best_valid_auc`, and boolean `best_epoch_completed`. Identity fields must
agree with original metadata and the loaded checkpoint. The resolved model
configuration must agree exactly. Current model source bytes must equal the
original model source; this entry does not implement a model migration.

The **compatibility review** requires `schema_version: 1`,
`status: "approved"`, nonempty `reviewed_by` and `statement`,
`training_protocol`, `evaluation_protocol`, `evaluation_source_sha256`, and
`candidates` exactly equal to the release's entire `runs` mapping. The review
must justify evaluating those old implementations with the new evaluator; the
program checks that the review binds the correct bytes, not its scientific
truth. Do not insert the new evaluator hash into old metadata/checkpoints.

The **analysis plan** requires `schema_version: 1`, the final-test `protocol`,
unique `run_ids` exactly matching the release candidate set,
`metrics: ["auc", "nll", "brier", "accuracy", "ece_15bins"]`,
`accuracy_threshold: 0.5`, `ece_bins: 15`, `kc_budget: 200`,
`test_selection: "all_eligible_complete_event_kc200"`, positive integer
`batch_size` and `max_rows_per_forward`, and absent/null `max_targets`.
Use 32/32 for the established inference budget unless a reviewed plan says
otherwise. Any additional prespecified comparisons/statistical methods are
also frozen by the plan hash; this entry does not execute them or claim that
their presence alone makes a comparison confirmatory.

Only completed, untested, non-debug runs with full shared validation and the
original epoch-zero-plus-strict-best selection policy qualify. An epoch 0
checkpoint is valid even when the run subsequently performed training: it must
retain `checkpoint_stage: "initial_untrained"` and
`best_epoch_completed: false`. The later training need not be absent.

## Data and prediction semantics

Preflight reads only the historical config, manifest JSON, metadata, frozen
control records and checkpoint. It deliberately does **not** call
`read_manifest()` or `verify_manifest_files()`, because those hash test files.
Release/plan/review paths cannot point inside prepared data. Checkpoint paths
cannot redirect into prepared data. Missing approval, changed identities,
debug/partial candidates and conflicting releases fail before test/mapping
reads or model construction.

The final dataset reuses the original event parser, compact target index,
longest complete contiguous history suffix and ordered KC deduplication.
It accepts only canonical `test.csv` with every source row at fold `-1`,
distinct student IDs and the manifest's existing vocabulary. The manifest's
`keyid2idx.json` receipt is checked; no IDs are fitted or remapped on test.
Both required files are hashed only after attempt registration. This assumes
the pinned preprocessing/split audit already established train/test student
disjointness; it does not reread development students to redo that audit.

All eligible events are evaluated with history plus the complete current
query at most 200 KCs, and a nonempty allowed history. Oversized/no-history
exclusions are explicitly counted. The source fold `-1` is used in event and
context hashes. `training_fold` is separate provenance, so training fold 0
and fold 1 see the same held-out target/history identities. Repeated attempts
at the same question retain separate event ordinals. AS15 questions remain
`None` until the existing adapters map them to dummy 0.

Current labels remain outside queries and context digests. Both tensor
representations preserve the existing fixed current-response 0 placeholder;
only earlier answers enter history. The model's original target-only shared
prediction function is reused. It has no optimization/loss return.

## Durable one-attempt policy and outputs

Before the first test or mapping read, the evaluator creates and fsyncs an
exclusive record under `<canonical data_dir>/.final-test-attempts/`.
The candidate attempt key binds original run ID, checkpoint hash, dataset
manifest hash and test-source hash. Outputs always use `<run_dir>/final_test/`.
Changing the release filename or copying the same frozen run to another
output directory cannot reset that canonical attempt.

A separate persistent dataset lock binds the entire release, compatibility
review, analysis plan, evaluator source and candidate set. Its dataset key
uses dataset name, test-source hash and mapping hash. Different frozen
models/folds/seeds in that same release may be evaluated sequentially. A
different plan, source, added candidate, changed checkpoint or changed release
is rejected by the normal entry after the first registration. Do not delete
these records. Explicit recovery or a new-research policy is future work,
not an automatic retry option in this implementation.

On failure, the attempt remains consumed and records the failure; on a process
crash it remains `started`. Even an error after registration but before label
reading conservatively consumes the attempt. Concurrent/reentrant calls
cannot acquire the same attempt. Partial output files never justify a retry.

On success, separate artifacts form this acyclic receipt chain:

1. `predictions.csv` contains full-precision probabilities, labels, event ID,
   question ID, KC count and individual target/context hashes.
2. `predictions.csv.manifest.json` binds CSV bytes, the complete test audit,
   original training identity, new evaluation identity and initial attempt hash.
3. `result.json` binds the CSV and sidecar hashes, metrics and the same identities.
4. The terminal attempt becomes `completed` and binds result/CSV/sidecar hashes.

The terminal attempt must be checked before treating any outputs as a final
result. A failed attempt may have incomplete files left for diagnosis. Original
metadata, validation predictions, histories and weights are never rewritten.
Receipts/source are checked again around export. If all labels have one class,
AUC is null and explicitly marked unavailable; other metrics remain defined.

## Saved final statistics and remaining boundary

The separate [final saved-prediction reader](SHARED_FINAL_PREDICTION_ANALYSIS.md)
checks terminal attempts, all output receipts, the complete frozen cohort and
per-event label/target/history alignment. Its `final_statistics` block must be
part of the hash-frozen analysis plan before the first test attempt. It supports
the complete multi-dataset 5-fold × 3-seed design through the existing pure
statistics function, without changing evaluator source or reopening raw data.
The older saved-prediction analysis remains development-only; changing its
protocol string is not a final-test analysis path.

Candidate selection and failed-attempt recovery remain outside these entries.
Neither statistical availability nor a positive interval independently proves
model originality, training authenticity or a SOTA result.
