# Frozen final-test saved-prediction analysis

This reader consumes completed final-test artifacts from the separately released
shared evaluator. It does not create a release, choose candidates, open raw
test/development data or mappings, read checkpoints, construct models, or execute
training/inference. Its acceptance fixtures are artificial CSV/JSON files in
temporary directories. Development-only analysis remains a separate entry.

## Invocation and complete cohort

After the entire previously frozen release has completed:

```sh
python scripts/analyze_shared_final_predictions.py \
  --release /absolute/path/to/shared-test-release.json \
  --output /absolute/path/to/new-final-statistics.json
```

The CLI offers no candidate, reference, metric, seed, bootstrap-count or subset
switch. It refuses existing output files. `analyze(release_file)` is the Python
entry. The release's existing `analysis_plan_file` must contain the following
additional `final_statistics` block **before the first test attempt**. The
evaluator already freezes the plan's complete file hash and accepts additional
plan fields. Neither its source nor its protocol is changed by this reader.

Every dataset × method must contain the complete five folds `[0,1,2,3,4]` and
three training seeds `[42,2026,3407]`. The cohort must equal the complete release
candidate set. Every candidate must participate in a declared comparison; no
dataset can silently disappear. Different datasets may use different baselines
and configurations. A missing fit/pair refuses the complete report.

## `final_statistics` version 1

| Field | Required value/meaning |
| --- | --- |
| `schema_version` | `1` |
| `method` | `paired-student-mean-run-auc-v1` |
| `reader_sha256` | Byte SHA of `scripts/analyze_shared_final_predictions.py` |
| `statistics_sha256` | Byte SHA of the unchanged `src/ktlab/statistics.py` |
| `metric_helpers_sha256` | Byte SHA of the pure NumPy `scripts/analyze_shared_predictions.py` |
| `folds`, `seeds` | Exactly the arrays above |
| `metrics` | `["auc","nll","brier","accuracy","ece_15bins"]` |
| `auxiliary_intervals` | `none` |
| `dispersion` | `sample_std_ddof1_descriptive` |
| `multiple_comparisons` | `bonferroni_auc_family` |
| `methods` | Mapping from method ID to the declaration below |
| `cohort` | Exact run records below, including canonical absolute run directories |
| `comparisons` | Exact comparisons and their explicit 15 matched pairs |
| `bootstrap` | The fixed block below |

Each `methods[method_id]` is exactly:

```json
{
  "label": "Frozen, human-readable method label",
  "model": "akt",
  "ukt_training_recipe": null,
  "mockt_training_recipe": null,
  "model_config_sha256": "<canonical resolved model_config SHA256>",
  "training_recipe_sha256": "<canonical training recipe SHA256>"
}
```

UKT recipes must match its explicit CL/augmentation switches. MoC's two explicit
training adaptations retain their complete recipe strings; absent recipe means
the original training path. Labels do not override these checks or allow an
identical method to be duplicated. This describes a method, not a novelty claim.

`training_recipe_sha256` hashes resolved config after excluding only
`run_id`, `fold`, `seed`, `device`, `dataset_manifest` (path), `_batch_approved`,
`batch_id`, `approved_batch_sha256`, `test_release_file` and `max_train_steps`.
The raw configuration fingerprint and dataset manifest are independently bound
to the original release. Epochs, patience, learning rate, weight decay, batch
size, wall limit, gradient clipping and validation settings remain in the method
fingerprint. The entire 15-fit method must match it.

The step cap is separately frozen per run: a legitimate full-epoch cap can be
`ceil(training_rows_for_fold / batch_size) × epochs`, which differs between
folds. It is not silently dropped, normalized or claimed to imply equal updates.
Each cohort record is:

```json
{
  "run_id": "<original run ID>",
  "run_dir": "/absolute/canonical/path/to/that/run",
  "dataset": "<prepared dataset name>",
  "method_id": "<declared method ID>",
  "fold": 0,
  "seed": 42,
  "max_train_steps": 4750
}
```

Each comparison declares `comparison_id`, `dataset`, `candidate_method`,
`reference_method`, and exactly 15 `pairs`. Each pair has `fold`, `seed`,
`candidate_run_id`, and `reference_run_id`; the IDs must match that exact method
and grid cell. Duplicate or reversed comparisons are rejected. Every dataset
comparison belongs to the frozen primary AUC family. The normal three-dataset,
one-comparison-per-dataset plan therefore has family size 3.

```json
{
  "n_resamples": 2000,
  "seed": 42,
  "seed_scheme": "reset_42_per_dataset",
  "confidence": 0.95,
  "family_size": 3,
  "max_attempts": 20000,
  "single_class_draws": "bounded_redraw_condition_on_two_classes"
}
```

`family_size` must equal the complete declared comparison count, even if an
entire dataset later has undefined AUC. Version 1 does not provide a different
resampling rule or fewer draws. Changing these defaults requires an explicitly
versioned analysis-method change and documentation before test access.

## Receipt and event checks

The reader verifies original config/metadata and manifest JSON hashes, the
release/plan/compatibility-review bytes, and the complete training and evaluation
identities. These sources remain separate. It never calls the development
`_read_run`, final-test `_preflight`, or a weight/data loader.

The manifest's canonical `data_dir` determines the registry and dataset lock.
The reader requires the exact completed terminal attempt, reconstructs the
initial serialized byte receipt (LF or Windows CRLF), and verifies the lock's
complete release/candidate/plan identity. An active/failed attempt, copied
registry, redirected artifact, partial output or changed source receipt is
rejected. Epoch 0 remains legitimate even after later positive optimizer steps;
its completion flag is false. A selected positive but partial epoch retains
`after_partial_training_epoch` rather than being silently relabeled complete.

Result, sidecar and CSV byte hashes are linked to the terminal attempt. Every
CSV independently retains the frozen ordered target/context hash sequence.
Within a dataset, all fits must have exactly the same event order and
`(uid,event_id,question_id,label,kc_count,target_hash,context_hash)`. Canonical
event ordinals, fold `-1`, AS15's empty question ID, full selected/excluded event
accounting, student counts, fixed KC/history semantics and source byte receipts
are checked. Current labels are compared explicitly, not inferred from hashes.
Saved scalar metrics are recomputed with strict same-CSV numerical tolerance.
All captured receipts are reread at the end to detect file changes during
analysis; no completed report is returned on that failure.

These are unsigned consistency receipts. The reader does not independently
prove training, weight contents, compatibility-review truth, actual past access
ordering, or the raw histories behind their hashes. It does not rewrite any
original artifact. Reports contain aggregates and hashes, not student/event IDs.

## Statistics and resource scope

The existing pure `paired_auc_bootstrap` is executed from its captured,
hash-verified source bytes. It sorts scores/ties once per fitted run. Each draw
resamples students and shares multiplicities across both methods and all 15
matched fits. It calculates each fitted model's AUC first, then averages the
15 paired differences equally. It never pools duplicated student events as
independent observations or averages probabilities into an ensemble.

Each dataset is analyzed separately. Calls reset seed 42 as the existing final
protocol specifies; comparisons on the same dataset therefore use the same
draws. Datasets are not pooled or paired; reset RNG streams are not claimed to
be mathematically independent. Bonferroni does not require independence.
The original bounded redraw rule records accepted draws, attempts and rejected
single-class draws, and conditions intervals on both classes. Exhausting the
bound fails rather than silently reducing the requested 2000 draws. An entirely
single-class dataset retains its full cohort and all auxiliary metrics, reports
AUC/interval unavailable, and cannot receive a positive AUC-interval conclusion.

For every metric the report gives each run, each matched pair, and descriptive
mean/sample standard deviation (`ddof=1`) of candidate, reference and difference.
Only AUC has bootstrap intervals. NLL/Brier/accuracy/ECE have no unregistered
intervals or p-values. Per-seed AUC output retains the existing descriptive
interpretation. The intervals condition on the fixed trained models and
test-student sampling; they do not quantify retraining uncertainty. Approximate
Bonferroni percentile coverage is not an exact finite-sample guarantee.

Processing is sequential by dataset and uses one weight vector per draw, not
a `2000 × events` matrix. The report gives score-array bytes (`events × fits × 8`),
accepted-draw event/fit visits (`events × 30 × 2000 × comparisons`) and the
20,000-attempt upper bound. These are cost estimates, not elapsed-time or total
RAM guarantees: parsed event identities, rank caches, statistical-library
validation copies and Python overhead also consume memory. No development-only
small event-visit cap is reused, and no automatic sampling, pair reduction or
probability aggregation reduces this work. An interrupted pure statistics job
does not rerun a model or reset a consumed final-test attempt.

## Acceptance

The concentrated artificial-file suite completed on 2026-10-15: **59 passed,
0 failed, 0 skipped**. Pytest reported **15.80 seconds**; `/usr/bin/time` reported
**16.40 seconds wall time** (11.69 user, 2.41 system). Exact PowerShell command:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/doc/knowledge_trace_graduation_project/knowledge-tracing-finaltest && mkdir -p .runtime && export PYTHONPATH=src CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && /usr/bin/time -p /home/ml/.venvs/knowledge-tracing/bin/python -m pytest tests/test_shared_final_prediction_analysis.py -q > .runtime/shared-final-analysis.stdout.log 2> .runtime/shared-final-analysis.stderr.log'
```

The suite uses only temporary synthetic CSV/JSON receipts. It includes three
datasets with complete 15-pair grids and actual 2000-draw bootstrap calls;
different scores across fits make mean per-fit AUC differ from ensemble AUC.
It also checks identical same-dataset RNG draws, mixed two-class/single-class
datasets without family reduction, epoch-0 and partial-epoch records, per-fold
step caps, recipes, receipt/identity/label/order tampering, file races, and
release of the previous dataset's score arrays. Synthetic raw-data, mapping and
weight sentinels are protected from reads. There were **0 model constructions,
0 forwards, 0 gradient calls, 0 optimizer steps**, and no real data, weights,
test labels or GPU use. No real final release or final-test statistics have
been produced by this work.

Independent static review by `frontier_audit` found no blocking defect in the
tested reader/test versions. This review and these fixtures establish the
documented saved-file contract; they do not prove future real releases or
independently verify training or historical test-access order.

| Artifact | SHA256 |
| --- | --- |
| `scripts/analyze_shared_final_predictions.py` | `88fa56b5d41b0e6c6ad030990fa648ab66479eb44a144a7c084cd6a4477adad7` |
| `tests/test_shared_final_prediction_analysis.py` | `afef36b58fa2003fa14acd54227bc65f8d2611de925769cfb28a6f809826c725` |
| Unchanged `scripts/analyze_shared_predictions.py` | `7ffd86b7910dfb3a970a9e4a942be59854d737cb21caeba916f1c3bfa3ae6ad7` |
| Unchanged `src/ktlab/statistics.py` | `7a070047587d1a4892fbfd9e563554dbb7b9456590a87fcf475f676cc9ec00e1` |
| Unchanged evaluation `source_fingerprint()` | `3a4b278994b1d94ef0776fe7415b3d0cde1e201f2e1d468d152661919acf34cb` |
| `.runtime/shared-final-analysis.stdout.log` | `d0dd1bac54f13e57930eeda29a0581683ffa64316c81854a890a8362f8e18dcd` |
| `.runtime/shared-final-analysis.stderr.log` | `ce527759a47794dc2dec1e5ca7a29e55c62a9df5113b533b85e4396872131c11` |

The private raw logs remain in this worktree's ignored `.runtime` directory;
stderr contains only the three timing lines. `git diff --check` passed, and
`src`, `third_party`, and `pyproject.toml` have no changes for this delivery.
