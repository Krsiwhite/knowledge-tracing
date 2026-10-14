# Dataset provenance and preparation

`ktlab.data.prepare_dataset(name, root)` downloads and prepares data locally;
`inspect_dataset(name, root)` verifies hashes and student membership without
training a model or calculating test metrics. `root` is the project directory.
The CLI also supports `python -m ktlab.data prepare assist2009 --root .`.

The only supported initial datasets are `assist2009`, `assist2015`, and
`nips_task34` (alias `nips34`). Source URLs, fixed SHA256 digests, and vocabulary
expectations live in `configs/datasets.json`. Interrupted HTTP transfers retain
`.part` files and resume with Range requests; if the server ignores Range, the
partial download is restarted. Hash mismatches stop preparation.

| Dataset | Author source | Local treatment |
| --- | --- | --- |
| ASSISTments 2009 | [Corrected, collapsed skill-builder release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | One source row per student/problem; all supplied concept tags are retained. |
| ASSISTments 2015 | [2015 skill-builder release](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data) | Concept-only: `num_q=0`; do not manufacture question identifiers. |
| NIPS34 | [Eedi official challenge page](https://www.eedischool.com/projects/neurips-education-challenge) | Only task 3/4 training answers and necessary metadata are extracted from the official archive. The pinned pyKT preprocessing uses all level-3 concept tags (57 concepts), not the previous project's last-tag-only 53-concept transformation. |

Eedi specifies [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/):
noncommercial use, attribution, and no redistribution of transformed material.
Question images have additional restrictions. This project publishes code,
aggregate audit counts and hashes only. All downloaded archives, original
records, transformed records and split membership remain in ignored `data/`.
ASSISTments requests attribution to the dataset page and its cited system paper.

Preprocessing calls the vendored pyKT commit
`eec087e0230520e7ae7bbd4f5433d71612df62fb` directly. The initial comparison
protocol uses minimum sequence length 3, maximum sequence length 200, an 80/20
student split and five development folds; both upstream shuffles use seed 1024.
IDs are mapped by the upstream implementation before splitting; this is an
in-distribution benchmark and does not establish unseen-item generalization.
Each held-out development fold is disjoint from its training students and from
the sealed test students. The original datasets and archived custom 80.6453%
result from the older project are not imported.

The official corrected ASSIST2009 CSV contains CP1252 descriptive text while
the pinned upstream reader requires UTF8. Preparation creates a local UTF8
copy, records both hashes, and verifies that all five consumed columns
(`user_id`, `order_id`, `problem_id`, `skill_id`, `correct`) remain unchanged
ASCII values. No rows, tags, identifiers or labels are repaired or replaced.
Timestamp preprocessing runs in UTC and package versions are recorded.

The manifest is `data/prepared/<dataset>/manifest.json` and provides:

- `data_dir`, `data_config`, and `files` (filenames, sizes, SHA256);
- `source` (download URL, content hash, raw file hashes) and `pykt_commit`;
- `protocol`, `preprocessing_sha256`, `split_audit` including assignment hash,
  student counts, per-fold counts and disjointness;
- `validation_question_files`, a mapping from fold strings to
  `validation_question_fold<fold>.csv` for datasets with questions.

Validation question files use upstream `generate_question_sequences` with
`window=False`, which delegates to `expand_question`. The `selectmasks`, `qidxs`,
`rest`, `orirow`, student and fold columns are retained. Concepts belonging to
the target question cannot appear as answered history for another concept of
that same target. The first question is not scored. ASSIST2015 uses the canonical
concept sequence validation rows selected by fold. Non-window primary test
evaluation uses the corresponding upstream question file when available.

Data preparation may read test records to transform and hash them, but produces
no predictions or performance measurements. Every training batch and final test
evaluation must follow the project's explicit user alignment workflow.
