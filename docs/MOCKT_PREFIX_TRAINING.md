# MoC target-prefix branch training: implementation and acceptance scope

Started 2026-10-15 in `research/mockt-prefix-training`, based on `1bc4111`.
Only this isolated worktree is edited. Vendor bytes and shared validation
mathematics remain unchanged. This is a training adaptation/performance
experiment, not a new model mechanism or an exact reproduction of the paper.

## Fixed design

The original model chooses one convolution/beta branch for every position of a
row from scalar `ground_seq`. Ordinary training supplies the full row's valid
length; shared evaluation supplies the current target's permitted prefix
length. The proposed training recipe performs three complete original-model
forwards, with scalar representatives 1, 51 and 151. At a selected target with
expanded-KC prefix length n, select the first output for n <= 50, the second
for 50 < n <= 150, and the third for n > 150. Select probabilities, not logits:
the original forward already applies sigmoid. Preserve the existing masked
BCE and add the original full-batch item L2 regularizer exactly once.

Two explicit, top-level configuration values are implemented:

- `mockt_training_recipe: "mockt-target-prefix-branches-v1"`: the three fixed
  representative lengths above.
- `mockt_training_recipe: "mockt-three-pass-ground-seq-control-v1"`: use the
  original per-row `ground_seq` for all three complete forwards, selecting
  positions with exactly the same masks as the prefix recipe. This controls
  the multiple-forward/dropout structure. With dropout disabled it must equal
  the original single forward.

An absent recipe retains the original one-forward path, including its saved
configuration identity. Unknown recipes, a recipe on another model/protocol,
and unsupported model switches are rejected before data/model construction.
Initial support is explicitly bounded to shared-dev MoC with
`qid_conv_ker_noexp`, kernel size 4, sequence length 200 and thresholds 50/150.
Recipe identity is part of resolved config, checkpoint config fingerprint and
saved-result comparison labels; arbitrary display names cannot create variants.

The three paths share the same original parameter objects. There is no detach,
prediction fusion or new parameter. All three graphs contribute to one masked
loss and one optimizer update. No optimizer is constructed by the helper.
The existing engine retains its one gradient clipping/optimizer step policy.
Peak training memory can increase because all three graphs are retained; this
must be measured in a later concrete experiment rather than assumed harmless.

## Equivalence and remaining limitations

For the active original architecture, fixed parameters, identical window start
and disabled dropout, the selected probability equals an original forward on
that target's cropped prefix with its actual prefix length, up to declared
floating-point tolerance. Selection is after three **complete** forwards, not
after only the convolution: each target's whole history keeps the same branch
as the original target-prefix call. Current/future answers must not affect the
current probability; permitted history should remain influential.

All active attention masks are causal, LayerNorm operates over features, and
Kerple uses relative positions. Its parameter lower-bound clamp must be in the
same state for oracle comparisons. Cache tensors must be created normally or
with `no_grad`, never `inference_mode`, to preserve later autograd use.
Dropout-enabled training is not expected to be bitwise identical to separately
sampled per-target forwards. The second recipe is the explicit structural
control. The L2 oracle adds one full-batch regularizer; it must not average or
sum repeated cropped-prefix regularizers, which define another objective.

“Prefix” here still means the original training block's expanded KC prefix.
Repeated-KC current-event siblings and answers can remain in training history;
block boundaries and event weighting also remain different from the shared
complete-event KC200 evaluation. This change aligns the length branch only.
It does not claim full train/evaluation protocol equivalence or solve every
causality concern in repeated-KC training. AS15 has no separate question ID;
it still uses concept observations and its original zero item regularizer.

## S006 acceptance budget and result

Code/tests were written before root recorded their actual hashes and the
existing standing authorization. The one-shot S006 runner then executed the
bounded acceptance below. No actual B006 run, real data/weights, GPU or
final-test access was included.

- CPU, single thread; seed42; d16, 1 block, 2 heads; two real original small
  MoC instances total, one `num_q=0` and one nonzero.
- At most 64 original-model forwards and at most 4 `torch.autograd.grad`
  computations; zero backward calls, optimizer constructions and steps.
  The written acceptance plan uses 58 original forwards (29 per model), four
  gradient calls and a batch-size-one tail check with the same parameters.
  Counters reject calls before exceeding limits and require two models/four
  gradients at teardown; the explicit execution flag must not be omitted.
- Use one model's shared parameters for fast/oracle comparisons, and reuse
  retained outputs/graphs where valid. Test the CPU originals with a narrowly
  scoped construction-only binding of the vendor's `.cuda()` tensor placement
  to CPU; no mathematical operation or vendor file is replaced. Production
  MoC's existing device guard remains unchanged.
- Synthetic tensors only: prefix boundaries 49/50/51/149/150/151, right padding,
  current/future response isolation, historical-response positive control,
  per-prefix probability/BCE/parameter-gradient oracle, and graph connectivity.
- Pure fake-model checks cover routing, one-time L2, dispatch/default behavior
  and pre-construction config rejection. Saved-prediction comparison tests use
  artificial file receipts only. They do not import or execute real models.
- Do not rerun the old full model/training suite. WSL execution uses
  `bash -lc`, `PYTHONPATH=src`, hidden accelerators, and
  `/home/ml/.venvs/knowledge-tracing/bin/python` after registration.

### Recorded acceptance, 2026-10-15 (Hong Kong)

S006 canonical manifest SHA256:
`f04a5126293b635e00a53ab3d25faba04797a8efc39ad219dcbeb12393a70ea7`.
Implementation fingerprint:
`0e9eb36f2ef023f51400ee74ff7bc3ae699aac0b52fbe1568bc73389b4cf07f0`.
The runner verified its bound files before and after execution and reported
`passed`, exit 0, no timeout, and unchanged source/test files. One execution;
no retry. Pytest: **196 passed, zero skipped, 14 warnings, 5.50 seconds**;
runner elapsed 7.70 seconds. The warnings are the pinned model's imported
Matplotlib/pyparsing deprecations; stderr rejection messages came from planned
artificial-file negative tests.

Actual counters: two original CPU models, **58 forwards, four autograd.grad**,
zero backward calls, optimizer constructions and steps. Both model modes had
zero prefix BCE difference and a maximum parameter-gradient absolute error of
`1.4901161193847656e-08`. The item model's full-batch L2 was nonzero
(`0.039639998227357864`), including original padding-ID behavior. History-response
perturbations changed the current probability by maxima `0.00011900067329406738`
(no item ID) and `0.00012475252151489258` (item ID); current/future response,
cropped-prefix, control, padding and singleton checks passed within their fixed
tolerances. Fake-model and saved-file checks are included in the 196 tests.

This establishes bounded implementation/oracle consistency, not a real-data
accuracy gain, GPU resource fit, originality or paper reproduction.
The later [B006 preparation](../reports/B006_MOCKT_PREFIX_TRAINING_PLAN_2026-10-15.md)
freezes six real-data configurations and the resource limits after a file-only
training-input audit. B006 has not started; its predecessor batches, immutable
receipts, idle GPU and standing-authority execution record are checked at launch.
