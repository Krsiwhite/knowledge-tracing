"""Artificial metadata and source files only: no data, torch or model execution."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_ukt_batch', ROOT / 'scripts/prepare_ukt_batch.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'artificial.py').write_text('VALUE = 2\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "artificial"\n')
    monkeypatch.setattr(PREPARE, 'git_revision', lambda root: 'artificial-git-revision')
    baseline = {'batch_id': 'B003-shared-convergence-baselines', 'runs': [],
                'limits': {'paid_services': False, 'automatic_retries': False},
                'selection': {'strict_best_auc': True, 'initial_validation': True, 'patience': 3}}
    inputs = {'test_accessed': False, 'model_executed': False, 'datasets': []}
    for dataset, rows, batches, wall in [('assist2009', 3016, 95, 14400),
                                        ('assist2015', 12344, 386, 14400),
                                        ('nips_task34', 6043, 189, 28800)]:
        baseline['runs'].append({'run_id': f'B003-akt-{dataset}-f0-s42',
                                 'dataset_manifest_sha256': dataset, 'train_sequence_rows': rows,
                                 'config': {'model': 'akt', 'dataset': dataset, 'fold': 0, 'seed': 42,
                                            'protocol': 'ktlab-v2-shared-event-context-kc200-dev',
                                            'max_wall_seconds': wall}})
        inputs['datasets'].append({'dataset': dataset, 'held_out_fold': 0, 'train_rows': rows,
                                   'dataset_manifest_sha256': dataset, 'batches_per_epoch': {'32': batches},
                                   'nonempty_left_prefix_binary_response_domain_passed': True,
                                   'author_supervision_mask_equals_project_on_all_training_rows': True})
    simple = copy.deepcopy(baseline['runs'][0])
    simple['run_id'] = 'B003-simplekt-assist2009-f0-s42'
    simple['config']['model'] = 'simplekt'
    baseline['runs'].append(simple)
    recovery = {'batch_id': 'B003-R1-shared-convergence-baselines', 'test_access': False,
                'runs': [{'run_id': 'B003-R1-mockt-assist2009-f0-s42'}],
                'recovery': {'original_batch_id': baseline['batch_id'],
                             'original_batch_sha256': PREPARE.canonical_hash(baseline),
                             'preserved_completed_runs': ['B003-akt-assist2009-f0-s42', simple['run_id']]}}
    old = PREPARE.prepare_batch(tmp_path, baseline, inputs, recovery)
    old['source_sha256'] = '1' * 64
    return tmp_path, baseline, inputs, recovery, old


def test_refreeze_preserves_configs_budgets_and_binds_actual_source(prepared):
    root, baseline, inputs, recovery, old = prepared
    before = copy.deepcopy((baseline, inputs, recovery, old))
    revised = PREPARE.prepare_batch(root, baseline, inputs, recovery, superseded=old)
    assert (baseline, inputs, recovery, old) == before
    assert revised['supersedes']['batch_sha256'] == PREPARE.canonical_hash(old)
    assert revised['supersedes']['source_sha256'] == old['source_sha256']
    assert revised['source_sha256'] == PREPARE.source_fingerprint(root)
    assert revised['revision'] == 1 and revised['status'] == 'prepared'
    for key in ('runs', 'limits', 'selection'):
        assert revised[key] == old[key]
    assert len(revised['runs']) == 6
    assert {r['config']['epochs'] for r in revised['runs']} == {50}
    assert {r['config']['patience'] for r in revised['runs']} == {3}
    assert revised['limits']['total_optimizer_steps_upper_bound'] == 67000
    assert revised['limits']['sum_run_wall_seconds'] == 115200
    assert revised['execution_predecessor']['batch_sha256'] == PREPARE.canonical_hash(recovery)
    assert revised['execution_predecessor']['carry_forward_run_ids'] == recovery['recovery']['preserved_completed_runs']
    assert any('GPU is idle' in x for x in revised['execution_dependencies'])


@pytest.mark.parametrize('change', ['status', 'config', 'limits', 'artifacts', 'unchanged_source'])
def test_refreeze_rejects_executed_or_changed_plan(prepared, change):
    root, baseline, inputs, recovery, old = prepared
    if change == 'status':
        old['status'] = 'running'
    elif change == 'config':
        old['runs'][0]['config']['model_config']['d_model'] = 256
    elif change == 'limits':
        old['limits']['total_optimizer_steps_upper_bound'] -= 1
    elif change == 'artifacts':
        (root / 'runs' / old['runs'][0]['run_id']).mkdir(parents=True)
    else:
        old['source_sha256'] = PREPARE.source_fingerprint(root)
    with pytest.raises(ValueError):
        PREPARE.prepare_batch(root, baseline, inputs, recovery, superseded=old)


@pytest.mark.parametrize('change', ['original_sha', 'carry_forward', 'already_rerun'])
def test_recovery_identity_must_preserve_original_results(prepared, change):
    root, baseline, inputs, recovery, old = prepared
    if change == 'original_sha':
        recovery['recovery']['original_batch_sha256'] = 'wrong'
    elif change == 'carry_forward':
        recovery['recovery']['preserved_completed_runs'].pop()
    else:
        recovery['runs'].append({'run_id': recovery['recovery']['preserved_completed_runs'][0]})
    with pytest.raises(ValueError, match='recovery'):
        PREPARE.prepare_batch(root, baseline, inputs, recovery, superseded=old)


@pytest.mark.parametrize('explicit_sha', [None, 'wrong'])
def test_cli_cannot_silently_replace_existing_output(prepared, monkeypatch, explicit_sha):
    root, _, _, _, old = prepared
    output = root / 'batch.json'
    PREPARE.write_json(output, old)
    before = output.read_bytes()
    argv = ['prepare', '--baseline-batch', 'unused.json', '--recovery-batch', 'unused.json', '--output', str(output)]
    if explicit_sha:
        argv += ['--supersedes', explicit_sha]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises((ValueError, FileExistsError)):
        PREPARE.main()
    assert output.read_bytes() == before
