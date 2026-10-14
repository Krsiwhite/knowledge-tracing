"""Prepare B004 under standing authorization; never launch it or make an approval."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ktlab.provenance import canonical_hash, git_revision, project_root, read_json, source_fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite a prepared/frozen batch; inspect its status first')
    root = project_root()
    previous = read_json(args.baseline_batch)
    inputs = read_json(root / 'reports/UKT_TRAINING_INPUT_AUDIT_2026-10-15.json')
    audits = {item['dataset']: item for item in inputs['datasets']}
    if inputs['test_accessed'] or inputs['model_executed']:
        raise ValueError('Unexpected scope for the training-input audit')
    model_config = {'d_model': 128, 'n_blocks': 2, 'd_ff': 256, 'num_attn_heads': 8,
                    'final_fc_dim': 256, 'final_fc_dim2': 256, 'dropout': .1,
                    'seq_len': 200, 'emb_type': 'stoc_qid', 'atten_type': 'w2',
                    'separate_qa': False, 'use_uncertainty_aug': True, 'cl_weight': .02}
    runs = []
    for entry in previous['runs']:
        if entry['config']['model'] != 'akt':
            continue
        dataset = entry['config']['dataset']
        audit = audits[dataset]
        if (entry['config'].get('fold') != 0 or entry['config'].get('seed') != 42
                or audit.get('held_out_fold') != entry['config'].get('fold')
                or entry['config'].get('protocol') != 'ktlab-v2-shared-event-context-kc200-dev'):
            raise ValueError('B004 requires the audited shared-protocol fold0/seed42 baseline')
        if (audit['dataset_manifest_sha256'] != entry['dataset_manifest_sha256']
                or audit['train_rows'] != entry['train_sequence_rows']
                or not audit['nonempty_left_prefix_binary_response_domain_passed']
                or not audit['author_supervision_mask_equals_project_on_all_training_rows']):
            raise ValueError('Training input audit does not match the declared data/augmentation contract')
        for label, use_cl, recipe in (
            ('authoraug', True, 'ukt-author-first-two-from-last-86ad9d8-v1'),
            ('no-cl', False, 'ukt-no-cl'),
        ):
            run = copy.deepcopy(entry)
            run['run_id'] = f'B004-ukt-{label}-{dataset}-f0-s42'
            config = run['config']
            config.update(model='ukt', model_config={**model_config, 'use_CL': use_cl},
                          ukt_training_recipe=recipe, learning_rate=1e-4, weight_decay=1e-5,
                          epochs=50, patience=3, early_stop_min_delta=0.,
                          batch_size=32, shuffle=True, grad_clip=1.,
                          purpose='ukt_objective_comparison_shared_validation_not_paper_best_or_sota')
            config['max_train_steps'] = audit['batches_per_epoch']['32'] * 50
            runs.append(run)
    if len(runs) != 6 or {r['config']['dataset'] for r in runs} != set(audits):
        raise ValueError('Expected exactly two UKT objectives on each of the three datasets')
    limits = copy.deepcopy(previous['limits'])
    limits.update(epochs_per_run=50, total_optimizer_steps_upper_bound=sum(r['config']['max_train_steps'] for r in runs),
                  sum_run_wall_seconds=sum(r['config']['max_wall_seconds'] for r in runs))
    result = {'schema_version': 1, 'batch_id': 'B004-ukt-shared-objectives', 'status': 'prepared',
              'purpose': 'Compare pinned UKT author-augmentation CL with no-CL under shared development validation',
              'source_sha256': source_fingerprint(root), 'git_commit_at_preparation': git_revision(root),
              'base_batch_sha256': canonical_hash(previous), 'input_audit_sha256': canonical_hash(inputs),
              'test_access': False, 'stop_on_noncompleted_run': True,
              'execution_dependencies': ['B003 terminal results inspected; no competing GPU job',
                                         'Bind the concrete manifest to the existing standing user authorization before launch'],
              'comparison_boundary': 'Same UKT architecture and epoch/update caps across objectives; CL costs extra compute. B003 comparisons require matching validation identities, not paper-score ranking.',
              'limits': limits, 'selection': {**previous['selection'], 'patience': 3}, 'runs': runs}
    if limits['total_optimizer_steps_upper_bound'] != 67000 or limits['sum_run_wall_seconds'] != 115200:
        raise ValueError('Unexpected B004 resource totals; inspect the plan before execution')
    write_json(args.output, result)
    print({'path': str(args.output), 'canonical_sha256': canonical_hash(result),
           'source_sha256': result['source_sha256'], 'runs': len(runs),
           'optimizer_steps_upper_bound': limits['total_optimizer_steps_upper_bound']})


if __name__ == '__main__':
    main()
