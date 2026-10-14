#!/usr/bin/env python3
"""Check UKT augmentation's input domain on training folds, without torch.

Reads only the public manifests and their development sequence CSVs. Other
folds are skipped before response parsing; no test file or model is opened.
Only aggregate counts and content hashes are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DATASETS = ('assist2009', 'assist2015', 'nips_task34')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_dataset(root, dataset, held_out_fold=0):
    directory = root / 'data/prepared' / dataset
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = manifest['data_config']
    require(manifest['dataset'] == dataset, 'Dataset identity mismatch')
    require(config['train_valid_file'] == 'train_valid_sequences.csv',
            'Only the declared development sequence filename is allowed')
    folds = set(config['folds'])
    require(held_out_fold in folds, 'Unknown held-out development fold')
    train_folds = folds - {held_out_fold}
    path = directory / config['train_valid_file']
    receipt = manifest['files'][path.name]
    source_hash = digest(path)
    require(source_hash == receipt['sha256'] and path.stat().st_size == receipt['bytes'],
            'Development sequence file differs from its frozen manifest')
    sizes, fold_rows, students = Counter(), Counter(), set()
    exclusions = targets = q_missing = 0
    with path.open(encoding='utf-8', newline='') as stream:
        for row_index, row in enumerate(csv.DictReader(stream), 2):
            fold = int(row['fold'])
            require(fold in folds, f'Unknown development fold at CSV row {row_index}')
            if fold not in train_folds:
                continue
            fields = {name: [int(value) for value in row[name].split(',')]
                      for name in ('concepts', 'responses', 'selectmasks')}
            concepts, responses, selected = (fields[name] for name in ('concepts', 'responses', 'selectmasks'))
            width = int(config['maxlen'])
            require(width == 200 and all(len(values) == width for values in fields.values()),
                    f'Expected aligned fixed T=200 at training row {row_index}')
            valid_length = sum(value != -1 for value in concepts)
            require(valid_length >= 2 and all(value != -1 for value in concepts[:valid_length])
                    and all(value == -1 for value in concepts[valid_length:]),
                    f'Augmentation requires nonempty left-aligned adjacent masks: row {row_index}')
            require(all(0 <= value < config['num_c'] for value in concepts[:valid_length]),
                    f'Concept ID outside manifest range: row {row_index}')
            require(all(value in (0, 1) for value in responses[:valid_length])
                    and all(value == -1 for value in responses[valid_length:]),
                    f'Response/padding domain mismatch: row {row_index}')
            require(all(value in (0, 1) for value in selected[:valid_length])
                    and all(value == -1 for value in selected[valid_length:]),
                    f'Selection/padding domain mismatch: row {row_index}')
            if row.get('questions') not in (None, '', 'NA'):
                questions = [int(value) for value in row['questions'].split(',')]
                require(config['num_q'] > 0 and len(questions) == width
                        and all(0 <= value < config['num_q'] for value in questions[:valid_length])
                        and all(value == -1 for value in questions[valid_length:]),
                        f'Question ID/padding domain mismatch: row {row_index}')
            else:
                require(config['num_q'] == 0, f'Missing real question IDs: row {row_index}')
                q_missing += 1
            selected_shifted = selected[1:valid_length]
            require(any(value == 1 for value in selected_shifted), f'No supervised target: row {row_index}')
            exclusions += selected_shifted.count(0)
            targets += selected_shifted.count(1)
            sizes[valid_length] += 1
            fold_rows[fold] += 1
            students.add(row['uid'])
    count = sum(fold_rows.values())
    require(count > 0, 'No eligible training rows')
    return {
        'dataset': dataset, 'held_out_fold': held_out_fold, 'training_folds': sorted(train_folds),
        'dataset_manifest_sha256': digest(manifest_path), 'training_sequence_sha256': source_hash,
        'train_rows': count, 'train_students': len(students), 'rows_by_fold': dict(sorted(fold_rows.items())),
        'sequence_width': 200, 'valid_positions_min': min(sizes), 'valid_positions_max': max(sizes),
        'supervised_shifted_targets': targets, 'zero_selection_shifted_positions': exclusions,
        'author_supervision_mask_equals_project_on_all_training_rows': exclusions == 0,
        'rows_with_question_placeholder': q_missing,
        'nonempty_left_prefix_binary_response_domain_passed': True,
        'batches_per_epoch': {str(size): (count + size - 1) // size for size in (32, 64)},
        'num_c': config['num_c'], 'num_q': config['num_q'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = {'audit': 'ukt-training-input-domain-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'held_out_fold': 0, 'scope': 'training folds only within development sequence files',
              'torch_imported': False, 'model_executed': False, 'optimizer_steps': 0, 'test_accessed': False,
              'limits': 'Checks source input eligibility, not a GPU forward, training quality, or author protocol reproduction.',
              'datasets': [audit_dataset(args.repo_root.resolve(), dataset) for dataset in DATASETS]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
