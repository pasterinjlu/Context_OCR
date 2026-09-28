"""Verify release files, labels, archived predictions, and optional image assets."""
import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from score import score
from common import classify

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', action='store_true')
    args = p.parse_args()
    inventory = json.loads((ROOT/'manifest.json').read_text())
    for name, item in inventory['files'].items():
        path = ROOT/name
        require(path.is_file() and not path.is_symlink(), f'Missing file: {name}')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'Changed file: {name}')
    datasets = {}
    for name in ['data/scenefaith', 'data/fixed_patch', 'examples']:
        records = json.loads((ROOT/name/'annotations.json').read_text())
        require(len(records)==len({r['id'] for r in records}), f'Duplicate IDs: {name}')
        for r in records:
            path = Path(r['image_path'])
            require(not path.is_absolute() and '..' not in path.parts, 'Invalid image path')
            require(r['observed_text'] != r['canonical_text'], 'Label collision')
        datasets[name] = records
    groups = defaultdict(list)
    for r in datasets['data/fixed_patch']:
        groups[r['adversarial_pair_id']].append(r)
    for pair, records in groups.items():
        require({r['condition'] for r in records} == {'aligned_full','identifier_full','masked_A','masked_B','shuffled_semantic'}, f'Conditions: {pair}')
        require(len({r['patch_pixel_sha256'] for r in records}) == 1, f'Patch mismatch: {pair}')
    predictions = [json.loads(line) for line in (ROOT/'results/clear_predictions.jsonl').read_text().splitlines()]
    labels = {r['id']:r for r in datasets['data/scenefaith']}
    for row in predictions:
        prediction, category = classify(row['raw_response'], labels[row['id']])
        require((prediction, category)==(row['prediction'], row['category']), 'Prediction label mismatch')
    computed = {r['model']:r for r in score(datasets['data/scenefaith'], predictions)}
    expected = json.loads((ROOT/'results/clear_summary.json').read_text())
    require(set(computed)=={r['model'] for r in expected}, 'Model set mismatch')
    for row in expected:
        actual = computed[row['model']]
        require(actual['counts']==row['counts'] and actual['n']==row['n'], 'Metric mismatch')
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text())
    checked = 0
    if args.images:
        for name, item in json.loads((ROOT/'data/assets.json').read_text()).items():
            path = ROOT/name
            require(path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'], f'Invalid image asset: {name}')
            checked += 1
    print(json.dumps({'passed':True, 'files_verified':len(inventory['files']),
                      'predictions_verified':len(predictions), 'image_assets_verified':checked}))


if __name__ == '__main__':
    main()
