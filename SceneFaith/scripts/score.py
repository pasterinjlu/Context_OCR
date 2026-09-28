"""Score JSONL predictions against literal and canonical annotations."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from common import classify


def score(annotations, predictions):
    labels = {r['id']: r for r in annotations}
    if len(labels) != len(annotations):
        raise ValueError('Duplicate annotation IDs')
    groups = defaultdict(lambda: {'L': 0, 'C': 0, 'O': 0, 'errors': 0})
    seen = set()
    for row in predictions:
        key = (row.get('model', 'model'), row['id'])
        if key in seen:
            raise ValueError(f'Duplicate prediction: {key}')
        seen.add(key)
        if row['id'] not in labels:
            raise ValueError(f'Unknown ID: {row["id"]}')
        counts = groups[key[0]]
        if row.get('error'):
            counts['errors'] += 1
            continue
        raw = row['raw_response']
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('Empty response without an error status')
        _, category = classify(raw, labels[row['id']])
        counts[category] += 1
    result = []
    for model, counts in sorted(groups.items()):
        n = sum(counts[k] for k in 'LCO')
        result.append({'model': model, 'n': n, 'counts': {k: counts[k] for k in 'LCO'},
                       'errors': counts['errors'], 'missing': len(labels) - n - counts['errors'],
                       'literal_accuracy_percent': 100 * counts['L'] / n if n else None,
                       'rewriting_rate_percent': 100 * counts['C'] / n if n else None,
                       'other_rate_percent': 100 * counts['O'] / n if n else None})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=Path('data/scenefaith/annotations.json'))
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    text = json.dumps(score(json.loads(args.annotations.read_text()), rows), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
