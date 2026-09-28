"""Evaluate scene text through an OpenAI-compatible chat-completions endpoint."""
import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlsplit
from common import PROMPTS, classify, prompt_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=Path('data/scenefaith/annotations.json'))
    p.add_argument('--model', required=True)
    p.add_argument('--prompt', choices=sorted(PROMPTS), default='neutral')
    p.add_argument('--output', type=Path, default=Path('outputs/predictions.jsonl'))
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.timeout < 1 or (args.limit is not None and args.limit < 1):
        p.error('Timeout and limit must be positive')
    rows = json.loads(args.annotations.read_text())
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate annotation IDs')
    if args.limit:
        rows = rows[:args.limit]
    prompt = prompt_for(args.model, args.prompt)
    images = []
    root = args.annotations.resolve().parent
    for row in rows:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path must stay inside the dataset folder')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['image_sha256']:
            raise ValueError(f'Image hash mismatch: {row["id"]}')
        classify(row['observed_text'], row)
        images.append(path)
    if args.dry_run:
        print(json.dumps({'images_verified': len(images), 'model': args.model, 'prompt': prompt}))
        return
    import requests
    key = os.environ.get('CONTEXT_OCR_API_KEY', '')
    base = os.environ.get('CONTEXT_OCR_BASE_URL', '').rstrip('/')
    parsed = urlsplit(base)
    if not key or parsed.scheme not in ('http', 'https') or not parsed.netloc:
        p.error('Set CONTEXT_OCR_API_KEY and CONTEXT_OCR_BASE_URL')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        p.error('Use a base URL without credentials, query, or fragment')
    endpoint = base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = {'model': args.model, 'prompt': prompt, 'temperature': 0.0, 'max_tokens': 2048,
                'annotations_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                'selected_ids': [r['id'] for r in rows]}
    # Exclusive creation prevents accidentally overwriting a prior evaluation.
    with args.output.open('x', encoding='utf-8') as stream, requests.Session() as session:
        args.output.with_suffix('.config.json').write_text(json.dumps(settings, indent=2) + '\n')
        session.trust_env = False
        for row, path in zip(rows, images):
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            url = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()
            payload = {'model': args.model, 'temperature': 0.0, 'max_tokens': 2048,
                       'messages': [{'role': 'user', 'content': [
                           {'type': 'text', 'text': prompt},
                           {'type': 'image_url', 'image_url': {'url': url}}]}]}
            result = {'id': row['id'], 'model': args.model}
            fatal = False
            try:
                response = session.post(endpoint, json=payload, headers={'Authorization': 'Bearer ' + key},
                                        timeout=(30, args.timeout), allow_redirects=False)
                if response.status_code != 200:
                    fatal = response.status_code in (400, 401, 402, 403, 404)
                    raise ValueError(f'HTTP {response.status_code}')
                body = response.json()
                choice = body['choices'][0]
                raw = choice['message']['content']
                if choice.get('finish_reason') != 'stop' or not isinstance(raw, str) or not raw.strip():
                    raise ValueError('Incomplete or empty response')
                raw = raw.replace(key, '[REDACTED]')
                prediction, category = classify(raw, row)
                result.update(raw_response=raw, prediction=prediction, category=category,
                              reported_model=body.get('model'), error='')
            except Exception as exc:
                result.update(raw_response='', error=type(exc).__name__)
                if isinstance(exc, ValueError) and str(exc).startswith('HTTP '):
                    result['error'] = str(exc)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            print(row['id'], result.get('category', result.get('error')), flush=True)
            if fatal:
                raise SystemExit('Evaluation stopped after an endpoint or authorization error')


if __name__ == '__main__':
    main()
