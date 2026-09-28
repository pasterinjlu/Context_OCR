# SceneFaith


SceneFaith evaluates whether vision-language models transcribe visible scene text faithfully when surrounding context suggests a different word.

## Contents

- `data/`: scene-text and fixed-patch annotations.
- `examples/`: sample images and annotations.
- `scripts/`: API evaluation, L/C/O scoring, and image perturbations.
- `results/`: archived predictions, summary results, and lexical scores.

Image paths are relative to each annotation file. Extract `SceneFaith-scenefaith-images.zip` and `SceneFaith-fixed_patch-images.zip` into the repository root to use the full datasets.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python scripts/verify.py
python scripts/evaluate.py --annotations examples/annotations.json --model YOUR_MODEL --dry-run
```

For evaluation, configure an OpenAI-compatible endpoint:

```bash
export CONTEXT_OCR_BASE_URL="https://your-provider.example/v1"
export CONTEXT_OCR_API_KEY="YOUR_API_KEY"
python scripts/evaluate.py --model YOUR_MODEL --output outputs/predictions.jsonl
python scripts/score.py --predictions outputs/predictions.jsonl
```

Score the archived predictions without API access:

```bash
python scripts/score.py --predictions results/clear_predictions.jsonl
```

Create an image variant (`gray`, `mild`, `moderate`, or `strong`):

```bash
python scripts/prepare_variant.py --variant moderate --output-dir outputs/moderate
```

## Labels and metrics

`observed_text` is the literal transcription; `canonical_text` is the conventional spelling. Predictions are classified as **Literal**, **Canonical**, or **Other** using the released parser. Literal accuracy and rewriting rate are the percentages of Literal and Canonical outputs, respectively.

<!-- ## Citation

```bibtex
@misc{cheng2026scenefaith,
  title={When VLMs Trust Context: Evaluating Scene Text Recognition under Misleading Context},
  author={Cheng, Yuxing and Wu, Yuan and Chang, Yi},
  year={2026}
}
``` -->
