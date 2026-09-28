# Results

- `clear_predictions.jsonl`: archived clear-image model responses and L/C/O labels.
- `clear_summary.json`: clear-image metrics by model.
- `run_settings.json`: archived prompts and generation settings.
- `blur.json`: metrics across blur levels.
- `prompts.json`: prompt comparison metrics.
- `fixed_patch.json`: fixed-patch comparison results.
- `lexical_scores.jsonl`: external DistilGPT2 lexical-prior scores.

Recompute clear-image metrics with `python scripts/score.py --predictions results/clear_predictions.jsonl`.
