# Data

`scenefaith/annotations.json` contains scene-text samples. `fixed_patch/annotations.json` groups matched conditions by `adversarial_pair_id`.

| Field | Meaning |
| --- | --- |
| `id` | Sample identifier |
| `image_path` | Image path relative to the annotation file |
| `observed_text` | Literal transcription target |
| `canonical_text` | Conventional spelling for rewrite analysis |
| `red_bbox_xywh` | Target rectangle in pixels: x, y, width, height |
| `patch_bbox_xyxy` | Fixed target patch: left, top, right, bottom |
| `condition` | Fixed-patch experimental condition |
| `image_sha256` | Image file checksum |

Extract `SceneFaith-scenefaith-images.zip` and `SceneFaith-fixed_patch-images.zip` at the repository root, then run `python scripts/verify.py --images`.
