"""Create full-canvas gray-mask or target-blur variants from original images."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from transforms import blur_target, inside_bbox


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=Path('data/scenefaith/annotations.json'))
    p.add_argument('--variant', required=True, choices=['gray', 'mild', 'moderate', 'strong'])
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    rows = json.loads(args.annotations.read_text())
    root = args.annotations.resolve().parent
    # Preflight before creating outputs.
    for row in rows:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError(f'Invalid image: {row["id"]}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'images').mkdir()
    output = []
    for index, row in enumerate(rows):
        with Image.open(root / row['image_path']) as im:
            source = im.convert('RGB')
        x, y, w, h = row['red_bbox_xywh']
        if args.variant == 'gray':
            box = (max(0, x), max(0, y), min(source.width, x+w), min(source.height, y+h))
            image = Image.new('RGB', source.size, (128, 128, 128))
            image.paste(source.crop(box), box[:2])
        else:
            ratio, minimum = {'mild': (.015, .6), 'moderate': (.030, 1.2), 'strong': (.060, 2.4)}[args.variant]
            region = inside_bbox(row['red_bbox_xywh'], source.size, .15, 2)
            image, _ = blur_target(source, region, max(minimum, ratio*h))
        rel = f'images/{index:04d}.png'
        path = args.output_dir / rel
        image.save(path)
        new = {k:v for k,v in row.items() if k not in ('image_sha256', 'image_rgb_sha256', 'image_path')}
        new.update(image_path=rel, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   image_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(), variant=args.variant,
                   source_image_sha256=row['image_sha256'])
        output.append(new)
    (args.output_dir / 'annotations.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'variant': args.variant, 'images': len(output)}))


if __name__ == '__main__':
    main()
