"""Mix pre-generated class and image attributes while preserving provenance."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def mix_attributes(template: dict, sources: dict, root: Path, seed: int, expected_count: int = 50) -> dict:
    if expected_count <= 0 or set(template) != set(sources):
        raise ValueError("Expected a positive count and exactly the support template's labels")
    result = {}
    for label in sorted(template):
        support_images = template[label]["support_images"]
        support = {str((root / image).resolve()) for image in support_images}
        if not support:
            raise ValueError(f"{label}: no support images")
        class_attributes = sources[label]["class_attributes"]
        image_attributes = sources[label]["image_attributes"]
        if not isinstance(class_attributes, list) or not isinstance(image_attributes, list) or not class_attributes or not image_attributes:
            raise ValueError(f"{label}: both attribute branches must be nonempty lists")
        if len(class_attributes) + len(image_attributes) != expected_count:
            raise ValueError(f"{label}: expected {expected_count} raw descriptions")
        if any(not isinstance(text, str) or not text.strip() for text in class_attributes):
            raise ValueError(f"{label}: class descriptions must be nonempty strings")
        mixed = [{"source": "class", "text": text} for text in class_attributes]
        for entry in image_attributes:
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
                raise ValueError(f"{label}: image description must contain nonempty text")
            image_ids = entry.get("image_ids")
            if not isinstance(image_ids, list) or not image_ids or not all(isinstance(path, str) for path in image_ids):
                raise ValueError(f"{label}: image description must contain image IDs")
            if not {str((root / path).resolve()) for path in image_ids} <= support:
                raise ValueError(f"{label}: image description references a non-support image")
            mixed.append({"source": "image", "text": entry["text"], "image_ids": image_ids})
        random.Random(f"{seed}:{label}").shuffle(mixed)
        result[label] = {"support_images": support_images, "class_attributes": class_attributes, "image_attributes": image_attributes, "mixed_attributes": mixed, "mix_seed": seed}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--support-template", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed for deterministic mixing, not the training seed")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    template = json.loads(args.support_template.read_text(encoding="utf-8"))
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    mixed = mix_attributes(template, sources, args.data_root.resolve(), args.seed, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(mixed, stream, ensure_ascii=False, indent=2)
    print(f"Wrote mixed attributes to {args.output}")


if __name__ == "__main__":
    main()
