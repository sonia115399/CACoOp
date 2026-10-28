"""Export the sampled support set and an empty per-run attribute template."""
import argparse
import json
from pathlib import Path

from .run import read_manifest, resolve_support, sample_support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--support", type=Path, help="optional CSV of the exact training support images")
    p.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.data_root.resolve()
    train = read_manifest(args.train, root)
    support = resolve_support(train, args.support, root, args.shots) if args.support else sample_support(train, args.shots, args.seed)
    library = {}
    for path, label in support:
        library.setdefault(label, {"support_images": [], "mixed_attributes": []})["support_images"].append(str(path))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote support set and attribute template to {args.output}")


if __name__ == "__main__":
    main()
