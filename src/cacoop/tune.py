"""Select a user-supplied weight combination using validation accuracy only."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .core import PAPER_WEIGHTS


def read_candidates(path: Path) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file must contain a nonempty candidates list")
    for weights in candidates:
        if not isinstance(weights, list) or len(weights) != 6 or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in weights):
            raise ValueError("Each candidate needs six finite nonnegative weights")
    if len({tuple(weights) for weights in candidates}) != len(candidates):
        raise ValueError("Candidate weights must be unique")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--attributes-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, help="optional per-run support CSV directory")
    parser.add_argument("--dataset", choices=sorted(PAPER_WEIGHTS), required=True)
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-attributes", action="store_true", help="allow non-paper attribute counts for development")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.shots) != len(set(args.shots)) or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Seeds and shot counts must be unique; epochs, batch size and learning rate must be positive")
    candidates = read_candidates(args.candidates)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    paths = []
    for shot in args.shots:
        for seed in args.seeds:
            stem = f"{args.dataset}_k{shot}_seed{seed}"
            attrs = args.attributes_dir / f"{stem}.json"
            support = args.support_dir / f"{stem}.csv" if args.support_dir else None
            if not attrs.is_file() or (support is not None and not support.is_file()):
                raise FileNotFoundError(f"Missing attributes or support manifest for {stem}")
            paths.append((shot, seed, attrs, support))
    args.output_dir.mkdir(parents=True)
    candidate_scores = []
    for index, weights in enumerate(candidates):
        scores = []
        for shot, seed, attributes, support in paths:
            output = args.output_dir / f"candidate_{index:04d}" / f"k{shot}_seed{seed}"
            command = [sys.executable, "-m", "cacoop.run", "--validation-only", "--data-root", str(args.data_root), "--train", str(args.train), "--val", str(args.val), "--attributes", str(attributes), "--dataset", args.dataset, "--shots", str(shot), "--seed", str(seed), "--output", str(output), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--precision", args.precision, "--weights", *[str(w) for w in weights]]
            if args.allow_incomplete_attributes:
                command.append("--allow-incomplete-attributes")
            if support is not None:
                command.extend(["--support", str(support)])
            print(f"Candidate {index + 1}/{len(candidates)}, {shot}-shot, seed {seed}", flush=True)
            subprocess.run(command, check=True)
            record = json.loads((output / "results.json").read_text(encoding="utf-8"))
            if record.get("test") is not None or not record.get("configuration", {}).get("validation_only"):
                raise ValueError("Tuning run did not use validation-only mode")
            score = float(record["val"]["accuracy"])
            if not math.isfinite(score):
                raise ValueError("Validation accuracy must be finite")
            scores.append({"shots": shot, "seed": seed, "accuracy": score})
        candidate_scores.append({"weights": weights, "validation_runs": scores, "mean_validation_accuracy": sum(row["accuracy"] for row in scores) / len(scores)})
    best = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i]["mean_validation_accuracy"])
    selected = {"dataset": args.dataset, "shots": args.shots, "seeds": args.seeds, "selection_rule": "mean validation accuracy across shots and seeds; first candidate wins ties", "weights": candidates[best], "candidates": candidate_scores}
    (args.output_dir / "selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"selected_weights": candidates[best], "mean_validation_accuracy": candidate_scores[best]["mean_validation_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
