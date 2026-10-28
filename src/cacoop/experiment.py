"""Run CACoOp for multiple shot counts and seeds using per-run attributes."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .core import PAPER_WEIGHTS


def plan_runs(dataset, shots, seeds, attributes_dir, output_dir):
    if len(shots) != len(set(shots)) or len(seeds) != len(set(seeds)):
        raise ValueError("Shot counts and seeds must be unique")
    runs = []
    for shot in shots:
        for seed in seeds:
            attributes = attributes_dir / f"{dataset}_k{shot}_seed{seed}.json"
            output = output_dir / dataset / f"k{shot}_seed{seed}"
            runs.append((shot, seed, attributes, output))
    missing = [str(attributes) for _, _, attributes, _ in runs if not attributes.is_file()]
    occupied = [str(output) for _, _, _, output in runs if (output / "results.json").exists() or (output / "checkpoint.pt").exists()]
    if missing:
        raise FileNotFoundError("Missing attribute files: " + ", ".join(missing))
    if occupied:
        raise FileExistsError("Run outputs already exist: " + ", ".join(occupied))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attributes-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, help="optional per-run support CSV directory")
    parser.add_argument("--weights-file", type=Path, help="selected weights JSON for this dataset")
    parser.add_argument("--dataset", choices=sorted(PAPER_WEIGHTS), required=True)
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-attributes", action="store_true", help="allow non-paper attribute counts for development")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--without-alignment", choices=["none", "visual", "teacher", "anchor"], default="none")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    runs = plan_runs(args.dataset, args.shots, args.seeds, args.attributes_dir, args.output_dir)
    support_files = {}
    if args.support_dir:
        for shot, seed, _, _ in runs:
            support = args.support_dir / f"{args.dataset}_k{shot}_seed{seed}.csv"
            if not support.is_file():
                raise FileNotFoundError(f"Missing support manifest: {support}")
            support_files[(shot, seed)] = support
    if args.weights_file and not args.weights_file.is_file():
        raise FileNotFoundError(f"Missing weights file: {args.weights_file}")
    for shot, seed, attributes, output in runs:
        command = [sys.executable, "-m", "cacoop.run", "--data-root", str(args.data_root), "--train", str(args.train), "--val", str(args.val), "--test", str(args.test), "--attributes", str(attributes), "--dataset", args.dataset, "--shots", str(shot), "--seed", str(seed), "--output", str(output), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--precision", args.precision, "--without-alignment", args.without_alignment]
        if args.allow_incomplete_attributes:
            command.append("--allow-incomplete-attributes")
        if args.support_dir:
            command.extend(["--support", str(support_files[(shot, seed)])])
        if args.weights_file:
            command.extend(["--weights-file", str(args.weights_file)])
        print(f"Running {args.dataset}: {shot}-shot, seed {seed}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
