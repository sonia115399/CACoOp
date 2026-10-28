"""Summarize independent runs without mixing datasets, shots, or ablations."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(records: list[dict], ddof: int = 0) -> list[dict]:
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    groups = defaultdict(list)
    for record in records:
        config = record.get("configuration", {})
        key = (record["dataset"], record["shots"], config.get("without_alignment", "none"))
        groups[key].append(record)
    output = []
    for (dataset, shots, ablation), runs in sorted(groups.items()):
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed for {dataset}, {shots}-shot, {ablation}")
        if len(runs) <= ddof:
            raise ValueError(f"Need more than {ddof} runs for standard deviation")
        config_keys = ("epochs", "batch_size", "learning_rate", "precision_requested", "model", "pretrained", "python", "torch", "open_clip_torch", "device", "allow_incomplete_attributes")
        configurations = {tuple(run.get("configuration", {}).get(key) for key in config_keys) for run in runs}
        if len(configurations) != 1:
            raise ValueError(f"Training configurations differ for {dataset}, {shots}-shot, {ablation}")
        split_hashes = {tuple(sorted(run.get("configuration", {}).get("split_sha256", {}).items())) for run in runs}
        effective_precision = {run.get("precision_effective") for run in runs}
        if len(split_hashes) != 1 or len(effective_precision) != 1:
            raise ValueError(f"Data splits or effective precision differ for {dataset}, {shots}-shot, {ablation}")
        weights = {tuple(run["weights_wg_wm_wr_lv_lt_lk"]) for run in runs}
        if len(weights) != 1:
            raise ValueError(f"Weights differ for {dataset}, {shots}-shot, {ablation}")
        summary = {"dataset": dataset, "shots": shots, "without_alignment": ablation, "seeds": sorted(seeds), "runs": len(runs), "std_ddof": ddof}
        for metric in ("accuracy", "macro_f1"):
            values = [float(run["test"][metric]) for run in runs]
            summary[metric] = {"mean": statistics.mean(values), "std": statistics.pstdev(values) if ddof == 0 else statistics.stdev(values)}
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="paths to individual results.json files")
    parser.add_argument("--ddof", type=int, choices=[0, 1], default=0, help="0: population standard deviation; 1: sample standard deviation")
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    print(json.dumps(summarize(records, args.ddof), indent=2))


if __name__ == "__main__":
    main()
