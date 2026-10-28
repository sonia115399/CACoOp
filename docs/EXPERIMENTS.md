# Experiments

## Single run

After preparing the three manifests and a matching attribute JSON, run:

```bash
python -m cacoop.run \
  --data-root data/corn --train data/corn/train.csv \
  --val data/corn/val.csv --test data/corn/test.csv \
  --attributes data/corn/attributes/corn_k4_seed1.json \
  --dataset corn --shots 4 --seed 1 --output outputs/corn/k4_seed1
```

## Shot and seed grid

Create one attribute file per run under `--attributes-dir`, named `<dataset>_k<K>_seed<S>.json`. For instance, create the templates for a two-seed, one-shot run with:

```bash
for seed in 1 2; do
  python -m cacoop.prepare --data-root data/corn --train data/corn/train.csv \
    --shots 1 --seed "$seed" --output "data/corn/attributes/corn_k1_seed${seed}.json"
done
```

Populate each template with its own generated attributes, then run:

```bash
python -m cacoop.experiment \
  --data-root data/corn --train data/corn/train.csv \
  --val data/corn/val.csv --test data/corn/test.csv \
  --attributes-dir data/corn/attributes --dataset corn \
  --shots 1 --seeds 1 2 --output-dir outputs
```

Before starting, the grid runner checks that all requested attribute files exist and output files do not. It stops on the first failed run; completed runs remain on disk. Use a new output directory or run the remaining settings separately after correcting the failure.

## Aggregate results

```bash
python -m cacoop.summarize outputs/corn/k1_seed1/results.json outputs/corn/k1_seed2/results.json --ddof 0
```

The output is JSON with means and standard deviations of test accuracy and Macro-F1 on a 0–1 scale. `--ddof 0` uses population standard deviation; `--ddof 1` uses sample standard deviation and requires at least two runs per group. Select the convention required by the table being compared. The summarizer groups by dataset, K, and alignment ablation; it rejects duplicate seeds and differences in recorded weights, split hashes, effective precision, or training configuration. It cannot detect data or weight differences that are absent from the records.

## Validation-only weight selection

Create a local JSON file containing a `candidates` array. Each entry is a six-number array in the order `wg, wm, wr, lambda_visual, lambda_teacher, lambda_anchor`; values must be finite and nonnegative.

```bash
python -m cacoop.tune \
  --data-root data/corn --train data/corn/train.csv --val data/corn/val.csv \
  --attributes-dir data/corn/attributes --dataset corn --shots 4 --seeds 1 2 3 \
  --candidates private_inputs/training_configs/candidates.json --output-dir outputs/tuning_corn
```

The tuner evaluates the supplied candidates at the specified shot counts and seeds, selects the highest mean validation accuracy across those runs, and resolves ties by candidate order. It does not accept, read, or evaluate a test manifest. Optional `--support-dir` supplies `corn_k4_seed1.csv` and corresponding per-seed support manifests. The generated `selected.json` can be passed to `cacoop.run` or `cacoop.experiment` using `--weights-file` for subsequent runs. Do not choose between weight combinations using test results.

## Alignment-loss ablation

Pass `--without-alignment visual`, `teacher`, or `anchor` to either training entry point. This sets the corresponding alignment weight to zero while retaining CAIP and all target construction, as in the manuscript's alignment-loss ablation. Use a separate `--output` or `--output-dir` for each configuration.

## Prediction

```bash
python -m cacoop.predict --checkpoint outputs/corn/k4_seed1/checkpoint.pt --images path/to/image.jpg
```

The command loads OpenAI CLIP ViT-B/32, computes image features and compares them with the checkpoint's stored class text features. It prints the predicted label and cosine similarity for each image. It does not regenerate attributes or perform CAIP selection at inference.
