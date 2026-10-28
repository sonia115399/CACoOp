# CACoOp: Visual-Evidence-Guided Attribute Selection for Knowledge-Regularized Context Optimization

CACoOp is a few-shot plant disease classification method built on frozen CLIP. It selects class attributes using labeled support images and trains shared context vectors with visual, selected-text, and attribute-anchor supervision.

## Contents

- [Installation](#installation)
- [Data and attributes](#data-and-attributes)
- [Training and evaluation](#training-and-evaluation)
- [Prediction](#prediction)
- [Repository structure](#repository-structure)

## Installation

Use Python 3.10 or later. Install the PyTorch and torchvision build appropriate for your system, then:

```bash
git clone https://github.com/sonia115399/CACoOp.git
cd CACoOp
python -m pip install -e '.[test]'
python -m pytest -q
```

## Data and attributes

Prepare disjoint `train.csv`, `val.csv`, and `test.csv` manifests with `image_path,label` columns. The splits must contain the same classes; labels use the `plant::class` form.

For each dataset, shot count, and seed, export the support set and an attribute template:

```bash
python -m cacoop.prepare \
  --data-root data/corn --train data/corn/train.csv \
  --shots 4 --seed 1 --output data/corn/attributes/corn_k4_seed1.json
```

Provide separately generated class-name and support-image descriptions and mix them with `cacoop.attributes`, or fill the template with descriptions in their fixed mixed order. Training requires 50 raw descriptions per class from both branches and checks that declared image sources match the support set. `--allow-incomplete-attributes` is available for development runs. See [dataset and attribute preparation](docs/DATASETS.md) for the JSON fields, filtering limits, and provenance requirements.

## Training and evaluation

```bash
python -m cacoop.run \
  --data-root data/corn --train data/corn/train.csv \
  --val data/corn/val.csv --test data/corn/test.csv \
  --attributes data/corn/attributes/corn_k4_seed1.json \
  --dataset corn --shots 4 --seed 1 --output outputs/corn/k4_seed1
```

For multi-seed runs, result aggregation, and the three alignment-loss ablations, see [experiment instructions](docs/EXPERIMENTS.md). The aggregation command supports population and sample standard deviations; check the reporting convention for each paper table.

## Prediction

```bash
python -m cacoop.predict \
  --checkpoint outputs/corn/k4_seed1/checkpoint.pt --images path/to/image.jpg
```

The output contains the predicted class and cosine similarity for each image. Prediction uses the checkpoint's stored class text features; it does not run attribute generation or CAIP selection.

## Repository structure

```text
src/cacoop/
  core.py        CAIP, attribute filtering, prototypes, and alignment loss
  prompt.py      shared context and CLIP text encoding
  prepare.py     support-set sampling or import and attribute template
  attributes.py  mix generated descriptions with support-image provenance
  run.py         target construction, training, and evaluation
  experiment.py  multi-shot and multi-seed runner
  tune.py        validation-only selection of user-supplied weights
  summarize.py   aggregation across runs
  predict.py     checkpoint-based image classification
docs/             data and experiment instructions
tests/            unit tests and a simulated training flow
private_inputs/   empty placeholders for local research inputs
```
