# Dataset and attribute preparation

The code expects one dataset per run. Keep images, split files, and attributes outside the Git repository or under an ignored local `data/` directory. This repository does not contain the six manuscript datasets or their original splits.

## Split manifests

Create UTF-8 CSV files for training, validation, and testing. Each file needs a header with `image_path,label`. An image path may be absolute or relative to `--data-root`. Use the same class labels across all splits, for example `plant::class`. The first part supplies the plant name in the attribute text template; the second part is the class name. Labels should be consistent with the names used to generate attributes.

Each manifest must have unique image paths, and the three splits must be disjoint. Each split must contain all classes; the training split needs at least K images per class.

## Support sets

For each K and seed, export a support-set template:

```bash
python -m cacoop.prepare \
  --data-root data/corn --train data/corn/train.csv \
  --shots 4 --seed 1 --output data/corn/attributes/corn_k4_seed1_template.json
```

The command uses the training manifest and records the selected image paths under each class. To use an existing support set, provide `--support private_inputs/support_sets/corn_k4_seed1.csv` to both `cacoop.prepare` and `cacoop.run`; it must use the same CSV columns and contain exactly K training images per class. The preparation command refuses to overwrite an existing file. Without a support manifest, the runner repeats the sampling and checks that the attribute template matches. With `--support`, it uses the supplied training subset and checks that the attribute template matches it. Changing the order or contents of the training CSV, K, or seed can change the sampled images. The original paper's support sets cannot be inferred from a seed alone unless the original sampling code and data order are also known.

## Attribute JSON

Each top-level key is an exact class label from the manifests. Its value has:

- `support_images`: the complete list of support image paths for that class.
- `mixed_attributes`: an ordered list of attribute objects. Each object has a `text` string and a `source` equal to `class` or `image`.
- `image_ids`: a nonempty list on every `image`-source object. Every path in this list must belong to the current class's support set. It is omitted for `class`-source objects.

Paths are resolved relative to `--data-root`; the template generated above contains absolute paths. Supply a separate file for each K and seed. Training requires exactly 50 raw descriptions from both branches for each class.

## Mix separately generated attributes

Keep source attributes in `private_inputs/attributes/`. The source JSON uses class labels as keys. Each label has `class_attributes` (a list of strings) and `image_attributes` (a list of objects with `text` and `image_ids`). Use the template created above to supply the support-image lists. Then run:

```bash
python -m cacoop.attributes \
  --data-root data/corn \
  --support-template data/corn/attributes/corn_k4_seed1_template.json \
  --sources private_inputs/attributes/corn_k4_seed1_sources.json \
  --seed 1 --output data/corn/attributes/corn_k4_seed1.json
```
