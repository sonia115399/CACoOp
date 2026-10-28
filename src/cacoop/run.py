"""Train one CACoOp dataset, support size and sampling seed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from .core import PAPER_WEIGHTS, alignment_loss, filter_attributes, select_caip, teacher_compatible, visual_prototypes
from .prompt import OfficialStylePromptLearner, encode_prompt_features


def read_manifest(path: Path, root: Path) -> list[tuple[Path, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = [( (root / row["image_path"]).resolve(), row["label"].strip()) for row in csv.DictReader(f)]
    if not rows or any(not label for _, label in rows):
        raise ValueError(f"Empty manifest or label: {path}")
    if len({p for p, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate images in {path}")
    return rows


def sample_support(records, shot, seed):
    rng = random.Random(seed)
    labels = sorted({label for _, label in records})
    support = []
    for label in labels:
        group = [row for row in records if row[1] == label]
        if len(group) < shot:
            raise ValueError(f"{label}: {len(group)} training images < {shot} shots")
        support.extend(rng.sample(group, shot))
    return support


def resolve_support(train_records, support_manifest: Path, root: Path, shot: int):
    support = read_manifest(support_manifest, root)
    train_pairs = set(train_records)
    if any(record not in train_pairs for record in support):
        raise ValueError("Support manifest contains an image or label outside the training split")
    train_labels = {label for _, label in train_records}
    support_counts = {label: sum(current == label for _, current in support) for label in train_labels}
    if any(count != shot for count in support_counts.values()) or {label for _, label in support} != train_labels:
        raise ValueError(f"Support manifest must contain exactly {shot} images per training class")
    return support


def clean_name(value: str) -> str:
    return " ".join(value.replace("___", " ").replace("_", " ").replace("-", " ").split()).lower()


def classname(label: str) -> str:
    if "::" not in label:
        return clean_name(label)
    plant, disease = (clean_name(x) for x in label.split("::", 1))
    if disease.startswith(plant + " "):
        disease = disease[len(plant) + 1:]
    return f"{plant} {disease}".strip()


def validate_library(library: dict, support, labels, root: Path, allow_incomplete: bool = False):
    if set(library) != set(labels):
        raise ValueError("Attribute-library keys must match the support labels exactly")
    for label in labels:
        item = library[label]
        expected = {str(path) for path, c in support if c == label}
        observed = {str((root / path).resolve()) for path in item["support_images"]}
        if expected != observed:
            raise ValueError(f"{label}: attribute support_images do not match sampled support set")
        entries = item["mixed_attributes"]
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError(f"{label}: mixed_attributes must be a list of objects")
        if not allow_incomplete and (len(entries) != 50 or {entry.get("source") for entry in entries} != {"class", "image"}):
            raise ValueError(f"{label}: expected 50 raw attributes from both branches; use --allow-incomplete-attributes only for development")
        for entry in entries:
            if entry.get("source") not in ("class", "image") or not isinstance(entry.get("text"), str):
                raise ValueError(f"{label}: each mixed attribute requires source (class/image) and text")
            if entry["source"] == "image":
                origins = entry.get("image_ids")
                if not isinstance(origins, list) or not origins or not {str((root / p).resolve()) for p in origins} <= expected:
                    raise ValueError(f"{label}: image attribute has missing or non-support image_ids")


def build_targets(model, tokenizer, labels, library, prototypes, device, amp, weights):
    teachers, anchors, report = [], [], {}
    with torch.no_grad():
        for c, label in enumerate(labels):
            plant = clean_name(label.split("::", 1)[0]) if "::" in label else "plant"
            name = classname(label)
            attrs = filter_attributes([x["text"] for x in library[label]["mixed_attributes"]])
            def prompt(attr):
                return f"a close-up photo of a {plant} leaf showing {attr}."
            candidates = [x for x in attrs if teacher_compatible(label, prompt(x))]
            def encode(prompts):
                with amp():
                    return F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1)
            base = encode([f"a leaf photo of {name}."])[0]
            if candidates:
                encoded = encode([prompt(x) for x in candidates])
                indices, trace = select_caip(encoded, base, prototypes, c, weights[:3])
                teacher = F.normalize(encoded[indices].mean(0), dim=0) if indices else base
            else:
                indices, trace, teacher = [], [], base
            if attrs:
                anchor = F.normalize(encode([prompt(x) for x in attrs[:4]]).mean(0), dim=0)
            else:
                anchor = F.normalize(encode([f"a photo of {name}.", f"a close-up photo of {name}."]).mean(0), dim=0)
            teachers.append(teacher)
            anchors.append(anchor)
            report[label] = {"raw_count": len(library[label]["mixed_attributes"]), "filtered_attributes": attrs, "teacher_candidates": candidates, "selected_attributes": [candidates[i] for i in indices], "selection_trace": trace, "teacher_fallback": not indices, "anchor_attributes": attrs[:4], "anchor_fallback": not bool(attrs)}
    return torch.stack(teachers), torch.stack(anchors), report


def encode_images(model, preprocess, records, labels, device, amp, batch_size):
    features = []
    targets = torch.tensor([labels.index(label) for _, label in records], device=device)
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            images = []
            for path, _ in records[start:start + batch_size]:
                with Image.open(path) as img:
                    images.append(preprocess(img.convert("RGB")))
            with amp():
                x = model.encode_image(torch.stack(images).to(device))
            features.append(F.normalize(x.float(), dim=-1))
    return torch.cat(features), targets


def metrics(logits, targets, nclasses):
    pred = logits.argmax(dim=1)
    f1 = []
    for i in range(nclasses):
        tp = ((pred == i) & (targets == i)).sum().item()
        fp = ((pred == i) & (targets != i)).sum().item()
        fn = ((pred != i) & (targets == i)).sum().item()
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": (pred == targets).float().mean().item(), "macro_f1": sum(f1) / len(f1)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path)
    p.add_argument("--validation-only", action="store_true", help="do not read or evaluate a test split")
    p.add_argument("--attributes", type=Path, required=True)
    p.add_argument("--support", type=Path, help="optional CSV of the exact training support images")
    p.add_argument("--allow-incomplete-attributes", action="store_true", help="allow fewer than 50 or single-branch raw attributes for development")
    p.add_argument("--dataset", choices=sorted(PAPER_WEIGHTS), required=True)
    p.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    weight_source = p.add_mutually_exclusive_group()
    weight_source.add_argument("--weights", type=float, nargs=6, metavar=("WG", "WM", "WR", "LV", "LT", "LK"))
    weight_source.add_argument("--weights-file", type=Path)
    p.add_argument("--without-alignment", choices=["none", "visual", "teacher", "anchor"], default="none")
    args = p.parse_args()
    if args.validation_only and args.test is not None:
        p.error("--test cannot be used with --validation-only")
    if not args.validation_only and args.test is None:
        p.error("--test is required unless --validation-only is set")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("epochs, batch size and learning rate must be positive")
    if (args.output / "checkpoint.pt").exists() or (args.output / "results.json").exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.weights_file:
        weight_payload = json.loads(args.weights_file.read_text(encoding="utf-8"))
        if isinstance(weight_payload, dict) and weight_payload.get("dataset", args.dataset) != args.dataset:
            raise ValueError("Weight file dataset does not match --dataset")
        weights = weight_payload.get("weights") if isinstance(weight_payload, dict) else weight_payload
    else:
        weights = args.weights if args.weights is not None else list(PAPER_WEIGHTS[args.dataset])
    if not isinstance(weights, list) or len(weights) != 6 or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("Weights must be six finite nonnegative numbers: wg wm wr lv lt lk")
    weights = list(weights)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = args.data_root.resolve()
    train, val = [read_manifest(x, root) for x in (args.train, args.val)]
    test = read_manifest(args.test, root) if args.test is not None else None
    splits = [train, val] + ([test] if test is not None else [])
    sets = [{path for path, _ in records} for records in splits]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("Train, validation and test manifests overlap")
    labels = sorted({label for _, label in train})
    if len(labels) < 2 or any({label for _, label in records} != set(labels) for records in splits[1:]):
        raise ValueError("All splits must contain the same set of at least two classes")
    support = resolve_support(train, args.support, root, args.shots) if args.support else sample_support(train, args.shots, args.seed)
    library = json.loads(args.attributes.read_text(encoding="utf-8"))
    validate_library(library, support, labels, root, allow_incomplete=args.allow_incomplete_attributes)
    import open_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = encode_images(model, preprocess, support, labels, device, amp, args.batch_size)
    val_x, val_y = encode_images(model, preprocess, val, labels, device, amp, args.batch_size)
    if test is not None:
        test_x, test_y = encode_images(model, preprocess, test, labels, device, amp, args.batch_size)
    visual = visual_prototypes(train_x, train_y, len(labels))
    if args.without_alignment != "none":
        weights[{"visual": 3, "teacher": 4, "anchor": 5}[args.without_alignment]] = 0.0
    teacher, anchor, report = build_targets(model, tokenizer, labels, library, visual, device, amp, weights)
    learner = OfficialStylePromptLearner(model, tokenizer, [classname(x) for x in labels], "a leaf photo of", 4).to(device)
    optimizer = torch.optim.SGD([learner.ctx], lr=args.lr)
    scale = model.logit_scale.exp().detach()
    best_acc, best_epoch, best_context = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        for idx in torch.randperm(len(train_x), device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                z = encode_prompt_features(model, learner).float()
                loss = F.cross_entropy(scale.float() * train_x[idx] @ z.T, train_y[idx]) + alignment_loss(z, visual, teacher, anchor, weights[3:])
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad(), amp():
                z = encode_prompt_features(model, learner).float()
                score = metrics(val_x @ z.T, val_y, len(labels))["accuracy"]
            if score > best_acc:
                best_acc, best_epoch, best_context = score, epoch, learner.ctx.detach().clone()
    with torch.no_grad():
        learner.ctx.copy_(best_context)
        with amp():
            z = encode_prompt_features(model, learner).float()
        val_metrics = metrics(val_x @ z.T, val_y, len(labels))
        test_metrics = metrics(test_x @ z.T, test_y, len(labels)) if test is not None else None
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"context": best_context.cpu(), "class_text_features": z.cpu(), "labels": labels, "model": "ViT-B-32", "pretrained": "openai"}, args.output / "checkpoint.pt")
    result = {"dataset": args.dataset, "shots": args.shots, "seed": args.seed, "weights_wg_wm_wr_lv_lt_lk": weights, "best_epoch": best_epoch, "val": val_metrics, "test": test_metrics, "precision_effective": "bf16" if use_bf16 else "fp32", "support": [{"image_path": str(path), "label": label} for path, label in support], "attribute_report": report}
    result["configuration"] = {
        "model": "ViT-B-32", "pretrained": "openai", "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "precision_requested": args.precision, "without_alignment": args.without_alignment,
        "split_sha256": {name: file_sha256(path) for name, path in (("train", args.train), ("val", args.val), ("test", args.test)) if path is not None},
        "validation_only": args.validation_only, "allow_incomplete_attributes": args.allow_incomplete_attributes,
        "attributes_sha256": file_sha256(args.attributes),
        "support_manifest_sha256": file_sha256(args.support) if args.support else None,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "open_clip_torch": package_version("open_clip_torch"), "device": str(device),
    }
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
