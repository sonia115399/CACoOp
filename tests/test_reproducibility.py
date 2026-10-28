import csv
import json
import sys

import pytest

from cacoop.attributes import mix_attributes
from cacoop.core import filter_attributes
from cacoop.run import read_manifest, resolve_support
from cacoop.tune import main as tune_main


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)


def test_support_manifest_must_be_in_training_set_and_have_k_per_class(tmp_path):
    train_path = tmp_path / "train.csv"
    support_path = tmp_path / "support.csv"
    write_manifest(train_path, [("a.jpg", "a"), ("a2.jpg", "a"), ("b.jpg", "b")])
    train = read_manifest(train_path, tmp_path)
    write_manifest(support_path, [("a2.jpg", "a"), ("b.jpg", "b")])
    assert resolve_support(train, support_path, tmp_path, 1) == [((tmp_path / "a2.jpg").resolve(), "a"), ((tmp_path / "b.jpg").resolve(), "b")]
    write_manifest(support_path, [("test.jpg", "a"), ("b.jpg", "b")])
    with pytest.raises(ValueError, match="outside the training"):
        resolve_support(train, support_path, tmp_path, 1)
    write_manifest(support_path, [("a.jpg", "a"), ("a2.jpg", "a"), ("b.jpg", "b")])
    with pytest.raises(ValueError, match="exactly 1"):
        resolve_support(train, support_path, tmp_path, 1)


def test_attribute_mixing_keeps_branches_support_and_order(tmp_path):
    template = {"crop::class": {"support_images": ["s.jpg"], "mixed_attributes": []}}
    sources = {"crop::class": {"class_attributes": [f"class {i}" for i in range(25)], "image_attributes": [{"text": f"image {i}", "image_ids": ["s.jpg"]} for i in range(25)]}}
    result = mix_attributes(template, sources, tmp_path, 19)
    assert result == mix_attributes(template, sources, tmp_path, 19)
    entry = result["crop::class"]
    assert len(entry["mixed_attributes"]) == 50
    assert entry["class_attributes"] == sources["crop::class"]["class_attributes"]
    sources["crop::class"]["image_attributes"][0]["image_ids"] = ["test.jpg"]
    with pytest.raises(ValueError, match="non-support"):
        mix_attributes(template, sources, tmp_path, 19)


def test_filter_excludes_acquisition_and_etiology_examples():
    assert filter_attributes(["camera glare", "fungal infection", "brown spots"]) == ["brown spots"]


def test_tuning_never_passes_or_reads_test_split(tmp_path, monkeypatch):
    attrs = tmp_path / "attributes"
    attrs.mkdir()
    for shot in (1, 2):
        for seed in (1, 2):
            (attrs / f"corn_k{shot}_seed{seed}.json").touch()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": [[0.1]*6, [0.2]*6]}), encoding="utf-8")
    commands = []
    def fake_run(command, check):
        commands.append(command)
        assert "--validation-only" in command and "--test" not in command
        output = tmp_path / command[command.index("--output") + 1]
        output.mkdir(parents=True)
        score = float(command[command.index("--weights") + 1])
        (output / "results.json").write_text(json.dumps({"test": None, "configuration": {"validation_only": True}, "val": {"accuracy": score}}), encoding="utf-8")
    monkeypatch.setattr("cacoop.tune.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["cacoop.tune", "--data-root", str(tmp_path), "--train", "train.csv", "--val", "val.csv", "--attributes-dir", str(attrs), "--dataset", "corn", "--shots", "1", "2", "--seeds", "1", "2", "--candidates", str(candidates), "--output-dir", str(tmp_path / "tuning")])
    tune_main()
    selected = json.loads((tmp_path / "tuning/selected.json").read_text())
    assert len(commands) == 8
    assert selected["shots"] == [1, 2]
    assert selected["weights"] == [0.2] * 6


def test_training_library_requires_50_attributes_from_both_sources(tmp_path):
    from cacoop.run import validate_library

    support = [(tmp_path / "s.jpg", "crop::class")]
    library = {"crop::class": {"support_images": ["s.jpg"], "mixed_attributes": [{"source": "class", "text": "brown spots"}]}}
    with pytest.raises(ValueError, match="expected 50 raw attributes"):
        validate_library(library, support, ["crop::class"], tmp_path)
    validate_library(library, support, ["crop::class"], tmp_path, allow_incomplete=True)
