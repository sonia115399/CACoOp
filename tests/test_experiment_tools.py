import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from cacoop.experiment import plan_runs
from cacoop.predict import predict
from cacoop.summarize import summarize


def run_result(seed, accuracy, macro_f1):
    return {
        "dataset": "corn", "shots": 4, "seed": seed,
        "weights_wg_wm_wr_lv_lt_lk": [0.7, 0.1, 0.1, 0.1, 0.2, 0.2],
        "test": {"accuracy": accuracy, "macro_f1": macro_f1},
        "configuration": {"without_alignment": "none", "epochs": 100, "batch_size": 4, "learning_rate": 0.0025, "precision_requested": "bf16", "model": "ViT-B-32", "pretrained": "openai"},
    }


def test_summary_respects_standard_deviation_and_rejects_duplicates():
    runs = [run_result(1, 0.5, 0.4), run_result(2, 0.7, 0.6)]
    result = summarize(runs, ddof=0)[0]
    assert result["accuracy"]["mean"] == pytest.approx(0.6)
    assert result["accuracy"]["std"] == pytest.approx(0.1)
    assert summarize(runs, ddof=1)[0]["accuracy"]["std"] == pytest.approx(math.sqrt(0.02))
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize([runs[0], runs[0]])
    runs[1]["configuration"]["epochs"] = 50
    with pytest.raises(ValueError, match="configurations differ"):
        summarize(runs)


def test_experiment_preflights_all_attribute_files_and_outputs(tmp_path):
    attrs = tmp_path / "attributes"
    attrs.mkdir()
    (attrs / "corn_k1_seed1.json").touch()
    with pytest.raises(FileNotFoundError, match="corn_k1_seed2"):
        plan_runs("corn", [1], [1, 2], attrs, tmp_path / "runs")
    (attrs / "corn_k1_seed2.json").touch()
    runs = plan_runs("corn", [1], [1, 2], attrs, tmp_path / "runs")
    assert len(runs) == 2
    runs[0][3].mkdir(parents=True)
    (runs[0][3] / "results.json").touch()
    with pytest.raises(FileExistsError, match="outputs already exist"):
        plan_runs("corn", [1], [1, 2], attrs, tmp_path / "runs")


def test_prediction_uses_checkpoint_label_order(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(image_path)
    class Model:
        def encode_image(self, tensor):
            return torch.tensor([[0.0, 2.0]])
    checkpoint = {"labels": ["first", "second"], "class_text_features": torch.eye(2)}
    result = predict(Model(), lambda image: torch.zeros((3, 1, 1)), checkpoint, [image_path], torch.device("cpu"))
    assert result[0]["label"] == "second"
    assert result[0]["cosine_similarity"] == pytest.approx(1.0)


def test_experiment_passes_ablation_and_matching_attributes(tmp_path, monkeypatch):
    import sys
    from cacoop.experiment import main

    attrs = tmp_path / "attributes"
    attrs.mkdir()
    (attrs / "corn_k1_seed1.json").touch()
    (attrs / "corn_k1_seed2.json").touch()
    commands = []
    monkeypatch.setattr("cacoop.experiment.subprocess.run", lambda command, check: commands.append(command))
    monkeypatch.setattr(sys, "argv", ["cacoop.experiment", "--data-root", str(tmp_path), "--train", "train.csv", "--val", "val.csv", "--test", "test.csv", "--attributes-dir", str(attrs), "--dataset", "corn", "--shots", "1", "--seeds", "1", "2", "--output-dir", str(tmp_path / "out"), "--without-alignment", "teacher"])
    main()
    assert len(commands) == 2
    for seed, command in enumerate(commands, 1):
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--attributes") + 1].endswith(f"corn_k1_seed{seed}.json")
        assert command[command.index("--without-alignment") + 1] == "teacher"


def test_experiment_passes_explicit_support_and_weight_file(tmp_path, monkeypatch):
    import sys
    from cacoop.experiment import main

    attrs, supports = tmp_path / "attributes", tmp_path / "support"
    attrs.mkdir()
    supports.mkdir()
    (attrs / "corn_k1_seed1.json").touch()
    support_file = supports / "corn_k1_seed1.csv"
    support_file.touch()
    weight_file = tmp_path / "weights.json"
    weight_file.touch()
    commands = []
    monkeypatch.setattr("cacoop.experiment.subprocess.run", lambda command, check: commands.append(command))
    monkeypatch.setattr(sys, "argv", ["cacoop.experiment", "--data-root", str(tmp_path), "--train", "train.csv", "--val", "val.csv", "--test", "test.csv", "--attributes-dir", str(attrs), "--support-dir", str(supports), "--weights-file", str(weight_file), "--dataset", "corn", "--shots", "1", "--seeds", "1", "--output-dir", str(tmp_path / "out")])
    main()
    assert commands[0][commands[0].index("--support") + 1] == str(support_file)
    assert commands[0][commands[0].index("--weights-file") + 1] == str(weight_file)
