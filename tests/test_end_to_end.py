import csv
import json
import sys
import types

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from cacoop.run import main, read_manifest, sample_support


class Transformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        return x + x.mean(dim=0, keepdim=True)


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(100, 8)
        self.transformer = Transformer()
        self.positional_embedding = nn.Parameter(torch.zeros(16, 8))
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.eye(8))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.attn_mask = None

    def encode_text(self, tokens):
        emb = self.token_embedding(tokens) + self.positional_embedding
        return self.ln_final(self.transformer(emb.permute(1, 0, 2)).permute(1, 0, 2)[:, -1]) @ self.text_projection

    def encode_image(self, images):
        return F.pad(images.mean((2, 3)), (0, 5))


def tokenize(prompts):
    rows = []
    for prompt in prompts:
        ids = [1] + [2 + sum(map(ord, word)) % 90 for word in prompt.split()][:14]
        rows.append((ids + [0] * (15 - len(ids))) + [99])
    return torch.tensor(rows)


@pytest.mark.parametrize("ablation, weight_index", [("visual", 3), ("teacher", 4), ("anchor", 5)])
def test_training_pipeline_with_stub_encoder(tmp_path, monkeypatch, ablation, weight_index):
    for split in ("train", "val", "test"):
        rows = []
        for i, label in enumerate(("corn::rust", "corn::healthy")):
            name = f"{split}_{i}.png"
            Image.new("RGB", (2, 2), (255 * i, 255 * (1 - i), 0)).save(tmp_path / name)
            rows.append({"image_path": name, "label": label})
        with (tmp_path / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
            writer.writeheader()
            writer.writerows(rows)
    support = sample_support(read_manifest(tmp_path / "train.csv", tmp_path), 1, 1)
    library = {label: {"support_images": [str(path)], "mixed_attributes": [{"source": "image", "image_ids": [str(path)], "text": "brown spots"}]} for path, label in support}
    attr_path = tmp_path / "attrs.json"
    attr_path.write_text(json.dumps(library))
    fake = types.SimpleNamespace(create_model_and_transforms=lambda *a, **kw: (FakeCLIP(), None, lambda im: torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(2, 2, 3).permute(2, 0, 1) / 255), get_tokenizer=lambda name: tokenize)
    monkeypatch.setitem(sys.modules, "open_clip", fake)
    monkeypatch.setattr(sys, "argv", ["cacoop", "--data-root", str(tmp_path), "--train", str(tmp_path / "train.csv"), "--val", str(tmp_path / "val.csv"), "--test", str(tmp_path / "test.csv"), "--attributes", str(attr_path), "--allow-incomplete-attributes", "--dataset", "corn", "--shots", "1", "--seed", "1", "--epochs", "2", "--without-alignment", ablation, "--output", str(tmp_path / "out")])
    main()
    result = json.loads((tmp_path / "out/results.json").read_text())
    checkpoint = torch.load(tmp_path / "out/checkpoint.pt", weights_only=True)
    assert result["best_epoch"] in (1, 2)
    assert result["configuration"]["without_alignment"] == ablation
    assert result["weights_wg_wm_wr_lv_lt_lk"][weight_index] == 0.0
    assert result["configuration"]["epochs"] == 2
    assert len(result["configuration"]["attributes_sha256"]) == 64
    assert checkpoint["context"].shape == (4, 8)
    assert checkpoint["class_text_features"].shape == (2, 8)
    assert set(result["attribute_report"]) == set(library)


def test_validation_only_uses_supplied_support_without_test_set(tmp_path, monkeypatch):
    from cacoop.run import main

    labels = ["corn::rust", "corn::healthy"]
    for split in ("train", "val"):
        with (tmp_path / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "label"])
            writer.writeheader()
            for index, label in enumerate(labels):
                name = f"{split}_{index}.png"
                Image.new("RGB", (2, 2), (index * 255, (1 - index) * 255, 0)).save(tmp_path / name)
                writer.writerow({"image_path": name, "label": label})
    support = tmp_path / "support.csv"
    support.write_text((tmp_path / "train.csv").read_text())
    library = {label: {"support_images": [f"train_{index}.png"], "mixed_attributes": [{"source": "class", "text": "brown spots"}]} for index, label in enumerate(labels)}
    attrs = tmp_path / "attrs.json"
    attrs.write_text(json.dumps(library))
    weight_file = tmp_path / "selected.json"
    weight_file.write_text(json.dumps({"dataset": "corn", "weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}))
    fake = types.SimpleNamespace(create_model_and_transforms=lambda *a, **kw: (FakeCLIP(), None, lambda im: torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(2, 2, 3).permute(2, 0, 1) / 255), get_tokenizer=lambda name: tokenize)
    monkeypatch.setitem(sys.modules, "open_clip", fake)
    monkeypatch.setattr(sys, "argv", ["cacoop.run", "--validation-only", "--data-root", str(tmp_path), "--train", str(tmp_path / "train.csv"), "--val", str(tmp_path / "val.csv"), "--support", str(support), "--attributes", str(attrs), "--allow-incomplete-attributes", "--dataset", "corn", "--shots", "1", "--seed", "1", "--weights-file", str(weight_file), "--epochs", "1", "--output", str(tmp_path / "validation")])
    main()
    result = json.loads((tmp_path / "validation/results.json").read_text())
    assert result["test"] is None
    assert result["weights_wg_wm_wr_lv_lt_lk"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert set(result["configuration"]["split_sha256"]) == {"train", "val"}
    assert result["configuration"]["support_manifest_sha256"]
