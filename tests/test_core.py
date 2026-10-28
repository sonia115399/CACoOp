import torch
from torch.nn import functional as F

from cacoop.core import alignment_loss, filter_attributes, select_caip, teacher_compatible, visual_prototypes
from cacoop.run import sample_support, validate_library


def test_greedy_selection_uses_gain_threshold_and_redundancy():
    p = torch.tensor([[1., 0.], [0., 1.]])
    candidates = F.normalize(torch.tensor([[1., 0.], [1., 0.], [0.8, 0.6], [-1., 0.]]), dim=-1)
    selected, trace = select_caip(candidates, torch.tensor([0.8, 0.6]), p, 0, (1., 0., 0.4), -0.05)
    assert selected == [0]
    assert trace[0]["redundancy"] == 0


def test_ties_retain_original_order_and_no_top_k():
    p = torch.tensor([[1., 0.], [0., 1.]])
    x = torch.tensor([[1., 0.]] * 6)
    selected, _ = select_caip(x, torch.tensor([1., 0.]), p, 0, (0., 0., 0.), -0.05)
    assert selected == list(range(6))


def test_filter_and_label_check_separate_from_anchor():
    attrs = filter_attributes(["Orange pustules.", "orange pustules", "whitefly insect", "white background", "brown spots"])
    assert attrs == ["Orange pustules", "whitefly insect", "brown spots"]
    assert not teacher_compatible("corn::rust", attrs[1])
    assert teacher_compatible("corn::whitefly infestation", attrs[1])
    assert not teacher_compatible("bugplant::rust", "a close-up photo of a bugplant leaf showing brown spots.")
    assert attrs[:2] == ["Orange pustules", "whitefly insect"]


def test_visual_prototypes_normalize_each_support_feature_first():
    x = torch.tensor([[10., 0.], [0., 1.], [-3., 0.]])
    p = visual_prototypes(x, torch.tensor([0, 0, 1]), 2)
    assert torch.allclose(p[0], torch.tensor([2**-0.5, 2**-0.5]))


def test_alignment_loss_is_three_cosine_terms():
    z = torch.tensor([[1., 0.]])
    assert torch.isclose(alignment_loss(z, torch.tensor([[0., 1.]]), z, -z, (0.1, 0.2, 0.3)), torch.tensor(0.7))


def test_image_provenance_must_match_support(tmp_path):
    import pytest
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    support = [(a, "x"), (b, "y")]
    library = {"x": {"support_images": ["a.jpg"], "mixed_attributes": [{"source": "image", "text": "brown spots", "image_ids": ["a.jpg"]}]}, "y": {"support_images": ["b.jpg"], "mixed_attributes": []}}
    validate_library(library, support, ["x", "y"], tmp_path, allow_incomplete=True)
    library["x"]["mixed_attributes"][0]["image_ids"] = ["b.jpg"]
    with pytest.raises(ValueError, match="non-support"):
        validate_library(library, support, ["x", "y"], tmp_path, allow_incomplete=True)


def test_teacher_filter_checks_complete_attribute_prompt():
    from contextlib import nullcontext
    from cacoop.run import build_targets

    class FakeModel:
        def encode_text(self, tokens):
            return torch.tensor([[1., 0.]] * len(tokens))

    labels = ["bugplant::rust", "corn::healthy"]
    library = {label: {"mixed_attributes": [{"text": "brown spots", "source": "class"}]} for label in labels}
    _, _, report = build_targets(FakeModel(), lambda prompts: torch.zeros((len(prompts), 1)), labels, library, torch.tensor([[1., 0.], [0., 1.]]), torch.device("cpu"), nullcontext, (0.7, 0.1, 0.1, 0.1, 0.2, 0.2))
    assert report[labels[0]]["teacher_candidates"] == []
    assert report[labels[0]]["anchor_attributes"] == ["brown spots"]
