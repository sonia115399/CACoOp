"""CACoOp target construction and objective (manuscript Section III)."""
from __future__ import annotations

import re

import torch
from torch.nn import functional as F

PEST_LABELS = ("infestation", "whitefly", "leafhopper", "leafhoppers", "diaspididae")
PEST_ATTRIBUTES = ("insect", "bug", "whitefly", "leafhopper", "leafhoppers", "diaspididae")
BLOCKED = ("jade jade", "painting", "man in", "man is", "holding a green leaf", "how to grow", "white background", "black background", "white flowers", "bird is", "tree branch", "four images", "col of four", "piece of wood", "piece of green", "plate", "hat", "marble", "water droplets", "laying on", "on a table", "next to a cup", "blue surface", "white surface", "shadow on", "patch of dirt", "small green leaf", "some leaves on it", "eaten by a fly", "words,", "background")

NON_VISUAL_WORDS = frozenset({"camera", "lighting", "illumination", "background", "shadow", "pathogen", "pathogens", "fungal", "bacterial", "viral", "infection", "disease"})

PAPER_WEIGHTS = {
    "corn": (0.7, 0.1, 0.1, 0.1, 0.2, 0.2),
    "datepalm": (0.3, 0.1, 0.2, 0.2, 0.2, 0.1),
    "tea": (0.1, 0.7, 0.1, 0.2, 0.2, 0.1),
    "duranta_erecta_gold": (0.3, 0.4, 0.1, 0.9, 0.5, 0.5),
    "murraya_exotica": (0.1, 0.5, 0.1, 0.2, 0.2, 0.1),
    "ruellia_simplex": (0.9, 0.7, 0.1, 0.2, 0.2, 0.1),
}


def normalize_attribute(value: str) -> str:
    return " ".join(value.strip().rstrip(" .;,").split())


def filter_attributes(values: list[str]) -> list[str]:
    result, seen = [], set()
    for value in values:
        text = normalize_attribute(value)
        key = text.casefold()
        words = key.split()
        word_set = set(re.findall(r"[a-z]+", key))
        if not words or key in seen or word_set & NON_VISUAL_WORDS or any(term in key for term in BLOCKED):
            continue
        if len(words) > 24 or len(set(words)) < max(2, len(words) // 3):
            continue
        seen.add(key)
        result.append(text)
    return result


def teacher_compatible(label: str, prompt: str) -> bool:
    """Apply the manuscript class check to the complete attribute prompt."""
    label = label.split("::", 1)[-1].casefold()
    return any(term in label for term in PEST_LABELS) or not any(term in prompt.casefold() for term in PEST_ATTRIBUTES)


def visual_prototypes(features: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    result = []
    for c in range(n_classes):
        group = features[labels == c]
        if not len(group):
            raise ValueError(f"Class {c} has no support images")
        result.append(F.normalize(group.mean(0), dim=0))
    return torch.stack(result)


def select_caip(candidates: torch.Tensor, base: torch.Tensor, prototypes: torch.Tensor, class_index: int, weights: tuple[float, float, float], threshold: float = -0.05) -> tuple[list[int], list[dict]]:
    if prototypes.shape[0] < 2:
        raise ValueError("CAIP class margin requires at least two classes")
    if len(candidates) == 0:
        return [], []
    candidates = F.normalize(candidates.float(), dim=-1)
    base = F.normalize(base.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    sims = candidates @ prototypes.T
    gain = sims[:, class_index] - base @ prototypes[class_index]
    mask = torch.arange(len(prototypes), device=sims.device) == class_index
    margin = sims[:, class_index] - sims.masked_fill(mask[None, :], float('-inf')).max(dim=1).values
    selected, trace = [], []
    remaining = list(range(len(candidates)))
    wg, wm, wr = weights
    while remaining:
        best, best_score, best_r = None, None, None
        for i in remaining:
            if gain[i].item() < 0:
                continue
            redundancy = (candidates[selected] @ candidates[i]).max().item() if selected else 0.0
            score = wg * gain[i].item() + wm * margin[i].item() - wr * redundancy
            if score >= threshold and (best_score is None or score > best_score):
                best, best_score, best_r = i, score, redundancy
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        trace.append({"index": best, "gain": gain[best].item(), "margin": margin[best].item(), "redundancy": best_r, "score": best_score})
    return selected, trace


def alignment_loss(text: torch.Tensor, visual: torch.Tensor, teacher: torch.Tensor, anchor: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    return sum(w * (1 - (text.float() * target.float()).sum(dim=-1)).mean() for w, target in zip(weights, (visual, teacher, anchor)))
