"""Classify images using the stored CACoOp class text features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F


def predict(model, preprocess, checkpoint, image_paths, device):
    labels = checkpoint["labels"]
    features = checkpoint["class_text_features"].to(device).float()
    if features.ndim != 2 or len(labels) != features.shape[0] or not labels:
        raise ValueError("Checkpoint class labels and text features do not match")
    text_features = F.normalize(features, dim=-1)
    results = []
    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            image_features = F.normalize(model.encode_image(tensor).float(), dim=-1)
            if image_features.shape[1] != text_features.shape[1]:
                raise ValueError("Checkpoint and image encoder feature dimensions differ")
            similarities = (image_features @ text_features.T)[0]
            best = int(similarities.argmax().item())
            results.append({"image_path": str(image_path), "label": labels[best], "cosine_similarity": float(similarities[best])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model", "ViT-B-32") != "ViT-B-32" or checkpoint.get("pretrained", "openai") != "openai":
        raise ValueError("The checkpoint is not compatible with OpenAI CLIP ViT-B/32")
    import open_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.to(device).eval()
    print(json.dumps(predict(model, preprocess, checkpoint, args.images, device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
