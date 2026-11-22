"""Run inference for one image."""
from __future__ import annotations

from pathlib import Path
import torch
from PIL import Image
from .dataset import eval_transform
from .train import build_model

ROOT = Path(__file__).resolve().parents[1]

def predict(image: Image.Image, model_path: Path = ROOT / "models" / "best_model.pt", top_k: int = 3):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(len(checkpoint["classes"]), pretrained=False); model.load_state_dict(checkpoint["model_state"]); model.eval()
    temperature = float(checkpoint.get("temperature", 1.0))
    with torch.inference_mode():
        logits = model(eval_transform()(image.convert("RGB")).unsqueeze(0))[0]
        probabilities = torch.softmax(logits / temperature, 0)
    top_k = min(top_k, len(checkpoint["classes"]))
    scores, indices = probabilities.topk(top_k)
    predictions = [(checkpoint["classes"][index.item()], score.item()) for score, index in zip(scores, indices)]
    return predictions, dict(zip(checkpoint["classes"], probabilities.tolist()))
