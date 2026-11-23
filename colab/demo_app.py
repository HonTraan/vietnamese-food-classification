"""Streamlit demo for a checkpoint of any architecture (mirrors app.py's interface).

Run: VNFOOD_MODEL=models/final.pt streamlit run colab/demo_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # `streamlit run colab/demo_app.py` puts only colab/ on the path
    sys.path.insert(0, str(REPO))

from colab.models_zoo import build  # noqa: E402
from src.dataset import eval_transform  # noqa: E402


def load_model(path: Path) -> tuple[torch.nn.Module, list[str], float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build(checkpoint.get("arch", "efficientnet_b0"), len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), classes, float(checkpoint.get("temperature", 1.0))


def predict(image: Image.Image, model, classes: list[str], temperature: float, top_k: int = 3) -> list[tuple[str, float]]:
    with torch.inference_mode():
        logits = model(eval_transform()(image.convert("RGB")).unsqueeze(0))[0]
        probabilities = torch.softmax(logits / temperature, 0)
    scores, indices = probabilities.topk(min(top_k, len(classes)))
    return [(classes[index], score) for index, score in zip(indices.tolist(), scores.tolist())]


def main() -> None:
    import streamlit as st

    model_path = Path(os.environ.get("VNFOOD_MODEL", REPO / "models" / "final.pt"))
    st.set_page_config(page_title="Vietnamese Food Classifier", page_icon="🍜")
    st.title("🍜 Phân loại món ăn Việt Nam")
    if not model_path.is_file():
        st.error(f"Chưa có checkpoint {model_path}. Chạy notebook 03 để chuẩn bị model.")
        return
    model, classes, temperature = st.cache_resource(load_model)(model_path)
    st.caption(f"Phân loại 30 món ăn Việt Nam · mô hình: {type(model).__name__}")
    threshold = st.sidebar.slider("Ngưỡng độ tin cậy", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
    uploaded = st.file_uploader("Tải ảnh món ăn", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return
    image = Image.open(uploaded)
    st.image(image, caption="Ảnh đầu vào", use_container_width=True)
    predictions = predict(image, model, classes, temperature)
    label, confidence = predictions[0]
    if confidence < threshold:
        st.warning("Chưa xác định được món ăn.")
        st.caption(f"Độ tin cậy cao nhất {confidence:.1%}, dưới ngưỡng {threshold:.0%}.")
        return
    st.success(f"Dự đoán: **{label}** ({confidence:.1%})")
    st.subheader("Top 3 dự đoán")
    st.table({"Hạng": list(range(1, len(predictions) + 1)), "Món ăn": [item[0] for item in predictions],
              "Confidence": [f"{item[1]:.1%}" for item in predictions]})
    st.bar_chart(dict(predictions))


if __name__ == "__main__":
    main()
