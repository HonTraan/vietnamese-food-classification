import torch
from PIL import Image

from colab import demo_app
from colab.models_zoo import build


def test_predict_probabilities_sum(tmp_path):
    classes = [f"class_{i}" for i in range(30)]
    path = tmp_path / "best.pt"
    torch.save({"model_state": build("mobilenet_v3_large", 30, pretrained=False).state_dict(),
                "classes": classes, "arch": "mobilenet_v3_large", "temperature": 1.5}, path)
    model, loaded_classes, temperature = demo_app.load_model(path)
    assert loaded_classes == classes and temperature == 1.5
    predictions = demo_app.predict(Image.new("RGB", (300, 200), "orange"), model, classes, temperature)
    assert len(predictions) == 3
    scores = [score for _, score in predictions]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) <= 1.0 + 1e-6


def test_load_model_defaults_to_efficientnet(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": build("efficientnet_b0", 30, pretrained=False).state_dict(),
                "classes": [str(i) for i in range(30)]}, path)
    _, _, temperature = demo_app.load_model(path)
    assert temperature == 1.0
