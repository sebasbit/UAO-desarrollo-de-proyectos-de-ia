import os

from PIL import Image

os.environ["USE_DUMMY_MODEL"] = "1"

from src.inference.service import TriageService


def test_inference_returns_known_category() -> None:
    service = TriageService()
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    predictions = service.predict(image)
    assert len(predictions) >= 1
    assert predictions[0].category_key in {
        "red_conectividad",
        "correo_office365",
        "aplicacion_errores",
        "otros",
    }
    assert 0.0 <= predictions[0].score <= 1.0
