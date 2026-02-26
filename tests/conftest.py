"""
Fixtures compartidas para todos los tests.

Aquí se centralizan los recursos reutilizables entre test_health.py,
test_predict.py y test_inference.py, evitando duplicación.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente HTTP para hacer peticiones a la app FastAPI en tests."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Imagen PNG mínima válida generada en memoria (sin disco)."""
    img = Image.new("RGB", (224, 224), color=(100, 149, 237))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
