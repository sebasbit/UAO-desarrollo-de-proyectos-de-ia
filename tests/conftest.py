import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ["USE_DUMMY_MODEL"] = "1"

from src.api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(100, 149, 237))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
