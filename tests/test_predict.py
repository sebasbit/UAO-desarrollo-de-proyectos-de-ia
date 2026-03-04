"""
RF-03: pruebas del endpoint POST /api/predict — T09.

ESTADO: en espera de T05
Cuando schemas.py y routes/predict.py estén implementados,
eliminar el skip y activar los tests.

Cubre:
  - Rechazo de archivos que no son imágenes (HTTP 422)
  - Respuesta válida con category, score, team y timestamp
  - Si score < 0.40 → human_review_required: true y category_key: "otros"
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.domain.categories import CATEGORY_KEYS
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.inference import classifier

pytestmark = pytest.mark.skip(
    reason="Esperando T05: src/api/schemas.py y src/api/routes/predict.py"
)


def _png_bytes(size=(224, 224), color=(100, 149, 237)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _make_artifact():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(40, 192)).astype(np.float32)
    labels = [CATEGORY_KEYS[i % len(CATEGORY_KEYS)] for i in range(40)]
    return classifier.train(embeddings, labels)


def _dummy_embedding(_image):
    return np.random.default_rng(1).normal(size=(192,)).astype(np.float32)


@pytest.fixture(scope="module")
def client_with_mocks():
    from src.api.main import app
    artifact = _make_artifact()
    with (
        patch("src.api.routes.predict.model.extract_embedding",
              side_effect=_dummy_embedding),
        patch("src.api.routes.predict._get_artifact", return_value=artifact),
    ):
        yield TestClient(app)


class TestRejectNonImage:
    def test_text_file_returns_422(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"image": ("doc.txt", b"texto", "text/plain")},
        )
        assert r.status_code == 422

    def test_pdf_returns_422(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 422


class TestValidResponse:
    def test_status_200(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_response_schema(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        ).json()
        for key in ("category", "score", "team", "timestamp", "human_review_required"):
            assert key in data

    def test_category_key_valid(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        ).json()
        assert data["category_key"] in CATEGORY_KEYS


class TestHumanReviewRule:
    def test_low_score_sets_human_review_true(self):
        from src.api.main import app
        artifact = _make_artifact()

        def _low(_art, _emb):
            return "otros", CONFIDENCE_THRESHOLD - 0.01

        with (
            patch("src.api.routes.predict.model.extract_embedding",
                  side_effect=_dummy_embedding),
            patch("src.api.routes.predict._get_artifact", return_value=artifact),
            patch("src.api.routes.predict.classifier.predict", side_effect=_low),
        ):
            data = TestClient(app).post(
                "/api/predict",
                files={"image": ("test.png", _png_bytes(), "image/png")},
            ).json()

        assert data["human_review_required"] is True
        assert data["category_key"] == "otros"
