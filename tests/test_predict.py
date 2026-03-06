"""
RF-03: pruebas del endpoint POST /api/predict — T09.

Adaptado al contrato real de Alexander Calambas (T05):
  - El campo del formulario es 'file', no 'image'
  - Formatos aceptados: JPG, PNG, WebP
  - Formato no soportado → HTTP 415
  - Imagen inválida     → HTTP 422
  - Modelo no entrenado → HTTP 503

Cubre:
  - Rechazo de archivos con formato no soportado (415)
  - Respuesta válida con category, score, team y timestamp
  - Si score < 0.40 → human_review_required: true
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.domain.categories import CATEGORY_KEYS
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.inference import classifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_bytes(size=(224, 224), color=(100, 149, 237)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(size=(224, 224), color=(100, 149, 237)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_artifact():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(40, 192)).astype(np.float32)
    labels = [CATEGORY_KEYS[i % len(CATEGORY_KEYS)] for i in range(40)]
    return classifier.train(embeddings, labels)


def _dummy_embedding(_image):
    return np.random.default_rng(1).normal(size=(192,)).astype(np.float32)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client_with_mocks():
    artifact = _make_artifact()
    with (
        patch("src.api.routes.predict.model.extract_embedding",
              side_effect=_dummy_embedding),
        patch("src.api.routes.predict.classifier.load",
              return_value=artifact),
    ):
        yield TestClient(app)


# ── Rechazo de formatos no soportados ────────────────────────────────────────

class TestRejectUnsupportedFormat:
    def test_text_file_returns_415(self, client_with_mocks):
        """Archivos que no son imagen deben retornar 415."""
        r = client_with_mocks.post(
            "/api/predict",
            files={"file": ("doc.txt", b"texto plano", "text/plain")},
        )
        assert r.status_code == 415

    def test_pdf_returns_415(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 415

    def test_gif_returns_415(self, client_with_mocks):
        """GIF no está en los formatos aceptados por Alexander."""
        r = client_with_mocks.post(
            "/api/predict",
            files={"file": ("anim.gif", b"GIF89a", "image/gif")},
        )
        assert r.status_code == 415


# ── Respuesta válida ──────────────────────────────────────────────────────────

class TestValidResponse:
    def test_png_returns_200(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_jpeg_returns_200(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.jpg", _jpg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_has_required_fields(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        ).json()
        for key in ("category", "score", "team", "timestamp", "human_review_required"):
            assert key in data, f"Campo faltante en la respuesta: '{key}'"

    def test_score_in_range(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        ).json()
        assert 0.0 <= data["score"] <= 1.0

    def test_category_nonempty(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        ).json()
        assert data["category"].strip()

    def test_team_nonempty(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        ).json()
        assert data["team"].strip()

    def test_human_review_required_is_bool(self, client_with_mocks):
        data = client_with_mocks.post(
            "/api/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        ).json()
        assert isinstance(data["human_review_required"], bool)


# ── Regla de revisión humana ──────────────────────────────────────────────────

class TestHumanReviewRule:
    def test_low_score_sets_human_review_true(self):
        """score < CONFIDENCE_THRESHOLD → human_review_required debe ser True."""
        artifact = _make_artifact()

        def _low_score(_art, _emb):
            return "otros", CONFIDENCE_THRESHOLD - 0.01

        with (
            patch("src.api.routes.predict.model.extract_embedding",
                  side_effect=_dummy_embedding),
            patch("src.api.routes.predict.classifier.load",
                  return_value=artifact),
            patch("src.api.routes.predict.classifier.predict",
                  side_effect=_low_score),
        ):
            data = TestClient(app).post(
                "/api/predict",
                files={"file": ("test.png", _png_bytes(), "image/png")},
            ).json()

        assert data["human_review_required"] is True

    def test_high_score_sets_human_review_false(self):
        """score >= CONFIDENCE_THRESHOLD → human_review_required debe ser False."""
        artifact = _make_artifact()

        def _high_score(_art, _emb):
            return "red_conectividad", CONFIDENCE_THRESHOLD + 0.20

        with (
            patch("src.api.routes.predict.model.extract_embedding",
                  side_effect=_dummy_embedding),
            patch("src.api.routes.predict.classifier.load",
                  return_value=artifact),
            patch("src.api.routes.predict.classifier.predict",
                  side_effect=_high_score),
        ):
            data = TestClient(app).post(
                "/api/predict",
                files={"file": ("test.png", _png_bytes(), "image/png")},
            ).json()

        assert data["human_review_required"] is False
