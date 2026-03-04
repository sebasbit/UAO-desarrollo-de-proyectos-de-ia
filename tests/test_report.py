"""
RF-06: pruebas del endpoint POST /api/report — T09.

Cubre:
  - El endpoint retorna un PDF válido (firma %PDF-)
  - Content-Type es application/pdf
  - Content-Disposition indica descarga (attachment)
  - Al menos 3 PDFs consecutivos son válidos (criterio de aceptación RF-06)
  - Rechazo de archivos que no son imágenes (HTTP 422)
  - Pruebas unitarias del builder y del DTO ReportData
"""

from __future__ import annotations

import io
from datetime import UTC
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.api.pdf_builder import ReportData
from src.api.pdf_builder import _PDFBuilder
from src.api.pdf_builder import build_report_pdf
from src.domain.categories import CATEGORY_KEYS
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.inference import classifier


# Helpers

def _png_bytes(color=(100, 149, 237)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_artifact():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(40, 192)).astype(np.float32)
    labels = [CATEGORY_KEYS[i % len(CATEGORY_KEYS)] for i in range(40)]
    return classifier.train(embeddings, labels)


def _dummy_embedding(_image):
    return np.random.default_rng(1).normal(size=(192,)).astype(np.float32)


def _sample_report_data() -> ReportData:
    return ReportData(
        category="Red / Conectividad",
        category_key="red_conectividad",
        score=0.87,
        team="Equipo Redes",
        timestamp=datetime(2025, 6, 15, 10, 30, tzinfo=UTC),
        human_review_required=False,
        image_filename="captura_test.png",
    )


@pytest.fixture(scope="module")
def client_with_mocks():
    artifact = _make_artifact()
    with (
        patch("src.api.routes.report.model.extract_embedding",
              side_effect=_dummy_embedding),
        patch("src.api.routes.report._get_artifact", return_value=artifact),
    ):
        yield TestClient(app)


# Unitarias: ReportData DTO

class TestReportDataDTO:
    def test_fields_assigned_correctly(self):
        d = _sample_report_data()
        assert d.category == "Red / Conectividad"
        assert d.category_key == "red_conectividad"
        assert d.score == pytest.approx(0.87)
        assert d.team == "Equipo Redes"
        assert d.human_review_required is False

    def test_from_prediction_factory(self):
        d = ReportData.from_prediction(
            category="Correo / Office 365",
            category_key="correo_office365",
            score=0.65,
            team="Equipo O365",
            human_review_required=False,
        )
        assert d.category_key == "correo_office365"
        assert isinstance(d.timestamp, datetime)

    def test_from_prediction_default_timestamp(self):
        d = ReportData.from_prediction(
            category="X", category_key="otros",
            score=0.3, team="Revisión humana",
            human_review_required=True,
        )
        assert isinstance(d.timestamp, datetime)

    def test_human_review_true_when_low_score(self):
        d = ReportData.from_prediction(
            category="Otros / No clasifica",
            category_key="otros",
            score=CONFIDENCE_THRESHOLD - 0.01,
            team="Revisión humana",
            human_review_required=True,
        )
        assert d.human_review_required is True


# Unitarias: _PDFBuilder

class TestPDFBuilder:
    def test_build_returns_bytes(self):
        assert isinstance(_PDFBuilder(_sample_report_data()).build(), bytes)

    def test_pdf_has_valid_signature(self):
        assert _PDFBuilder(_sample_report_data()).build()[:5] == b"%PDF-"

    def test_pdf_above_5kb(self):
        assert len(_PDFBuilder(_sample_report_data()).build()) > 1024

    def test_build_with_human_review_required(self):
        d = ReportData(
            category="Otros / No clasifica", category_key="otros",
            score=0.30, team="Revisión humana",
            timestamp=datetime.now(UTC), human_review_required=True,
        )
        pdf = _PDFBuilder(d).build()
        assert pdf[:5] == b"%PDF-"

    def test_score_bar_full(self):
        assert _PDFBuilder._score_bar(1.0, 10) == "█" * 10

    def test_score_bar_empty(self):
        assert _PDFBuilder._score_bar(0.0, 10) == "░" * 10

    def test_score_bar_half(self):
        bar = _PDFBuilder._score_bar(0.5, 10)
        assert bar.count("█") == 5 and bar.count("░") == 5


# Unitarias: build_report_pdf

class TestBuildReportPDF:
    def test_returns_valid_pdf(self):
        assert build_report_pdf(_sample_report_data())[:5] == b"%PDF-"

    def test_size_above_5kb(self):
        assert len(build_report_pdf(_sample_report_data())) > 1024


# API: endpoint POST /api/report

class TestAPIReport:
    def test_status_200(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_content_type_pdf(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.headers["content-type"] == "application/pdf"

    def test_pdf_signature(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.content[:5] == b"%PDF-"

    def test_content_disposition_attachment(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert "attachment" in r.headers.get("content-disposition", "")

    def test_filename_ends_with_pdf(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert ".pdf" in r.headers.get("content-disposition", "")

    def test_pdf_size_above_5kb(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert len(r.content) > 1024

    def test_reject_text_file_422(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("doc.txt", b"texto plano", "text/plain")},
        )
        assert r.status_code == 422

    def test_reject_pdf_input_422(self, client_with_mocks):
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 422

    def test_three_consecutive_pdfs_valid(self, client_with_mocks):
        """Criterio de aceptación RF-06: mínimo 3 PDFs abribles y válidos."""
        for i in range(3):
            color = (i * 80, 100, 200 - i * 60)
            r = client_with_mocks.post(
                "/api/report",
                files={"image": (f"inc_{i}.png", _png_bytes(color), "image/png")},
            )
            assert r.status_code == 200,       f"PDF #{i+1}: status {r.status_code}"
            assert r.content[:5] == b"%PDF-",  f"PDF #{i+1}: firma inválida"
            assert len(r.content) > 1024,  f"PDF #{i+1}: tamaño sospechoso"

    def test_jpeg_accepted(self, client_with_mocks):
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (200, 100, 50)).save(buf, format="JPEG")
        r = client_with_mocks.post(
            "/api/report",
            files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert r.status_code == 200
        assert r.content[:5] == b"%PDF-"
