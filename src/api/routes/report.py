"""
RF-06: POST /api/report — genera y descarga el reporte PDF de la clasificación.

Recibe la imagen, invoca la inferencia y
delega la construcción del PDF a src.api.pdf_builder.

No importa nada de src.api.schemas.
Usa ReportData, el DTO propio de pdf_builder, para ser completamente
autónomo. Cuando T05 esté listo, la integración será transparente.
"""

from __future__ import annotations

import io
from datetime import UTC
from datetime import datetime

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from src.api.pdf_builder import ReportData
from src.api.pdf_builder import build_report_pdf
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.domain.categories import get_category
from src.inference import classifier
from src.inference import model

router = APIRouter()

_artifact = None


def _get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = classifier.load()
    return _artifact


@router.post("/report")
async def download_report(image: UploadFile = File(...)) -> StreamingResponse:
    """
    Clasifica la imagen y retorna un PDF descargable con el reporte de triage.

    El PDF incluye: categoría, score, equipo, timestamp y sugerencia
    de enrutamiento según las reglas del catálogo (RF-06).
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="El archivo debe ser una imagen (image/*).",
        )

    try:
        from io import BytesIO

        from PIL import Image

        raw = await image.read()
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen. Formato inválido o corrupto.",
        ) from exc

    embedding = model.extract_embedding(img)
    category_key, score = classifier.predict(_get_artifact(), embedding)
    cat = get_category(category_key)

    data = ReportData.from_prediction(
        category=cat.label,
        category_key=cat.key,
        score=round(score, 4),
        team=cat.team,
        human_review_required=score < CONFIDENCE_THRESHOLD,
        image_filename=image.filename or "imagen_incidente",
        timestamp=datetime.now(UTC),
    )

    pdf_bytes = build_report_pdf(data)
    filename = f"triage_{data.timestamp:%Y%m%d_%H%M%S}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
