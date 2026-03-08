"""
RF-03 / RF-04: POST /api/predict — clasificación de imagen de incidencia TI.

Responsabilidad única: recibir la imagen HTTP, validar el formato,
delegar la inferencia a la capa de inferencia y retornar la respuesta.
No contiene lógica de modelo ni de dominio.
"""

from __future__ import annotations

import io
from datetime import datetime
from datetime import timezone
from typing import Annotated

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError

from src.api.schemas import PredictionResponse
from src.domain.categories import CONFIDENCE_THRESHOLD
from src.domain.categories import get_category
from src.inference import classifier
from src.inference import model

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: Annotated[UploadFile, File()],
) -> PredictionResponse:
    """
    Recibe una imagen (multipart/form-data) y retorna la categoría
    de incidencia TI predicha con score de confianza y equipo sugerido.

    Errores:
        415 — formato de imagen no soportado (solo JPG, PNG, WebP).
        422 — el archivo no puede procesarse como imagen.
        503 — el modelo no está entrenado aún (ejecutar 'make train').
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Formato no soportado: '{file.content_type}'. Use JPG, PNG o WebP."
            ),
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="El archivo no es una imagen válida.",
        ) from None

    try:
        artifact = classifier.load()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo aún no está entrenado. "
                "Ejecuta 'make train' para entrenarlo primero."
            ),
        ) from None

    embedding = model.extract_embedding(image)
    category_key, score = classifier.predict(artifact, embedding)
    cat = get_category(category_key)

    return PredictionResponse(
        category=cat.label,
        score=round(score, 4),
        team=cat.team,
        timestamp=datetime.now(timezone.utc),
        human_review_required=(score < CONFIDENCE_THRESHOLD),
    )
