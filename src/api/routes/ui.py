"""
RF-05 / RF-07: Rutas HTML — interfaz web para el triage de soporte TI.

Maneja las rutas que devuelven HTML con Jinja2.
No importa nada de src.api.schemas.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import File
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Página principal con el formulario de carga de imagen."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/", response_class=HTMLResponse)
async def predict_ui(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    """
    Recibe la imagen del formulario e intenta clasificarla.
    Si la inferencia aún no está lista (T05 pendiente), muestra
    un mensaje claro en lugar de un traceback.
    """
    from src.domain.categories import CONFIDENCE_THRESHOLD
    from src.domain.categories import get_category
    from src.inference import classifier
    from src.inference import model

    error = None
    result = None

    if not image.content_type or not image.content_type.startswith("image/"):
        error = "El archivo debe ser una imagen (image/*)."
    else:
        try:
            import io as _io

            from PIL import Image

            raw = await image.read()
            img = Image.open(_io.BytesIO(raw))
            img.load()

            artifact = classifier.load()
            embedding = model.extract_embedding(img)
            category_key, score = classifier.predict(artifact, embedding)
            cat = get_category(category_key)

            result = {
                "category": cat.label,
                "category_key": cat.key,
                "score": round(score, 4),
                "team": cat.team,
                "human_review_required": score < CONFIDENCE_THRESHOLD,
            }
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
            "error": error,
            "image_filename": image.filename,
        },
    )
