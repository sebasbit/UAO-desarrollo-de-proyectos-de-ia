from __future__ import annotations

from fastapi import APIRouter
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.deps import cfg
from src.api.deps import triage_service
from src.api.routes.predict import _predict

router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": cfg.app_title,
            "info": triage_service.info(),
            "results": None,
            "error": None,
        },
    )


@router.post("/predict", response_class=HTMLResponse)
def predict_html(request: Request, image: UploadFile) -> HTMLResponse:
    try:
        response = _predict(image)
        results = [
            {
                "category": item.category,
                "score": item.score,
                "routed_to": item.team,
            }
            for item in response.predictions
        ]
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": cfg.app_title,
                "info": response.model.model_dump(),
                "results": results,
                "error": None,
            },
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        message = getattr(exc, "detail", "Error inesperado durante la clasificación.")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": cfg.app_title,
                "info": triage_service.info(),
                "results": None,
                "error": message,
            },
            status_code=status_code,
        )
