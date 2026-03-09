"""
Punto de entrada de la aplicación FastAPI.

Responsabilidad única: configurar la app, registrar routers
y montar archivos estáticos. No contiene lógica de negocio.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.deps import triage_service
from src.api.routes import health
from src.api.routes import predict
from src.api.routes import ui

app = FastAPI(
    title="Triage de Soporte TI",
    description="Clasificación de imágenes de incidencias TI mediante DeiT-Tiny",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

app.include_router(health.router)
app.include_router(predict.router, prefix="/api")
app.include_router(ui.router)


@app.on_event("startup")
def startup_event() -> None:
    triage_service.load()
