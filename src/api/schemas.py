"""
Modelos Pydantic que definen el contrato HTTP de la API.

Responsabilidad única: serialización/deserialización y validación
de los datos que entran y salen por HTTP. No contiene lógica de
negocio ni de inferencia.

Depende de: nada (tipos estándar de Python + Pydantic)
"""

from datetime import datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Respuesta del endpoint POST /api/predict."""

    category: str
    score: float
    team: str
    timestamp: datetime
    human_review_required: bool