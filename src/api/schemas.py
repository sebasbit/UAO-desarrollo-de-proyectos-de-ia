"""
Modelos Pydantic que definen el contrato HTTP de la API.

Responsabilidad única: serialización/deserialización y validación
de los datos que entran y salen por HTTP. No contiene lógica de
negocio ni de inferencia.

Depende de: src.domain.categories (solo para documentar los valores posibles)
"""

# TODO (T05): definir PredictionResponse con category, score, team, timestamp
