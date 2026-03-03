"""
Pruebas unitarias de la capa de inferencia.

Cubre:
  - Extracción de embeddings: el vector tiene la dimensión esperada
  - Clasificador: retorna una clave válida del catálogo de categorías
  - Regla de seguridad: score < 0.40 → categoría 'otros'

TODO (T09): implementar una vez que src/inference/ esté completo (T04)
"""
