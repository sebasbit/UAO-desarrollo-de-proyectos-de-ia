"""
RF-03: pruebas del endpoint POST /api/predict.

Cubre:
  - Rechazo de archivos que no son imágenes (4xx)
  - Respuesta válida con categoría y score para una imagen real
  - Activación de revisión humana cuando score < 0.40

TODO (T09): implementar una vez que RF-03 y RF-04 estén completos
"""
