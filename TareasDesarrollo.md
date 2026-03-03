# Plan de Desarrollo — Tareas Pendientes

Documento de planificación para el equipo. Describe las tareas que faltan según el cronograma del proyecto, cómo se pueden dividir entre los 4 integrantes y qué debe entregar cada uno.

---

## Índice

1. [Resumen del estado](#1-resumen-del-estado)
2. [Tareas pendientes](#2-tareas-pendientes)
3. [División sugerida del trabajo](#3-división-sugerida-del-trabajo)
4. [Detalle por integrante](#4-detalle-por-integrante)
5. [Dependencias entre tareas](#5-dependencias-entre-tareas)
6. [Criterios de aceptación por tarea](#6-criterios-de-aceptación-por-tarea)

---

## 1. Resumen del estado

| Tarea | Descripción | Estado |
|---|---|---|
| T01 | Catálogo de categorías y reglas de enrutamiento |  Completo |
| T02 | Recolección y etiquetado del dataset |  Pendiente |
| T03 | Configuración del entorno y estructura del proyecto |  Completo |
| T04 | Módulo de inferencia DeiT-Tiny + clasificador sklearn |  Completo |
| T05 | Endpoints FastAPI: `POST /api/predict` y `GET /health` |  Pendiente |
| T06 | Prueba de baseline del modelo y ajuste de umbrales |  Pendiente |
| T07 | Interfaz web HTML/Jinja2 + exportar reporte PDF |  Pendiente |
| T08 | Mejora del modelo si Macro-F1 < 0.70 |  Condicional |
| T09 | Pruebas automatizadas (unitaria, API, integración) |  Pendiente |
| T10 | Pipeline CI en GitHub Actions |  Estructura lista — activar y ajustar |
| T11 | Dockerfile final y prueba de contenedor en CPU |  Base lista — completar al final |
| T12 | Evidencias, demo y organización del repositorio |  Pendiente |

---

## 2. Tareas pendientes

### T02 — Recolección del dataset
**Fechas:** 26/02 – 28/02
**Entregable:** carpetas `data/raw/{categoria}/` con 20–50 imágenes por categoría

Recolectar imágenes para cada una de las 8 categorías. Las imágenes deben ser capturas de pantalla simuladas, imágenes públicas (Unsplash, Pixabay, Wikimedia) o screenshots genéricos de interfaces. **Ninguna imagen puede contener datos personales (PII).**

Estructura esperada:
```
data/raw/
├── red_conectividad/
├── acceso_contrasenas/
├── correo_office365/
├── impresion_perifericos/
├── aplicacion_errores/
├── hardware_equipo/
├── vpn_remoto/
└── otros/
```

---

### T05 — Endpoints FastAPI
**Fechas:** 02/03 – 03/03
**Entregable:** servicio probado con curl o Postman

Implementar en `src/api/`:
- `schemas.py` — modelos Pydantic para request y response de `/api/predict`
- `routes/predict.py` — lógica del endpoint `POST /api/predict`: recibir imagen, validar formato, invocar `model.extract_embedding()` + `classifier.predict()`, retornar JSON con categoría, score, equipo y timestamp
- Manejo de errores: archivo no imagen → HTTP 422, imagen corrupta → HTTP 400

Respuesta esperada del endpoint:
```json
{
  "category": "Red / Conectividad",
  "category_key": "red_conectividad",
  "score": 0.87,
  "team": "Equipo Redes",
  "timestamp": "2026-03-03T10:45:00",
  "human_review_required": false
}
```

---

### T06 — Baseline del modelo y ajuste de umbrales
**Fechas:** 03/03 – 04/03
**Entregable:** reporte con Accuracy, Macro-F1, Top-1 routing accuracy

Implementar `scripts/evaluate.py`:
- Cargar el clasificador entrenado desde `models/classifier.pkl`
- Correr inferencia sobre el split de test del dataset
- Reportar: Accuracy, Macro-F1, matriz de confusión, Top-1 y Top-2 routing accuracy
- Si Macro-F1 < 0.70 → documentar el valor real y activar T08

---

### T07 — Interfaz web HTML/Jinja2 y exportación PDF
**Fechas:** 04/03 – 05/03
**Entregable:** UI funcional en el navegador con botón de descarga de reporte

Implementar en `src/api/`:
- `routes/ui.py` — rutas HTML para la página principal y resultados
- `src/api/templates/` — plantillas Jinja2:
  - `index.html` — formulario de carga de imagen
  - `result.html` — visualización de categoría, score, equipo y timestamp
- `src/api/static/` — estilos CSS básicos
- Botón para descargar el reporte de clasificación como PDF (RF-06): el PDF debe incluir la imagen analizada, categoría predicha, score, timestamp y equipo sugerido
- En caso de error (formato no soportado, falla de API) mostrar mensaje claro sin traceback

---

### T08 — Mejora del modelo (condicional)
**Fechas:** 04/03 – 06/03
**Entregable:** modelo con Macro-F1 >= 0.70

Se activa solo si el Macro-F1 del baseline es menor a 0.70. Estrategias a aplicar:
- Aumentar el número de imágenes por categoría
- Aplicar data augmentation (rotaciones, recortes, cambios de brillo)
- Ajustar hiperparámetros del clasificador (C en LogisticRegression, kernel en SVM)
- Como último recurso: cambiar de LogisticRegression a SVM

---

### T09 — Pruebas automatizadas
**Fechas:** 05/03 – 06/03
**Entregable:** mínimo 3 tests pasando localmente

Implementar en `tests/`:
- `test_predict.py`:
  - El endpoint rechaza archivos que no son imágenes con HTTP 422
  - El endpoint retorna `category`, `score` y `team` para una imagen válida
  - Si score < 0.40 → `human_review_required: true` y `category_key: "otros"`
- `test_inference.py`:
  - El embedding tiene la dimensión correcta (192)
  - `classifier.predict()` retorna una clave válida del catálogo
  - La regla de seguridad: score < 0.40 → retorna `"otros"` siempre

---

### T12 — Evidencias, demo y organización del repositorio
**Fechas:** 10/03 – 12/03
**Entregable:** proyecto completamente funcional y documentado

- Organizar el repositorio: limpiar archivos temporales, revisar README
- Tomar capturas de pantalla de la UI funcionando
- Grabar o documentar una demostración completa del flujo
- Verificar que el CI pasa en GitHub Actions
- Verificar que `docker build` y `docker run` funcionan correctamente
- Subir evidencias de despliegue en DigitalOcean Droplet (healthcheck desde URL pública)

---

## 3. División sugerida del trabajo

La división está pensada para que cada integrante tenga tareas independientes que pueda desarrollar en paralelo, respetando las dependencias mínimas.

| Integrante | Tareas asignadas | Foco |
|---|---|---|
| **Alexander Calambas** | T05 + T09 | API y calidad |
| **Angelo Parra** | T07 | Frontend e interfaz web |
| **Oscar Portela** | T02 + T06 | Dataset y evaluación del modelo |
| **Sebastian Torres** | T02 + T08 + T11 + T12 | Dataset, modelo y despliegue |

> Esta es una sugerencia. El equipo puede ajustar la distribución según disponibilidad. Lo importante es respetar el orden de dependencias.

---

## 4. Detalle por integrante

### Alexander Calambas — API y calidad

**T05 — Endpoints FastAPI**

Archivos a modificar:
- [src/api/schemas.py](src/api/schemas.py) — definir `PredictionResponse` con Pydantic
- [src/api/routes/predict.py](src/api/routes/predict.py) — implementar `POST /api/predict`

Cómo empezar:
```bash
# Verificar que el entorno funciona
make test

# Levantar el servidor y probar con curl
make start
curl http://localhost:8000/health
```

**T09 — Pruebas automatizadas**

Archivos a modificar:
- [tests/test_predict.py](tests/test_predict.py)
- [tests/test_inference.py](tests/test_inference.py)

Esperar a que T05 esté completo antes de implementar `test_predict.py`.

---

### Angelo Parra — Frontend e interfaz web

**T07 — Interfaz web HTML/Jinja2 + PDF**

Archivos a modificar / crear:
- [src/api/routes/ui.py](src/api/routes/ui.py) — rutas HTML
- `src/api/templates/index.html` — formulario de carga
- `src/api/templates/result.html` — visualización del resultado
- `src/api/static/style.css` — estilos básicos

Esperar a que T05 esté listo (schemas y endpoint de predicción).

Referencia para generación de PDF desde Python:
- Librería sugerida: `reportlab` o `fpdf2` (agregar a `pyproject.toml` si se elige esta ruta)
- Alternativa más simple: generar el PDF desde el navegador con `window.print()` y CSS `@media print`

---

### Oscar Portela — Dataset y evaluación

**T02 — Recolección del dataset**

- Recolectar 20–50 imágenes por categoría y organizarlas en `data/raw/`
- Verificar que ninguna imagen contenga PII antes de usarla
- Fuentes sugeridas: Unsplash, Pixabay, Wikimedia, capturas propias en modo demo

**T06 — Baseline y evaluación**

Archivos a modificar:
- [scripts/evaluate.py](scripts/evaluate.py) — implementar reporte de métricas

Cómo ejecutar una vez que el dataset esté listo:
```bash
make train     # genera models/classifier.pkl
make evaluate  # reporta Accuracy y Macro-F1
```

---

### Sebastian Torres — Dataset, modelo y despliegue

**T02 — Recolección del dataset** (en paralelo con Oscar)

Apoyar la recolección de imágenes para las categorías que le correspondan.

**T08 — Mejora del modelo** _(solo si Macro-F1 < 0.70 tras T06)_

- Estrategia 1: aumentar muestras del dataset
- Estrategia 2: data augmentation en `scripts/train.py`
- Estrategia 3: ajustar hiperparámetros del clasificador

**T11 — Dockerfile y despliegue**

- Verificar que `docker build -t triage-ti .` completa sin errores con el modelo entrenado
- Verificar que `docker run -p 8000:8000 triage-ti` levanta la UI y la API
- Documentar el proceso de despliegue en DigitalOcean Droplet

**T12 — Cierre del proyecto**

- Coordinar la recolección de evidencias del equipo
- Verificar estado del CI en GitHub Actions
- Organizar el repositorio para la entrega final

---

## 5. Dependencias entre tareas

```
T02 (dataset)
    └── T06 (baseline)
            └── T08 (mejora, si aplica)

T04 (inferencia) ← ya completo
    └── T05 (API predict)
            ├── T07 (UI web)
            └── T09 (tests)

T05 + T07
    └── T11 (Docker final)
            └── T12 (cierre)
```

**Tareas que pueden empezar hoy sin esperar nada:**
- T02 — recolección del dataset (todos pueden contribuir)
- T05 — endpoints FastAPI (T04 ya está completo)

---

## 6. Criterios de aceptación por tarea

| Tarea | Criterio de aceptación |
|---|---|
| T02 | Al menos 20 imágenes por categoría en `data/raw/`, sin PII |
| T05 | `POST /api/predict` responde con `{category, score, team, timestamp}` y rechaza no-imágenes con 4xx |
| T06 | `make evaluate` imprime reporte con Macro-F1 >= 0.70 (o valor real documentado) |
| T07 | UI funciona en el navegador: se puede subir imagen, ver resultado y descargar PDF |
| T08 | Modelo reentrenado alcanza Macro-F1 >= 0.70 |
| T09 | `make test` pasa los 3 tests mínimos sin errores |
| T10 | Cada push a `main` dispara el workflow en GitHub y pasa lint + tests |
| T11 | `docker build` y `docker run` levantan la app completa en CPU |
| T12 | Repositorio organizado, CI verde, evidencias de despliegue documentadas |
