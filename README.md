# Triage de Soporte TI mediante Clasificación de Imágenes

Proyecto integrado sobre el esqueleto **UAO-desarrollo-de-proyectos-de-ia** para que el triage visual funcione siguiendo el árbol del repo base.

## Qué se integró

Se tomó el proyecto `triage_deit_cpu_fastapi_do_grpc_mlflow_uv_v1` y se acomodó dentro de la estructura del repo académico:

- `src/api/`: FastAPI, rutas HTTP, UI Jinja2 y esquemas de respuesta.
- `src/inference/`: extractor DeiT-Tiny, carga del clasificador y servicio de predicción.
- `src/domain/`: catálogo de categorías y equipos de enrutamiento.
- `src/grpc/`: servidor, cliente y stubs gRPC.
- `scripts/`: entrenamiento y evaluación.
- `proto/`: contrato `triage.proto`.
- `tests/`: pruebas para health, predict, inferencia y gRPC.
- `docker-compose.yml`: despliegue web + gRPC.

## Árbol principal

```text
.
├── main.py
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── proto/
│   └── triage.proto
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── deps.py
│   │   ├── schemas.py
│   │   ├── utils.py
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   └── ui.py
│   │   ├── static/
│   │   └── templates/
│   ├── config.py
│   ├── domain/
│   ├── grpc/
│   │   ├── client.py
│   │   ├── server.py
│   │   └── stubs/
│   └── inference/
│       ├── classifier.py
│       ├── model.py
│       └── service.py
└── tests/
```

## Cómo correrlo local con uv

### 1. Instalar dependencias

```bash
uv sync --group dev
```

### 2. Modo rápido para pruebas locales

Usa modelo dummy y evita descargar DeiT o requerir `models/` entrenados.

**Windows PowerShell**
```powershell
$env:USE_DUMMY_MODEL="1"
uv run uvicorn src.api.main:app --reload
```

**Git Bash / Linux / macOS**
```bash
USE_DUMMY_MODEL=1 uv run uvicorn src.api.main:app --reload
```

Abrir:
- UI: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`

### 3. Pruebas

```bash
USE_DUMMY_MODEL=1 uv run pytest
```

## Cómo correrlo con gRPC

### Terminal 1
```bash
USE_DUMMY_MODEL=1 uv run python -m src.grpc.server
```

### Terminal 2
```bash
USE_DUMMY_MODEL=1 USE_GRPC_BACKEND=1 GRPC_TARGET=127.0.0.1:50051 uv run uvicorn src.api.main:app --reload
```

## Docker Compose

```bash
docker compose up --build
```

Servicios:
- Web FastAPI: `http://localhost:8000`
- gRPC: `localhost:50051`

## Entrenamiento real

Estructura esperada:

```text
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

Ejecutar:

```bash
uv run python scripts/train.py
```

Esto genera:
- `models/classifier.pkl`
- `models/labels.json`
- `models/val_report.txt`
- `models/confusion_matrix.csv`
- `mlruns/` con tracking de MLflow

## Variables de entorno útiles

- `USE_DUMMY_MODEL=1`: evita descargar y cargar el modelo real.
- `USE_GRPC_BACKEND=1`: hace que FastAPI consulte el backend gRPC.
- `GRPC_TARGET=127.0.0.1:50051`: destino del servidor gRPC.
- `CLASSIFIER_PATH=models/classifier.pkl`
- `LABELS_PATH=models/labels.json`
- `MIN_CONFIDENCE=0.40`
- `TOP_K=3`

## Endpoints

- `GET /health`
- `POST /api/predict`
- `GET /`
- `POST /predict`

## Nota importante

La integración ya quedó acomodada al árbol del repo base. Lo que no pude validar dentro del contenedor fue la ejecución real de `pytest` porque el entorno aquí no tiene instalada la dependencia `grpcio`; sí pude validar que todos los archivos Python compilan correctamente con `compileall`.
