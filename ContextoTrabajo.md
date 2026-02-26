# Contexto de Trabajo — Triage de Soporte TI

Documento de referencia para el equipo de desarrollo. Explica la estructura del proyecto, cómo configurar el entorno, los comandos disponibles y el estado actual de cada módulo.

---

## Índice

1. [Requisitos previos](#1-requisitos-previos)
2. [Configuración inicial](#2-configuración-inicial)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Arquitectura y capas](#4-arquitectura-y-capas)
5. [Comandos disponibles](#5-comandos-disponibles)
6. [Estado actual del desarrollo](#6-estado-actual-del-desarrollo)
7. [Flujo de trabajo en equipo](#7-flujo-de-trabajo-en-equipo)

---

## 1. Requisitos previos

| Herramienta | Versión mínima | Cómo instalar |
|---|---|---|
| Python | >= 3.10 | [python.org](https://www.python.org/downloads/) |
| uv | >= 0.10 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git | cualquiera | [git-scm.com](https://git-scm.com/) |
| Docker | >= 24.0 | [docker.com](https://www.docker.com/) _(opcional, para probar el contenedor)_ |

> **Windows:** todo funciona de la misma manera. Si no tienes WSL2, asegúrate de correr los comandos desde PowerShell o Git Bash.
>
> **Mac Apple Silicon (M-series):** compatible nativamente. `uv` detecta la plataforma y configura PyTorch automáticamente.

---

## 2. Configuración inicial

Ejecuta estos comandos **una sola vez** después de clonar el repositorio.

### Mac (Apple Silicon / Intel)

`make` viene preinstalado en macOS. Ejecuta directamente desde la terminal:

```bash
# 1. Instalar todas las dependencias (detecta tu plataforma automáticamente)
make install

# 2. Activar los hooks de calidad de código (pre-commit)
make pre-commit-install

# 3. Verificar que todo funciona
make test
```

### Windows

`make` no está disponible en PowerShell por defecto. Hay dos opciones:

**Opción A — Git Bash (recomendada)**

Si tienen [Git for Windows](https://git-scm.com/) instalado, abrir **Git Bash** y usar los mismos comandos `make`:

```bash
make install
make pre-commit-install
make test
```

**Opción B — PowerShell / CMD (sin make)**

Usar `uv` directamente como equivalente:

```powershell
uv sync                                           # equivale a make install
uv run pre-commit install                         # equivale a make pre-commit-install
uv run pytest tests/                              # equivale a make test
```

---

Si el último comando muestra `1 passed`, el entorno está listo.

### Nota sobre PyTorch

No es necesario instalar PyTorch manualmente. `uv` lee el archivo `pyproject.toml` y descarga el build correcto según tu sistema operativo:

| Plataforma | Build instalado |
|---|---|
| Mac (Apple Silicon / Intel) | CPU build estándar |
| Windows | CPU-only build |
| Linux / Docker | CPU-only build |

---

## 3. Estructura del proyecto

```
UAO-desarrollo-de-proyectos-de-ia/
│
├── src/                        ← código fuente principal
│   ├── domain/                 ← conocimiento del negocio (sin frameworks)
│   │   └── categories.py       ← catálogo de 8 categorías + reglas de enrutamiento
│   │
│   ├── inference/              ← capa ML (solo importa de domain/)
│   │   ├── model.py            ← carga DeiT-Tiny, extrae embeddings
│   │   └── classifier.py       ← entrenamiento, serialización y predicción sklearn
│   │
│   └── api/                    ← capa HTTP (solo importa de inference/ y domain/)
│       ├── main.py             ← app FastAPI: registra routers y monta estáticos
│       ├── schemas.py          ← modelos Pydantic (contrato HTTP)
│       ├── routes/
│       │   ├── health.py       ← GET /health
│       │   ├── predict.py      ← POST /api/predict
│       │   └── ui.py           ← vistas HTML y descarga de PDF
│       ├── templates/          ← plantillas Jinja2 (.html)
│       └── static/             ← archivos estáticos (CSS, JS)
│
├── tests/                      ← pruebas automatizadas
│   ├── conftest.py             ← fixtures compartidas (client, sample_image_bytes)
│   ├── test_health.py          ← pruebas de GET /health
│   ├── test_predict.py         ← pruebas de POST /api/predict
│   └── test_inference.py       ← pruebas unitarias de model.py y classifier.py
│
├── scripts/                    ← scripts de entrenamiento y evaluación
│   ├── train.py                ← entrena el clasificador y guarda models/classifier.pkl
│   └── evaluate.py             ← evalúa el modelo y reporta Accuracy y Macro-F1
│
├── data/                       ← dataset de imágenes (no se sube al repo)
│   └── raw/
│       ├── red_conectividad/
│       ├── acceso_contrasenas/
│       ├── correo_office365/
│       ├── impresion_perifericos/
│       ├── aplicacion_errores/
│       ├── hardware_equipo/
│       ├── vpn_remoto/
│       └── otros/
│
├── models/                     ← artefactos del modelo (no se suben al repo)
│   ├── classifier.pkl          ← clasificador entrenado (generado por make train)
│   └── hf_cache/               ← caché local de DeiT-Tiny (descargado automáticamente)
│
├── .github/
│   └── workflows/
│       └── ci.yml              ← pipeline GitHub Actions (lint + tests)
│
├── main.py                     ← entrypoint: levanta el servidor localmente
├── Dockerfile                  ← imagen de producción (python:3.11-slim)
├── pyproject.toml              ← dependencias y configuración del proyecto
├── Makefile                    ← comandos del equipo
└── .pre-commit-config.yaml     ← hooks de calidad (ruff, check-yaml, etc.)
```

---

## 4. Arquitectura y capas

El proyecto sigue una **dirección de dependencia unidireccional** para garantizar alta cohesión y bajo acoplamiento:

```
domain  ←  inference  ←  api
```

| Capa | Qué contiene | Qué NO puede importar |
|---|---|---|
| `domain/` | Catálogo de categorías, threshold 0.40, reglas de enrutamiento | Nada de FastAPI, PyTorch ni sklearn |
| `inference/` | Carga del modelo DeiT-Tiny, clasificador sklearn | Nada de FastAPI |
| `api/` | Rutas HTTP, schemas Pydantic, templates HTML | Puede importar todo lo anterior |

Esto significa que si mañana se cambia FastAPI por otro framework, **`domain/` e `inference/` no se tocan**. Si se cambia el modelo de DeiT-Tiny a otro, **`api/` no se toca**.

---

## 5. Comandos disponibles

Todos los comandos se ejecutan desde la raíz del proyecto.

### Entorno

```bash
make install             # instala todas las dependencias con uv
make pre-commit-install  # activa los hooks de pre-commit en el repo local
```

### Desarrollo

```bash
make start    # levanta el servidor FastAPI en http://localhost:8000 con recarga automática
make test     # corre todos los tests con pytest
make lint     # verifica el código con ruff (sin modificar)
make format   # formatea el código con ruff (modifica los archivos)
```

### Modelo

```bash
make train     # entrena el clasificador con las imágenes de data/raw/ y guarda models/classifier.pkl
make evaluate  # evalúa el modelo y muestra Accuracy, Macro-F1 y matriz de confusión
```

### Docker

```bash
docker build -t triage-ti .          # construye la imagen de producción
docker run -p 8000:8000 triage-ti    # lanza el contenedor en http://localhost:8000
```

---

## 6. Estado actual del desarrollo

### Listo y funcional

| Módulo | Archivo | Descripción |
|---|---|---|
| Dominio | `src/domain/categories.py` | Catálogo de 8 categorías, threshold 0.40, función `get_team()` |
| Modelo | `src/inference/model.py` | Carga DeiT-Tiny con lazy loading, extrae embedding CLS de 192 dimensiones |
| Clasificador | `src/inference/classifier.py` | `train`, `save`, `load`, `predict` con regla de confianza integrada |
| API base | `src/api/main.py` | FastAPI configurada con routers y archivos estáticos |
| Healthcheck | `src/api/routes/health.py` | `GET /health` → `{"status": "ok"}` |
| Entrenamiento | `scripts/train.py` | Pipeline completo: lee `data/raw/`, extrae embeddings, entrena, guarda `.pkl` |
| Test base | `tests/test_health.py` | 1 test pasando |
| CI | `.github/workflows/ci.yml` | Pipeline GitHub Actions activo |

### Pendiente de implementar

| Tarea | Archivo(s) | Bloqueado por |
|---|---|---|
| T02 | `data/raw/{categoria}/` | Recolección del dataset (equipo) |
| T05 | `src/api/schemas.py`, `routes/predict.py` | Nada — puede empezarse ahora |
| T06 | `scripts/evaluate.py` | T02 + `make train` |
| T07 | `routes/ui.py`, `templates/`, `static/` | T05 |
| T09 | `test_predict.py`, `test_inference.py` | T05 |

---

## 7. Flujo de trabajo en equipo

### Para agregar una nueva funcionalidad

1. Crear una rama desde `main`: `git checkout -b feature/nombre-de-la-tarea`
2. Escribir el código en el módulo correspondiente.
3. Correr `make lint` y `make test` antes de hacer commit.
4. Los hooks de pre-commit se ejecutan automáticamente al hacer `git commit`.
5. Abrir un Pull Request hacia `main` — el CI (GitHub Actions) corre lint + tests automáticamente.
6. Si todos los checks pasan, se puede hacer merge.

### Para entrenar el modelo

1. Asegurarse de tener imágenes en `data/raw/{categoria}/` (20–50 por categoría).
2. Correr `make train` — guarda el artefacto en `models/classifier.pkl`.
3. Correr `make evaluate` — imprime Accuracy, Macro-F1 y matriz de confusión.
4. El archivo `models/classifier.pkl` **no se sube al repo** (está en `.gitignore`).

### Dataset — estructura esperada

Cada integrante que recolecte imágenes debe organizarlas así:

```
data/raw/
├── red_conectividad/       ← imágenes de routers, Wi-Fi, errores de red
├── acceso_contrasenas/     ← pantallas de login, MFA, bloqueos
├── correo_office365/       ← Outlook, OWA, errores de sincronización
├── impresion_perifericos/  ← impresoras, spooler, escáneres
├── aplicacion_errores/     ← stack traces, crashes, errores de app
├── hardware_equipo/        ← PC sin encender, pantallas dañadas
├── vpn_remoto/             ← cliente VPN, túneles, RDP
└── otros/                  ← casos que no encajan en las anteriores
```

Formatos aceptados: `.jpg`, `.jpeg`, `.png`, `.webp`
Mínimo recomendado: **20 imágenes por categoría**
Restricción: **ninguna imagen puede contener datos personales (PII)**
