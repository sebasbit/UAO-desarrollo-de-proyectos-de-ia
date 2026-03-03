.PHONY: help install start test lint format pre-commit-install train evaluate

help:
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install           Instala dependencias con uv (cross-platform)"
	@echo "  make start             Levanta el servidor FastAPI con recarga automática"
	@echo "  make test              Ejecuta los tests con pytest"
	@echo "  make lint              Verifica el código con ruff"
	@echo "  make format            Formatea el código con ruff"
	@echo "  make pre-commit-install Instala los hooks de pre-commit en el repo local"
	@echo "  make train             Entrena el clasificador y guarda el .pkl"
	@echo "  make evaluate          Evalúa el modelo y reporta Macro-F1"
	@echo ""

install:
	uv sync --locked

start:
	uv run main.py

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

pre-commit-install:
	uv run pre-commit install

train:
	uv run scripts/train.py

evaluate:
	uv run scripts/evaluate.py
