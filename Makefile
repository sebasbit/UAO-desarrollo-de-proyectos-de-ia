.PHONY: install test

help:
	@echo Commandos
	@echo     make install : Sincroniza el entorno usando uv
	@echo     make test    : Ejecuta los tests con Pytest

install:
	uv sync --locked

start:
	uv run main.py

test:
	uv run pytest
