FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
