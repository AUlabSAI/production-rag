FROM python:3.12-slim

# Pin to the version printed by `uv --version` on your machine, e.g. ghcr.io/astral-sh/uv:0.9.5
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# Non-root user; pre-create directories that will be mounted as volumes
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/chroma_db /cache/huggingface \
    && chown -R app:app /app/chroma_db /cache
USER app

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]