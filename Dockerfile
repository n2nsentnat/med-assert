# med-assert API — Python 3.13, non-root, uvicorn on 0.0.0.0:8000
# Build: docker build -t med-assert:latest .
# Run:  docker run --rm -p 8000:8000 --env-file .env med-assert:latest

FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# lxml wheels usually suffice; add libs only if wheel install fails on your platform
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install ".[specter]"

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
    && mkdir -p /app/med_assert_output \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "med_assert.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
