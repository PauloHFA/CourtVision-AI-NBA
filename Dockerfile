FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY migrations ./migrations
COPY data/fixtures ./data/fixtures

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src:/app
EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn apps.api.main:app --host 0.0.0.0 --port 8000"]
