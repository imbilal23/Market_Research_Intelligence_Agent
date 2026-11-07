FROM python:3.12.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY backend/requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt
COPY backend/app/ ./app/
ENV DATA_DIR=/data HOST=0.0.0.0 PORT=8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
