FROM python:3.12.11-slim-bookworm
WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/streamlit/*.py ./streamlit/
COPY frontend/.streamlit/config.toml .streamlit/config.toml
ENV STREAMLIT_SERVER_HEADLESS=true PORT=8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/_stcore/health', timeout=4)"
CMD ["sh", "-c", "exec streamlit run streamlit/main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
